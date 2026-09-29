"""Construcción de las unidades base a partir de la cartografía del Censo 2024.

Unidades base:
  * zona censal urbana (prefijo Z), ~3.200 habitantes en promedio;
  * localidad rural (prefijo L), ~250 habitantes en promedio;
  * territorio sin población (prefijo S): partes de cada comuna que no cubre
    ninguna zona ni localidad (campos de hielo, desierto, islas deshabitadas).
    Se incluyen para que el mapa no tenga huecos y los distritos sean contiguos.

Cada unidad lleva su centroide ponderado por población (calculado con las
manzanas o entidades que contiene), que sirve de origen para los tiempos de viaje.
"""
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from .config import GDB_CENSO

# Superficie mínima (km²) para conservar un trozo de territorio sin población.
SUPERFICIE_MINIMA_HUECO_KM2 = 0.05

COLUMNAS_COMUNES = ["CUT", "COD_REGION", "REGION", "PROVINCIA", "COMUNA"]


def _leer(capa: str, columnas: list[str], geometria: bool = True, crs: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gdf = pyogrio.read_dataframe(GDB_CENSO, layer=capa, columns=columnas,
                                     read_geometry=geometria)
    if geometria and crs:
        gdf = gdf.to_crs(crs)
        gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    return gdf


def _centroides_ponderados(puntos: gpd.GeoDataFrame, clave: str) -> pd.DataFrame:
    """Centroide ponderado por población de los puntos agrupados por `clave`."""
    p = puntos[puntos["n_per"] > 0]
    df = pd.DataFrame({clave: p[clave].values, "w": p["n_per"].values,
                       "x": p.geometry.x.values, "y": p.geometry.y.values})
    df["wx"], df["wy"] = df.w * df.x, df.w * df.y
    g = df.groupby(clave)[["w", "wx", "wy"]].sum()
    return pd.DataFrame({"cx": g.wx / g.w, "cy": g.wy / g.w})


def construir_unidades(crs: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Devuelve (unidades, centros_urbanos)."""
    # --- Centros urbanos: conurbación INE o, si no tiene, la entidad urbana.
    lim = _leer("Limite_Urbano_CPV24",
                ["ID_ENTIDAD", "ENTIDAD", "COMUNA", "CATEGORIA", "CONURBACION", "n_per"],
                geometria=False)
    conurb = lim["CONURBACION"].fillna("").str.strip()
    lim["centro_urbano"] = conurb.where(conurb != "", lim["ENTIDAD"])
    centros = (lim.groupby("centro_urbano")
               .agg(n_per=("n_per", "sum"),
                    comunas=("COMUNA", lambda s: ", ".join(sorted(set(s)))),
                    n_entidades=("ID_ENTIDAD", "count"))
               .sort_values("n_per", ascending=False).reset_index())
    centro_de_entidad = lim.set_index("ID_ENTIDAD")["centro_urbano"]

    # --- Zonas urbanas
    zonas = _leer("Zonal_CPV24", COLUMNAS_COMUNES + ["ID_ZONA", "ID_ENTIDAD", "ENTIDAD", "n_per"],
                  crs=crs)
    zonas["id_unidad"] = "Z" + zonas["ID_ZONA"].astype("int64").astype(str)
    zonas["tipo"] = "zona_urbana"
    zonas["nombre"] = zonas["ENTIDAD"]
    zonas["centro_urbano"] = zonas["ID_ENTIDAD"].map(centro_de_entidad)

    manz = _leer("Manzanas_CPV24", ["ID_ZONA", "n_per"], crs=crs)
    manz["geometry"] = manz.geometry.representative_point()
    cz = _centroides_ponderados(manz, "ID_ZONA")
    zonas = zonas.join(cz, on="ID_ZONA")

    # --- Localidades rurales
    locs = _leer("Localidades_CPV24", COLUMNAS_COMUNES + ["ID_LOCALIDAD", "LOCALIDAD", "n_per"],
                 crs=crs)
    locs["id_unidad"] = "L" + locs["ID_LOCALIDAD"].astype("int64").astype(str)
    locs["tipo"] = "localidad_rural"
    locs["nombre"] = locs["LOCALIDAD"]
    locs["centro_urbano"] = None

    ent = _leer("Entidades_CPV24", ["ID_LOCALIDAD", "n_per"], crs=crs)
    ent["geometry"] = ent.geometry.representative_point()
    # Las aldeas (manzanas rurales) también pertenecen a una localidad.
    ald = _leer("Manzanas_CPV24", ["ID_LOCALIDAD", "AREA_C", "n_per"], crs=crs)
    ald = ald[ald["AREA_C"] == "RURAL"].drop(columns="AREA_C")
    ald["geometry"] = ald.geometry.representative_point()
    cl = _centroides_ponderados(pd.concat([ent, ald], ignore_index=True), "ID_LOCALIDAD")
    locs = locs.join(cl, on="ID_LOCALIDAD")

    cols = ["id_unidad", "tipo", "nombre", "centro_urbano", *COLUMNAS_COMUNES,
            "n_per", "cx", "cy", "geometry"]
    unidades = pd.concat([zonas[cols], locs[cols]], ignore_index=True)

    # --- Territorio sin población (huecos dentro de cada comuna)
    comunas = _leer("Comunal_CPV24", COLUMNAS_COMUNES, crs=crs)
    huecos = []
    for _, c in comunas.iterrows():
        dentro = unidades.geometry[unidades["CUT"] == c["CUT"]].values
        resto = shapely.difference(c.geometry, shapely.union_all(dentro))
        partes = shapely.get_parts(shapely.make_valid(resto))
        partes = [p for p in partes
                  if p.geom_type in ("Polygon", "MultiPolygon")
                  and p.area / 1e6 >= SUPERFICIE_MINIMA_HUECO_KM2]
        for i, p in enumerate(partes):
            huecos.append({"id_unidad": f"S{c['CUT']}_{i}", "tipo": "sin_poblacion",
                           "nombre": None, "centro_urbano": None,
                           **{k: c[k] for k in COLUMNAS_COMUNES},
                           "n_per": 0.0, "cx": np.nan, "cy": np.nan, "geometry": p})
    if huecos:
        unidades = pd.concat([unidades, gpd.GeoDataFrame(huecos, crs=crs)], ignore_index=True)

    unidades = gpd.GeoDataFrame(unidades, geometry="geometry", crs=crs)
    unidades["n_per"] = unidades["n_per"].fillna(0).astype("int64")
    unidades["area_km2"] = unidades.area / 1e6
    # Unidades sin población: el centroide es un punto interior del polígono.
    sin_c = unidades["cx"].isna()
    rp = unidades.geometry[sin_c].representative_point()
    unidades.loc[sin_c, "cx"] = rp.x.values
    unidades.loc[sin_c, "cy"] = rp.y.values
    return unidades, centros
