"""Ríos principales y su coincidencia con los bordes entre unidades.

Para cada par de unidades vecinas se mide cuántos metros de su borde común
corren a lo largo de un río principal. El optimizador penaliza los límites de
distrito que pasan por esos bordes.
"""
import geopandas as gpd
import networkx as nx
import numpy as np
import osmium
import pandas as pd
import shapely

from .adyacencia import TOLERANCIA_M


def leer_rios(pbf, principales: list, largo_minimo_km: float, crs: str) -> gpd.GeoDataFrame:
    """Tramos de OSM de los ríos principales, agrupados por nombre y continuidad.

    `principales` admite nombres o dicts {nombre, largo_minimo_km}.
    """
    minimo = {}
    for r in principales:
        if isinstance(r, dict):
            minimo[r["nombre"]] = r.get("largo_minimo_km", largo_minimo_km)
        else:
            minimo[r] = largo_minimo_km
    buscados = set(minimo)
    filas = []
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.TagFilter(("waterway", "river"))))
    for w in fp:
        if not w.is_way() or w.tags.get("name") not in buscados:
            continue
        try:
            pts = [(n.ref, n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            continue
        if len(pts) < 2:
            continue
        filas.append({"nombre": w.tags["name"], "a": pts[0][0], "b": pts[-1][0],
                      "geometry": shapely.LineString([(p[1], p[2]) for p in pts])})
    tramos = gpd.GeoDataFrame(filas, crs="EPSG:4326").to_crs(crs)

    # Agrupa tramos del mismo nombre que se tocan; descarta grupos cortos (homónimos).
    partes = []
    for nombre, sub in tramos.groupby("nombre"):
        g = nx.Graph()
        for i, r in sub.iterrows():
            g.add_edge(("n", r.a), ("t", i))
            g.add_edge(("n", r.b), ("t", i))
        for comp in nx.connected_components(g):
            idx = [k for tipo, k in comp if tipo == "t"]
            geom = shapely.line_merge(shapely.union_all(sub.loc[idx].geometry.values))
            if geom.length / 1000 >= minimo[nombre]:
                partes.append({"nombre": nombre, "largo_km": round(geom.length / 1000, 1),
                               "geometry": geom})
    return gpd.GeoDataFrame(partes, crs=crs)


def bordes_sobre_rios(unidades: gpd.GeoDataFrame, ady: pd.DataFrame,
                      rios: gpd.GeoDataFrame, distancia_m: float) -> pd.Series:
    """Metros del borde común de cada par vecino que corren junto a un río principal."""
    franja = shapely.union_all(shapely.buffer(rios.geometry.values, distancia_m))
    geom = pd.Series(unidades.geometry.values, index=unidades["id_unidad"].values)
    ga, gb = geom[ady["origen"]].values, geom[ady["destino"]].values
    # Solo se examinan los pares cuyo borde puede tocar la franja del río.
    cerca = shapely.intersects(ga, franja) & shapely.intersects(gb, franja)
    largo = np.zeros(len(ady))
    idx = np.nonzero(cerca)[0]
    shapely.prepare(franja)
    for k in range(0, len(idx), 2000):
        i = idx[k:k + 2000]
        comun = shapely.intersection(shapely.boundary(ga[i]),
                                     shapely.buffer(gb[i], TOLERANCIA_M, quad_segs=2))
        largo[i] = shapely.length(shapely.intersection(comun, franja))
    return pd.Series(largo.round(0), index=ady.index)
