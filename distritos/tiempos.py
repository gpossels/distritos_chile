"""Tiempos de viaje entre unidades y centros urbanos sobre la red vial.

Cada unidad (y cada centro urbano) se ubica en el vértice vial más cercano a su
centroide ponderado por población. El tiempo de acceso desde el centroide al
vértice se estima en línea recta con un factor de desvío, a 20 km/h.

Resultados:
  * tiempo de cada centro urbano a cada unidad (matriz centros x unidades);
  * tiempo entre cada par de unidades vecinas;
  * para cada centro urbano, el centro mayor más cercano (jerarquía urbana);
  * para cada unidad, el centro urbano más cercano y el centro grande más cercano.
"""
import multiprocessing as mp

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .red_vial import RedVial, _a_xyz

VELOCIDAD_ACCESO_KMH = 20
FACTOR_DESVIO = 1.3
# Componentes de la red con al menos estos vértices se consideran "red troncal".
TAMANO_MIN_COMPONENTE = 500
# Se prefiere un vértice de la red troncal aunque esté hasta esta distancia (m)
# más lejos que el vértice más cercano (evita caer en tramos viales aislados).
TOLERANCIA_TRONCAL_M = 2000
# Unidades sin conexión vial ni marítima: línea recta a 15 km/h más 24 h.
VELOCIDAD_SIN_CONEXION_KMH = 15
PENALIDAD_SIN_CONEXION_MIN = 24 * 60

_RED: RedVial | None = None


def _minutos_acceso(dist_m):
    return dist_m * FACTOR_DESVIO / 1000 / VELOCIDAD_ACCESO_KMH * 60


def ubicar_en_red(red: RedVial, lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """Vértice asignado y minutos de acceso para cada punto."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    v_any, d_any = red.vertice_mas_cercano(lon, lat)
    tam = np.bincount(red.componente)
    troncal = np.where(tam[red.componente] >= TAMANO_MIN_COMPONENTE)[0]
    arbol = cKDTree(_a_xyz(red.lon[troncal], red.lat[troncal]))
    d_tr, k_tr = arbol.query(_a_xyz(lon, lat))
    usar_troncal = d_tr <= d_any + TOLERANCIA_TRONCAL_M
    v = np.where(usar_troncal, troncal[k_tr], v_any)
    d = np.where(usar_troncal, d_tr, d_any)
    return v, _minutos_acceso(d)


def _a_lonlat(x, y, crs):
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return tr.transform(np.asarray(x), np.asarray(y))


def _init(red):
    global _RED
    _RED = red


def _dijkstra_lote(args):
    """Dijkstra desde `fuentes`; devuelve solo d[filas, columnas] (o d[:, columnas])."""
    fuentes, limite, filas, columnas = args
    d = dijkstra(_RED.grafo, directed=False, indices=fuentes, limit=limite)
    return d[:, columnas] if filas is None else d[filas, columnas]


def _distancia_recta_km(lon1, lat1, lon2, lat2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def calcular_tiempos(red: RedVial, unidades: pd.DataFrame, ady: pd.DataFrame,
                     centros: pd.DataFrame, crs: str, umbral_grande: int,
                     procesos: int | None = None) -> dict:
    procesos = procesos or max(1, mp.cpu_count())

    # --- Ubicación de unidades en la red.
    u_lon, u_lat = _a_lonlat(unidades["cx"].values, unidades["cy"].values, crs)
    u_v, u_acc = ubicar_en_red(red, u_lon, u_lat)

    # --- Punto de cada centro urbano: centroide ponderado de sus zonas.
    z = unidades[unidades["centro_urbano"].notna()].copy()
    z["wx"], z["wy"] = z["cx"] * z["n_per"], z["cy"] * z["n_per"]
    g = z.groupby("centro_urbano")[["wx", "wy", "n_per"]].sum()
    g = g[g["n_per"] > 0]
    c = centros.set_index("centro_urbano").loc[g.index].copy()
    c["x"], c["y"] = g["wx"] / g["n_per"], g["wy"] / g["n_per"]
    c_lon, c_lat = _a_lonlat(c["x"].values, c["y"].values, crs)
    c_v, c_acc = ubicar_en_red(red, c_lon, c_lat)
    c = c.reset_index()
    c["lon"], c["lat"] = c_lon, c_lat

    # --- Matriz centros x unidades (minutos).
    lotes = [(c_v[i:i + 8], np.inf, None, u_v) for i in range(0, len(c_v), 8)]
    T = np.empty((len(c), len(unidades)), dtype=np.float32)
    with mp.get_context("fork").Pool(procesos, initializer=_init, initargs=(red,)) as pool:
        fila = 0
        for d in pool.imap(_dijkstra_lote, lotes):
            T[fila:fila + len(d)] = d
            fila += len(d)
    T += c_acc[:, None].astype(np.float32) + u_acc[None, :].astype(np.float32)

    # Sin conexión (islas sin transbordador): línea recta + penalidad.
    sin_con = ~np.isfinite(T)
    if sin_con.any():
        ii, jj = np.nonzero(sin_con)
        km = _distancia_recta_km(c_lon[ii], c_lat[ii], u_lon[jj], u_lat[jj])
        T[ii, jj] = km / VELOCIDAD_SIN_CONEXION_KMH * 60 + PENALIDAD_SIN_CONEXION_MIN

    # --- Tiempos entre unidades vecinas: primero con límite corto, luego sin límite.
    pos = pd.Series(np.arange(len(unidades)), index=unidades["id_unidad"].values)
    io, idd = pos[ady["origen"]].values, pos[ady["destino"]].values
    t_ady = np.full(len(ady), np.inf)
    pendientes = np.arange(len(ady))
    with mp.get_context("fork").Pool(procesos, initializer=_init, initargs=(red,)) as pool:
        for limite in (120.0, np.inf):
            if len(pendientes) == 0:
                break
            fuentes = np.unique(u_v[io[pendientes]])
            lotes, pares = [], []
            for i in range(0, len(fuentes), 32):
                lote = fuentes[i:i + 32]
                # Pares pendientes cuyo origen está en este lote.
                p = pendientes[np.isin(u_v[io[pendientes]], lote)]
                fila_local = pd.Series(np.arange(len(lote)), index=lote)
                lotes.append((lote, limite, fila_local[u_v[io[p]]].values, u_v[idd[p]]))
                pares.append(p)
            for p, t in zip(pares, pool.imap(_dijkstra_lote, lotes)):
                t_ady[p] = t
            pendientes = pendientes[~np.isfinite(t_ady[pendientes])]
    t_ady = t_ady + u_acc[io] + u_acc[idd]
    sin_con_ady = ~np.isfinite(t_ady)
    if sin_con_ady.any():
        km = _distancia_recta_km(u_lon[io], u_lat[io], u_lon[idd], u_lat[idd])[sin_con_ady]
        t_ady[sin_con_ady] = km / VELOCIDAD_SIN_CONEXION_KMH * 60 + PENALIDAD_SIN_CONEXION_MIN
    ady = ady.copy()
    ady["minutos"] = t_ady.round(1)
    ady["sin_conexion"] = sin_con_ady

    # --- Jerarquía urbana: centro mayor más cercano de cada centro.
    # Tiempo centro -> centro: se mide hasta la unidad más poblada del otro centro.
    principal = (z.sort_values("n_per", ascending=False)
                 .drop_duplicates("centro_urbano").set_index("centro_urbano")["id_unidad"])
    col_c = pos[principal.loc[c["centro_urbano"]].values].values
    TC = T[:, col_c]  # TC[i, j]: del centro i a la unidad principal del centro j
    pob = c["n_per"].values
    mayor, t_mayor = [], []
    for j in range(len(c)):
        cand = np.where(pob > pob[j])[0]
        if len(cand) == 0:
            mayor.append(None)
            t_mayor.append(np.nan)
            continue
        k = cand[np.argmin(TC[cand, j])]
        mayor.append(c.loc[k, "centro_urbano"])
        t_mayor.append(float(TC[k, j]))
    c["centro_mayor_cercano"] = mayor
    c["minutos_a_mayor"] = np.round(t_mayor, 1)

    # --- Centro más cercano y centro grande más cercano de cada unidad.
    grande = pob >= umbral_grande
    k_any = T.argmin(axis=0)
    Tg = T[grande]
    k_g = Tg.argmin(axis=0)
    nombres = c["centro_urbano"].values
    u = pd.DataFrame({
        "id_unidad": unidades["id_unidad"].values,
        "vertice": u_v, "minutos_acceso": u_acc.round(1),
        "centro_cercano": nombres[k_any],
        "minutos_centro_cercano": T[k_any, np.arange(T.shape[1])].round(1),
        "centro_grande_cercano": nombres[grande][k_g],
        "minutos_centro_grande": Tg[k_g, np.arange(T.shape[1])].round(1),
    })
    return {"matriz": T, "centros": c, "adyacencia": ady, "unidades": u}
