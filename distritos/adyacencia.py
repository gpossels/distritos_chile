"""Grafo de adyacencia entre unidades (comparten borde de longitud positiva).

Las islas y otras unidades aisladas quedan como componentes separados; más
adelante se conectan mediante las rutas de transbordador de OpenStreetMap.
"""
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely

# Tolerancia (m) para considerar vecinas dos unidades separadas por un sliver.
TOLERANCIA_M = 5.0


def calcular_adyacencia(unidades: gpd.GeoDataFrame) -> pd.DataFrame:
    geoms = unidades.geometry.values
    arbol = shapely.STRtree(geoms)
    i, j = arbol.query(geoms, predicate="dwithin", distance=TOLERANCIA_M)
    m = i < j
    i, j = i[m], j[m]
    # Longitud aproximada del borde compartido: borde de A dentro del buffer de B.
    bordes = shapely.boundary(geoms)
    zonas = shapely.buffer(geoms, TOLERANCIA_M, quad_segs=2)
    largo = np.empty(len(i))
    for k in range(0, len(i), 5000):
        a, b = i[k:k + 5000], j[k:k + 5000]
        largo[k:k + 5000] = shapely.length(shapely.intersection(bordes[a], zonas[b]))
    ids = unidades["id_unidad"].values
    ady = pd.DataFrame({"origen": ids[i], "destino": ids[j], "borde_m": largo})
    # Descarta contactos en un punto (esquinas).
    return ady[ady["borde_m"] > 2 * TOLERANCIA_M].reset_index(drop=True)


def componentes(unidades: gpd.GeoDataFrame, ady: pd.DataFrame) -> pd.Series:
    g = nx.Graph()
    g.add_nodes_from(unidades["id_unidad"])
    g.add_edges_from(zip(ady["origen"], ady["destino"]))
    comp = {}
    for n, c in enumerate(sorted(nx.connected_components(g), key=len, reverse=True)):
        for u in c:
            comp[u] = n
    return unidades["id_unidad"].map(comp)
