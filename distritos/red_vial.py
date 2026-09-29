"""Red vial y de transbordadores a partir de OpenStreetMap.

Construye un grafo no dirigido cuyos vértices son las intersecciones y extremos
de las vías transitables en vehículo, y cuyas aristas llevan el tiempo de viaje
en minutos. Los transbordadores se agregan como aristas cuyo tiempo es la
travesía más la espera media (la mitad del intervalo entre salidas).

Para distritar basta con tiempos aproximados, así que el grafo se trata como no
dirigido (se ignoran los sentidos únicos).
"""
import re

import numpy as np
import osmium
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# Velocidad (km/h) por tipo de vía.
VELOCIDADES = {
    "motorway": 100, "motorway_link": 50,
    "trunk": 80, "trunk_link": 50,
    "primary": 70, "primary_link": 45,
    "secondary": 60, "secondary_link": 40,
    "tertiary": 50, "tertiary_link": 35,
    "unclassified": 40, "road": 30,
    "residential": 30, "living_street": 15,
    "service": 20, "track": 25,
}
# Tope de velocidad en caminos de ripio o tierra.
VELOCIDAD_MAX_SIN_PAVIMENTO = 40
SUPERFICIES_SIN_PAVIMENTO = {
    "unpaved", "gravel", "fine_gravel", "compacted", "dirt", "earth", "ground",
    "sand", "mud", "grass", "pebblestone",
}
# Velocidad supuesta (km/h) de un transbordador sin duración en OSM.
VELOCIDAD_TRANSBORDADOR = 15
# Distancia máxima (m) para enlazar un terminal de transbordador con la red vial.
DISTANCIA_MAX_TERMINAL_M = 3000

RADIO_TIERRA_M = 6371008.8


def _duracion_horas(valor: str | None) -> float | None:
    """Convierte 'HH:MM', 'H:MM:SS' o minutos ('45') a horas."""
    if not valor:
        return None
    valor = valor.strip()
    m = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", valor)
    if m:
        h, mi, s = int(m[1]), int(m[2]), int(m[3] or 0)
        return h + mi / 60 + s / 3600
    if re.fullmatch(r"\d+(\.\d+)?", valor):
        return float(valor) / 60
    return None


def _largo_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Largo (m) de cada tramo consecutivo de una polilínea (haversine)."""
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = np.diff(la), np.diff(lo)
    a = np.sin(dla / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(dlo / 2) ** 2
    return 2 * RADIO_TIERRA_M * np.arcsin(np.sqrt(a))


def _leer_osm(pbf: str):
    """Lee vías transitables y rutas de transbordador del archivo OSM."""
    vias = []          # (ids de nodos, lon, lat, velocidad)
    transbordadores = []
    fp = (osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
          .with_locations()
          .with_filter(osmium.filter.KeyFilter("highway", "route")))
    for w in fp:
        if not w.is_way():
            continue
        t = w.tags
        es_ferry = t.get("route") == "ferry"
        tipo = t.get("highway")
        if not es_ferry and tipo not in VELOCIDADES:
            continue
        if not es_ferry and (t.get("access") in ("private", "no")
                             or t.get("motor_vehicle") == "no"
                             or t.get("area") == "yes"):
            continue
        try:
            nodos = [(n.ref, n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            nodos = [(n.ref, n.lon, n.lat) for n in w.nodes if n.location.valid()]
        if len(nodos) < 2:
            continue
        ids = np.array([n[0] for n in nodos], dtype=np.int64)
        lon = np.array([n[1] for n in nodos])
        lat = np.array([n[2] for n in nodos])
        if es_ferry:
            transbordadores.append({
                "osm_id": w.id, "nombre": t.get("name", ""), "ids": ids[[0, -1]],
                "duracion_osm": t.get("duration", ""),
                "intervalo_osm": t.get("interval", ""),
                "lon": lon, "lat": lat,
            })
        else:
            v = VELOCIDADES[tipo]
            if t.get("surface") in SUPERFICIES_SIN_PAVIMENTO or tipo == "track":
                v = min(v, VELOCIDAD_MAX_SIN_PAVIMENTO)
            vias.append((ids, lon, lat, v))
    return vias, transbordadores


def _intervalo_por_defecto(duracion_h: float, par: dict) -> float:
    """Intervalo entre salidas (h) cuando OSM no lo informa, según la travesía."""
    d = par["intervalo_por_defecto_horas"]
    if duracion_h <= 1.5:
        return d["corto"]
    if duracion_h <= 8:
        return d["medio"]
    return d["largo"]


class RedVial:
    """Grafo de tiempos de viaje (minutos) entre vértices de la red."""

    def __init__(self, grafo: sp.csr_matrix, lon: np.ndarray, lat: np.ndarray,
                 transbordadores: pd.DataFrame):
        self.grafo = grafo
        self.lon, self.lat = lon, lat
        self.transbordadores = transbordadores
        self._xyz = _a_xyz(lon, lat)
        self._arbol = cKDTree(self._xyz)
        n, etiquetas = connected_components(grafo, directed=False)
        self.componente = etiquetas
        tam = np.bincount(etiquetas)
        self.componente_principal = int(np.argmax(tam))

    def vertice_mas_cercano(self, lon, lat):
        """Vértice de la red más cercano a cada punto y distancia (m)."""
        d, i = self._arbol.query(_a_xyz(np.asarray(lon), np.asarray(lat)))
        return i, d

    def guardar(self, ruta):
        np.savez_compressed(ruta, data=self.grafo.data, indices=self.grafo.indices,
                            indptr=self.grafo.indptr, shape=self.grafo.shape,
                            lon=self.lon, lat=self.lat)

    @classmethod
    def cargar(cls, ruta, transbordadores: pd.DataFrame | None = None):
        z = np.load(ruta)
        g = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return cls(g, z["lon"], z["lat"], transbordadores)


def _a_xyz(lon, lat):
    """Coordenadas cartesianas (m) sobre la esfera, para búsquedas de vecinos."""
    la, lo = np.radians(lat), np.radians(lon)
    return RADIO_TIERRA_M * np.column_stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def construir_red(pbf: str, par_transbordadores: dict,
                  ajustes_transbordadores: dict | None = None) -> RedVial:
    vias, ferries = _leer_osm(pbf)

    # Vértices: extremos de cada vía y nodos compartidos por más de una vía.
    todos = np.concatenate([v[0] for v in vias])
    unicos, cuenta = np.unique(todos, return_counts=True)
    compartidos = set(unicos[cuenta > 1].tolist())
    del todos, unicos, cuenta

    indice: dict[int, int] = {}
    v_lon, v_lat = [], []

    def vertice(osm_id, lon, lat):
        k = indice.get(osm_id)
        if k is None:
            k = indice[osm_id] = len(v_lon)
            v_lon.append(lon)
            v_lat.append(lat)
        return k

    orig, dest, minutos = [], [], []
    for ids, lon, lat, vel in vias:
        seg = _largo_m(lon, lat)
        acum = np.concatenate([[0.0], np.cumsum(seg)])
        corte = [0] + [k for k in range(1, len(ids) - 1) if int(ids[k]) in compartidos] + [len(ids) - 1]
        for a, b in zip(corte[:-1], corte[1:]):
            va = vertice(int(ids[a]), lon[a], lat[a])
            vb = vertice(int(ids[b]), lon[b], lat[b])
            if va == vb:
                continue
            orig.append(va)
            dest.append(vb)
            minutos.append((acum[b] - acum[a]) / 1000 / vel * 60)

    # Transbordadores: cada tramo une sus nodos extremos. Los tramos consecutivos
    # de una misma ruta comparten nodo y quedan encadenados. Los terminales que
    # no están sobre la red vial se enlazan con el vértice vial más cercano
    # (acceso a 20 km/h); los que no tienen camino cerca quedan como paradas
    # (caletas sin acceso vial).
    n_vial = len(v_lon)
    arbol = cKDTree(_a_xyz(np.array(v_lon), np.array(v_lat)))
    ajustes = ajustes_transbordadores or {}
    filas, terminales = [], set()
    for f in ferries:
        largo_km = _largo_m(f["lon"], f["lat"]).sum() / 1000
        dur = _duracion_horas(f["duracion_osm"]) or largo_km / VELOCIDAD_TRANSBORDADOR
        intervalo = _duracion_horas(f["intervalo_osm"])
        fuente_intervalo = "osm" if intervalo else "por_defecto"
        if not intervalo:
            intervalo = _intervalo_por_defecto(dur, par_transbordadores)
        aj = ajustes.get(f["osm_id"]) or {}
        if "intervalo_horas" in aj:
            intervalo, fuente_intervalo = float(aj["intervalo_horas"]), "ajuste"
        if "duracion_horas" in aj:
            dur = float(aj["duracion_horas"])
        espera = intervalo / 2
        excluido = bool(aj.get("excluir", False))
        if not excluido:
            va = vertice(int(f["ids"][0]), f["lon"][0], f["lat"][0])
            vb = vertice(int(f["ids"][1]), f["lon"][-1], f["lat"][-1])
            if va != vb:
                orig.append(va)
                dest.append(vb)
                minutos.append((dur + espera) * 60)
                terminales.update(k for k in (va, vb) if k >= n_vial)
        filas.append({"osm_id": f["osm_id"], "nombre": f["nombre"],
                      "largo_km": round(largo_km, 1), "travesia_h": round(dur, 2),
                      "intervalo_h": intervalo, "fuente_intervalo": fuente_intervalo,
                      "espera_h": round(espera, 2), "excluido": excluido})

    v_lon, v_lat = np.array(v_lon), np.array(v_lat)
    if terminales:
        t = np.array(sorted(terminales))
        d, k = arbol.query(_a_xyz(v_lon[t], v_lat[t]))
        cerca = d <= DISTANCIA_MAX_TERMINAL_M
        orig.extend(t[cerca].tolist())
        dest.extend(k[cerca].tolist())
        minutos.extend((d[cerca] / 1000 / 20 * 60).tolist())

    n = len(v_lon)
    o, d_, w = np.array(orig), np.array(dest), np.array(minutos)
    w = np.maximum(w, 1e-3)  # scipy trata el peso 0 como ausencia de arista
    g = _minimo_por_par(o, d_, w, n)
    return RedVial(g, v_lon, v_lat, pd.DataFrame(filas))


def _minimo_por_par(o, d, w, n) -> sp.csr_matrix:
    """Matriz simétrica; si hay aristas repetidas entre dos vértices, conserva la más rápida."""
    a, b = np.minimum(o, d), np.maximum(o, d)
    df = pd.DataFrame({"a": a, "b": b, "w": w}).groupby(["a", "b"], sort=False).w.min().reset_index()
    fil = np.concatenate([df.a.values, df.b.values])
    col = np.concatenate([df.b.values, df.a.values])
    val = np.concatenate([df.w.values, df.w.values])
    return sp.csr_matrix((val, (fil, col)), shape=(n, n))
