"""Optimizador de distritos.

1. Grafo de unidades: vecinas por borde común, más enlaces "virtuales" que unen
   islas y otros trozos aislados con la unidad más cercana de otro trozo.
2. Plan inicial: semillas (k-medias ponderadas por población) — los centros
   urbanos grandes reciben sus propias semillas — y crecimiento por tiempo de
   viaje con tope de población.
3. Búsqueda local (recocido simulado): mueve unidades de borde entre distritos
   vecinos, sin romper la contigüidad, minimizando el puntaje ponderado.

Puntaje de un plan (menor es mejor), con los pesos de parametros.yaml:
  equilibrio_poblacion  ((P - objetivo) / 10.000)² por distrito, más una
                        penalidad grande fuera de [mínimo, máximo]
  compacidad            tiempo medio (min / 30) de la población al centro
                        principal del distrito, más el largo del borde (km / 100)
  centralidad_centro    tiempo cuadrático medio (min / 30) al centro principal
  tiempo_viaje          población (/ 10.000) de pueblos separados del distrito
                        de su ciudad grande más cercana
  evitar_rios           km de límite de distrito que siguen un río principal / 10
  no_dividir_comunas    número de trozos adicionales de cada comuna
"""
import heapq
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

PENALIDAD_FUERA_RANGO = 100.0


# ---------------------------------------------------------------------------
# Datos del problema
# ---------------------------------------------------------------------------
class Problema:
    """Unidades a distritar, su grafo y los datos para el puntaje."""

    def __init__(self, unidades: pd.DataFrame, ady: pd.DataFrame, T: np.ndarray,
                 centros: pd.DataFrame, par: dict, minutos_pueblo_ciudad: float = 45):
        self.u = unidades.reset_index(drop=True)
        n = len(self.u)
        self.n = n
        self.pob = self.u["n_per"].to_numpy(dtype=np.int64)
        self.xy = self.u[["cx", "cy"]].to_numpy()
        pos = pd.Series(np.arange(n), index=self.u["id_unidad"].values)

        # Aristas reales (ambas unidades dentro del problema).
        a = ady[ady["origen"].isin(pos.index) & ady["destino"].isin(pos.index)]
        ea, eb = pos[a["origen"]].to_numpy(), pos[a["destino"]].to_numpy()
        minutos = a["minutos"].to_numpy(dtype=float)
        rio_km = a["borde_rio_m"].to_numpy(dtype=float) / 1000
        borde_km = a["borde_m"].to_numpy(dtype=float) / 1000
        # Enlaces virtuales entre trozos desconectados.
        va, vb, vmin = _enlaces_virtuales(n, ea, eb, self.xy)
        self.ea = np.concatenate([ea, va])
        self.eb = np.concatenate([eb, vb])
        self.e_min = np.concatenate([minutos, vmin])
        self.e_rio = np.concatenate([rio_km, np.zeros(len(va))])
        self.e_borde = np.concatenate([borde_km, np.zeros(len(va))])
        self.n_virtuales = len(va)
        # Listas de vecinos: (vecino, índice de arista).
        self.vecinos: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for k, (i, j) in enumerate(zip(self.ea, self.eb)):
            self.vecinos[i].append((int(j), k))
            self.vecinos[j].append((int(i), k))

        # Comunas.
        self.comuna = pd.factorize(self.u["CUT"])[0]

        # Centros urbanos: fila de la matriz T de cada centro.
        self.T = T  # centros x unidades (ya restringida a estas unidades)
        self.centros = centros.reset_index(drop=True)
        fila_centro = pd.Series(np.arange(len(centros)), index=centros["centro_urbano"].values)
        cu = self.u["centro_urbano"]
        self.centro_de = np.where(cu.notna(), cu.map(fila_centro).fillna(-1), -1).astype(int)
        self.pob_centro = self.centros["n_per"].to_numpy()

        # Ciudades grandes y pueblos que deberían ir con ellas.
        umbral = par["centros_urbanos"]["umbral_distrito_propio"]
        self.es_grande = self.pob_centro >= umbral
        grande_cercana = self.u["centro_grande_cercano"].map(fila_centro).to_numpy()
        minutos_g = self.u["minutos_centro_grande"].to_numpy()
        propia = np.where(self.centro_de >= 0, self.es_grande[np.maximum(self.centro_de, 0)], False)
        # Unidad de ciudad grande: la ciudad a la que pertenece (o -1).
        self.ciudad = np.where(propia, self.centro_de, -1)
        # Objetivo: ciudad grande cercana para unidades fuera de ella.
        self.objetivo = np.where(~propia & (minutos_g <= minutos_pueblo_ciudad),
                                 grande_cercana, -1).astype(int)
        self.centro_cercano = self.u["centro_cercano"].map(fila_centro).to_numpy().astype(int)

        p = par["poblacion"]
        self.objetivo_pob, self.min_pob, self.max_pob = p["objetivo"], p["minimo"], p["maximo"]
        self.w = par["pesos"]


def _enlaces_virtuales(n, ea, eb, xy):
    """Une trozos desconectados (Borůvka sobre distancia entre centroides)."""
    padre = np.arange(n)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i, j in zip(ea, eb):
        ri, rj = raiz(i), raiz(j)
        if ri != rj:
            padre[ri] = rj
    arbol = cKDTree(xy)
    va, vb, vmin = [], [], []
    while True:
        comp = np.array([raiz(i) for i in range(n)])
        grupos = np.unique(comp)
        if len(grupos) == 1:
            break
        mejor = {}
        for g in grupos:
            miembros = np.nonzero(comp == g)[0]
            k = min(64, n)
            while True:
                d, idx = arbol.query(xy[miembros], k=k)
                d, idx = np.atleast_2d(d), np.atleast_2d(idx)
                fuera = comp[idx] != g
                if fuera.any() or k >= n:
                    break
                k = min(k * 4, n)
            dd = np.where(fuera, d, np.inf)
            f, c = np.unravel_index(np.argmin(dd), dd.shape)
            mejor[g] = (dd[f, c], miembros[f], idx[f, c])
        for g, (d, i, j) in mejor.items():
            ri, rj = raiz(i), raiz(j)
            if ri != rj:
                padre[ri] = rj
                va.append(i)
                vb.append(j)
                vmin.append(d / 1000 / 15 * 60)  # cruce a 15 km/h
    return np.array(va, dtype=int), np.array(vb, dtype=int), np.array(vmin)


# ---------------------------------------------------------------------------
# Plan inicial
# ---------------------------------------------------------------------------
def _kmedias(xy, w, k, rng, iteraciones=30):
    k = min(k, len(xy))
    p = w / w.sum()
    c = xy[rng.choice(len(xy), size=k, replace=False, p=p)]
    for _ in range(iteraciones):
        _, lab = cKDTree(c).query(xy)
        for j in range(k):
            m = lab == j
            if m.any() and w[m].sum() > 0:
                c[j] = np.average(xy[m], axis=0, weights=w[m])
    _, lab = cKDTree(c).query(xy)
    return c, lab


def distritos_por_ciudad(pob_ciudad: float, objetivo: int, maximo: int) -> int:
    return max(math.ceil(pob_ciudad / maximo), round(pob_ciudad / objetivo), 1)


def plan_inicial(pr: Problema, k_total: int, rng) -> np.ndarray:
    semillas = []
    en_ciudad = np.zeros(pr.n, dtype=bool)
    # Semillas de ciudades grandes.
    for c in np.nonzero(pr.es_grande)[0]:
        m = np.nonzero(pr.ciudad == c)[0]
        if len(m) == 0:
            continue
        en_ciudad[m] = True
        k = distritos_por_ciudad(pr.pob[m].sum(), pr.objetivo_pob, pr.max_pob)
        cen, _ = _kmedias(pr.xy[m], pr.pob[m].astype(float) + 1, k, rng)
        _, idx = cKDTree(pr.xy[m]).query(cen)
        semillas.extend(m[np.unique(idx)])
    # Semillas del resto del territorio.
    resto = np.nonzero(~en_ciudad & (pr.pob > 0))[0]
    k_resto = k_total - len(semillas)
    if k_resto > 0 and len(resto):
        cen, _ = _kmedias(pr.xy[resto], pr.pob[resto].astype(float), k_resto, rng)
        _, idx = cKDTree(pr.xy[resto]).query(cen)
        semillas.extend(resto[np.unique(idx)])
    semillas = list(dict.fromkeys(int(s) for s in semillas))

    # Crecimiento por tiempo de viaje con tope de población.
    asig = np.full(pr.n, -1)
    pob_d = np.zeros(len(semillas))
    cola = []
    for d, s in enumerate(semillas):
        heapq.heappush(cola, (0.0, d, s))
    while cola:
        t, d, u = heapq.heappop(cola)
        if asig[u] != -1:
            continue
        if pob_d[d] + pr.pob[u] > pr.objetivo_pob and pr.pob[u] > 0 and pob_d[d] > 0:
            continue
        asig[u] = d
        pob_d[d] += pr.pob[u]
        for v, k in pr.vecinos[u]:
            if asig[v] == -1:
                heapq.heappush(cola, (t + pr.e_min[k], d, v))
    # Unidades sobrantes: al distrito vecino con menos población.
    while (asig == -1).any():
        for u in np.nonzero(asig == -1)[0]:
            ds = [asig[v] for v, _ in pr.vecinos[u] if asig[v] != -1]
            if ds:
                d = min(ds, key=lambda x: pob_d[x])
                asig[u] = d
                pob_d[d] += pr.pob[u]
    return asig


def _contiguidad_biparticion(parte, nodos, pob, conexas):
    """Los trozos sueltos de cada parte pasan a la otra, hasta que ambas son conexas."""
    parte = parte.copy()
    for _ in range(10):
        cambio = False
        for lado in (True, False):
            idx = np.nonzero(parte == lado)[0]
            if len(idx) == 0:
                continue
            lab = conexas(nodos[idx])
            if lab.max() == 0:
                continue
            pl = np.bincount(lab, weights=pob[idx] + 1e-6)
            parte[idx[lab != np.argmax(pl)]] = not lado
            cambio = True
        if not cambio:
            break
    return parte


def _atomos(pr: Problema) -> np.ndarray:
    """Agrupa unidades en átomos: cada trozo conexo de una comuna de hasta el
    máximo de población es un átomo; en comunas mayores, cada unidad es un átomo."""
    pob_com = np.bincount(pr.comuna, weights=pr.pob)
    atomo = np.full(pr.n, -1)
    misma = pr.comuna[pr.ea] == pr.comuna[pr.eb]
    adj = sp.csr_matrix((np.ones(misma.sum()), (pr.ea[misma], pr.eb[misma])), shape=(pr.n, pr.n))
    _, lab = connected_components(adj, directed=False)
    entera = pob_com[pr.comuna] <= pr.max_pob
    atomo[entera] = lab[entera]
    sueltas = np.nonzero(~entera)[0]
    atomo[sueltas] = lab.max() + 1 + np.arange(len(sueltas))
    return pd.factorize(atomo)[0]


def plan_biseccion(pr: Problema, k_total: int) -> np.ndarray:
    """Plan inicial por bisección recursiva según población.

    Cada paso divide un conjunto conexo en dos partes conexas cuyas poblaciones
    son proporcionales al número de distritos que recibe cada una. Así cada zona
    del país recibe el número correcto de distritos desde el inicio. Los cortes
    siguen límites comunales: se trabaja con átomos (comunas enteras de hasta el
    máximo de población; unidades sueltas en las comunas mayores).
    """
    at = _atomos(pr)
    n = at.max() + 1
    pob_at = np.bincount(at, weights=pr.pob)
    xy_at = np.column_stack([
        np.bincount(at, weights=pr.xy[:, i] * (pr.pob + 1e-6)) / np.bincount(at, weights=pr.pob + 1e-6)
        for i in (0, 1)])
    ea, eb = at[pr.ea], at[pr.eb]
    m = ea != eb
    adj = sp.csr_matrix((np.ones(2 * m.sum()),
                         (np.concatenate([ea[m], eb[m]]), np.concatenate([eb[m], ea[m]]))),
                        shape=(n, n))
    asig = np.full(n, -1)
    siguiente = [0]

    def conexas(nodos):
        sub = adj[nodos][:, nodos]
        _, lab = connected_components(sub, directed=False)
        return lab

    def dividir(nodos, k):
        if k == 1:
            asig[nodos] = siguiente[0]
            siguiente[0] += 1
            return
        k1 = k // 2
        pob = pob_at[nodos]
        meta = pob.sum() * k1 / k
        xy = xy_at[nodos]
        # Eje de corte: la dirección de mayor dispersión de la población.
        c = np.average(xy, axis=0, weights=pob + 1e-9)
        mejor = None
        for ang in np.radians([0, 30, 60, 90, 120, 150]):
            eje = np.array([np.cos(ang), np.sin(ang)])
            proy = (xy - c) @ eje
            disp = np.average(proy ** 2, weights=pob + 1e-9)
            if mejor is None or disp > mejor[0]:
                mejor = (disp, proy)
        orden = np.argsort(mejor[1])
        acum = np.cumsum(pob[orden])
        # Se recorta y se corrige la contigüidad; si la población de cada parte
        # se aleja de la meta, se desplaza el corte y se repite.
        objetivo_corte = meta
        mejor_parte, mejor_err = None, np.inf
        for _ in range(25):
            corte = int(np.clip(np.searchsorted(acum, objetivo_corte), 0, len(nodos) - 2))
            parte = np.zeros(len(nodos), dtype=bool)
            parte[orden[:corte + 1]] = True
            parte = _contiguidad_biparticion(parte, nodos, pob, conexas)
            err = pob[parte].sum() - meta
            if abs(err) < mejor_err and parte.any() and (~parte).any():
                mejor_parte, mejor_err = parte, abs(err)
            if abs(err) <= 0.01 * pob.sum() / k:
                break
            objetivo_corte -= err
        parte = mejor_parte
        # Número de distritos de cada parte según su población real.
        k1 = int(np.clip(round(pob[parte].sum() / (pob.sum() / k)), 1, k - 1))
        dividir(nodos[parte], k1)
        dividir(nodos[~parte], k - k1)

    # El grafo es conexo gracias a los enlaces virtuales.
    dividir(np.arange(n), k_total)
    return asig[at]


# ---------------------------------------------------------------------------
# Estado y puntaje
# ---------------------------------------------------------------------------
class Estado:
    def __init__(self, pr: Problema, asig: np.ndarray):
        self.pr = pr
        self.asig = asig.copy()
        self.k = int(asig.max()) + 1
        self.recalcular()

    # --- cálculo completo
    def recalcular(self):
        pr, a, k = self.pr, self.asig, self.k
        self.P = np.bincount(a, weights=pr.pob, minlength=k)
        self.miembros = [set() for _ in range(k)]
        for u, d in enumerate(a):
            self.miembros[d].add(u)
        # Centro principal: el centro urbano con más población en el distrito.
        self.centro = np.zeros(k, dtype=int)
        self.ancla = np.zeros(k, dtype=int)
        for d in range(k):
            m = np.fromiter(self.miembros[d], dtype=int)
            cs = pr.centro_de[m]
            if (cs >= 0).any():
                pc = pd.Series(pr.pob[m][cs >= 0]).groupby(cs[cs >= 0]).sum()
                c = int(pc.idxmax())
                self.centro[d] = c
                mm = m[cs == c]
            else:
                u0 = m[np.argmax(pr.pob[m])]
                self.centro[d] = pr.centro_cercano[u0]
                mm = m
            # Ancla: unidad del centro principal más cercana al centro de población
            # del distrito (no se mueve, para que el distrito conserve su centro).
            w = pr.pob[m].astype(float) + 1e-6
            c0 = np.average(pr.xy[m], axis=0, weights=w)
            self.ancla[d] = mm[np.argmin(((pr.xy[mm] - c0) ** 2).sum(axis=1))]
        t = pr.T[self.centro[a], np.arange(pr.n)].astype(float)
        self.S1 = np.bincount(a, weights=pr.pob * t, minlength=k)
        self.S2 = np.bincount(a, weights=pr.pob * t * t, minlength=k)
        # Comunas: unidades por (comuna, distrito).
        self.cnt_com = defaultdict(int)
        for u, d in enumerate(a):
            self.cnt_com[(pr.comuna[u], d)] += 1
        # Ciudades grandes presentes en cada distrito y pueblos que las buscan.
        self.pob_ciudad = defaultdict(float)   # (d, ciudad) -> población de la ciudad en d
        self.pob_obj = defaultdict(float)      # (d, ciudad) -> población que busca esa ciudad
        for u, d in enumerate(a):
            if pr.ciudad[u] >= 0:
                self.pob_ciudad[(d, pr.ciudad[u])] += pr.pob[u] + 1e-6
            if pr.objetivo[u] >= 0:
                self.pob_obj[(d, pr.objetivo[u])] += pr.pob[u]

    def _f_pob(self, P):
        pr = self.pr
        x = ((P - pr.objetivo_pob) / 10000.0) ** 2
        fuera = np.maximum(pr.min_pob - P, 0) + np.maximum(P - pr.max_pob, 0)
        return x + PENALIDAD_FUERA_RANGO * (fuera / 1000.0 + (fuera > 0))

    def _f_viaje(self, P, S1, S2):
        w = self.pr.w
        Pn = np.maximum(P, 1)
        return (w["compacidad"] * (S1 / Pn) / 30.0
                + w["centralidad_centro"] * np.sqrt(np.maximum(S2 / Pn, 0)) / 30.0)

    # Versiones escalares (más rápidas) para el cálculo incremental.
    def _fp(self, P) -> float:
        pr = self.pr
        x = ((P - pr.objetivo_pob) / 10000.0) ** 2
        fuera = max(pr.min_pob - P, 0) + max(P - pr.max_pob, 0)
        return x + (PENALIDAD_FUERA_RANGO * (fuera / 1000.0 + 1) if fuera > 0 else 0.0)

    def _fv(self, P, S1, S2) -> float:
        w = self.pr.w
        Pn = max(P, 1)
        return (w["compacidad"] * (S1 / Pn) / 30.0
                + w["centralidad_centro"] * math.sqrt(max(S2 / Pn, 0)) / 30.0)

    def componentes_puntaje(self) -> dict:
        pr, a = self.pr, self.asig
        corte = a[pr.ea] != a[pr.eb]
        comunas_extra = sum(1 for (c, d), n in self.cnt_com.items() if n > 0) - len(set(pr.comuna))
        separados = sum(p for (d, c), p in self.pob_obj.items()
                        if p > 0 and self.pob_ciudad.get((d, c), 0) <= 0)
        return {
            "equilibrio_poblacion": float(self._f_pob(self.P).sum()),
            "viaje_centro": float(self._f_viaje(self.P, self.S1, self.S2).sum()),
            "borde_km": float(pr.e_borde[corte].sum()),
            "rios_km": float(pr.e_rio[corte].sum()),
            "comunas_extra": int(comunas_extra),
            "pob_pueblos_separados": float(separados),
        }

    def puntaje(self) -> float:
        c, w = self.componentes_puntaje(), self.pr.w
        return (w["equilibrio_poblacion"] * c["equilibrio_poblacion"]
                + c["viaje_centro"]
                + w["compacidad"] * c["borde_km"] / 100.0
                + w["evitar_rios"] * c["rios_km"] / 10.0
                + w["no_dividir_comunas"] * c["comunas_extra"]
                + w["tiempo_viaje"] * c["pob_pueblos_separados"] / 10000.0)

    # --- cambio de puntaje al mover u de A a B
    def delta(self, u, A, B) -> float:
        pr, w = self.pr, self.pr.w
        p = pr.pob[u]
        PA, PB = self.P[A], self.P[B]
        d = w["equilibrio_poblacion"] * (self._fp(PA - p) + self._fp(PB + p)
                                         - self._fp(PA) - self._fp(PB))
        tA, tB = float(pr.T[self.centro[A], u]), float(pr.T[self.centro[B], u])
        s1A, s1B, s2A, s2B = self.S1[A], self.S1[B], self.S2[A], self.S2[B]
        d += (self._fv(PA - p, s1A - p * tA, s2A - p * tA * tA)
              + self._fv(PB + p, s1B + p * tB, s2B + p * tB * tB)
              - self._fv(PA, s1A, s2A) - self._fv(PB, s1B, s2B))
        # Bordes y ríos.
        dr = db = 0.0
        for v, k in pr.vecinos[u]:
            dv = self.asig[v]
            antes_c, despues_c = dv != A, dv != B
            if antes_c != despues_c:
                s = 1 if despues_c else -1
                dr += s * pr.e_rio[k]
                db += s * pr.e_borde[k]
        d += w["evitar_rios"] * dr / 10.0 + w["compacidad"] * db / 100.0
        # Comunas.
        c = pr.comuna[u]
        dc = (-1 if self.cnt_com[(c, A)] == 1 else 0) + (1 if self.cnt_com[(c, B)] == 0 else 0)
        d += w["no_dividir_comunas"] * dc
        # Pueblos separados de su ciudad.
        d += w["tiempo_viaje"] * self._delta_pueblos(u, A, B) / 10000.0
        return d

    def _delta_pueblos(self, u, A, B) -> float:
        pr = self.pr
        p = pr.pob[u]
        dd = 0.0
        ob = pr.objetivo[u]
        if ob >= 0:
            dd -= p if self.pob_ciudad.get((A, ob), 0) <= 0 else 0
            dd += p if self.pob_ciudad.get((B, ob), 0) <= 0 else 0
        ci = pr.ciudad[u]
        if ci >= 0:
            pu = p + 1e-6
            if self.pob_ciudad.get((A, ci), 0) - pu <= 1e-9:   # A pierde la ciudad
                dd += self.pob_obj.get((A, ci), 0)
            if self.pob_ciudad.get((B, ci), 0) <= 0:           # B gana la ciudad
                dd -= self.pob_obj.get((B, ci), 0)
        return dd

    def contiguo_sin(self, u, A) -> bool:
        """¿Sigue conexo el distrito A si se le quita u?"""
        vec = [v for v, _ in self.pr.vecinos[u] if self.asig[v] == A]
        if not vec:
            return len(self.miembros[A]) == 1
        faltan = set(vec)
        faltan.discard(vec[0])
        if not faltan:
            return True
        vistos = {vec[0], u}
        cola = deque([vec[0]])
        while cola:
            x = cola.popleft()
            for y, _ in self.pr.vecinos[x]:
                if y not in vistos and self.asig[y] == A:
                    vistos.add(y)
                    faltan.discard(y)
                    if not faltan:
                        return True
                    cola.append(y)
        return False

    def mover(self, u, A, B):
        pr = self.pr
        p = pr.pob[u]
        tA, tB = float(pr.T[self.centro[A], u]), float(pr.T[self.centro[B], u])
        self.P[A] -= p
        self.P[B] += p
        self.S1[A] -= p * tA
        self.S1[B] += p * tB
        self.S2[A] -= p * tA * tA
        self.S2[B] += p * tB * tB
        self.miembros[A].discard(u)
        self.miembros[B].add(u)
        self.asig[u] = B
        c = pr.comuna[u]
        self.cnt_com[(c, A)] -= 1
        self.cnt_com[(c, B)] += 1
        if pr.ciudad[u] >= 0:
            self.pob_ciudad[(A, pr.ciudad[u])] -= p + 1e-6
            self.pob_ciudad[(B, pr.ciudad[u])] += p + 1e-6
        if pr.objetivo[u] >= 0:
            self.pob_obj[(A, pr.objetivo[u])] -= p
            self.pob_obj[(B, pr.objetivo[u])] += p


# ---------------------------------------------------------------------------
# Búsqueda local
# ---------------------------------------------------------------------------
MAX_TROZO_UNIDADES = 80
MAX_TROZO_POB = 30000


def _trozo_comuna(est: Estado, u: int, A: int) -> list[int]:
    """Trozo conexo de la comuna de u dentro del distrito A (incluye u)."""
    pr = est.pr
    c = pr.comuna[u]
    trozo, vistos, cola = [u], {u}, deque([u])
    while cola:
        x = cola.popleft()
        for y, _ in pr.vecinos[x]:
            if y not in vistos and est.asig[y] == A and pr.comuna[y] == c:
                vistos.add(y)
                trozo.append(y)
                if len(trozo) > MAX_TROZO_UNIDADES:
                    return trozo
                cola.append(y)
    return trozo


def _conexo(est: Estado, A: int) -> bool:
    """¿Es conexo el distrito A? (BFS desde su ancla)."""
    pr = est.pr
    inicio = est.ancla[A]
    vistos, cola = {inicio}, deque([inicio])
    while cola:
        x = cola.popleft()
        for y, _ in pr.vecinos[x]:
            if y not in vistos and est.asig[y] == A:
                vistos.add(y)
                cola.append(y)
    return len(vistos) == len(est.miembros[A])


def _mover_trozo(est: Estado, u: int, A: int, B: int, temp: float, rng) -> bool:
    """Intenta mover a B todo el trozo de la comuna de u que está en A."""
    pr = est.pr
    trozo = _trozo_comuna(est, u, A)
    if (len(trozo) > MAX_TROZO_UNIDADES or est.ancla[A] in trozo
            or len(trozo) >= len(est.miembros[A]) or pr.pob[trozo].sum() > MAX_TROZO_POB):
        return False
    d = 0.0
    for x in trozo:           # orden BFS desde u: cada unidad toca B al moverse
        d += est.delta(x, A, B)
        est.mover(x, A, B)
    if (d <= 0 or rng.random() < math.exp(-d / temp)) and _conexo(est, A):
        return True
    for x in reversed(trozo):
        est.mover(x, B, A)
    return False
def recocido(est: Estado, iteraciones: int, rng, t_ini=0.5, t_fin=0.002,
             recalcular_cada=50000, informar=None, prob_trozo=0.15) -> Estado:
    pr = est.pr
    ne = len(pr.ea)
    lote = 4096
    enteros = rng.integers(0, ne, size=lote)
    reales = rng.random(size=lote)
    for it in range(iteraciones):
        if it and it % recalcular_cada == 0:
            est.recalcular()
            if informar:
                informar(it, est)
        j = it % lote
        if j == 0 and it:
            enteros = rng.integers(0, ne, size=lote)
            reales = rng.random(size=lote)
        k = enteros[j]
        a, b = pr.ea[k], pr.eb[k]
        if est.asig[a] == est.asig[b]:
            continue
        # Mueve a hacia el distrito de b o b hacia el de a.
        u, v = (a, b) if reales[j] < 0.5 else (b, a)
        A, B = est.asig[u], est.asig[v]
        if u == est.ancla[A] or len(est.miembros[A]) == 1:
            continue
        temp = t_ini * (t_fin / t_ini) ** (it / iteraciones)
        if reales[(j + 1) % lote] < prob_trozo:
            _mover_trozo(est, u, A, B, temp, rng)
            continue
        d = est.delta(u, A, B)
        if d > 0 and rng.random() >= math.exp(-d / temp):
            continue
        if not est.contiguo_sin(u, A):
            continue
        est.mover(u, A, B)
    est.recalcular()
    return est


def resumen_distritos(est: Estado, prefijo: str = "D") -> pd.DataFrame:
    pr = est.pr
    filas = []
    for d in range(est.k):
        m = np.fromiter(est.miembros[d], dtype=int)
        P = est.P[d]
        t = pr.T[est.centro[d], m].astype(float)
        w = pr.pob[m]
        comunas = pr.u.loc[m, "COMUNA"]
        pc = pd.Series(w).groupby(comunas.values).sum().sort_values(ascending=False)
        filas.append({
            "distrito": f"{prefijo}{d + 1:03d}",
            "poblacion": int(P),
            "centro_principal": pr.centros.loc[est.centro[d], "centro_urbano"],
            "minutos_medio_al_centro": round(float((w * t).sum() / max(P, 1)), 1),
            "minutos_max_al_centro": round(float(t[w > 0].max()) if (w > 0).any() else 0.0, 1),
            "comunas": ", ".join(f"{c} ({int(p):,})" for c, p in pc.items() if p > 0),
            "n_unidades": len(m),
            "en_rango": bool(pr.min_pob <= P <= pr.max_pob),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Reparación de población (cadenas de traspaso)
# ---------------------------------------------------------------------------
def _grafo_distritos(est: Estado) -> dict[int, set[int]]:
    pr, a = est.pr, est.asig
    g = defaultdict(set)
    corte = a[pr.ea] != a[pr.eb]
    for x, y in zip(a[pr.ea[corte]], a[pr.eb[corte]]):
        g[int(x)].add(int(y))
        g[int(y)].add(int(x))
    return g


def _trozo_colgante(est: Estado, u: int, X: int) -> list[int]:
    """Unidades de X que quedan separadas del ancla si se quita u (incluye u), en orden BFS desde u."""
    pr = est.pr
    vistos = {u}
    cola = deque([est.ancla[X]])
    vistos.add(est.ancla[X])
    while cola:
        x = cola.popleft()
        for y, _ in pr.vecinos[x]:
            if y not in vistos and est.asig[y] == X:
                vistos.add(y)
                cola.append(y)
    alcanzados = vistos - {u}
    trozo, cola, marcados = [u], deque([u]), {u}
    while cola:
        x = cola.popleft()
        for y, _ in pr.vecinos[x]:
            if est.asig[y] == X and y not in alcanzados and y not in marcados:
                marcados.add(y)
                trozo.append(y)
                cola.append(y)
    return trozo


def transferir(est: Estado, X: int, Y: int, cantidad: float) -> float:
    """Mueve unas `cantidad` personas de X a Y por su borde común (mejor puntaje primero).

    Si una unidad del borde es la única unión de un trozo de X con el resto del
    distrito, se mueve junto con ese trozo.
    """
    pr = est.pr
    movido = 0.0
    while movido < cantidad:
        resta = cantidad - movido
        mejor, mejor_d = None, math.inf
        for u in list(est.miembros[X]):
            if u == est.ancla[X]:
                continue
            if not any(est.asig[v] == Y for v, _ in pr.vecinos[u]):
                continue
            if est.contiguo_sin(u, X):
                trozo = [u]
            else:
                trozo = _trozo_colgante(est, u, X)
                if est.ancla[X] in trozo:
                    continue
            p = pr.pob[trozo].sum()
            if p > resta + 5000 or len(trozo) >= len(est.miembros[X]):
                continue
            d = est.delta(u, X, Y) / max(len(trozo), 1)
            if d < mejor_d:
                mejor, mejor_d = trozo, d
        if mejor is None:
            break
        for u in mejor:
            est.mover(u, X, Y)
        movido += pr.pob[mejor].sum()
    return movido


def reparar(est: Estado, rondas: int = 1000) -> int:
    """Corrige distritos fuera de rango pasando población por cadenas de distritos.

    En cada ronda se busca (por cercanía en el grafo de distritos) el primer
    distrito que pueda ceder o recibir población sin salir de rango, y se pasa
    lo que ese distrito permite (hasta lo que falta) a lo largo del camino.
    """
    pr = est.pr
    margen = 2000
    for _ in range(rondas):
        fuera = np.nonzero((est.P < pr.min_pob) | (est.P > pr.max_pob))[0]
        if len(fuera) == 0:
            break
        D = int(fuera[np.argmax(np.abs(est.P[fuera] - pr.objetivo_pob))])
        falta = est.P[D] < pr.min_pob
        necesario = (pr.min_pob + margen - est.P[D]) if falta else (est.P[D] - pr.max_pob + margen)
        g = _grafo_distritos(est)
        previo = {D: None}
        cola = deque([D])
        fin, cantidad = None, 0.0
        while cola:
            x = cola.popleft()
            if x != D:
                holgura = (est.P[x] - pr.min_pob - margen) if falta else (pr.max_pob - margen - est.P[x])
                if holgura >= 1000:
                    fin, cantidad = x, min(necesario, holgura)
                    break
            for y in g[x]:
                if y not in previo:
                    previo[y] = x
                    cola.append(y)
        if fin is None:
            break
        camino = [fin]
        while previo[camino[-1]] is not None:
            camino.append(previo[camino[-1]])
        # camino: fin -> ... -> D. Falta: fluye de fin hacia D. Sobra: de D hacia fin.
        if not falta:
            camino = camino[::-1]
        for X, Y in zip(camino[:-1], camino[1:]):
            if transferir(est, X, Y, cantidad) <= 0:
                break
    est.recalcular()
    return int(((est.P < pr.min_pob) | (est.P > pr.max_pob)).sum())
