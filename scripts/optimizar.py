"""Paso 4: genera planes de distritos.

Uso:  python scripts/optimizar.py [--planes 3] [--iteraciones 3000000] [--semilla 1]
Requiere los pasos 1 a 3.
Salidas en datos/resultados/:
  plan_N.csv            unidad -> distrito
  plan_N_resumen.csv    población, centro principal, tiempos y comunas por distrito
  plan_N.gpkg           polígonos de los distritos
  planes.csv            puntaje de cada plan (menor es mejor)
"""
import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.config import DATOS, PROCESADOS, cargar_especiales, cargar_parametros  # noqa: E402
from distritos.optimizador import (Estado, Problema, plan_biseccion,  # noqa: E402
                                   plan_inicial, recocido, reparar, resumen_distritos)

RESULTADOS = DATOS / "resultados"
INICIO = "biseccion"


def cargar():
    u = gpd.read_file(PROCESADOS / "unidades.gpkg")
    c = u.geometry.centroid
    u = pd.DataFrame(u.drop(columns="geometry")).assign(gx=c.x.values, gy=c.y.values)
    ut = pd.read_csv(PROCESADOS / "unidades_tiempos.csv")
    u = u.merge(ut, on="id_unidad", how="left")
    ady = pd.read_csv(PROCESADOS / "adyacencia_tiempos.csv")
    rios = pd.read_csv(PROCESADOS / "adyacencia_rios.csv")[["origen", "destino", "borde_rio_m"]]
    ady = ady.merge(rios, on=["origen", "destino"], how="left").fillna({"borde_rio_m": 0})
    z = np.load(PROCESADOS / "tiempos_centros.npz", allow_pickle=True)
    centros = pd.read_csv(PROCESADOS / "centros_urbanos_tiempos.csv")
    assert list(z["centros"]) == list(centros["centro_urbano"])
    col = pd.Series(np.arange(len(z["unidades"])), index=z["unidades"])
    return u, ady, z["minutos"], col, centros


def resolver(u, ady, T, col, centros, par, k, iteraciones, rng, etiqueta, prefijo):
    Tsub = np.ascontiguousarray(T[:, col[u["id_unidad"]].values])
    pr = Problema(u, ady, Tsub, centros, par)
    asig = plan_biseccion(pr, k) if INICIO == "biseccion" else plan_inicial(pr, k, rng)
    est = Estado(pr, asig)
    print(f"  [{etiqueta}] {est.k} distritos, {pr.n:,} unidades, "
          f"{pr.n_virtuales} enlaces entre islas; puntaje inicial {est.puntaje():.1f}")

    def informar(it, e):
        fuera = int(((e.P < pr.min_pob) | (e.P > pr.max_pob)).sum())
        print(f"    iteración {it:>9,}: puntaje {e.puntaje():9.1f}, fuera de rango {fuera}")

    # Recocido, reparación de población y pulido final a baja temperatura.
    est = recocido(est, iteraciones, rng, recalcular_cada=max(iteraciones // 10, 1),
                   informar=informar)
    for ronda in range(3):
        fuera = reparar(est)
        print(f"    reparación {ronda + 1}: fuera de rango {fuera}")
        est = recocido(est, iteraciones // 3, rng, t_ini=0.05,
                       recalcular_cada=max(iteraciones // 3, 1))
        if fuera == 0 and not ((est.P < pr.min_pob) | (est.P > pr.max_pob)).any():
            break
    plan = pd.DataFrame({"id_unidad": u["id_unidad"].values,
                         "distrito": [f"{prefijo}{d + 1:03d}" for d in est.asig]})
    return plan, resumen_distritos(est, prefijo), est


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes", type=int, default=3)
    ap.add_argument("--iteraciones", type=int, default=3_000_000)
    ap.add_argument("--semilla", type=int, default=1)
    ap.add_argument("--inicio", choices=["biseccion", "semillas"], default="biseccion",
                    help="plan inicial: bisección por población o semillas por ciudad")
    args = ap.parse_args()
    global INICIO
    INICIO = args.inicio

    t0 = time.time()
    par = cargar_parametros()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    u, ady, T, col, centros = cargar()
    geo = gpd.read_file(PROCESADOS / "unidades.gpkg")[["id_unidad", "geometry"]]

    especiales = cargar_especiales()
    normales = u[u["distrito_especial"].isna()].reset_index(drop=True)
    k = round(normales["n_per"].sum() / par["poblacion"]["objetivo"])

    puntajes = []
    for n in range(1, args.planes + 1):
        rng = np.random.default_rng(args.semilla * 1000 + n)
        print(f"Plan {n}")
        plan, resumen, est = resolver(normales, ady, T, col, centros, par, k,
                                      args.iteraciones, rng, "nacional", "D")
        partes, resumenes = [plan], [resumen]

        # Distritos especiales.
        for e in especiales:
            ue = u[u["distrito_especial"] == e["nombre"]].reset_index(drop=True)
            nd = e.get("n_distritos", 1)
            if len(ue) == 0:
                resumenes.append(pd.DataFrame([{"distrito": e["nombre"], "poblacion": 0,
                                                "centro_principal": None, "especial": True}]))
                continue
            if nd == 1:
                partes.append(pd.DataFrame({"id_unidad": ue["id_unidad"], "distrito": e["nombre"]}))
                resumenes.append(pd.DataFrame([{
                    "distrito": e["nombre"], "poblacion": int(ue["n_per"].sum()),
                    "comunas": ", ".join(sorted(ue["COMUNA"].unique())), "en_rango": None}]))
                continue
            # Especial dividido en varios distritos (Punta Arenas).
            pe = int(ue["n_per"].sum())
            par_e = {**par, "poblacion": {"objetivo": pe / nd, "minimo": 0.85 * pe / nd,
                                          "maximo": 1.15 * pe / nd}}
            pl, rs, _ = resolver(ue, ady, T, col, centros, par_e, nd,
                                 max(args.iteraciones // 20, 20000), rng, e["nombre"],
                                 e["nombre"] + " ")
            partes.append(pl)
            resumenes.append(rs.assign(en_rango=None))

        plan = pd.concat(partes, ignore_index=True)
        resumen = pd.concat(resumenes, ignore_index=True)
        resumen["especial"] = ~resumen["distrito"].str.match(r"^D\d{3}$")
        plan.to_csv(RESULTADOS / f"plan_{n}.csv", index=False)
        resumen.to_csv(RESULTADOS / f"plan_{n}_resumen.csv", index=False)
        dis = geo.merge(plan, on="id_unidad").dissolve(by="distrito").reset_index()
        # Compacidad geométrica: Polsby-Popper (4πA/P²) y área / envolvente convexa.
        g = dis.geometry.simplify(100)
        dis["polsby_popper"] = (4 * np.pi * g.area / g.length ** 2).round(3)
        dis["envolvente_convexa"] = (g.area / g.convex_hull.area).round(3)
        resumen = resumen.merge(dis[["distrito", "polsby_popper", "envolvente_convexa"]],
                                on="distrito", how="left")
        resumen.to_csv(RESULTADOS / f"plan_{n}_resumen.csv", index=False)
        dis = dis.merge(resumen[["distrito", "poblacion", "centro_principal"]], on="distrito")
        dis.to_file(RESULTADOS / f"plan_{n}.gpkg", layer="distritos", driver="GPKG")

        comp = est.componentes_puntaje()
        normales_r = resumen[~resumen["especial"]]
        puntajes.append({"plan": n, "puntaje": round(est.puntaje(), 2),
                         "distritos": len(resumen),
                         "fuera_de_rango": int((~normales_r["en_rango"].astype(bool)).sum()),
                         "polsby_popper_mediana": float(normales_r["polsby_popper"].median()),
                         "envolvente_mediana": float(normales_r["envolvente_convexa"].median()),
                         **{k2: round(v, 2) for k2, v in comp.items()}})
        print(f"  Plan {n}: puntaje {est.puntaje():.1f}, {len(resumen)} distritos, "
              f"fuera de rango {puntajes[-1]['fuera_de_rango']} ({time.time() - t0:.0f} s)")

    pd.DataFrame(puntajes).sort_values("puntaje").to_csv(RESULTADOS / "planes.csv", index=False)
    print(pd.DataFrame(puntajes).sort_values("puntaje").to_string(index=False))


if __name__ == "__main__":
    main()
