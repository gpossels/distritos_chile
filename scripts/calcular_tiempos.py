"""Paso 2: red vial de OSM y tiempos de viaje.

Uso:  python scripts/calcular_tiempos.py
Requiere haber corrido scripts/preparar_unidades.py.
Salidas en datos/procesados/:
  red_vial.npz                 grafo de tiempos (minutos)
  transbordadores.csv          rutas, travesía, intervalo y espera usados
  tiempos_centros.npz          matriz centros urbanos x unidades (minutos)
  centros_urbanos_tiempos.csv  jerarquía: centro mayor más cercano de cada centro
  adyacencia_tiempos.csv       tiempo entre unidades vecinas
  unidades_tiempos.csv         centro más cercano y centro grande más cercano por unidad
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.config import PBF_OSM, PROCESADOS, RAIZ, cargar_parametros  # noqa: E402
from distritos.red_vial import construir_red  # noqa: E402
from distritos.tiempos import calcular_tiempos  # noqa: E402


def cargar_ajustes_transbordadores() -> dict:
    ruta = RAIZ / "config" / "transbordadores.yaml"
    if not ruta.exists():
        return {}
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    return {int(k): v for k, v in (datos.get("ajustes") or {}).items()}


def main() -> None:
    t0 = time.time()
    par = cargar_parametros()

    print("Construyendo red vial desde OSM ...")
    red = construir_red(PBF_OSM, par["transbordadores"], cargar_ajustes_transbordadores())
    red.guardar(PROCESADOS / "red_vial.npz")
    red.transbordadores.to_csv(PROCESADOS / "transbordadores.csv", index=False)
    print(f"  {red.grafo.shape[0]:,} vértices ({time.time() - t0:.0f} s)")

    print("Leyendo unidades ...")
    unidades = gpd.read_file(PROCESADOS / "unidades.gpkg", ignore_geometry=True)
    ady = pd.read_csv(PROCESADOS / "adyacencia.csv")
    centros = pd.read_csv(PROCESADOS / "centros_urbanos.csv")

    print("Calculando tiempos de viaje ...")
    r = calcular_tiempos(red, unidades, ady, centros, par["proyeccion_metrica"],
                         par["centros_urbanos"]["umbral_distrito_propio"])
    np.savez_compressed(PROCESADOS / "tiempos_centros.npz", minutos=r["matriz"],
                        centros=r["centros"]["centro_urbano"].values,
                        unidades=unidades["id_unidad"].values)
    r["centros"].to_csv(PROCESADOS / "centros_urbanos_tiempos.csv", index=False)
    r["adyacencia"].to_csv(PROCESADOS / "adyacencia_tiempos.csv", index=False)
    r["unidades"].to_csv(PROCESADOS / "unidades_tiempos.csv", index=False)

    a = r["adyacencia"]
    print(f"  Pares vecinos sin conexión (islas sin transbordador): {int(a['sin_conexion'].sum()):,}")
    print(f"  Tiempo mediano entre unidades vecinas: {a['minutos'].median():.1f} min")
    print(f"Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
