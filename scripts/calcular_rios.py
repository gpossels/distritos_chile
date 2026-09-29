"""Paso 3: ríos principales y bordes entre unidades que siguen un río.

Uso:  python scripts/calcular_rios.py
Requiere haber corrido scripts/preparar_unidades.py.
Salidas en datos/procesados/:
  rios_principales.gpkg    ríos principales usados
  adyacencia_rios.csv      por par vecino: metros de borde junto a un río principal
"""
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.config import PBF_OSM, PROCESADOS, cargar_parametros  # noqa: E402
from distritos.rios import bordes_sobre_rios, leer_rios  # noqa: E402


def main() -> None:
    t0 = time.time()
    par = cargar_parametros()
    pr = par["rios"]

    print("Leyendo ríos principales desde OSM ...")
    rios = leer_rios(PBF_OSM, pr["principales"], pr["largo_minimo_km"], par["proyeccion_metrica"])
    rios.to_file(PROCESADOS / "rios_principales.gpkg", layer="rios", driver="GPKG")
    encontrados = set(rios["nombre"])
    for r in pr["principales"]:
        nombre = r["nombre"] if isinstance(r, dict) else r
        km = rios.loc[rios["nombre"] == nombre, "largo_km"].sum()
        print(f"  {nombre:22s} {km:7.1f} km" if nombre in encontrados
              else f"  {nombre:22s}   no encontrado en OSM")

    print("Midiendo bordes sobre ríos ...")
    unidades = gpd.read_file(PROCESADOS / "unidades.gpkg")
    ady = pd.read_csv(PROCESADOS / "adyacencia.csv")
    ady["borde_rio_m"] = bordes_sobre_rios(unidades, ady, rios, pr["distancia_m"])
    ady["fraccion_rio"] = (ady["borde_rio_m"] / ady["borde_m"]).clip(0, 1).round(3)
    ady.to_csv(PROCESADOS / "adyacencia_rios.csv", index=False)
    sobre = ady["fraccion_rio"] >= 0.5
    print(f"  Pares vecinos cuyo borde sigue un río principal (>= 50 %): {int(sobre.sum()):,}")
    print(f"Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
