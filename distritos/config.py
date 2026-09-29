"""Rutas del proyecto y lectura de la configuración."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
ORIGINALES = DATOS / "originales"
PROCESADOS = DATOS / "procesados"
GDB_CENSO = ORIGINALES / "Cartografia_censo2024_Pais.gdb"
PBF_OSM = ORIGINALES / "chile-latest.osm.pbf"


def cargar_parametros() -> dict:
    with open(RAIZ / "config" / "parametros.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_especiales() -> list[dict]:
    with open(RAIZ / "config" / "distritos_especiales.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["distritos_especiales"]
