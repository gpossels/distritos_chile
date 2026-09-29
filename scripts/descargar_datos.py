"""Descarga los datos originales desde el release `datos-v1` del repositorio.

Uso:  python scripts/descargar_datos.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.config import GDB_CENSO, ORIGINALES, PBF_OSM  # noqa: E402

BASE = "https://github.com/gpossels/distritos_chile/releases/download/datos-v1/"
ARCHIVOS = {
    "Cartografia_censo2024_Pais.gdb.zip": GDB_CENSO,
    "chile-latest.osm.pbf": PBF_OSM,
}


def descargar(nombre: str, destino: Path) -> None:
    print(f"Descargando {nombre} ...")
    urllib.request.urlretrieve(BASE + nombre, destino)


def main() -> None:
    ORIGINALES.mkdir(parents=True, exist_ok=True)
    for nombre, final in ARCHIVOS.items():
        if final.exists():
            print(f"Ya existe: {final.name}")
            continue
        destino = ORIGINALES / nombre
        try:
            descargar(nombre, destino)
        except Exception as e:  # noqa: BLE001
            print(f"  No se pudo descargar {nombre}: {e}")
            continue
        if nombre.endswith(".zip"):
            with zipfile.ZipFile(destino) as z:
                z.extractall(ORIGINALES)
            destino.unlink()
    print("Listo.")


if __name__ == "__main__":
    main()
