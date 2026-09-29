"""Dibuja un plan de distritos: país en tres franjas y las tres áreas metropolitanas.

Uso:  python scripts/mapa_plan.py [N]      (por defecto el plan 1)
Salidas en datos/resultados/: plan_N_pais.png y plan_N_ciudades.png
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from shapely.geometry import box  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.config import DATOS  # noqa: E402

RESULTADOS = DATOS / "resultados"
# Paleta categórica (8 tonos). En un mapa solo distingue distritos vecinos;
# la identidad la dan las etiquetas.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONDO = "#ffffff"
TINTA = "#1f1f1e"

# Ventanas (UTM 19S, metros): (título, xmin, ymin, xmax, ymax)
FRANJAS = [
    ("Norte", 180_000, 6_800_000, 620_000, 8_080_000),
    ("Centro", 50_000, 5_700_000, 560_000, 6_820_000),
    ("Sur", -150_000, 3_850_000, 560_000, 5_720_000),
]
CIUDADES = [
    ("Gran Santiago", 320_000, 6_265_000, 380_000, 6_320_000),
    ("Gran Valparaíso", 222_000, 6_318_000, 280_000, 6_372_000),
    ("Gran Concepción", 125_000, 5_885_000, 180_000, 5_945_000),
]


def colorear(gdf: gpd.GeoDataFrame) -> list[str]:
    """Coloreo voraz: distritos vecinos con colores distintos."""
    vec = gpd.sjoin(gdf[["geometry"]], gdf[["geometry"]], predicate="intersects")
    vec = vec[vec.index != vec["index_right"]]
    vecinos = vec.groupby(level=0)["index_right"].apply(set).to_dict()
    orden = sorted(gdf.index, key=lambda i: -len(vecinos.get(i, ())))
    color = {}
    for i in orden:
        usados = {color[j] for j in vecinos.get(i, ()) if j in color}
        color[i] = next((c for c in range(len(COLORES)) if c not in usados), 0)
    return [COLORES[color[i]] for i in gdf.index]


def etiqueta(nombre: str) -> str:
    return nombre[1:].lstrip("0") if nombre.startswith("D") and nombre[1:].isdigit() else nombre


def dibujar(ax, gdf, ventana, tam_etiqueta):
    titulo, x0, y0, x1, y1 = ventana
    v = gdf[gdf.intersects(box(x0, y0, x1, y1))]
    v.plot(ax=ax, color=v["color"], edgecolor=FONDO, linewidth=0.6)
    for _, r in v.iterrows():
        p = r.geometry.representative_point()
        if x0 <= p.x <= x1 and y0 <= p.y <= y1:
            ax.text(p.x, p.y, etiqueta(r["distrito"]), ha="center", va="center",
                    fontsize=tam_etiqueta, color=TINTA,
                    path_effects=[pe.withStroke(linewidth=2, foreground=FONDO)])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(titulo, fontsize=12, color=TINTA, loc="left")


def main() -> None:
    n = sys.argv[1] if len(sys.argv) > 1 else "1"
    gdf = gpd.read_file(RESULTADOS / f"plan_{n}.gpkg")
    gdf["geometry"] = gdf.geometry.simplify(150)
    gdf["color"] = colorear(gdf)

    fig, axes = plt.subplots(1, 3, figsize=(18, 16), facecolor=FONDO)
    for ax, v in zip(axes, FRANJAS):
        dibujar(ax, gdf, v, 6)
    fig.suptitle(f"Plan {n}: {len(gdf)} distritos", fontsize=15, color=TINTA, x=0.02, ha="left")
    fig.savefig(RESULTADOS / f"plan_{n}_pais.png", dpi=110, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), facecolor=FONDO)
    for ax, v in zip(axes, CIUDADES):
        dibujar(ax, gdf, v, 8)
    fig.savefig(RESULTADOS / f"plan_{n}_ciudades.png", dpi=110, bbox_inches="tight")
    print(f"Mapas en {RESULTADOS}")


if __name__ == "__main__":
    main()
