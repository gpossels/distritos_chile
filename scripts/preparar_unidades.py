"""Paso 1: construye las unidades base, su adyacencia y un resumen de validación.

Uso:  python scripts/preparar_unidades.py
Salidas en datos/procesados/:
  unidades.gpkg           unidades base con población y centroide ponderado
  adyacencia.csv          pares de unidades vecinas y largo del borde compartido
  centros_urbanos.csv     centros urbanos ordenados por población
  resumen_unidades.txt    totales y controles
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distritos.adyacencia import calcular_adyacencia, componentes  # noqa: E402
from distritos.censo import construir_unidades  # noqa: E402
from distritos.config import PROCESADOS, cargar_especiales, cargar_parametros  # noqa: E402


def main() -> None:
    t0 = time.time()
    par = cargar_parametros()
    PROCESADOS.mkdir(parents=True, exist_ok=True)

    print("Construyendo unidades ...")
    unidades, centros = construir_unidades(par["proyeccion_metrica"])
    print(f"  {len(unidades):,} unidades ({time.time() - t0:.0f} s)")

    print("Calculando adyacencia ...")
    ady = calcular_adyacencia(unidades)
    unidades["componente"] = componentes(unidades, ady)
    print(f"  {len(ady):,} pares vecinos ({time.time() - t0:.0f} s)")

    # Marca los distritos especiales.
    especial = {}
    for d in cargar_especiales():
        for cut in d["comunas"]:
            especial[cut] = d["nombre"]
    unidades["distrito_especial"] = unidades["CUT"].map(especial)

    umbral = par["centros_urbanos"]["umbral_distrito_propio"]
    centros["distrito_propio"] = centros["n_per"] >= umbral

    unidades.to_file(PROCESADOS / "unidades.gpkg", layer="unidades", driver="GPKG")
    ady.to_csv(PROCESADOS / "adyacencia.csv", index=False)
    centros.to_csv(PROCESADOS / "centros_urbanos.csv", index=False)

    pob = par["poblacion"]
    total = int(unidades["n_per"].sum())
    esp = unidades[unidades["distrito_especial"].notna()]
    normal = total - int(esp["n_per"].sum())
    ncomp = unidades["componente"].nunique()
    lineas = [
        f"Población georreferenciada total: {total:,}",
        f"Unidades: {len(unidades):,}  " + str(unidades['tipo'].value_counts().to_dict()),
        f"Pares vecinos: {len(ady):,}",
        f"Componentes conexos (islas y otros aislados): {ncomp}",
        "",
        "Distritos especiales:",
        *[f"  {n:32s} {int(p):>9,}" for n, p in esp.groupby("distrito_especial")["n_per"].sum().items()],
        "",
        f"Población para distritos normales: {normal:,}",
        f"Distritos normales estimados (a {pob['objetivo']:,}): {normal / pob['objetivo']:.1f}",
        f"Rango posible: {normal / pob['maximo']:.0f} a {normal / pob['minimo']:.0f}",
        "",
        f"Centros urbanos con distrito propio (>= {umbral:,}): {int(centros['distrito_propio'].sum())}",
        *[f"  {r.centro_urbano:55s} {int(r.n_per):>10,}" for r in centros[centros.distrito_propio].itertuples()],
    ]
    texto = "\n".join(lineas)
    (PROCESADOS / "resumen_unidades.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\nTiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
