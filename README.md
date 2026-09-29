# Distritos Chile

Una app para redistritar Chile: un diputado por distrito, con distritos de
**90.000 ± 10.000 habitantes** (Censo 2024), sin considerar los límites regionales.

## Criterios

En orden de prioridad por defecto (los pesos se ajustan en `config/parametros.yaml`):

1. **Equilibrio de población**: entre 80.000 y 100.000 habitantes (límite duro).
2. **Tiempo de viaje**: los pueblos pequeños quedan en el distrito de su ciudad
   grande más cercana (preferencia fuerte). Se incluyen transbordadores, contando la espera.
3. **Compacidad**: medida por tiempo de viaje; la forma es un criterio secundario.
4. **Centralidad**: el centro urbano principal, cerca del centro del distrito.
5. **Límites naturales**: los límites evitan seguir los ríos principales
   (preferencia fuerte; lista editable en `config/parametros.yaml`). Los cordones
   montañosos quedan como límite de forma natural: sin caminos que los crucen,
   el tiempo de viaje entre sus dos lados es alto.
6. **Comunas**: se respetan sus límites siempre que se pueda; las ciudades de más
   de 100.000 habitantes se dividen en distritos dentro de la ciudad.

Todo centro urbano de 80.000 habitantes o más tiene distrito propio.

## Distritos especiales

Definidos en `config/distritos_especiales.yaml`: Rapa Nui, Juan Fernández,
Tierra del Fuego, Puerto Williams, Territorio Chileno Antártico, Cochrane,
Coyhaique (resto de Aysén), Natales, y Punta Arenas (dos distritos).

## Datos

| Archivo | Fuente |
|---|---|
| `Cartografia_censo2024_Pais.gdb` | INE, cartografía del Censo 2024 con población por manzana, zona, localidad y entidad |
| `chile-latest.osm.pbf` | OpenStreetMap (Geofabrik, 28-09-2026): red vial, transbordadores y ríos |

Se usa la población **georreferenciada** (18.226.208 habitantes), la única que
se puede asignar a un lugar del mapa.

Los datos se publican en el release `datos-v1` del repositorio y no se guardan en git.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/descargar_datos.py   # descarga los datos a datos/originales/
python scripts/preparar_unidades.py # paso 1: unidades base y adyacencia (~4 min)
python scripts/calcular_tiempos.py  # paso 2: red vial y tiempos de viaje (~7 min)
python scripts/calcular_rios.py     # paso 3: bordes que siguen ríos principales (~1 min)
python scripts/optimizar.py --planes 3   # paso 4: planes de distritos (~4 min por plan)
python scripts/mapa_plan.py 1            # mapas PNG de un plan
```

Resultados en `datos/resultados/`: `plan_N.csv` (unidad → distrito),
`plan_N_resumen.csv` (población, centro principal, tiempos y comunas por distrito),
`plan_N.gpkg` (polígonos, se abre en QGIS) y `planes.csv` (puntaje de cada plan).

## Unidades base

- **Zona censal urbana** (~3.200 hab. en promedio).
- **Localidad rural** (~250 hab. en promedio).
- **Territorio sin población**: partes de cada comuna sin zona ni localidad
  (campos de hielo, desierto, islotes). Evitan huecos en el mapa.

Cada unidad tiene su centroide ponderado por población, que se usa como origen
de los tiempos de viaje.

## Tiempos de viaje

- Red vial de OSM tratada como no dirigida, con velocidad por tipo de vía y tope
  de 40 km/h en ripio y tierra.
- Transbordadores: travesía más espera media (mitad del intervalo entre salidas).
  Si OSM no trae el intervalo se usa uno por defecto según la travesía (hasta 1,5 h:
  cada 1 h; hasta 8 h: diario; más: semanal). Se corrigen en `config/transbordadores.yaml`.
- Cada unidad se ubica en el vértice vial más cercano a su centroide ponderado;
  el acceso se estima en línea recta × 1,3 a 20 km/h.
- Islas sin transbordador: línea recta a 15 km/h más 24 h de penalidad.

## Optimizador

1. **Plan inicial**: bisección recursiva por población, cortando por límites
   comunales (comunas de hasta 100.000 hab. enteras). Cada zona del país recibe
   el número de distritos que corresponde a su población.
2. **Recocido simulado**: mueve unidades de borde, o trozos enteros de comuna,
   entre distritos vecinos sin romper la contigüidad, minimizando el puntaje
   ponderado (pesos en `config/parametros.yaml`).
3. **Reparación**: si un distrito queda fuera de 80.000–100.000, traspasa
   población por una cadena de distritos hasta uno con holgura.

Las islas se unen a la unidad más cercana de tierra firme para que un distrito
pueda cruzar un canal (p. ej. Chacao). Punta Arenas se divide en dos distritos
con el mismo método.

## Estado

- [x] Paso 1: unidades base, adyacencia y centros urbanos
- [x] Paso 2: tiempos de viaje (red vial y transbordadores de OSM)
- [x] Paso 3: ríos principales
- [x] Paso 4: optimizador de distritos
- [ ] Paso 5: mapa web local para ajustar los límites a mano
