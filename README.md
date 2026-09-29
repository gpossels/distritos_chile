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
5. **Límites naturales**: se prefieren divisorias de aguas (cordillera de la Costa,
   Nahuelbuta, cordones transversales del Norte Chico); se evitan los ríos (preferencia fuerte).
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
| `chile-latest.osm.pbf` | OpenStreetMap (Geofabrik): red vial y transbordadores |

Se usa la población **georreferenciada** (18.226.208 habitantes), la única que
se puede asignar a un lugar del mapa.

Los datos se publican en el release `datos-v1` del repositorio y no se guardan en git.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/descargar_datos.py   # descarga los datos a datos/originales/
python scripts/preparar_unidades.py # paso 1: unidades base y adyacencia
```

## Unidades base

- **Zona censal urbana** (~3.200 hab. en promedio).
- **Localidad rural** (~250 hab. en promedio).
- **Territorio sin población**: partes de cada comuna sin zona ni localidad
  (campos de hielo, desierto, islotes). Evitan huecos en el mapa.

Cada unidad tiene su centroide ponderado por población, que se usa como origen
de los tiempos de viaje.

## Estado

- [x] Paso 1: unidades base, adyacencia y centros urbanos
- [ ] Paso 2: tiempos de viaje (red vial y transbordadores de OSM)
- [ ] Paso 3: divisorias de aguas y ríos
- [ ] Paso 4: optimizador de distritos
- [ ] Paso 5: mapa web local para ajustar los límites a mano
