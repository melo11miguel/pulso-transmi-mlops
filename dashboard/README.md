# Dashboard (bono)

Observabilidad del pipeline: accuracy en el tiempo, carrera del puesto, error por estación y
horizonte, drift, estado operativo, versiones del modelo y posición en el leaderboard.

**En vivo:** https://pulso-transmi-dashboard.vercel.app

## Mapa 3D de la red

Las 12 estaciones se dibujan en su **posición geográfica real** (coordenadas de la tabla
`stations`, proyectadas a kilómetros corrigiendo la longitud por el coseno de la latitud, de modo
que la forma del corredor no se deforma). La altura de cada torre es la **demanda promedio** a esa
hora y el reproductor recorre el día completo en franjas de media hora.

- Datos: `public.red_3d()`, que agrega el histórico por estación × media hora **en hora de Bogotá**
  y lo separa en día hábil y fin de semana. El pico sale donde debe: 06:30–07:30 y 17:00–18:00.
- Los dos tipos de día se normalizan con un **mismo máximo**; si cada uno usara el suyo, un domingo
  se vería tan lleno como un martes a las siete.
- El color alterna entre intensidad de demanda y **accuracy oficial de esa estación**, así se ve de
  un vistazo si fallamos donde hay mucha gente o donde hay poca.
- Se consulta **una sola vez** al cargar, no en cada refresco: es un agregado histórico y rehacerlo
  cada minuto tiraría la cámara que el usuario haya ajustado.
- Render propio sobre three.js, con controles de órbita escritos a mano (la distribución de cdnjs
  no incluye `OrbitControls`). La rueda sola no captura el scroll de la página: el zoom pide Ctrl.

## Por qué no necesita secretos

Es una página **estática**. Llama a `public.dashboard_public()` en Supabase con la **clave
publicable**, que está diseñada para viajar en el navegador:

- Esa clave solo puede ejecutar esa función. Verificado: no puede leer `observations`,
  `submissions`, `model_versions` ni `leaderboard_snapshots` (todas responden 42501), ni llamar a
  la función interna `dashboard_state()`.
- La `service_role` de Supabase y `PULSO_API_KEY` nunca entran aquí: viven solo en GitHub Actions.
- El leaderboard lo captura el monitor (que sí tiene la API key) en `leaderboard_snapshots`, así el
  dashboard lo lee sin credenciales. Solo se guarda nuestra posición y el mejor accuracy de la
  cohorte: **no se almacenan ni publican nombres de otros participantes**.

`red_3d()` sigue la misma regla: es `security definer`, solo devuelve **agregados** del histórico
que el propio reto publica más nuestro desempeño, y se le concede `execute` a `anon` sin abrir
ninguna tabla.

Migraciones relacionadas: `20260923165931_dashboard_state_function.sql`,
`20260923194119_dashboard_public_access.sql` y `20260923203631_red_3d.sql`.

Los nombres llevan el sello de version que usa Supabase. Es obligatorio: la
integracion de GitHub compara el historial remoto contra los archivos locales y
marca el commit en rojo si no coinciden.
