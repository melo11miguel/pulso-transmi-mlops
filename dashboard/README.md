# Dashboard (bono)

Observabilidad del pipeline: accuracy en el tiempo, carrera del puesto, error por estación y
horizonte, drift, estado operativo, versiones del modelo y posición en el leaderboard.

**En vivo:** https://pulso-transmi-dashboard.vercel.app

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

Migraciones relacionadas: `supabase/migrations/0004_dashboard_state.sql` y `0005_dashboard_public.sql`.
