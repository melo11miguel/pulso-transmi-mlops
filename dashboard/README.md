# Dashboard (bono)

Observabilidad del pipeline: accuracy en el tiempo, por estación y horizonte, drift, estado
operativo, versiones del modelo y posición en el leaderboard.

## Seguridad

- `api/state.js` es la **única** capa con acceso a credenciales. Corre en el servidor de Vercel.
- El navegador recibe únicamente JSON agregado: ni la `service_role` de Supabase ni `PULSO_API_KEY`
  salen del servidor, y no se usa ninguna variable `NEXT_PUBLIC_*`.
- No se publican nombres de otros participantes: solo nuestra posición y el mejor accuracy de la
  cohorte como referencia numérica.
- Toda la agregación vive en la función SQL `public.dashboard_state()` (migración 0004), cuyo
  permiso de ejecución es exclusivo de `service_role`.

## Variables de entorno en Vercel

| Variable | Uso |
|---|---|
| `SUPABASE_URL` | Proyecto de Supabase |
| `SUPABASE_SERVICE_ROLE_KEY` | Lectura del estado (solo servidor) |
| `PULSO_API_KEY` | Leaderboard e identidad vía `/v1/me` (solo servidor) |
| `PULSO_API_URL` | Opcional; por defecto la API pública del reto |
