-- Estado agregado del sistema para el dashboard (bono). Aplicada el 2026-09-23.
-- Devuelve solo datos agregados: ninguna credencial, ningún payload completo.
-- La ejecuta la función de servidor de Vercel con service_role; anon/authenticated no pueden.
-- El cuerpo vigente está en el proyecto; se reproduce con:
--   select pg_get_functiondef('public.dashboard_state()'::regprocedure);
-- Resumen de lo que expone:
--   accuracy_global, targets_evaluados, por_estacion, por_horizonte, serie_accuracy,
--   ciclos {vistos, entregados, perdidos}, modelo (champion), versiones, drift (nivel por
--   estación con su umbral), decisiones de reentrenamiento y operacion (últimas ejecuciones).
-- Permisos aplicados junto con la función:
revoke execute on function public.dashboard_state() from public, anon, authenticated;
grant  execute on function public.dashboard_state() to service_role;
