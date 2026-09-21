# Runbook

## 1. Puesta en marcha (una sola vez)

**Supabase** (proyecto `pulso-transmi`, ya creado y migrado con `supabase/migrations/0001…0003`)

1. Dashboard → *Project Settings → API Keys* → copia la clave **service_role** (secreta).
2. Local: pégala en `.env` como `SUPABASE_SERVICE_ROLE_KEY` (el archivo está en `.gitignore`).
3. Si algún día se recrea el proyecto: aplicar las migraciones en orden y cargar el histórico
   (`observations`, `context`, `stations`) desde `/v1/downloads/*`.

**API de Pulso**

1. Portal <https://pulso-transmi.72-60-245-2.sslip.io/> → ingresa con correo institucional y documento.
2. Genera tu API key (se muestra **una sola vez**; guárdala) y ponla en `.env` como `PULSO_API_KEY`.

**GitHub** (Settings → Secrets and variables → Actions)

| Tipo | Nombre | Valor |
|---|---|---|
| Secret | `PULSO_API_KEY` | tu API key |
| Secret | `SUPABASE_URL` | `https://tjqrgdzvhfxfovmwgzgv.supabase.co` |
| Secret | `SUPABASE_SERVICE_ROLE_KEY` | clave service_role |
| Variable | `PIPELINE_ENABLED` | `true` (activa los horarios; sin ella solo corre lo manual) |

**Champion inicial**: Actions → *Entrenar candidato* → *Run workflow*. Sin champion, la inferencia
falla con un mensaje claro (`No hay modelo champion`).

## 2. Operación normal

| Workflow | Cuándo | Qué hace |
|---|---|---|
| Recolectar y predecir | minuto 07 de cada hora (UTC) | recoge datos → si hay ciclo abierto, predice y envía |
| Recolectar y monitorear | minuto 37 de cada hora (UTC) | recoge datos → evalúa → decide → dispara entrenamiento si corresponde |
| Entrenar candidato | a mano, diario 08:20 UTC, o por el monitor | entrena, valida, promueve solo si mejora |

Ningún cron es la fuente de verdad del tiempo: la API decide si hay ciclo abierto y cuándo cierra.

## 3. Consultas útiles (SQL en Supabase)

```sql
-- ¿Qué modelo opera y por qué?
select version, status, validation->>'accuracy' acc, reason from model_versions order by created_at desc;
-- Entregas recientes
select cycle_id, status, attempt, submission_id, is_official, error from submissions order by id desc limit 20;
-- Últimas decisiones del monitor
select decided_at, decision, reason, evidence from retrain_decisions order by id desc limit 10;
-- Salud del collector
select started_at, status, rows_received, rows_upserted, error from ingestion_runs order by id desc limit 20;
```

## 4. Diagnóstico

| Síntoma | Causa probable | Acción |
|---|---|---|
| `No hay modelo champion` | no se ha entrenado | ejecutar *Entrenar candidato* |
| `401 invalid_api_key` | secret ausente/mal pegado/revocado | revisar `PULSO_API_KEY` (no imprimirla) |
| `409 cycle_closed` | la ejecución llegó tarde | queda como `missed`; revisar retrasos del cron |
| `422 invalid_target_set` | payload distinto del ciclo | no reintentar: revisar `predict.build_predictions` |
| `Artefacto entrenado con scikit-learn X` | versión distinta a la del entrenamiento | instalar de `requirements.txt` o reentrenar |
| decisión `investigate` por operación | retraso del collector, cobertura < 95 % o ≥ 3 fallos en 24 h | corregir la operación primero |
| el collector falla con `DataQualityError` | el stream trajo un lote inválido | el cursor no avanzó; inspeccionar el lote (estación desconocida, demanda fuera de rango) |

## 5. Rollback de modelo

```sql
select promote_model('<version_anterior>', 'rollback: <motivo>');
```

Es atómico (un solo champion) y las versiones anteriores siguen en Storage con su hash.
