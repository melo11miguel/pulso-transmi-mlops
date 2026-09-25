-- Índices que cubren las llaves foráneas (recomendación del linter de Supabase).
create index if not exists model_versions_parent_idx      on public.model_versions (parent_version);
create index if not exists predictions_cycle_idx          on public.predictions (cycle_id);
create index if not exists retrain_decisions_model_idx    on public.retrain_decisions (model_version);
create index if not exists retrain_decisions_newmodel_idx on public.retrain_decisions (new_model_version);
create index if not exists submissions_model_idx          on public.submissions (model_version);
