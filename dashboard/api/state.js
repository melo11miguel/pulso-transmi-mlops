// Función de servidor: única capa con acceso a credenciales.
// El navegador jamás recibe la service_role de Supabase ni la API key de Pulso, solo el JSON
// agregado que devuelve este endpoint (regla de la guía operativa v2.0).
//
// Del leaderboard se publica nuestra posición y el mejor accuracy de la cohorte, pero NO los
// nombres de los demás estudiantes: esta URL es pública y esos nombres no son nuestros para
// publicarlos.

const TIMEOUT_MS = 15000;

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function leaderboard(base, key, window, yo) {
  const data = await fetchJson(`${base}/v1/leaderboard?window=${window}`, {
    headers: { Authorization: `Bearer ${key}` },
  });
  const rows = data.data || [];
  // La identidad viene de /v1/me (la deriva el servidor de la API key), no de un nombre quemado.
  const me = rows.find((r) => r.is_me) || rows.find((r) => r.display_name === yo);
  const best = rows[0] || null;
  return {
    ventana: window,
    participantes: rows.length,
    puesto: me ? me.rank : null,
    accuracy: me ? Number(me.accuracy.toFixed(2)) : null,
    cobertura: me ? Number(me.coverage.toFixed(3)) : null,
    // Sin nombre: solo la referencia numérica del mejor de la cohorte.
    mejor_accuracy: best ? Number(best.accuracy.toFixed(2)) : null,
    mejor_cobertura: best ? Number(best.coverage.toFixed(3)) : null,
    somos_lideres: me && best ? me.rank === 1 : null,
  };
}

export default async function handler(req, res) {
  const base = (process.env.PULSO_API_URL || "https://pulso-transmi.72-60-245-2.sslip.io").replace(/\/$/, "");
  const supabaseUrl = (process.env.SUPABASE_URL || "").replace(/\/$/, "");
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const apiKey = process.env.PULSO_API_KEY;

  if (!supabaseUrl || !supabaseKey) {
    return res.status(500).json({ error: "Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY" });
  }

  try {
    const headers = { apikey: supabaseKey, "Content-Type": "application/json" };
    if (supabaseKey.startsWith("eyJ")) headers.Authorization = `Bearer ${supabaseKey}`;

    const yo = apiKey
      ? await fetchJson(`${base}/v1/me`, { headers: { Authorization: `Bearer ${apiKey}` } })
          .then((d) => d.display_name)
          .catch(() => null)
      : null;

    const [estado, acumulada, rolling] = await Promise.all([
      fetchJson(`${supabaseUrl}/rest/v1/rpc/dashboard_state`, {
        method: "POST", headers, body: "{}",
      }),
      apiKey ? leaderboard(base, apiKey, "cumulative", yo).catch(() => null) : null,
      apiKey ? leaderboard(base, apiKey, "rolling_24h", yo).catch(() => null) : null,
    ]);

    // 60 s de caché: el pipeline entrega como mucho una vez por hora.
    res.setHeader("Cache-Control", "public, s-maxage=60, stale-while-revalidate=300");
    return res.status(200).json({ ...estado, leaderboard: { acumulada, rolling } });
  } catch (error) {
    return res.status(502).json({ error: `No se pudo leer el estado: ${error.message}` });
  }
}
