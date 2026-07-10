// Vector Gridiron — ESPN Fantasy read proxy.
//
// Why a proxy at all: ESPN's fantasy API sets no permissive CORS headers, so a
// static page can't call it from the browser, and private leagues require the
// user's `espn_s2` + `SWID` cookies — which browsers won't attach cross-site.
// This same-origin serverless function forwards a read-only GET to ESPN with
// those cookies (passed per-request, never stored) and relays the JSON back.
//
// Cookies are accepted via query OR headers (X-Espn-S2 / X-Espn-Swid). Headers
// are preferred — espn_s2 is long and %-encoded; query strings double-encode
// and occasionally truncate on intermediaries.
//
// Modes:
//   (default) league snapshot — teams/rosters/settings/draft for one season
//   mode=players — resolve ESPN playerIds → name/position (for draft history)
//   mode=playerPool — full season player directory (fallback name resolve)
module.exports = async (req, res) => {
  if (req.method !== "GET")
    return res.status(405).json({ error: "GET only" });

  const q = req.query || {};
  const { leagueId, year, mode, ids, views } = q;
  // Prefer headers for cookies (avoids URL encoding / length issues).
  const hdr = req.headers || {};
  const espn_s2 = String(hdr["x-espn-s2"] || q.espn_s2 || "").trim();
  const swid = String(hdr["x-espn-swid"] || q.swid || "").trim();

  if (!year || !/^\d{4}$/.test(String(year)))
    return res.status(400).json({ error: "year (YYYY) required" });

  const headers = { "User-Agent": "vector-gridiron/1.0", Accept: "application/json" };
  if (espn_s2 && swid) {
    const swidBraced = /^\{.*\}$/.test(swid) ? swid : `{${swid}}`;
    // espn_s2 is often already URL-encoded from DevTools copy; decode once if so.
    let s2 = espn_s2;
    try {
      if (/%[0-9A-Fa-f]{2}/.test(s2)) s2 = decodeURIComponent(s2);
    } catch (_) { /* keep raw */ }
    headers.Cookie = `espn_s2=${s2}; SWID=${swidBraced}`;
  }

  let url;
  if (mode === "playerPool") {
    url = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/players?scoringPeriodId=0&view=players_wl`;
    headers["X-Fantasy-Filter"] = JSON.stringify({
      players: { limit: 5000, sortPercOwned: { sortAsc: false, sortPriority: 1 } },
    });
  } else if (mode === "players") {
    if (!ids || !/^-?\d+(,-?\d+)*$/.test(String(ids)))
      return res.status(400).json({ error: "ids=comma-separated playerIds required" });
    const idList = String(ids).split(",").map(Number).filter((n) => Number.isFinite(n));
    if (!idList.length || idList.length > 300)
      return res.status(400).json({ error: "ids must be 1–300 playerIds" });
    url = `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/players?scoringPeriodId=0&view=players_wl`;
    headers["X-Fantasy-Filter"] = JSON.stringify({ filterIds: { value: idList } });
  } else {
    if (!leagueId || !/^\d+$/.test(String(leagueId)))
      return res.status(400).json({ error: "leagueId required" });
    const viewList = (views && String(views).split(",").filter(Boolean).length)
      ? String(views).split(",").map((v) => v.trim()).filter(Boolean)
      : ["mTeam", "mRoster", "mSettings"];
    const viewQs = viewList.map((v) => `view=${encodeURIComponent(v)}`).join("&");
    url = Number(year) < 2018
      ? `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/leagueHistory/${leagueId}?${viewQs}&seasonId=${year}`
      : `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/segments/0/leagues/${leagueId}?${viewQs}`;
  }

  try {
    const r = await fetch(url, { headers });
    const body = await r.text();
    if (!r.ok) {
      const hint = r.status === 401
        ? " (private league — add valid espn_s2 + SWID cookies)"
        : r.status === 404 ? " (check League ID and season year)" : "";
      return res.status(r.status).json({
        error: `ESPN ${r.status}${hint}`,
        detail: body.slice(0, 200),
        hasCookies: !!(espn_s2 && swid),
      });
    }
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate=600");
    return res.status(200).send(body);
  } catch (e) {
    return res.status(502).json({ error: "upstream fetch failed", detail: String(e).slice(0, 200) });
  }
};
