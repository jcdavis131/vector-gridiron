// Vector Gridiron — ESPN Fantasy read proxy.
//
// Why a proxy at all: ESPN's fantasy API sets no permissive CORS headers, so a
// static page can't call it from the browser, and private leagues require the
// user's `espn_s2` + `SWID` cookies — which browsers won't attach cross-site.
// This same-origin serverless function forwards a read-only GET to ESPN with
// those cookies (passed per-request, never stored) and relays the JSON back.
//
// It only ever talks to lm-api-reads.fantasy.espn.com and only forwards the
// cookies the caller supplied for their own league. Nothing is persisted.
//
// Modes:
//   (default) league snapshot — teams/rosters/settings/draft for one season
//   mode=players — resolve ESPN playerIds → name/position (for draft history)
module.exports = async (req, res) => {
  if (req.method !== "GET")
    return res.status(405).json({ error: "GET only" });

  const { leagueId, year, espn_s2, swid, mode, ids, views } = req.query || {};
  if (!year || !/^\d{4}$/.test(String(year)))
    return res.status(400).json({ error: "year (YYYY) required" });

  const headers = { "User-Agent": "vector-gridiron/1.0", Accept: "application/json" };
  if (espn_s2 && swid) {
    const swidBraced = /^\{.*\}$/.test(swid) ? swid : `{${swid}}`;
    headers.Cookie = `espn_s2=${espn_s2}; SWID=${swidBraced}`;
  }

  let url;
  if (mode === "players") {
    // Resolve a batch of ESPN playerIds (draft picks) to names/positions.
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
    // ESPN: seasons path for 2018+ (including completed years); leagueHistory
    // only for 2017 and earlier. The old "year < now → leagueHistory" rule 404s.
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
      return res.status(r.status).json({ error: `ESPN ${r.status}${hint}`, detail: body.slice(0, 200) });
    }
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "s-maxage=120, stale-while-revalidate=600");
    return res.status(200).send(body);
  } catch (e) {
    return res.status(502).json({ error: "upstream fetch failed", detail: String(e).slice(0, 200) });
  }
};
