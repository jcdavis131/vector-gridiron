/* Vector Gridiron — fantasy cockpit client.
   Static, zero-backend for reads except the ESPN cookie proxy (api/espn.js).
   Sleeper is CORS-open and called directly. All league creds stay in
   localStorage and go straight to the platform (or our thin proxy). */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const POS = ['QB', 'RB', 'WR', 'TE'];                     // skill (vector space / MTNN)
const ALL_POS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];     // every fantasy lineup position
const SLOT_POS = ALL_POS;                                // dedicated lineup slots
const LINEUP_POS = new Set(SLOT_POS);                    // startable positions
const POS_COLOR = { QB: '#e8b23a', RB: '#4bd0a0', WR: '#5aa0f0', TE: '#c77dff', K: '#9aa4b2', DST: '#6b7c8f' };
const FLEX_ELIG = new Set(['RB', 'WR', 'TE']);
// ESPN proTeamId → nflverse/kdst team abbrev (for D/ST name joins)
const ESPN_TEAM = {
  1: 'ATL', 2: 'BUF', 3: 'CHI', 4: 'CIN', 5: 'CLE', 6: 'DAL', 7: 'DEN', 8: 'DET',
  9: 'GB', 10: 'TEN', 11: 'IND', 12: 'KC', 13: 'LV', 14: 'LA', 15: 'MIA', 16: 'MIN',
  17: 'NE', 18: 'NO', 19: 'NYG', 20: 'NYJ', 21: 'PHI', 22: 'ARI', 23: 'PIT', 24: 'LAC',
  25: 'SF', 26: 'SEA', 27: 'TB', 28: 'WAS', 29: 'CAR', 30: 'JAX', 33: 'BAL', 34: 'HOU',
};

const STATE = {
  vectors: null, proj: null, next: null, adp: null, lookback: null, kdst: null,
  projByKey: new Map(), vecByKey: new Map(), nextByKey: new Map(), adpByKey: new Map(),
  lookbackBySeason: new Map(),
  lookbackSeed: null,              // pristine seeded mock DB (restored when league disconnects)
  lookbackLive: null,              // {leagueName, seasons:Map, status} when real history loaded
  names: [], latestSeason: null,   // autocomplete pool (latest season)
  scoring: 'ppr',
  league: null,                    // {name, slots, teams:[{id,name,roster,starters}], myId}
  mineKeys: new Set(),             // normKeys on the user's team
  rosterOwner: new Map(),          // normKey -> team name that rosters the player
  espnCreds: null,                 // {s2, swid} for historic ESPN fetches
};
function rebuildOwnership() {
  STATE.mineKeys = new Set();
  STATE.rosterOwner = new Map();
  const lg = STATE.league;
  if (!lg) return;
  for (const t of lg.teams) for (const pl of t.roster) {
    const k = normKey(pl.name, pl.pos);
    STATE.rosterOwner.set(k, t.name);
    if (t.id === lg.myId) STATE.mineKeys.add(k);
  }
}
const isMine = (name, pos) => STATE.mineKeys.has(normKey(name, pos));
const ownerOf = (name, pos) => STATE.rosterOwner.get(normKey(name, pos)) || null;
const nextFor = (name, pos) => STATE.nextByKey.get(normKey(name, pos)) || null;
const adpFor = (name, pos) => STATE.adpByKey.get(normKey(name, pos)) || null;
// draft-day value = model's positional rank vs the market's positional ADP rank.
// positive edge = model ranks him higher AT HIS POSITION than the room does.
function valueEdge(p) {
  const a = adpFor(p.name, p.pos);
  return a && a.pos_rank && p.rank_pos ? a.pos_rank - p.rank_pos : null;
}
function valueBadge(p) {
  const e = valueEdge(p);
  if (e == null) return '';
  if (e >= 8) return `<span class="vg-val vg-val--up" title="model ${p.pos}${p.rank_pos} vs ADP ${p.pos}${adpFor(p.name, p.pos).pos_rank}">VALUE +${e}</span>`;
  if (e <= -8) return `<span class="vg-val vg-val--dn" title="model ${p.pos}${p.rank_pos} vs ADP ${p.pos}${adpFor(p.name, p.pos).pos_rank}">REACH ${e}</span>`;
  return `<span class="vg-val vg-val--fair">fair</span>`;
}
const adpText = p => { const a = adpFor(p.name, p.pos); return a ? a.adp.toFixed(1) : '<span class="vg-unmatched">—</span>'; };

// availability: BYE / injury report (Out/Doubtful/Questionable) / roster status
const OUT_FLAGS = new Set(['Out', 'BYE', 'RES', 'CUT', 'Doubtful', 'IR', 'PUP', 'SUS', 'RET']);
function isUnavailable(rec) { return !!(rec && rec.avail && OUT_FLAGS.has(rec.avail)); }
function availBadge(rec) {
  const a = rec && rec.avail;
  if (!a) return '';
  const cls = a === 'BYE' ? 'bye' : OUT_FLAGS.has(a) ? 'out' : 'q';
  return ` <span class="vg-avail vg-avail--${cls}" title="availability">${escapeHtml(a)}</span>`;
}
function movedBadge(rec) {
  return rec && rec.moved
    ? ` <span class="vg-moved" title="changed teams this offseason (was ${escapeHtml(rec.prev_team || '')})">↔ ${escapeHtml(rec.team)}</span>` : '';
}
const byeText = rec => rec && rec.bye ? String(rec.bye) : '<span class="vg-unmatched">—</span>';

/* ---------- name matching (mirrors pipeline norm_key) ---------- */
function normKey(name, pos) {
  let s = (name || '').toLowerCase();
  s = s.replace(/[.'`]/g, '');
  s = s.replace(/\s+(jr|sr|ii|iii|iv|v)$/i, '');
  s = s.replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim();
  return `${s}|${pos}`;
}
const projFor = (name, pos) => STATE.projByKey.get(normKey(name, pos)) || null;

/* ---------- boot ---------- */
async function boot() {
  try {
    const [vec, proj, next, adp, look, kdst] = await Promise.all([
      fetch('assets/vectors.json').then(r => r.json()),
      fetch('assets/projections.json').then(r => r.json()),
      fetch('assets/nextgame.json').then(r => r.json()).catch(() => null),
      fetch('assets/adp.json').then(r => r.json()).catch(() => null),
      fetch('assets/lookback_seasons.json').then(r => r.json()).catch(() => null),
      fetch('assets/kdst.json').then(r => r.json()).catch(() => null),
    ]);
    STATE.vectors = vec;
    STATE.proj = proj;
    STATE.next = next;
    STATE.adp = adp;
    STATE.lookback = look;
    STATE.lookbackSeed = look;
    STATE.kdst = kdst;
    if (look) for (const s of look.seasons) STATE.lookbackBySeason.set(s.season, s);
    // K/DST get real projections everywhere (roster, start/sit, waivers) — they
    // live in kdst.json (nflverse zeroes kicker points), merged as proj records.
    if (kdst) for (const [arr, pos] of [[kdst.kickers, 'K'], [kdst.dst, 'DST']]) {
      for (const p of (arr || [])) {
        const key = normKey(p.name, pos);
        STATE.projByKey.set(key, { key, name: p.name, pos, team: p.team,
          proj: p.proj, floor: Math.max(0, +(p.proj - 4).toFixed(1)), ceil: +(p.proj + 4).toFixed(1),
          bye: p.bye ?? null, comps: [] });
      }
    }
    $('#foot-built').textContent = proj.built || vec.built || '';
    for (const p of proj.players) STATE.projByKey.set(p.key, p);
    if (next) for (const p of next.players) STATE.nextByKey.set(p.key, p);
    if (adp) for (const p of adp.players) STATE.adpByKey.set(p.key, p);
    const latest = vec.seasons[vec.seasons.length - 1];
    STATE.latestSeason = latest;
    for (const p of vec.players) {
      if (p.season === latest) STATE.vecByKey.set(normKey(p.name, p.pos), p);
    }
    // search pool = projection board + K/DST (kdst.json)
    STATE.names = proj.players.map(p => ({ name: p.name, pos: p.pos, team: p.team }));
    if (kdst) {
      for (const p of (kdst.kickers || [])) STATE.names.push({ name: p.name, pos: 'K', team: p.team });
      for (const p of (kdst.dst || [])) STATE.names.push({ name: p.name, pos: 'DST', team: p.team });
    }
    labelSeasons();
    buildCareer();
    initMap(vec, latest);
    initLookback();
    initDraft();
    initNextGame();
    restoreSession();
    maybeOnboard();
  } catch (e) {
    showError('Could not load model artifacts. Run the pipeline: '
      + 'python pipeline/build_vectors.py && python pipeline/train_models.py');
    console.error(e);
  }
}

// season-aware labels sprinkled through the header/footer/tabs
function labelSeasons() {
  const last = STATE.latestSeason, up = STATE.proj.proj_season || last + 1;
  $('#draft-season-label') && ($('#draft-season-label').textContent = up);
  $('#nextgame-week-label') && (STATE.next
    ? ($('#nextgame-week-label').textContent = `${STATE.next.season} · Week ${STATE.next.week}`)
    : ($('#nextgame-week-label').textContent = ''));
  const r = STATE.proj.model?.report;
  if (r) $('#foot-model').textContent =
    `MTNN next-game MAE ${r.model_fpts_mae} (R² ${r.model_fpts_r2}) · ${last} data`;
}

function showError(msg) {
  const b = $('#error-banner'); b.textContent = msg; b.hidden = false;
}

/* ================================================================
   TABS
   ================================================================ */
$$('.vg-tab').forEach(t => t.addEventListener('click', () => selectView(t.dataset.view)));
function selectView(view) {
  $$('.vg-tab').forEach(t => t.setAttribute('aria-selected', String(t.dataset.view === view)));
  $$('.vg-view').forEach(v => v.classList.toggle('is-active', v.id === `view-${view}`));
}
document.addEventListener('click', e => {
  const j = e.target.closest('[data-jump]');
  if (j) { e.preventDefault(); selectView(j.dataset.jump); }
});

/* ================================================================
   CONNECT MODAL
   ================================================================ */
let PLATFORM = 'espn';
const openConnect = () => { $('#connect-backdrop').hidden = false; };
const closeConnect = () => { $('#connect-backdrop').hidden = true; };
$('#connect-btn').addEventListener('click', openConnect);
$('#connect-btn-2')?.addEventListener('click', openConnect);
$('#connect-cancel').addEventListener('click', closeConnect);
$('#connect-backdrop').addEventListener('click', e => { if (e.target.id === 'connect-backdrop') closeConnect(); });

$$('#platform-pick button').forEach(b => b.addEventListener('click', () => {
  PLATFORM = b.dataset.platform;
  $$('#platform-pick button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  $('#fields-espn').hidden = PLATFORM !== 'espn';
  $('#fields-sleeper').hidden = PLATFORM !== 'sleeper';
}));

$('#connect-go').addEventListener('click', onConnect);

function setStatus(msg, kind, html) {
  const s = $('#connect-status');
  s.className = 'vg-status' + (kind ? ` vg-status--${kind}` : '');
  if (html) s.innerHTML = msg; else s.textContent = msg;
}

/* Parse espn_s2 + SWID out of anything the user pastes: the bookmarklet output
   (espn_s2=…; SWID=…), a whole document.cookie string, OR the two raw values
   copied straight from DevTools (in any order, on separate lines). */
const isSwid = x => /^\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?$/i.test(x);
function parseEspnCookies(text) {
  const t = String(text || '').trim();
  let s2 = (t.match(/espn_s2=([^;\s]+)/i) || [])[1] || '';
  let swid = (t.match(/SWID=(\{[^}]*\}|[^;\s]+)/i) || [])[1] || '';
  if (!s2 || !swid) {                       // fall back: infer from raw tokens
    const tokens = t.split(/[\s;,]+/).map(x => x.replace(/^(espn_s2|SWID)=/i, '').trim()).filter(Boolean);
    for (const tok of tokens) {
      if (!swid && isSwid(tok)) swid = tok;
      else if (!s2 && tok.length > 40 && !isSwid(tok)) s2 = tok;
    }
  }
  return { s2: s2.trim(), swid: swid.trim() };
}
(function espnCookieHelpers() {
  const box = $('#espn-cookies');
  if (!box) return;
  box.addEventListener('input', () => {
    const { s2, swid } = parseEspnCookies(box.value);
    $('#espn-s2').value = s2;
    $('#espn-swid').value = swid;
    const st = $('#espn-cookie-status');
    if (s2 && swid) { st.textContent = '✓ found espn_s2 and SWID — ready'; st.className = 'vg-cookie-status is-ok'; }
    else if (s2 || swid) { st.textContent = `found ${s2 ? 'espn_s2' : 'SWID'} — still need ${s2 ? 'SWID' : 'espn_s2'}`; st.className = 'vg-cookie-status is-warn'; }
    else { st.textContent = box.value.trim() ? "couldn't find espn_s2 / SWID in that paste" : ''; st.className = 'vg-cookie-status' + (box.value.trim() ? ' is-warn' : ''); }
  });
  // clicking the bookmarklet here would run against OUR page, not ESPN — so copy it instead
  $('#espn-bookmarklet')?.addEventListener('click', e => {
    e.preventDefault();
    navigator.clipboard?.writeText(e.currentTarget.getAttribute('href'));
    const st = $('#espn-cookie-status');
    st.textContent = 'Bookmarklet copied — save it as a browser bookmark, then click it while on fantasy.espn.com.';
    st.className = 'vg-cookie-status is-warn';
  });
})();

async function onConnect() {
  try {
    let league;
    if (PLATFORM === 'espn') {
      const leagueId = $('#espn-league').value.trim();
      const year = $('#espn-year').value.trim() || '2024';
      if (!leagueId) return setStatus('Enter your ESPN League ID.', 'err');
      setStatus('Loading ESPN league…', 'busy');
      league = await loadEspn(leagueId, year, $('#espn-s2').value.trim(), $('#espn-swid').value.trim());
    } else {
      const leagueId = $('#sleeper-league').value.trim();
      if (!leagueId) return setStatus('Enter your Sleeper League ID.', 'err');
      setStatus('Loading Sleeper league…', 'busy');
      league = await loadSleeper(leagueId);
    }
    STATE.league = league;
    STATE.scoring = league.scoring;
    populateTeamSelect(league);
    setStatus(`Loaded "${league.name}" — ${league.teams.length} teams. Pick your team above, then it saves.`, 'ok');
    persistSession();
    afterLeagueLoaded();
  } catch (e) {
    console.error(e);
    if (PLATFORM === 'espn' && e.status === 401) {
      $('#espn-cookie-adv').open = true;
      setStatus(
        "This is a <b>private</b> league, so ESPN needs proof you're in it:<br>"
        + "• <b>Easiest (works on your phone):</b> if you're the league manager, make the "
        + "league public — see the tip above. Then no cookies needed.<br>"
        + "• <b>Otherwise:</b> paste your espn_s2 + SWID cookies below, then Load again.",
        'err', true);
    } else if (PLATFORM === 'espn' && e.status === 404) {
      setStatus("No league found for that ID + year. Check the League ID, and try season year 2025.", 'err');
    } else {
      setStatus('Could not load league: ' + (e.message || e), 'err');
    }
  }
}

function populateTeamSelect(league) {
  const sel = $('#my-team');
  sel.innerHTML = '<option value="">— pick your team —</option>'
    + league.teams.map(t => `<option value="${t.id}">${escapeHtml(t.name)}</option>`).join('');
  sel.value = league.myId || '';
  sel.onchange = () => {
    league.myId = sel.value;
    persistSession();
    afterLeagueLoaded();
  };
}

/* ---------- SLEEPER (direct, CORS-open) ---------- */
let SLEEPER_PLAYERS = null;
async function sleeperPlayers() {
  if (SLEEPER_PLAYERS) return SLEEPER_PLAYERS;
  const cached = localStorage.getItem('vg_sleeper_players_v2');
  const cachedAt = +localStorage.getItem('vg_sleeper_players_v2_at') || 0;
  if (cached && Date.now() - cachedAt < 7 * 864e5) {
    SLEEPER_PLAYERS = JSON.parse(cached); return SLEEPER_PLAYERS;
  }
  const all = await fetch('https://api.sleeper.app/v1/players/nfl').then(r => r.json());
  const slim = {};
  for (const [id, p] of Object.entries(all)) {
    if (!p.full_name && p.position !== 'DEF') continue;
    const team = (p.team || '').trim();
    if (p.position === 'DEF') {
      // Join key matches kdst.json ("BAL DST"), not ESPN-style "Ravens D/ST".
      slim[id] = { name: team ? `${team} DST` : (p.full_name || 'DST'), pos: 'DST', team };
    } else if (p.full_name) {
      slim[id] = { name: p.full_name, pos: p.position, team };
    }
  }
  try {
    localStorage.setItem('vg_sleeper_players_v2', JSON.stringify(slim));
    localStorage.setItem('vg_sleeper_players_v2_at', String(Date.now()));
  } catch (_) { /* 5MB may exceed quota; fine, keep in-memory */ }
  SLEEPER_PLAYERS = slim; return slim;
}

async function loadSleeper(leagueId) {
  const base = 'https://api.sleeper.app/v1/league/' + leagueId;
  const [lg, rosters, users, players] = await Promise.all([
    fetch(base).then(r => { if (!r.ok) throw new Error('league not found'); return r.json(); }),
    fetch(base + '/rosters').then(r => r.json()),
    fetch(base + '/users').then(r => r.json()),
    sleeperPlayers(),
  ]);
  const userById = Object.fromEntries(users.map(u => [u.user_id, u]));
  const rp = lg.scoring_settings?.rec ?? 0;
  const scoring = rp >= 1 ? 'ppr' : rp >= 0.5 ? 'half' : 'std';
  const slots = countSlots(lg.roster_positions || []);
  const teams = rosters.map(r => {
    const u = userById[r.owner_id];
    const mk = id => {
      const p = players[id]; if (!p) return null;
      return { name: p.name, pos: p.pos, team: p.team };
    };
    const roster = (r.players || []).map(mk).filter(Boolean);
    const starters = (r.starters || []).filter(x => x && x !== '0').map(mk).filter(Boolean);
    return {
      id: String(r.roster_id),
      name: u?.metadata?.team_name || u?.display_name || `Team ${r.roster_id}`,
      avatar: u?.avatar ? `https://sleepercdn.com/avatars/thumbs/${u.avatar}` : '',
      ownerId: r.owner_id != null ? String(r.owner_id) : null,
      ownerName: u?.display_name || null,
      manager: u?.display_name || '',
      roster, starters,
    };
  });
  return {
    platform: 'sleeper', name: lg.name, scoring, slots, teams, myId: null, leagueId,
    year: String(lg.season || ''), previousLeagueId: lg.previous_league_id || null,
  };
}

/** Walk Sleeper previous_league_id chain and pull each season's completed draft. */
async function loadSleeperSeasonHistory(leagueId) {
  const players = await sleeperPlayers();
  const seasons = [];
  let id = leagueId;
  const seen = new Set();
  while (id && !seen.has(id) && seasons.length < 12) {
    seen.add(id);
    try {
      const [lg, users, drafts] = await Promise.all([
        fetch('https://api.sleeper.app/v1/league/' + id).then(r => r.ok ? r.json() : null),
        fetch('https://api.sleeper.app/v1/league/' + id + '/users').then(r => r.ok ? r.json() : []),
        fetch('https://api.sleeper.app/v1/league/' + id + '/drafts').then(r => r.ok ? r.json() : []),
      ]);
      if (!lg) break;
      const draft = (drafts || []).find(d => d.status === 'complete')
        || (drafts || []).find(d => d.status === 'drafting' || d.status === 'paused')
        || (drafts || [])[0];
      if (draft?.draft_id) {
        const picksRaw = await fetch('https://api.sleeper.app/v1/draft/' + draft.draft_id + '/picks')
          .then(r => r.ok ? r.json() : []);
        if (!(picksRaw || []).length) {
          id = lg.previous_league_id || null;
          continue;
        }
        const userById = Object.fromEntries((users || []).map(u => [u.user_id, u]));
        // Also need roster_id → owner for standings; fetch rosters
        const rosters = await fetch('https://api.sleeper.app/v1/league/' + id + '/rosters')
          .then(r => r.ok ? r.json() : []);
        const teamsByRoster = {};
        for (const r of rosters) {
          const u = userById[r.owner_id];
          teamsByRoster[r.roster_id] = {
            id: String(r.roster_id),
            name: u?.metadata?.team_name || u?.display_name || `Team ${r.roster_id}`,
            ownerId: r.owner_id != null ? String(r.owner_id) : null,
            ownerName: u?.display_name || null,
            wins: r.settings?.wins || 0,
            losses: r.settings?.losses || 0,
            ties: r.settings?.ties || 0,
            points_for: +((r.settings?.fpts || 0) + (r.settings?.fpts_decimal || 0) / 100).toFixed(1),
            seed: r.settings?.rank || 99,
            champion: false,
            picks: [],
          };
        }
        for (const pk of (picksRaw || [])) {
          const tm = teamsByRoster[pk.roster_id];
          if (!tm) continue;
          const pl = players[pk.player_id];
          if (!pl || !ALL_POS.includes(pl.pos)) continue;
          tm.picks.push({
            pick: pk.pick_no,
            name: pl.name,
            pos: pl.pos,
            team: pl.team || '',
            key: normKey(pl.name, pl.pos),
          });
        }
        for (const tm of Object.values(teamsByRoster)) tm.picks.sort((a, b) => a.pick - b.pick);
        const teams = Object.values(teamsByRoster).filter(t => t.picks.length);
        if (teams.length) {
          const ranked = [...teams].sort((a, b) => a.seed - b.seed || b.wins - a.wins || b.points_for - a.points_for);
          if (ranked[0]) ranked[0].champion = true;
          seasons.push({ season: +(lg.season || draft.season), leagueName: lg.name, teams });
        }
      }
      id = lg.previous_league_id || null;
    } catch (e) {
      console.warn('sleeper history walk failed', e);
      break;
    }
  }
  return seasons;
}
function countSlots(arr) {
  const s = { QB: 0, RB: 0, WR: 0, TE: 0, FLEX: 0, K: 0, DST: 0 };
  for (const p of arr) {
    if (p === 'FLEX' || p === 'WRRB_FLEX' || p === 'REC_FLEX' || p === 'SUPER_FLEX') s.FLEX++;
    else if (p === 'DEF' || p === 'DST') s.DST++;
    else if (s[p] !== undefined) s[p]++;
  }
  return s;
}

/* ---------- ESPN (via thin serverless proxy at /api/espn) ---------- */
const ESPN_POS = { 1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST' };
const ESPN_PLAYER_POOL = new Map(); // year → Map(playerId → {name,pos,team})

async function espnProxy(params) {
  // Cookies go in headers (preferred) — espn_s2 is long; query strings double-encode.
  const { espn_s2, swid, ...rest } = params;
  const qs = new URLSearchParams(rest);
  const headers = {};
  if (espn_s2) headers['X-Espn-S2'] = espn_s2;
  if (swid) headers['X-Espn-Swid'] = swid;
  const res = await fetch('/api/espn?' + qs.toString(), { headers });
  if (!res.ok) {
    let msg = `ESPN returned ${res.status}`;
    try { msg = (JSON.parse(await res.text()).error) || msg; } catch (_) {}
    const err = new Error(msg); err.status = res.status; throw err;
  }
  let data = await res.json();
  // leagueHistory (≤2017) returns [{league}]; players / playerPool return [player,…].
  // Only unwrap the leagueHistory shape — never collapse a player list to data[0].
  if (Array.isArray(data) && rest.mode !== 'players' && rest.mode !== 'playerPool') {
    data = data[0];
  }
  return data;
}

function espnCredParams(s2, swid) {
  return {
    ...(s2 ? { espn_s2: s2 } : {}),
    ...(swid ? { swid } : {}),
  };
}

async function loadEspn(leagueId, year, s2, swid) {
  STATE.espnCreds = { s2: s2 || '', swid: swid || '' };
  const data = await espnProxy({
    leagueId, year,
    views: 'mTeam,mRoster,mSettings',
    ...espnCredParams(s2, swid),
  });
  const settings = data.settings || {};
  const scoring = detectEspnScoring(settings);
  const slots = espnSlots(settings);
  const membersById = espnMembersById(data);
  const teams = (data.teams || []).map(t => {
    const entries = t.roster?.entries || [];
    const roster = [], starters = [];
    for (const e of entries) {
      const pl = e.playerPoolEntry?.player || e.player;
      if (!pl) continue;
      const pos = ESPN_POS[pl.defaultPositionId] || (pl.id < 0 ? 'DST' : '?');
      const rec = { name: espnPlayerName(pl, pos), pos, team: ESPN_TEAM[pl.proTeamId] || '' };
      roster.push(rec);
      if (e.lineupSlotId !== 20 && e.lineupSlotId !== 21) starters.push(rec); // 20 bench,21 IR
    }
    const own = espnOwnerFields(t, membersById);
    return {
      id: String(t.id), name: espnTeamName(t), avatar: t.logo || '',
      ownerId: own.ownerId, ownerName: own.ownerName, manager: own.ownerName || '',
      roster, starters,
    };
  });
  return { platform: 'espn', name: settings.name || `ESPN ${leagueId}`, scoring, slots, teams, myId: null, leagueId, year };
}

/** Canonical display/join name — DST becomes "BAL DST" to match kdst.json. */
function espnPlayerName(pl, pos) {
  if (pos === 'DST') {
    const abbr = ESPN_TEAM[pl.proTeamId] || ESPN_TEAM[Math.abs(pl.id) - 16000];
    return abbr ? `${abbr} DST` : (pl.fullName || 'DST');
  }
  return pl.fullName || '';
}

function espnPlayerRec(p) {
  const pos = ESPN_POS[p.defaultPositionId] || (p.id < 0 ? 'DST' : '?');
  return { name: espnPlayerName(p, pos), pos, team: ESPN_TEAM[p.proTeamId] || '' };
}

/** Full-season player directory (cached) — fallback when filterIds resolve is thin. */
async function espnPlayerPool(year, s2, swid) {
  const y = String(year);
  if (ESPN_PLAYER_POOL.has(y)) return ESPN_PLAYER_POOL.get(y);
  const out = new Map();
  try {
    const rows = await espnProxy({ mode: 'playerPool', year: y, ...espnCredParams(s2, swid) });
    for (const p of (Array.isArray(rows) ? rows : [])) {
      if (p?.id == null) continue;
      out.set(p.id, espnPlayerRec(p));
    }
  } catch (e) { console.warn('espn playerPool failed', y, e); }
  ESPN_PLAYER_POOL.set(y, out);
  return out;
}

/** Resolve ESPN playerIds → {id, name, pos} via players mode, then playerPool fill. */
async function espnResolvePlayers(year, playerIds, s2, swid) {
  const uniq = [...new Set(playerIds.filter(id => id != null))];
  const out = new Map();
  const chunk = 80;
  for (let i = 0; i < uniq.length; i += chunk) {
    const ids = uniq.slice(i, i + chunk);
    try {
      const rows = await espnProxy({
        mode: 'players', year, ids: ids.join(','),
        ...espnCredParams(s2, swid),
      });
      for (const p of (Array.isArray(rows) ? rows : [])) {
        if (p?.id == null) continue;
        out.set(p.id, espnPlayerRec(p));
      }
    } catch (e) { console.warn('espn player resolve failed', e); }
  }
  const missing = uniq.filter(id => !out.has(id));
  if (missing.length) {
    const pool = await espnPlayerPool(year, s2, swid);
    for (const id of missing) {
      const rec = pool.get(id);
      if (rec) out.set(id, rec);
    }
  }
  return out;
}

/** One completed ESPN season: draft picks + standings, names resolved. */
async function loadEspnSeasonHistory(leagueId, year, s2, swid) {
  const data = await espnProxy({
    leagueId, year,
    views: 'mDraftDetail,mTeam,mSettings',
    ...espnCredParams(s2, swid),
  });
  const picks = data.draftDetail?.picks || [];
  // Prefer drafted=true, but accept any season that still has pick rows (offseason quirks).
  if (!picks.length) return null;
  const resolved = await espnResolvePlayers(year, picks.map(p => p.playerId), s2, swid);
  const membersById = espnMembersById(data);
  const teamsById = {};
  for (const t of (data.teams || [])) {
    const rec = t.record?.overall || {};
    const own = espnOwnerFields(t, membersById);
    teamsById[t.id] = {
      id: String(t.id),
      name: espnTeamName(t),
      ownerId: own.ownerId,
      ownerName: own.ownerName,
      wins: rec.wins || 0,
      losses: rec.losses || 0,
      ties: rec.ties || 0,
      points_for: +(t.points || rec.pointsFor || 0).toFixed(1),
      seed: t.playoffSeed || t.rankCalculatedFinal || 99,
      champion: t.rankCalculatedFinal === 1,
      picks: [],
    };
  }
  for (const pk of picks) {
    const tm = teamsById[pk.teamId];
    if (!tm) continue;
    const pl = resolved.get(pk.playerId);
    if (!pl || !ALL_POS.includes(pl.pos)) continue;
    tm.picks.push({
      pick: pk.overallPickNumber,
      name: pl.name,
      pos: pl.pos,
      team: pl.team || '',
      key: normKey(pl.name, pl.pos),
    });
  }
  for (const tm of Object.values(teamsById)) tm.picks.sort((a, b) => a.pick - b.pick);
  const teams = Object.values(teamsById).filter(t => t.picks.length);
  if (!teams.length) return null;
  // If ESPN didn't mark a champion (rankFinal often 0 offseason), take #1 seed / best record.
  if (!teams.some(t => t.champion)) {
    const ranked = [...teams].sort((a, b) => a.seed - b.seed || b.wins - a.wins || b.points_for - a.points_for);
    if (ranked[0]) ranked[0].champion = true;
  }
  return { season: +year, leagueName: data.settings?.name || '', teams };
}
function espnTeamName(t) {
  return t.name || [t.location, t.nickname].filter(Boolean).join(' ').trim() || `Team ${t.id}`;
}
/** Map ESPN member SWID → display name (stable across team renames). */
function espnMembersById(data) {
  const m = new Map();
  for (const mem of (data.members || [])) {
    if (!mem?.id) continue;
    const full = [mem.firstName, mem.lastName].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
    const name = full || mem.displayName || String(mem.id);
    m.set(mem.id, { id: mem.id, name, displayName: mem.displayName || name });
  }
  return m;
}
function espnOwnerFields(t, membersById) {
  const ownerId = t.primaryOwner || (Array.isArray(t.owners) && t.owners[0]) || null;
  const mem = ownerId != null ? membersById.get(ownerId) : null;
  return {
    ownerId: ownerId != null ? String(ownerId) : null,
    ownerName: mem?.name || null,
  };
}
function detectEspnScoring(settings) {
  const items = settings.scoringSettings?.scoringItems || [];
  const rec = items.find(i => i.statId === 53);   // 53 = receptions
  const pts = rec ? (rec.points ?? rec.pointsOverrides?.['16'] ?? 0) : 0;
  return pts >= 1 ? 'ppr' : pts >= 0.5 ? 'half' : 'std';
}
function espnSlots(settings) {
  const c = settings.rosterSettings?.lineupSlotCounts || {};
  return {
    QB: c[0] || 0, RB: c[2] || 0, WR: c[4] || 0, TE: c[6] || 0,
    FLEX: (c[23] || 0) + (c[7] || 0),   // 23 RB/WR/TE flex, 7 OP treated as flex
    K: c[17] || 0, DST: c[16] || 0,     // 17 K, 16 D/ST
  };
}

/* ---------- DEMO league (synthetic, from the real projection board) ---------- */
function loadDemo() {
  const board = [...STATE.projByKey.values()].sort((a, b) => b.proj - a.proj);
  const byPos = p => board.filter(x => x.pos === p);
  const pools = { QB: byPos('QB'), RB: byPos('RB'), WR: byPos('WR'), TE: byPos('TE'), K: byPos('K'), DST: byPos('DST') };
  const T = 10, need = { QB: 2, RB: 5, WR: 6, TE: 2, K: 1, DST: 1 };  // full roster incl K/DST
  const teams = Array.from({ length: T }, (_, i) => ({
    id: String(i + 1),
    name: DEMO_NAMES[i],
    ownerId: `demo-owner-${i + 1}`,
    ownerName: DEMO_NAMES[i],
    manager: DEMO_NAMES[i],
    roster: [],
    starters: [],
  }));
  // snake draft for realism: each round, teams pick the best remaining at each
  // position, alternating draft order round-to-round.
  const cursor = { QB: 0, RB: 0, WR: 0, TE: 0, K: 0, DST: 0 };
  for (let r = 0; r < Math.max(...Object.values(need)); r++) {
    for (const pos of SLOT_POS) {
      if (r >= need[pos]) continue;
      const seq = r % 2 === 0 ? teams : [...teams].reverse();
      for (const t of seq) {
        const pick = pools[pos][cursor[pos]++];
        if (pick) t.roster.push({ name: pick.name, pos: pick.pos, team: pick.team });
      }
    }
  }
  const slots = { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1 };
  for (const t of teams) t.starters = optimalLineup(t.roster, slots).starters.map(s => s.player);
  STATE.league = { platform: 'demo', name: 'Vector Gridiron Demo (PPR)', scoring: 'ppr', slots, teams, myId: '1', leagueId: 'demo' };
  STATE.scoring = 'ppr';
  closeConnect();
  afterLeagueLoaded();
}
const DEMO_NAMES = ['Gridiron Gang', 'The Analysts', 'Vector Victors', 'Cache Money',
  'Regression Kings', 'Zero RB Zealots', 'Waiver Wire Warlocks', 'Ceiling Chasers',
  'Floor Generals', 'Model Behavior'];

/* ================================================================
   OPTIMIZER — greedy lineup fill by projection
   ================================================================ */
function proj(player) {
  const p = projFor(player.name, player.pos);
  return p ? p.proj : null;
}
// greedy: fill dedicated positional slots first (by value), then FLEX.
// `withVal` = [{player:{pos,...}, p:<number>}]. Shared by the start/sit
// optimizer (value = projection) and the Lookback what-if (value = actual ppg).
function greedyFill(withVal, slots) {
  const wp = [...withVal].sort((a, b) => b.p - a.p);
  const slotList = [];
  for (const p of SLOT_POS) for (let i = 0; i < (slots[p] || 0); i++) slotList.push(p);
  for (let i = 0; i < (slots.FLEX || 0); i++) slotList.push('FLEX');
  const used = new Set(), starters = [];
  for (const slot of slotList.filter(s => s !== 'FLEX')) {
    const c = wp.find(x => !used.has(x) && x.player.pos === slot);
    if (c) { used.add(c); starters.push({ slot, ...c }); }
    else starters.push({ slot, player: null, p: 0 });
  }
  for (const _ of slotList.filter(s => s === 'FLEX')) {
    const c = wp.find(x => !used.has(x) && FLEX_ELIG.has(x.player.pos));
    if (c) { used.add(c); starters.push({ slot: 'FLEX', ...c }); }
    else starters.push({ slot: 'FLEX', player: null, p: 0 });
  }
  const total = starters.reduce((s, x) => s + (x.p || 0), 0);
  return { starters, total, used };
}
function optimalLineup(roster, slots) {
  const withProj = roster
    .map(pl => ({ player: pl, p: proj(pl) }))
    .filter(x => x.p != null && LINEUP_POS.has(x.player.pos));
  const { starters, total, used } = greedyFill(withProj, slots);
  const bench = withProj.filter(x => !used.has(x)).sort((a, b) => b.p - a.p);
  return { starters, bench, total };
}

/* ================================================================
   RENDER
   ================================================================ */
function updateConnectButton() {
  const lg = STATE.league;
  const connected = !!(lg && lg.platform !== 'demo');
  for (const id of ['connect-btn', 'connect-btn-2']) {
    const b = $('#' + id);
    if (!b) continue;
    if (id === 'connect-btn') {
      b.textContent = connected ? 'Reconnect' : 'Connect league';
      b.classList.toggle('vg-btn--primary', !connected);
      b.classList.toggle('vg-btn--ghost', connected);
    } else {
      b.textContent = connected ? 'Switch / reconnect league' : 'Connect your league';
    }
  }
}

function afterLeagueLoaded() {
  const lg = STATE.league;
  rebuildOwnership();
  updateScoringBadge();
  updateConnectButton();
  if (lg) {
    $('#league-badge').hidden = false;
    const me = lg.teams.find(t => t.id === lg.myId);
    $('#league-badge').innerHTML = `${escapeHtml(lg.name)}${me ? ' · <b>' + escapeHtml(me.name) + '</b>' : ''}`;
  } else {
    $('#league-badge').hidden = true;
  }
  renderTeam(); renderStartSit(); renderWaivers(); renderTrades(); renderDraft();
  renderOwnerCareers();
  if (MAP) { buildMapPts(); if (MAP.paused) drawMap(); }
  refreshLookbackSeasonPicker();
  if (LB.season != null) renderLookback();
  if (lg && lg.myId) selectView('team');
  if (lg && lg.platform !== 'demo') refreshLookbackFromLeague(lg);
  else if (!lg) restoreLookbackSeed();
  else {
    // demo: keep seed but still show owner board from seed seasons
    STATE.lookbackLive = null;
    renderOwnerCareers();
  }
}
function updateScoringBadge() {
  $('#scoring-badge').innerHTML = 'scoring <b>' + STATE.scoring.toUpperCase() + '</b>';
}
// ppg lives on VECTOR records (per player-season). projFor records carry the
// forward-looking model output (proj/floor/ceil/line/tier), not actual ppg.
const ppgOf = (rec) => rec && rec.ppg ? (rec.ppg[STATE.scoring] ?? rec.ppg.ppr) : null;
const vecFor = (name, pos) => STATE.vecByKey.get(normKey(name, pos)) || null;
const lastPpg = (name, pos) => ppgOf(vecFor(name, pos));
function tierChip(p) {
  return p && p.tier ? `<span class="vg-tier">T${p.tier}·${p.pos}${p.rank_pos}</span>` : '·';
}

function myTeam() {
  const lg = STATE.league;
  return lg && lg.myId ? lg.teams.find(t => t.id === lg.myId) : null;
}

function projCell(pl) {
  const p = projFor(pl.name, pl.pos);
  if (!p) return `<td class="vg-num vg-unmatched">—</td><td class="vg-num vg-unmatched">—</td><td></td>`;
  return `<td class="vg-num">${p.proj.toFixed(1)}</td>`
    + `<td class="vg-num" title="floor / ceiling">${p.floor.toFixed(0)}–${p.ceil.toFixed(0)}</td>`
    + `<td class="vg-num">${tierChip(p)}</td>`;
}
function nameCell(pl) {
  const v = STATE.vecByKey.get(normKey(pl.name, pl.pos));
  const hs = pl.headshot || (v && v.headshot);
  const head = hs ? `<img class="vg-headshot" src="${hs}" alt="" loading="lazy">` : '';
  const rp = pl.rookie !== undefined || pl.avail !== undefined ? pl : projFor(pl.name, pl.pos);
  const rook = rp && rp.rookie ? ` <span class="vg-rookie" title="rookie — draft-capital projection">R</span>` : '';
  const star = isMine(pl.name, pl.pos) ? `<span class="vg-mine" title="on your team">★</span> ` : '';
  return `<div class="vg-playername">${star}${head}<button class="vg-linkname" data-name="${escapeAttr(pl.name)}" data-pos="${pl.pos}" `
    + `style="background:none;border:none;color:inherit;cursor:pointer;text-align:left;padding:0;font:inherit">`
    + `${escapeHtml(pl.name)}</button>${rook}${availBadge(rp)}${movedBadge(rp)}</div>`;
}

// summed projection of a team's optimal starters (its "strength")
function teamStrength(t) {
  const withVal = t.roster.map(pl => ({ player: pl, p: proj(pl) })).filter(x => x.p != null && LINEUP_POS.has(x.player.pos));
  return greedyFill(withVal, STATE.league.slots).total;
}
function powerRankings() {
  const lg = STATE.league;
  return lg.teams.map(t => ({ t, s: teamStrength(t) })).sort((a, b) => b.s - a.s);
}

function renderTeam() {
  const t = myTeam();
  $('#team-unconnected').hidden = !!t;
  $('#team-connected').hidden = !t;
  if (!t) return;
  const lg = STATE.league;
  const ranks = powerRankings();
  const myRank = ranks.findIndex(r => r.t.id === t.id) + 1;
  const av = t.avatar ? `<img class="vg-teamlogo" src="${t.avatar}" alt="">`
    : `<div class="vg-teamlogo vg-teamlogo--mono">${escapeHtml(t.name.slice(0, 2).toUpperCase())}</div>`;

  // personalized banner
  $('#team-banner').innerHTML =
    `<div class="vg-banner">
       ${av}
       <div class="vg-banner__txt">
         <div class="vg-banner__league">${escapeHtml(lg.name)}</div>
         <div class="vg-banner__team">${escapeHtml(t.name)}</div>
         <div class="vg-banner__sub">${lg.teams.length}-team ${STATE.scoring.toUpperCase()} ·
           <b>#${myRank}</b> of ${lg.teams.length} by projected strength</div>
       </div>
       <div class="vg-banner__metric"><div class="vg-metric__n">${teamStrength(t).toFixed(0)}</div>
         <div class="vg-metric__l">starter proj/wk</div></div>
     </div>`;

  const yr = String(STATE.latestSeason).slice(2);
  const rows = [...t.roster]
    .sort((a, b) => (proj(b) ?? -1) - (proj(a) ?? -1))
    .map(pl => `<tr><td>${nameCell(pl)}</td><td><span class="vg-pos ${pl.pos}">${pl.pos}</span></td>`
      + `<td class="vg-num">${fmtPpg(lastPpg(pl.name, pl.pos))}</td>${projCell(pl)}`
      + `<td class="vg-num">${byeText(projFor(pl.name, pl.pos))}</td></tr>`).join('');
  $('#team-title').textContent = `${t.name} — roster (${t.roster.length})`;
  $('#roster-body').innerHTML = table(
    ['Player', 'Pos', `${STATE.scoring.toUpperCase()}/g '${yr}`, `Proj '${String(STATE.proj.proj_season).slice(2)}`, 'Fl–Ce', 'Tier', 'Bye'],
    rows) + unmatchedNote(t);

  // league power rankings
  $('#power-body').innerHTML = table(['#', 'Team', 'Proj strength/wk'],
    ranks.map((r, i) => `<tr class="${r.t.id === t.id ? 'vg-row--mine' : ''}">`
      + `<td class="vg-num">${i + 1}</td>`
      + `<td>${r.t.avatar ? `<img class="vg-headshot" src="${r.t.avatar}" alt="">` : ''} ${escapeHtml(r.t.name)}${r.t.id === t.id ? ' <span class="vg-mine">★</span>' : ''}</td>`
      + `<td class="vg-num">${r.s.toFixed(0)}</td></tr>`).join(''));
  bindNameLinks();
}
function fmtPpg(v) { return v == null ? '<span class="vg-unmatched">—</span>' : v.toFixed(1); }

// availability record + weekly projection for a rostered player
const availRec = pl => nextFor(pl.name, pl.pos) || projFor(pl.name, pl.pos);
const weekProj = pl => { const n = nextFor(pl.name, pl.pos); return n ? n.proj : proj(pl); };

function renderStartSit() {
  const t = myTeam();
  const body = $('#startsit-body');
  if (!t) { body.innerHTML = '<div class="vg-empty">Connect a league to optimize your lineup.</div>'; return; }
  const lg = STATE.league;
  const wk = STATE.next ? `Week ${STATE.next.week}` : 'this week';

  // exclude players on BYE / Out / IR from the startable pool
  const unavailable = t.roster.filter(pl => LINEUP_POS.has(pl.pos) && isUnavailable(availRec(pl)));
  const startable = t.roster.filter(pl => LINEUP_POS.has(pl.pos) && !isUnavailable(availRec(pl)));
  const withVal = startable.map(pl => ({ player: pl, p: weekProj(pl) })).filter(x => x.p != null);
  const opt = greedyFill(withVal, lg.slots);
  const bench = withVal.filter(x => !opt.used.has(x)).sort((a, b) => b.p - a.p);

  const curKeys = new Set((t.starters || []).map(s => normKey(s.name, s.pos)));
  const startRows = opt.starters.map(s => {
    if (!s.player) return `<tr><td><span class="vg-slot">${s.slot}</span></td><td class="vg-unmatched">— empty —</td><td></td><td></td></tr>`;
    const wasStarting = curKeys.size ? curKeys.has(normKey(s.player.name, s.player.pos)) : true;
    const flag = curKeys.size && !wasStarting ? `<span class="vg-rec vg-rec--start">START</span>` : '';
    return `<tr><td><span class="vg-slot">${s.slot}</span></td>`
      + `<td>${nameCell(s.player)} <span class="vg-pos ${s.player.pos}">${s.player.pos}</span></td>`
      + `<td class="vg-num">${s.p.toFixed(1)}</td><td>${flag}</td></tr>`;
  }).join('');
  const benchRows = bench.slice(0, 12).map(b => {
    const wasStarting = curKeys.has(normKey(b.player.name, b.player.pos));
    const flag = wasStarting ? `<span class="vg-rec vg-rec--sit">SIT</span>` : '';
    return `<tr><td><span class="vg-slot">BN</span></td>`
      + `<td>${nameCell(b.player)} <span class="vg-pos ${b.player.pos}">${b.player.pos}</span></td>`
      + `<td class="vg-num">${b.p.toFixed(1)}</td><td>${flag}</td></tr>`;
  }).join('');
  const unavailRows = unavailable.map(pl =>
    `<tr><td><span class="vg-slot">${availRec(pl).avail}</span></td>`
    + `<td>${nameCell(pl)} <span class="vg-pos ${pl.pos}">${pl.pos}</span></td>`
    + `<td class="vg-num vg-unmatched">out</td><td></td></tr>`).join('');

  body.innerHTML =
    `<div class="vg-lineup-total">
       <div class="vg-metric"><div class="vg-metric__n">${opt.total.toFixed(1)}</div><div class="vg-metric__l">optimal proj — ${wk}</div></div>
       <div class="vg-metric"><div class="vg-metric__n">${opt.starters.filter(s => s.player).length}</div><div class="vg-metric__l">slots filled</div></div>
       ${unavailable.length ? `<div class="vg-metric"><div class="vg-metric__n">${unavailable.length}</div><div class="vg-metric__l">unavailable</div></div>` : ''}
     </div>
     <div class="vg-card__title">Start</div>${table(['Slot', 'Player', 'Proj', ''], startRows)}
     <div class="vg-card__title" style="margin-top:14px">Bench</div>${table(['Slot', 'Player', 'Proj', ''], benchRows)}
     ${unavailable.length ? `<div class="vg-card__title" style="margin-top:14px">Can't start — bye / injured / IR</div>${table(['Why', 'Player', '', ''], unavailRows)}` : ''}
     <p class="vg-note" style="margin-top:10px">${STATE.next ? `Uses the ${wk} next-game MTNN (real matchup + weather + Vegas)` : 'Uses season projection'};
     players on bye or ruled out are excluded automatically. START/SIT flags compare the optimal lineup to your platform's current starters.</p>`;
  bindNameLinks();
}

function rosteredKeys() {
  const s = new Set();
  for (const t of STATE.league.teams) for (const pl of t.roster) s.add(normKey(pl.name, pl.pos));
  return s;
}
function renderWaivers() {
  const body = $('#waivers-body');
  if (!STATE.league) { body.innerHTML = '<div class="vg-empty">Connect a league to see who\'s available.</div>'; return; }
  const rostered = rosteredKeys();
  const avail = [...STATE.projByKey.values()]
    .filter(p => !rostered.has(p.key))
    .sort((a, b) => b.proj - a.proj);
  const mk = list => list.slice(0, 12).map(p =>
    `<tr><td>${nameCell(p)}</td><td><span class="vg-pos ${p.pos}">${p.pos}</span></td>`
    + `<td class="vg-num">${p.proj.toFixed(1)}</td><td class="vg-num">${p.floor.toFixed(0)}–${p.ceil.toFixed(0)}</td>`
    + `<td>${tierChip(p)}</td></tr>`).join('');
  const head = ['Player', 'Pos', 'Proj', 'Fl–Ce', 'Tier'];
  body.innerHTML =
    `<p class="vg-note" style="margin-bottom:12px">Best players by projection <b>not rostered in your league</b>.
      Sorted overall; per-position boards below. (Projected ${STATE.proj.proj_season} value — verify NFL status before claiming.)</p>`
    + `<div class="vg-card__title">Overall</div>${table(head, mk(avail))}`
    + ALL_POS.map(pos => `<div class="vg-card__title" style="margin-top:14px">${pos}</div>`
      + table(head, mk(avail.filter(p => p.pos === pos)))).join('');
  bindNameLinks();
}

function renderTrades() {
  const body = $('#trades-body');
  const me = myTeam();
  if (!me) { body.innerHTML = '<div class="vg-empty">Connect a league and pick your team to scan for trades.</div>'; return; }
  const lg = STATE.league;
  const careers = ownerCareerMap();
  const strength = t => {
    const s = { QB: 0, RB: 0, WR: 0, TE: 0, K: 0, DST: 0 };
    const byPos = { QB: [], RB: [], WR: [], TE: [], K: [], DST: [] };
    for (const pl of t.roster) { const p = proj(pl); if (p != null && byPos[pl.pos]) byPos[pl.pos].push(p); }
    for (const pos of ALL_POS) {
      byPos[pos].sort((a, b) => b - a);
      const n = Math.max(1, (lg.slots[pos] || 0) + (pos === 'K' || pos === 'DST' ? 0 : 1));
      s[pos] = byPos[pos].slice(0, n).reduce((a, b) => a + b, 0);
    }
    return s;
  };
  const leagueAvg = {}; ALL_POS.forEach(p => leagueAvg[p] = 0);
  const all = lg.teams.map(strength);
  all.forEach(s => ALL_POS.forEach(p => leagueAvg[p] += s[p] / all.length));
  const myS = strength(me);
  const tradePos = ALL_POS;
  const mySurplus = tradePos.filter(p => myS[p] > leagueAvg[p] * 1.12).sort((a, b) => (myS[b] - leagueAvg[b]) - (myS[a] - leagueAvg[a]));
  const myNeed = tradePos.filter(p => myS[p] < leagueAvg[p] * 0.9).sort((a, b) => (leagueAvg[a] - myS[a]) - (leagueAvg[b] - myS[b]));

  const ideas = [];
  for (const give of mySurplus) {
    for (const get of myNeed) {
      for (let ti = 0; ti < lg.teams.length; ti++) {
        const partner = lg.teams[ti];
        if (partner.id === me.id) continue;
        const ps = all[ti];
        if (!(ps[give] < leagueAvg[give] * 0.95 && ps[get] > leagueAvg[get] * 1.05)) continue;
        const myGive = bestBench(me, give);
        const theirGive = bestBench(partner, get);
        if (myGive && theirGive && Math.abs((proj(myGive) || 0) - (proj(theirGive) || 0)) < 4) {
          ideas.push({ partner, give, get, myGive, theirGive });
        }
      }
    }
  }
  const uniq = ideas.slice(0, 8);
  if (!mySurplus.length && !myNeed.length) {
    body.innerHTML = `<div class="vg-empty">Your roster is balanced vs the league — no obvious surplus/need to arbitrage. Nice draft.</div>`;
    return;
  }
  const histNote = careers.size
    ? ` · partner career draft grades from ${STATE.lookbackLive?.leagueName || 'Lookback'}`
    : '';
  const summary = `<p class="vg-note" style="margin-bottom:12px">`
    + (mySurplus.length ? `Surplus: ${mySurplus.map(p => `<span class="vg-pos ${p}">${p}</span>`).join(' ')} · ` : '')
    + (myNeed.length ? `Need: ${myNeed.map(p => `<span class="vg-pos ${p}">${p}</span>`).join(' ')}` : '')
    + histNote
    + `</p>`;
  const partnerLabel = t => {
    const c = careers.get(ownerCareerKey(t));
    if (!c) return escapeHtml(t.name);
    const who = t.ownerName ? `${escapeHtml(t.ownerName)} <span style="opacity:.65">(${escapeHtml(t.name)})</span>` : escapeHtml(t.name);
    return `${who} <span class="vg-owner-chip">${c.avgGrade} · ${c.titles}🏆 · ${c.seasons}y</span>`;
  };
  const rows = uniq.map(x =>
    `<tr><td>${partnerLabel(x.partner)}</td>`
    + `<td>send ${nameCell(x.myGive)} <span class="vg-pos ${x.myGive.pos}">${x.myGive.pos}</span></td>`
    + `<td>get ${nameCell(x.theirGive)} <span class="vg-pos ${x.theirGive.pos}">${x.theirGive.pos}</span></td>`
    + `<td class="vg-num">${(proj(x.myGive) || 0).toFixed(1)}→${(proj(x.theirGive) || 0).toFixed(1)}</td></tr>`).join('');
  body.innerHTML = summary + (uniq.length
    ? table(['Partner', 'You send', 'You get', 'Proj swap'], rows)
    : `<div class="vg-empty">Found surplus/need but no clean 1-for-1 with a matching partner. Try the Vector Map for stylistic swaps.</div>`)
    + `<p class="vg-note" style="margin-top:10px">Heuristic: your positional surplus → a partner's need, matched on projection (±4 pts).`
    + (careers.size ? ` Career chips = avg Lookback draft grade · titles · seasons graded.` : ` Connect + wait for Lookback history to see owner draft grades on partners.`)
    + `</p>`;
  bindNameLinks();
}
function bestBench(team, pos) {
  const lg = STATE.league;
  const cand = team.roster.filter(p => p.pos === pos && proj(p) != null).sort((a, b) => proj(b) - proj(a));
  return cand[lg.slots[pos]] || cand[cand.length - 1] || null; // first non-guaranteed-starter
}

/* ---------- shared render helpers ---------- */
function table(headers, rows) {
  return `<div class="vg-tablewrap"><table class="vg-tbl"><thead><tr>`
    + headers.map(h => `<th>${h}</th>`).join('') + `</tr></thead><tbody>`
    + (rows || `<tr><td colspan="${headers.length}" class="vg-empty">No rows.</td></tr>`)
    + `</tbody></table></div>`;
}
function unmatchedNote(t) {
  const un = t.roster.filter(pl => !projFor(pl.name, pl.pos) && ALL_POS.includes(pl.pos));
  if (!un.length) return '';
  return `<p class="vg-note" style="margin-top:8px">No projection for ${un.length} player(s)
    (rookies / players without a ${STATE.vectors.seasons.at(-1)} sample): ${un.map(p => escapeHtml(p.name)).join(', ')}.</p>`;
}
function bindNameLinks() {
  $$('.vg-linkname').forEach(b => b.addEventListener('click', () => {
    selectView('explorer');
    showProfile(b.dataset.name, b.dataset.pos);
    $('#explore-input').value = b.dataset.name;
  }));
}

/* ================================================================
   LOOKBACK — retro grades + what-if perfect roster + career arcs
   ================================================================ */
let CAREER = new Map();                    // normKey -> [player-seasons asc]
function buildCareer() {
  CAREER = new Map();
  for (const p of STATE.vectors.players) {
    const k = normKey(p.name, p.pos);
    if (!CAREER.has(k)) CAREER.set(k, []);
    CAREER.get(k).push(p);
  }
  for (const arr of CAREER.values()) arr.sort((a, b) => a.season - b.season);
}
const GRADE_TIERS = [[.95, 'A+'], [.88, 'A'], [.80, 'A-'], [.72, 'B+'], [.62, 'B'],
  [.52, 'B-'], [.42, 'C+'], [.32, 'C'], [.22, 'C-'], [.12, 'D'], [0, 'F']];
const letterGrade = pct => (GRADE_TIERS.find(([t]) => pct >= t) || [0, 'F'])[1];
const gradeClass = g => 't' + g[0];
const GRADE_ORD = { 'A+': 12, A: 11, 'A-': 10, 'B+': 9, B: 8, 'B-': 7, 'C+': 6, C: 5, 'C-': 4, D: 3, F: 1 };
const ORD_GRADE = Object.entries(GRADE_ORD).sort((a, b) => b[1] - a[1]);
function gradeFromOrd(mean) {
  if (!Number.isFinite(mean)) return '—';
  let best = 'F', dist = Infinity;
  for (const [g, o] of ORD_GRADE) {
    const d = Math.abs(o - mean);
    if (d < dist) { dist = d; best = g; }
  }
  return best;
}
function ownerNorm(name) {
  return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}
/** Stable career key: manager id first (survives team renames), then owner name, then team name. */
function ownerCareerKey(t) {
  if (!t) return null;
  if (t.ownerId) return 'oid:' + String(t.ownerId);
  if (t.ownerName) return 'oname:' + ownerNorm(t.ownerName);
  if (t.name) return 'tname:' + ownerNorm(t.name);
  return null;
}
function isMyLookbackTeam(t) {
  const lg = STATE.league;
  if (!lg || !lg.myId) return false;
  const me = lg.teams.find(x => x.id === lg.myId);
  if (!me) return false;
  if (me.ownerId && t.ownerId && String(me.ownerId) === String(t.ownerId)) return true;
  if (t.key && me.ownerId && t.key === 'oid:' + String(me.ownerId)) return true;
  if (t.id != null && String(t.id) === String(me.id)) return true;
  const myLabel = ownerNorm(me.ownerName || me.name);
  const theirLabel = ownerNorm(t.ownerName || t.name);
  return !!(myLabel && theirLabel && myLabel === theirLabel);
}

/** Resolve focus key: 'me' → connected manager, else an ownerCareerKey. */
function resolveFocusKey() {
  if (LB.focusKey && LB.focusKey !== 'me') return LB.focusKey;
  const lg = STATE.league;
  if (!lg?.myId) return null;
  const me = lg.teams.find(x => x.id === lg.myId);
  return ownerCareerKey(me);
}

function isFocusedLookbackTeam(t) {
  const fk = resolveFocusKey();
  if (!fk) return isMyLookbackTeam(t);
  return ownerCareerKey(t) === fk || (t.key && t.key === fk);
}

function myLookbackTeamLabel() {
  const lg = STATE.league;
  if (!lg?.myId) return null;
  const me = lg.teams.find(x => x.id === lg.myId);
  if (!me) return null;
  return me.ownerName || me.name || 'You';
}

/** Pin focused owner to top; stable secondary sort via cmp(a,b). */
function sortFocusFirst(list, cmp) {
  const fk = resolveFocusKey();
  return [...list].sort((a, b) => {
    const am = isFocusedLookbackTeam(a) ? 1 : 0;
    const bm = isFocusedLookbackTeam(b) ? 1 : 0;
    if (am !== bm) return bm - am;
    return cmp(a, b);
  });
}

/** Multi-season owner rollup from live lookback (falls back to seed if no live). */
function ownerCareerMap() {
  const live = STATE.lookbackLive;
  const seasons = [];
  if (live?.status === 'ready' && live.seasons?.size) {
    for (const rec of live.seasons.values()) seasons.push(rec);
  } else if (STATE.league && STATE.lookbackBySeason.size) {
    // Prefer only source=league records when mixed with seed overlay.
    for (const rec of STATE.lookbackBySeason.values()) {
      if (rec.source === 'league') seasons.push(rec);
    }
  }
  const by = new Map();
  for (const rec of seasons) {
    for (const t of rec.teams || []) {
      const k = ownerCareerKey(t);
      if (!k) continue;
      let row = by.get(k);
      if (!row) {
        row = {
          key: k,
          name: t.ownerName || t.name,
          ownerId: t.ownerId || null,
          ownerName: t.ownerName || null,
          id: t.id || null,
          teamNames: [],
          seasons: 0, titles: 0, pf: 0, gradeOrds: [],
          bestGrade: null, worstGrade: null, years: [],
        };
        by.set(k, row);
      }
      row.seasons += 1;
      row.years.push(rec.season);
      if (t.champion) row.titles += 1;
      row.pf += t.points_for || 0;
      const ord = GRADE_ORD[t.grade];
      if (ord != null) {
        row.gradeOrds.push(ord);
        if (!row.bestGrade || ord > (GRADE_ORD[row.bestGrade] || 0)) row.bestGrade = t.grade;
        if (!row.worstGrade || ord < (GRADE_ORD[row.worstGrade] || 99)) row.worstGrade = t.grade;
      }
      if (t.id != null) row.id = t.id;
      if (t.ownerId) row.ownerId = t.ownerId;
      if (t.ownerName) {
        row.ownerName = t.ownerName;
        row.name = t.ownerName;
      }
      if (t.name && !row.teamNames.includes(t.name)) row.teamNames.push(t.name);
    }
  }
  for (const row of by.values()) {
    const mean = row.gradeOrds.length
      ? row.gradeOrds.reduce((a, b) => a + b, 0) / row.gradeOrds.length
      : NaN;
    row.avgGrade = gradeFromOrd(mean);
    row.avgPf = row.seasons ? row.pf / row.seasons : 0;
    row.currentTeam = row.teamNames[row.teamNames.length - 1] || null;
  }
  // Prefer current connected-league labels when owner ids match.
  if (STATE.league) {
    for (const t of STATE.league.teams) {
      const row = by.get(ownerCareerKey(t));
      if (!row) continue;
      row.id = t.id;
      if (t.ownerName) { row.ownerName = t.ownerName; row.name = t.ownerName; }
      row.currentTeam = t.name;
      if (t.name && !row.teamNames.includes(t.name)) row.teamNames.push(t.name);
    }
  }
  return by;
}

function renderOwnerCareers() {
  const el = $('#ll-owners');
  if (!el) return;
  const live = STATE.lookbackLive;
  if (!STATE.league) {
    el.innerHTML = '<div class="vg-empty">Connect your league to evaluate owners historically.</div>';
    return;
  }
  if (live?.status === 'loading') {
    el.innerHTML = `<div class="vg-empty">Loading draft history for ${escapeHtml(live.leagueName || 'your league')}…</div>`;
    return;
  }
  let careers = [...ownerCareerMap().values()];
  if (!careers.length) {
    const msg = live?.status === 'empty' || live?.status === 'err' || live?.status === 'auth'
      ? (live.message || 'No historic drafts graded yet.')
      : 'No graded seasons yet — Lookback will fill this once draft history loads.';
    const cta = live?.status === 'auth'
      ? ` <button type="button" class="vg-btn vg-btn--primary" id="ll-owners-reconnect" style="margin-top:10px">Reconnect with ESPN cookies</button>`
      : '';
    el.innerHTML = `<div class="vg-empty">${escapeHtml(msg)}${cta}</div>`;
    $('#ll-owners-reconnect')?.addEventListener('click', openConnect);
    return;
  }
  const mode = LB.rankMode === 'career' ? 'draft' : LB.rankMode;
  const cmp = {
    draft: (a, b) => (GRADE_ORD[b.avgGrade] || 0) - (GRADE_ORD[a.avgGrade] || 0)
      || b.titles - a.titles || b.avgPf - a.avgPf,
    titles: (a, b) => b.titles - a.titles || (GRADE_ORD[b.avgGrade] || 0) - (GRADE_ORD[a.avgGrade] || 0),
    pf: (a, b) => b.avgPf - a.avgPf || b.titles - a.titles,
    playoffs: (a, b) => b.titles - a.titles || b.avgPf - a.avgPf,
  }[mode] || ((a, b) => (GRADE_ORD[b.avgGrade] || 0) - (GRADE_ORD[a.avgGrade] || 0));
  careers = sortFocusFirst(careers, cmp);
  const rows = careers.map((c, i) => {
    const mine = isFocusedLookbackTeam(c);
    const aka = (c.teamNames || []).filter(n => n && n !== c.currentTeam);
    const teamLine = c.currentTeam
      ? `<div style="font-size:11px;opacity:.7;margin-top:2px">${escapeHtml(c.currentTeam)}`
        + (aka.length ? ` · aka ${escapeHtml(aka.slice(0, 3).join(', '))}${aka.length > 3 ? '…' : ''}` : '')
        + `</div>`
      : '';
    return `<tr class="${mine ? 'vg-row--mine' : ''}"><td class="vg-num">${i + 1}</td>`
      + `<td>${mine ? '★ ' : ''}<b>${escapeHtml(c.name)}</b>${teamLine}</td>`
      + `<td><span class="vg-grade ${gradeClass(c.avgGrade)}">${c.avgGrade}</span></td>`
      + `<td class="vg-num">${c.seasons}</td>`
      + `<td class="vg-num">${c.titles}</td>`
      + `<td class="vg-num">${c.avgPf.toFixed(0)}</td>`
      + `<td><span class="vg-grade ${gradeClass(c.bestGrade || 'F')}">${c.bestGrade || '—'}</span>`
      + ` / <span class="vg-grade ${gradeClass(c.worstGrade || 'F')}">${c.worstGrade || '—'}</span></td>`
      + `<td class="vg-num" style="font-size:11px;text-align:left">${c.years.slice().sort((a, b) => b - a).join(', ')}</td></tr>`;
  }).join('');
  el.innerHTML = table(
    ['#', 'Owner', 'Avg draft', 'Seasons', 'Titles', 'Avg PF', 'Best / worst', 'Years'],
    rows,
  ) + `<p class="vg-note" style="margin-top:8px">Aligned by <b>manager identity</b> (not team name).
    ★ = focused owner (default: you). Start/sit &amp; bench skill need weekly lineup history — not graded yet.</p>`;
}

function refreshLookbackOwnerPicker() {
  const sel = $('#lookback-owner');
  if (!sel) return;
  const lg = STATE.league;
  const careers = [...ownerCareerMap().values()]
    .sort((a, b) => String(a.name).localeCompare(String(b.name)));
  const meLabel = myLookbackTeamLabel();
  const opts = [];
  if (lg?.myId && meLabel) {
    opts.push(`<option value="me">★ You — ${escapeHtml(meLabel)}</option>`);
  } else {
    opts.push(`<option value="me">★ You (connect a league)</option>`);
  }
  for (const c of careers) {
    if (isMyLookbackTeam(c)) continue;
    const label = c.currentTeam ? `${c.name} · ${c.currentTeam}` : c.name;
    opts.push(`<option value="${escapeAttr(c.key)}">${escapeHtml(label)}</option>`);
  }
  // Also offer current-league teams not yet in career map
  if (lg) {
    const seen = new Set(careers.map(c => c.key));
    for (const t of lg.teams) {
      const k = ownerCareerKey(t);
      if (!k || seen.has(k) || isMyLookbackTeam(t)) continue;
      opts.push(`<option value="${escapeAttr(k)}">${escapeHtml(t.ownerName || t.name)}</option>`);
      seen.add(k);
    }
  }
  const prev = LB.focusKey || 'me';
  sel.innerHTML = opts.join('');
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  else { sel.value = 'me'; LB.focusKey = 'me'; }
}

function renderYouCard(season) {
  const wrap = $('#ll-you-card');
  const title = $('#ll-you-title');
  const sub = $('#ll-you-sub');
  if (!wrap) return;
  const rec = STATE.lookbackBySeason.get(season);
  const fk = resolveFocusKey();
  const careers = ownerCareerMap();
  const career = fk ? careers.get(fk) : null;
  const team = rec?.teams?.find(t => isFocusedLookbackTeam(t));
  const label = team
    ? (team.ownerName || team.name)
    : (career?.name || myLookbackTeamLabel() || 'Focused owner');
  const isYou = LB.focusKey === 'me' || isMyLookbackTeam(team || career || {});
  if (title) title.textContent = isYou ? 'Your report card' : `Report card — ${label}`;
  if (sub) {
    sub.textContent = team
      ? `${season} draft grade · record · PF · best steal / bust`
      : (STATE.league
        ? `No ${season} draft on file for this owner — career rollup below when available.`
        : 'Connect a league to pin your card first, then toggle any owner.');
  }
  if (!team && !career) {
    wrap.innerHTML = '<div class="vg-empty">Connect your league — Lookback leads with your card.</div>';
    return;
  }
  const g = team?.grade || career?.avgGrade || '—';
  const metrics = [];
  if (team) {
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n"><span class="vg-grade ${gradeClass(g)}">${g}</span></div><div class="vg-metric__l">draft grade</div></div>`);
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n">${escapeHtml(team.record || '—')}</div><div class="vg-metric__l">record</div></div>`);
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n">${(team.points_for || 0).toFixed(0)}</div><div class="vg-metric__l">points for</div></div>`);
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n">${(team.draft_value || 0).toFixed(1)}</div><div class="vg-metric__l">draft VOR</div></div>`);
  }
  if (career) {
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n"><span class="vg-grade ${gradeClass(career.avgGrade)}">${career.avgGrade}</span></div><div class="vg-metric__l">career draft</div></div>`);
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n">${career.titles}</div><div class="vg-metric__l">titles</div></div>`);
    metrics.push(`<div class="vg-metric"><div class="vg-metric__n">${career.seasons}</div><div class="vg-metric__l">seasons graded</div></div>`);
  }
  const bp = team?.best_pick, wp = team?.worst_pick;
  const picks = (bp || wp)
    ? `<div class="vg-you__picks"><span class="vg-val vg-val--up">STEAL</span> ${bp ? escapeHtml(bp.name) + ` <span class="vg-slot">pk${bp.pick}</span>` : '—'}
         &nbsp; <span class="vg-val vg-val--dn">BUST</span> ${wp ? escapeHtml(wp.name) + ` <span class="vg-slot">pk${wp.pick}</span>` : '—'}</div>`
    : '';
  const champ = team?.champion ? '<span class="vg-you__champ">👑 Champion</span>' : '';
  wrap.innerHTML =
    `<div class="vg-you">
      <div class="vg-you__hd"><b>${escapeHtml(label)}</b> ${champ}
        ${team?.name && team.name !== label ? `<span class="vg-owner-chip">${escapeHtml(team.name)}</span>` : ''}</div>
      <div class="vg-lineup-total">${metrics.join('')}</div>
      ${picks}
      ${team?.roast?.line ? `<p class="vg-roast__line" style="margin-top:8px">${escapeHtml(team.roast.line)}</p>` : ''}
    </div>`;
}

function renderLeagueRankings(season) {
  const el = $('#ll-rankings');
  const note = $('#ll-rank-note');
  if (!el) return;
  const mode = LB.rankMode || 'draft';
  const notes = {
    draft: 'Ranked by draft VOR captured that season · ★ = focused owner.',
    titles: 'Career titles (multi-season) · season view still shows that year\'s record.',
    pf: 'Season points for · higher scoring managers rise.',
    playoffs: 'Champions first, then playoff seed / wins · ★ = focused owner.',
    career: 'Career average draft grade across graded seasons.',
  };
  if (note) note.textContent = notes[mode] || notes.draft;

  if (mode === 'career' || mode === 'titles') {
    const careers = [...ownerCareerMap().values()];
    if (!careers.length) {
      el.innerHTML = '<div class="vg-empty">Connect + load Lookback history for career standings.</div>';
      return;
    }
    const cmp = mode === 'titles'
      ? (a, b) => b.titles - a.titles || (GRADE_ORD[b.avgGrade] || 0) - (GRADE_ORD[a.avgGrade] || 0)
      : (a, b) => (GRADE_ORD[b.avgGrade] || 0) - (GRADE_ORD[a.avgGrade] || 0) || b.titles - a.titles;
    const ranked = [...careers].sort(cmp);
    const rows = ranked.map((c, i) => {
      const mine = isFocusedLookbackTeam(c);
      return `<tr class="${mine ? 'vg-row--mine' : ''}"><td class="vg-num">${i + 1}</td>`
        + `<td>${mine ? '★ ' : ''}<b>${escapeHtml(c.name)}</b></td>`
        + `<td><span class="vg-grade ${gradeClass(c.avgGrade)}">${c.avgGrade}</span></td>`
        + `<td class="vg-num">${c.titles}</td>`
        + `<td class="vg-num">${c.seasons}</td>`
        + `<td class="vg-num">${c.avgPf.toFixed(0)}</td></tr>`;
    }).join('');
    el.innerHTML = table(['#', 'Owner', 'Avg draft', 'Titles', 'Seasons', 'Avg PF'], rows);
    return;
  }

  const rec = STATE.lookbackBySeason.get(season);
  if (!rec?.teams?.length) {
    el.innerHTML = '<div class="vg-empty">No league season loaded for rankings.</div>';
    return;
  }
  const cmp = {
    draft: (a, b) => b.draft_value - a.draft_value,
    pf: (a, b) => b.points_for - a.points_for,
    playoffs: (a, b) => (b.champion ? 1 : 0) - (a.champion ? 1 : 0)
      || (a.seed || 99) - (b.seed || 99) || b.wins - a.wins || b.points_for - a.points_for,
  }[mode] || ((a, b) => b.draft_value - a.draft_value);
  const ranked = [...rec.teams].sort(cmp);
  const rows = ranked.map((t, i) => {
    const mine = isFocusedLookbackTeam(t);
    return `<tr class="${t.champion ? 'vg-row--champ' : ''}${mine ? ' vg-row--mine' : ''}">`
      + `<td class="vg-num">${i + 1}</td>`
      + `<td>${t.champion ? '👑 ' : ''}${mine ? '★ ' : ''}${escapeHtml(t.ownerName || t.name)}</td>`
      + `<td><span class="vg-grade ${gradeClass(t.grade)}">${t.grade}</span></td>`
      + `<td class="vg-num">${(t.draft_value || 0).toFixed(1)}</td>`
      + `<td class="vg-num">${escapeHtml(t.record || '—')}</td>`
      + `<td class="vg-num">${(t.points_for || 0).toFixed(0)}</td></tr>`;
  }).join('');
  el.innerHTML = table(['#', 'Owner', 'Grade', 'Draft VOR', 'Rec', 'PF'], rows)
    + `<p class="vg-note" style="margin-top:8px">Start/sit ranking needs weekly starter history — coming after matchup ingest.</p>`;
}

function refreshLookbackSeasonPicker() {
  const sel = $('#lookback-season');
  if (!sel || !STATE.vectors) return;
  const seasons = [...STATE.vectors.seasons].sort((a, b) => b - a);
  const live = STATE.lookbackLive;
  const cur = LB.season;
  sel.innerHTML = seasons.map(s => {
    const isLive = live?.status === 'ready' && live.seasons?.has(s);
    const label = live?.status === 'ready'
      ? (isLive ? `${s} · live` : `${s} · mock`)
      : String(s);
    return `<option value="${s}" class="${isLive ? 'vg-opt--live' : ''}">${label}</option>`;
  }).join('');
  if (cur != null && [...sel.options].some(o => +o.value === cur)) sel.value = String(cur);
  else if (seasons.length) {
    LB.season = seasons[0];
    sel.value = String(LB.season);
  }
}

// percentile of ppg within each (season, position) pool -> letter grade
function seasonGrades(season) {
  const pool = lookbackPool(season);
  const byPos = {};
  for (const p of pool) (byPos[p.pos] ||= []).push(p);
  const grade = new Map();
  for (const pos in byPos) {
    const arr = byPos[pos].map(p => ({ p, v: ppgOf(p) })).sort((a, b) => a.v - b.v);
    const n = arr.length;
    arr.forEach((x, i) => grade.set(x.p.id, { pct: n > 1 ? i / (n - 1) : 1, letter: letterGrade(n > 1 ? i / (n - 1) : 1) }));
  }
  return { pool, grade };
}

const LB = { season: null, pos: 'ALL', focusKey: 'me', rankMode: 'draft' };
function initLookback() {
  refreshLookbackSeasonPicker();
  const sel = $('#lookback-season');
  sel.addEventListener('change', () => { LB.season = +sel.value; renderLookback(); });
  $$('#lookback-posfilter button').forEach(b => b.addEventListener('click', () => {
    LB.pos = b.dataset.pos;
    $$('#lookback-posfilter button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    renderLookback();
  }));
  $('#lookback-owner')?.addEventListener('change', e => {
    LB.focusKey = e.target.value || 'me';
    renderLookback();
  });
  $$('#ll-rank-mode button').forEach(b => b.addEventListener('click', () => {
    LB.rankMode = b.dataset.mode;
    $$('#ll-rank-mode button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    renderLeagueRankings(LB.season);
    renderOwnerCareers();
  }));
  renderLookback();
}

function renderLookback() {
  if (!STATE.vectors || LB.season == null) return;
  refreshLookbackOwnerPicker();
  const { pool, grade } = seasonGrades(LB.season);
  const sorted = [...pool].sort((a, b) => ppgOf(b) - ppgOf(a));
  const leader = pos => sorted.find(p => p.pos === pos);
  const sc = STATE.scoring.toUpperCase();

  const sup = [['Overall #1', sorted[0]], ['QB1', leader('QB')], ['RB1', leader('RB')],
    ['WR1', leader('WR')], ['TE1', leader('TE')], ['K1', leader('K')], ['DST1', leader('DST')]];
  $('#lookback-superlatives').innerHTML = '<div class="vg-super">' + sup.map(([l, p]) => p
    ? `<div class="vg-super__card"><div class="vg-super__l">${l}</div>
       <div class="vg-super__n">${escapeHtml(p.name)}</div>
       <div class="vg-super__v">${ppgOf(p).toFixed(1)} ${sc}/g</div></div>` : '').join('') + '</div>';

  renderYouCard(LB.season);
  renderLeagueRankings(LB.season);

  // what-if: the best startable lineup of that season by ACTUAL ppg (incl K/DST)
  const slots = (STATE.league && STATE.league.slots) || { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1 };
  const withVal = sorted.filter(p => ALL_POS.includes(p.pos)).map(p => ({ player: p, p: ppgOf(p) }));
  const { starters, total } = greedyFill(withVal, slots);
  $('#lookback-perfect-title').textContent = `The perfect roster anyone could've started — ${LB.season}`;
  $('#lookback-perfect').innerHTML =
    `<div class="vg-lineup-total"><div class="vg-metric"><div class="vg-metric__n">${total.toFixed(1)}</div>
      <div class="vg-metric__l">${sc} pts / week</div></div></div>`
    + table(['Slot', 'Player', 'Pos', `${sc}/g`], starters.map(s => s.player
      ? `<tr><td><span class="vg-slot">${s.slot}</span></td><td>${nameCell(s.player)}</td>
         <td><span class="vg-pos ${s.player.pos}">${s.player.pos}</span></td>
         <td class="vg-num">${s.p.toFixed(1)}</td></tr>` : '').join(''));

  let rows = LB.pos === 'ALL' ? sorted : sorted.filter(p => p.pos === LB.pos);
  const body = rows.map(p => {
    const g = grade.get(p.id) || { letter: '?' };
    const arch = p._src === 'kdst' ? (p.pos === 'K' ? 'Kicker' : 'Team defense')
      : (STATE.vectors.clusters[p.c] || '—');
    return `<tr><td>${nameCell(p)}</td><td><span class="vg-pos ${p.pos}">${p.pos}</span></td>`
      + `<td class="vg-num">${escapeHtml(p.team)}</td><td class="vg-num">${p.games}</td>`
      + `<td class="vg-num">${ppgOf(p).toFixed(1)}</td>`
      + `<td><span class="vg-grade ${gradeClass(g.letter)}">${g.letter}</span></td>`
      + `<td class="vg-num" style="font-size:11px;text-align:left">${escapeHtml(arch)}</td></tr>`;
  }).join('');
  $('#lookback-grades').innerHTML =
    table(['Player', 'Pos', 'Team', 'G', `${sc}/g`, 'Grade', 'Archetype'], body)
    + `<p class="vg-note" style="margin-top:8px">Grade = fantasy points-per-game percentile
       within position that season (A+ = top 5%). ${rows.length} qualified players
       (skill from vectors · K/DST from kdst history).</p>`;
  renderLeagueSeason(LB.season);
  renderOwnerCareers();
  bindNameLinks();
}

// blog-style narratives: render **bold** markdown, escape the rest
const mdBold = s => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');

function renderLeagueSeason(season) {
  $('#ll-season-a').textContent = season;
  $('#ll-season-b').textContent = season;
  const live = STATE.lookbackLive;
  const rec = STATE.lookbackBySeason.get(season);
  const statusEl = $('#ll-live-status');
  if (statusEl) {
    if (live?.status === 'loading') statusEl.textContent = `Loading ${live.leagueName || 'league'} draft history…`;
    else if (live?.status === 'ready' && live.seasons?.has(season))
      statusEl.textContent = `Showing ${live.leagueName} — real draft grades for ${season}`;
    else if (live?.status === 'ready')
      statusEl.textContent = `${live.leagueName}: no draft on file for ${season} — seeded mock below`;
    else if (live?.status === 'empty')
      statusEl.textContent = live.message || 'No historic drafts found for this league — showing seeded mock.';
    else if (live?.status === 'auth')
      statusEl.textContent = live.message || 'ESPN cookies required for draft history — use Reconnect.';
    else if (live?.status === 'err')
      statusEl.textContent = live.message || 'Could not load league history — showing seeded mock.';
    else statusEl.textContent = 'Seeded mock league (connect yours to grade real drafts).';
  }
  if (!rec) {
    $('#ll-cards').innerHTML = '<div class="vg-empty">No seeded league season available.</div>';
    $('#ll-story').innerHTML = ''; $('#ll-weeks').innerHTML = '';
    return;
  }
  const isLive = !!(rec.source === 'league' || (live?.seasons?.has(season)));
  const teams = sortFocusFirst(rec.teams, (a, b) => b.draft_value - a.draft_value);
  const rows = teams.map(t => {
    const bp = t.best_pick, wp = t.worst_pick;
    const mine = isFocusedLookbackTeam(t);
    return `<tr class="${t.champion ? 'vg-row--champ' : ''}${mine ? ' vg-row--mine' : ''}"><td><span class="vg-grade ${gradeClass(t.grade)}">${t.grade}</span></td>`
      + `<td>${t.champion ? '👑 ' : ''}${mine ? '★ ' : ''}${escapeHtml(t.ownerName || t.name)}</td>`
      + `<td class="vg-num">${t.draft_value.toFixed(1)}</td>`
      + `<td class="vg-num">${escapeHtml(t.record)}</td>`
      + `<td class="vg-num">${t.points_for.toFixed(0)}</td>`
      + `<td style="text-align:left;font-size:11px">${bp ? escapeHtml(bp.name) + ` <span class="vg-slot">pk${bp.pick}</span>` : '—'}</td>`
      + `<td style="text-align:left;font-size:11px" class="vg-unmatched">${wp ? escapeHtml(wp.name) + ` <span class="vg-slot">pk${wp.pick}</span>` : '—'}</td></tr>`;
  }).join('');
  $('#ll-cards').innerHTML =
    table(['Grade', 'Owner', 'Draft VOR', 'Rec', 'PF', 'Best steal', 'Biggest bust'], rows)
    + `<p class="vg-note" style="margin-top:8px">${isLive
      ? `Your league's real ${season} draft, graded on value-over-replacement from actual NFL ${STATE.scoring.toUpperCase()} production (👑 = champion · ★ = focused owner, pinned top).`
      : `Best-ball PPR · drafts graded on value-over-replacement captured (👑 = champion · ★ = focused). Seeded from real ${season} results; your league's actual draft history plugs into this same engine once connected.`}</p>`;
  $('#ll-story').innerHTML =
    `<p class="vg-story">${mdBold(rec.narratives.draft)}</p><p class="vg-story">${mdBold(rec.narratives.season)}</p>`;
  const weeks = rec.narratives.weeks || [];
  $('#ll-weeks').innerHTML = weeks.length
    ? `<details class="vg-weeks"><summary>${weeks.length} weekly recaps ▾</summary>`
      + weeks.map(w => `<p class="vg-weekline">${mdBold(w.text)}</p>`).join('') + `</details>`
    : (isLive
      ? `<p class="vg-note">Weekly recaps stay on the seeded mock — your league's week-by-week story needs matchup history (coming next).</p>`
      : '');
  renderRoast(season, rec);
}

// trash-talk: each owner's biggest hit + miss + the pick they whiffed on, with
// copy buttons so you can paste the roast straight into the league group chat.
function renderRoast(season, rec) {
  $('#ll-season-c').textContent = season;
  const teams = sortFocusFirst(
    [...rec.teams].filter(t => t.roast),
    (a, b) => (a.roast.miss ? a.roast.miss.vor : 0) - (b.roast.miss ? b.roast.miss.vor : 0),
  );
  const chip = pl => pl ? `${escapeHtml(pl.name)} <span class="vg-slot">pk${pl.pick}</span>` : '—';
  $('#ll-roast').innerHTML =
    `<div style="margin-bottom:12px"><button class="vg-btn vg-btn--primary" id="ll-roast-copyall">📋 Copy the whole roast</button>
       <span class="vg-note" id="ll-roast-msg" style="margin-left:8px"></span></div>`
    + teams.map(t => {
      const mine = isFocusedLookbackTeam(t);
      return `<div class="vg-roast${mine ? ' vg-roast--mine' : ''}">
        <div class="vg-roast__hd"><b>${mine ? '★ ' : ''}${escapeHtml(t.ownerName || t.name)}</b> <span class="vg-grade ${gradeClass(t.grade)}">${t.grade}</span>
          <button class="vg-btn vg-btn--ghost vg-roast__copy" data-name="${escapeAttr(t.name)}">copy</button></div>
        <p class="vg-roast__line">${escapeHtml(t.roast.line)}</p>
        <div class="vg-roast__pk"><span class="vg-val vg-val--dn">MISS</span> ${chip(t.roast.miss)}
          &nbsp; <span class="vg-val vg-val--up">HIT</span> ${chip(t.roast.hit)}
          &nbsp; <span class="vg-slot">redraft →</span> ${chip(t.roast.redraft)}</div>
      </div>`;
    }).join('')
    + `<p class="vg-note" style="margin-top:8px">Best-ball hindsight: MISS = worst pick vs its draft cost · redraft = the best player still on the board when they took it.${rec.source === 'league' ? ` Graded from your league's real ${season} draft.` : ` Seeded from real ${season}; your league's real draft roasts through the same engine once connected.`}</p>`;
  const lineFor = t => `${t.ownerName || t.name} (${t.grade}): ${t.roast.line}`;
  $('#ll-roast-copyall').addEventListener('click', () => {
    copyText(`🔥 ${season} Draft Roast 🔥\n\n` + teams.map(lineFor).join('\n\n'), '#ll-roast-msg', 'Roast copied — go start something.');
  });
  $$('.vg-roast__copy').forEach(b => b.addEventListener('click', () => {
    const t = teams.find(x => x.name === b.dataset.name);
    copyText(lineFor(t), '#ll-roast-msg', `Copied the ${t.name} roast.`);
  }));
}
function copyText(text, msgSel, okMsg) {
  (navigator.clipboard?.writeText(text) || Promise.reject())
    .then(() => { if (msgSel) $(msgSel).textContent = okMsg; })
    .catch(() => { if (msgSel) $(msgSel).textContent = 'Copy failed — select + copy manually.'; });
}

/* ================================================================
   LOOKBACK LIVE — grade a connected league's real draft history
   against actual NFL production (same VOR engine as the seeded DB).
   ================================================================ */
const LB_REPL = { QB: 12, RB: 30, WR: 36, TE: 12, K: 12, DST: 12 };
const LB_MISS = [
  '{tm} used pick {pk} on {name} and got a cool {vor} VOR for their trouble',
  'Somebody tell {tm} that pick {pk} — {name} — returned {vor} VOR',
  '{tm} slammed the table at pick {pk} for {name}. It returned {vor} VOR',
  'Pick {pk}, {tm} on the clock, hard-forces {name}: {vor} VOR',
];
const LB_HIT = [
  'credit though — {name} at {pk} was grand larceny ({vor} VOR)',
  'at least {name} at pick {pk} was a heist ({vor} VOR)',
  'saving grace: {name} ({pk}) printed {vor} VOR',
];

/** Season pool for lookback grades: skill vectors + K/DST history from kdst.json. */
function lookbackPool(season) {
  const skill = (STATE.vectors?.players || [])
    .filter(p => p.season === season && POS.includes(p.pos))
    .map(p => ({ ...p, _src: 'vec' }));
  const kdst = [];
  const push = (arr, pos) => {
    for (const p of (arr || [])) {
      const hist = p.history || {};
      const ppg = hist[season] ?? hist[String(season)];
      if (ppg == null) continue;
      kdst.push({
        id: `kdst:${p.key || normKey(p.name, pos)}:${season}`,
        name: p.name, pos, team: p.team || '', season,
        games: 16, headshot: '',
        ppg: { std: ppg, half: ppg, ppr: ppg },
        c: null, _src: 'kdst',
      });
    }
  };
  if (STATE.kdst) {
    push(STATE.kdst.kickers, 'K');
    push(STATE.kdst.dst, 'DST');
  }
  return skill.concat(kdst);
}

function vecPool(season) {
  return lookbackPool(season);
}
function replacementPpg(pool) {
  const out = {};
  for (const pos of ALL_POS) {
    const ppgs = pool.filter(p => p.pos === pos && p.games >= 4)
      .map(p => ppgOf(p)).sort((a, b) => b - a);
    const r = LB_REPL[pos] ?? 12;
    out[pos] = ppgs.length > r ? ppgs[r] : (ppgs[ppgs.length - 1] || 0);
  }
  return out;
}
function actualVorMap(season) {
  const pool = lookbackPool(season);
  const repl = replacementPpg(pool);
  const m = new Map();
  for (const p of pool) {
    if (p.games < 3) continue;
    m.set(normKey(p.name, p.pos), {
      vor: +(ppgOf(p) - (repl[p.pos] || 0)).toFixed(2),
      ppg: +ppgOf(p).toFixed(1),
      total: +(ppgOf(p) * p.games).toFixed(1),
      name: p.name, pos: p.pos, team: p.team, headshot: p.headshot || '',
      games: p.games,
    });
  }
  return m;
}
const draftCost = pick => Math.max(0, 14 - 0.09 * pick);

function playerBrief(vorMap, pick) {
  if (!pick) return null;
  const a = vorMap.get(pick.key);
  return {
    name: pick.name, pos: pick.pos, team: a?.team || '', headshot: a?.headshot || '',
    pick: pick.pick, ppg: a?.ppg ?? 0, total: a?.total ?? 0, vor: a?.vor ?? 0,
  };
}

/** Grade one season of real draft picks → lookback_seasons.json-shaped record. */
function gradeLeagueSeason(hist) {
  const vorMap = actualVorMap(hist.season);
  if (!vorMap.size) return null;
  const n = hist.teams.length;
  const scored = hist.teams.map(tm => {
    const value = tm.picks.reduce((s, pk) => s + Math.max(0, vorMap.get(pk.key)?.vor || 0), 0);
    const bestSteal = tm.picks.length
      ? [...tm.picks].sort((a, b) => {
          const va = vorMap.get(a.key)?.vor || -9;
          const vb = vorMap.get(b.key)?.vor || -9;
          return (vb - va) || (b.pick - a.pick);
        })[0]
      : null;
    const worstBust = tm.picks.length
      ? [...tm.picks].sort((a, b) => {
          const sa = (vorMap.get(a.key)?.vor || 0) - draftCost(a.pick);
          const sb = (vorMap.get(b.key)?.vor || 0) - draftCost(b.pick);
          return sa - sb;
        })[0]
      : null;
    return {
      id: tm.id || null,
      name: tm.name,
      ownerId: tm.ownerId || null,
      ownerName: tm.ownerName || null,
      wins: tm.wins,
      points_for: tm.points_for,
      seed: tm.seed,
      champion: !!tm.champion,
      draft_value: +value.toFixed(1),
      best_pick: playerBrief(vorMap, bestSteal),
      worst_pick: playerBrief(vorMap, worstBust),
      record: `${tm.wins}-${tm.losses}${tm.ties ? `-${tm.ties}` : ''}`,
      roster: tm.picks.slice(0, 8).map(pk => playerBrief(vorMap, pk)),
      _picks: tm.picks,
    };
  });
  const ranked = [...scored].sort((a, b) => b.draft_value - a.draft_value);
  ranked.forEach((t, i) => { t.grade = letterGrade(n > 1 ? 1 - i / (n - 1) : 1); });

  const pickOf = new Map();
  const teamOfKey = new Map();
  for (const t of scored) for (const pk of t._picks) {
    pickOf.set(pk.key, pk.pick);
    teamOfKey.set(pk.key, t.name);
  }
  for (const t of scored) {
    t.roast = roastTeam(t._picks, t.name, vorMap, pickOf, hist.season);
    delete t._picks;
  }

  const champ = scored.find(t => t.champion) || ranked[0];
  const worst = ranked[ranked.length - 1];
  let stealLeague = null, bustLeague = null;
  for (const [key, pick] of pickOf) {
    const a = vorMap.get(key);
    if (!a) continue;
    if (pick > n * 4 && (!stealLeague || a.vor > stealLeague.vor))
      stealLeague = { name: a.name, pos: a.pos, vor: a.vor, pick, team: teamOfKey.get(key) };
    if (pick <= n * 2 && (!bustLeague || a.vor < bustLeague.vor))
      bustLeague = { name: a.name, pos: a.pos, vor: a.vor, pick, team: teamOfKey.get(key) };
  }

  const draftNarr = [
    `**The ${hist.season} Draft, graded.** ${ranked[0].name} ran away with the class — an ${ranked[0].grade} draft that banked ${ranked[0].draft_value.toFixed(1)} points of value over replacement, anchored by ${ranked[0].best_pick?.name || 'their top pick'}. At the other end, ${worst.name} earned a ${worst.grade}: ${worst.worst_pick?.name || 'their worst pick'} never returned the draft capital.`,
    stealLeague ? `Steal of the draft: **${stealLeague.name}** (${stealLeague.pos}), scooped at pick ${stealLeague.pick} by ${stealLeague.team} and worth ${stealLeague.vor.toFixed(1)} VOR.` : '',
    bustLeague ? `Biggest reach: **${bustLeague.name}** went at pick ${bustLeague.pick} to ${bustLeague.team} and returned just ${bustLeague.vor.toFixed(1)} VOR.` : '',
  ].filter(Boolean).join(' ');

  const draftedKeys = new Set(teamOfKey.keys());
  const mvp = [...vorMap.entries()]
    .filter(([k]) => draftedKeys.has(k))
    .map(([, a]) => a)
    .sort((a, b) => b.total - a.total)[0];
  const draftRank = ranked.indexOf(champ) + 1;
  const seasonNarr = [
    `**${hist.season}: ${champ.name} are your champions.** They finished the regular season ${champ.record} (${champ.points_for.toFixed(1)} PF).`,
    mvp ? `League MVP among drafted skill players: **${mvp.name}** (${mvp.pos}), ${mvp.total.toFixed(1)} total points.` : '',
    draftRank <= 3
      ? `The title traced straight to the draft — ${champ.name} had a top-${draftRank} class (${champ.grade}).`
      : `Proof the draft isn't destiny: ${champ.name}'s class only graded ${champ.grade} (#${draftRank}), but the trophy is the trophy.`,
  ].filter(Boolean).join(' ');

  scored.sort((a, b) => b.wins - a.wins || b.points_for - a.points_for);
  return {
    season: hist.season,
    source: 'league',
    leagueName: hist.leagueName || '',
    champion: champ.name,
    teams: scored,
    narratives: { draft: draftNarr, season: seasonNarr, weeks: [] },
  };
}

function roastTeam(picks, teamName, vorMap, pickOf, season) {
  const list = (picks || []).filter(p => p && p.pick != null);
  if (!list.length) return { line: `${teamName}: no skill picks to roast.`, miss: null, hit: null, redraft: null };
  const miss = [...list].sort((a, b) => {
    const sa = (vorMap.get(a.key)?.vor || 0) - draftCost(a.pick);
    const sb = (vorMap.get(b.key)?.vor || 0) - draftCost(b.pick);
    return sa - sb;
  })[0];
  const hit = [...list].sort((a, b) => {
    const va = (vorMap.get(a.key)?.vor || -9) + 0.02 * a.pick;
    const vb = (vorMap.get(b.key)?.vor || -9) + 0.02 * b.pick;
    return vb - va;
  })[0];
  let redraft = null;
  if (miss) {
    const P = miss.pick;
    let best = null;
    for (const [key, pk] of pickOf) {
      if (pk <= P) continue;
      const a = vorMap.get(key);
      if (!a || a.vor <= 0) continue;
      if (!best || a.vor > best.vor) best = { ...a, pick: pk, key };
    }
    redraft = best;
  }
  const seed = [...`${season}-${teamName}`].reduce((h, c) => ((h << 5) - h + c.charCodeAt(0)) | 0, 0);
  const pickOpener = LB_MISS[Math.abs(seed) % LB_MISS.length];
  const pickHit = LB_HIT[Math.abs(seed >> 3) % LB_HIT.length];
  const parts = [];
  if (miss) {
    const v = vorMap.get(miss.key)?.vor || 0;
    parts.push(pickOpener
      .replace('{tm}', teamName).replace('{pk}', miss.pick)
      .replace('{name}', miss.name).replace('{vor}', (v >= 0 ? '+' : '') + v.toFixed(1)));
    if (redraft) {
      parts.push(`${redraft.name} (went ${redraft.pick}, ${redraft.vor >= 0 ? '+' : ''}${redraft.vor.toFixed(1)} VOR) was right there`);
    }
  }
  if (hit && hit.key !== miss?.key) {
    const v = vorMap.get(hit.key)?.vor || 0;
    parts.push(pickHit
      .replace('{name}', hit.name).replace('{pk}', hit.pick)
      .replace('{vor}', (v >= 0 ? '+' : '') + v.toFixed(1)));
  }
  return {
    line: (parts.join(' — ') || `${teamName} somehow drafted a blank slate.`) + '.',
    miss: miss ? playerBrief(vorMap, miss) : null,
    hit: hit ? playerBrief(vorMap, hit) : null,
    redraft: redraft ? {
      name: redraft.name, pos: redraft.pos, team: redraft.team || '', headshot: redraft.headshot || '',
      pick: redraft.pick, ppg: redraft.ppg, total: redraft.total, vor: redraft.vor,
    } : null,
  };
}

function restoreLookbackSeed() {
  STATE.lookbackLive = null;
  STATE.lookbackBySeason = new Map();
  const seed = STATE.lookbackSeed;
  STATE.lookback = seed;
  if (seed) for (const s of seed.seasons) STATE.lookbackBySeason.set(s.season, s);
  refreshLookbackSeasonPicker();
  if (LB.season != null) renderLookback();
  renderOwnerCareers();
  renderTrades();
}

function applyLiveLookback(leagueName, graded) {
  const bySeason = new Map();
  const seed = STATE.lookbackSeed;
  if (seed) for (const s of seed.seasons) bySeason.set(s.season, s);
  for (const rec of graded) bySeason.set(rec.season, rec);
  STATE.lookbackBySeason = bySeason;
  STATE.lookbackLive = {
    status: 'ready',
    leagueName,
    seasons: new Map(graded.map(g => [g.season, g])),
  };
  const latestLive = Math.max(...graded.map(g => g.season));
  if (Number.isFinite(latestLive)) LB.season = latestLive;
  refreshLookbackSeasonPicker();
  if (LB.season != null) renderLookback();
  renderOwnerCareers();
  renderTrades();
}

function lookbackFallbackSeed(lg, status, message) {
  STATE.lookbackBySeason = new Map();
  const seed = STATE.lookbackSeed;
  STATE.lookback = seed;
  if (seed) for (const s of seed.seasons) STATE.lookbackBySeason.set(s.season, s);
  STATE.lookbackLive = { status, leagueName: lg.name, seasons: new Map(), message };
  refreshLookbackSeasonPicker();
  if (LB.season != null) renderLookback();
  renderOwnerCareers();
  renderTrades();
}

let _lbRefreshToken = 0;
async function refreshLookbackFromLeague(lg) {
  const token = ++_lbRefreshToken;
  STATE.lookbackLive = { status: 'loading', leagueName: lg.name, seasons: new Map() };
  renderOwnerCareers();
  if (LB.season != null) renderLookback();
  try {
    let raw = [];
    let authFailed = false;
    let resolveThin = false;
    if (lg.platform === 'espn') {
      const creds = STATE.espnCreds || {};
      const start = Math.min(+lg.year || new Date().getFullYear(), (STATE.vectors?.seasons || []).at(-1) || 2025);
      // Walk back through completed seasons that have vector actuals (newest first).
      // Sequential to stay under ESPN rate limits; stop after a few consecutive misses.
      const years = (STATE.vectors?.seasons || []).filter(y => y <= start && y >= start - 8).sort((a, b) => b - a);
      let misses = 0;
      for (const y of years) {
        if (token !== _lbRefreshToken) return;
        try {
          const rec = await loadEspnSeasonHistory(lg.leagueId, String(y), creds.s2, creds.swid);
          if (rec) { raw.push(rec); misses = 0; }
          else if (++misses >= 3 && raw.length) break;
        } catch (e) {
          if (e.status === 401) { authFailed = true; break; }
          if (e.status === 404) { if (++misses >= 3 && raw.length) break; continue; }
          console.warn('espn history', y, e);
          // A year that returns picks but zero resolved names looks like a miss — note it.
          if (String(e.message || '').includes('resolve')) resolveThin = true;
        }
      }
    } else if (lg.platform === 'sleeper') {
      raw = await loadSleeperSeasonHistory(lg.leagueId);
    }
    if (token !== _lbRefreshToken) return;   // a newer connect superseded us
    if (authFailed && !raw.length) {
      const hasCreds = !!(STATE.espnCreds?.s2 && STATE.espnCreds?.swid);
      lookbackFallbackSeed(lg, 'auth',
        hasCreds
          ? `ESPN auth failed for ${lg.name} — cookies expired or invalid. Reconnect and re-paste espn_s2 + SWID.`
          : `Private league: paste espn_s2 + SWID via Reconnect to load draft history for ${lg.name}.`);
      return;
    }
    const graded = raw.map(gradeLeagueSeason).filter(Boolean);
    if (!graded.length) {
      const why = resolveThin
        ? `Draft picks found but player names could not be resolved for ${lg.name}.`
        : `No completed drafts found for ${lg.name} — showing seeded mock.`;
      lookbackFallbackSeed(lg, 'empty', why);
      return;
    }
    applyLiveLookback(lg.name, graded);
    try {
      localStorage.setItem('vg_lookback_' + lg.platform + '_' + lg.leagueId,
        JSON.stringify({ at: Date.now(), seasons: graded }));
    } catch (_) { /* quota */ }
  } catch (e) {
    if (token !== _lbRefreshToken) return;
    console.warn('lookback refresh failed', e);
    lookbackFallbackSeed(lg, 'err',
      `Could not load draft history (${e.message || e}) — showing seeded mock.`);
  }
}

/* ================================================================
   DRAFT PREP — season projection board (2026)
   ================================================================ */
const DRAFT = { pos: 'ALL' };
function initDraft() {
  $$('#draft-posfilter button').forEach(b => b.addEventListener('click', () => {
    DRAFT.pos = b.dataset.pos;
    $$('#draft-posfilter button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    renderDraft();
  }));
  renderDraft();
}
function renderDraft() {
  if (!STATE.proj) return;
  // Merge K/DST projections onto the board when filtering those positions (or All).
  const kdstRows = [];
  if (STATE.kdst) {
    for (const [arr, pos] of [[STATE.kdst.kickers, 'K'], [STATE.kdst.dst, 'DST']]) {
      for (const p of (arr || [])) {
        kdstRows.push({
          key: p.key || normKey(p.name, pos), name: p.name, pos, team: p.team,
          proj: p.proj, floor: Math.max(0, +(p.proj - 4).toFixed(1)), ceil: +(p.proj + 4).toFixed(1),
          tier: Math.min(10, Math.ceil((p.rank_pos || 99) / 3)), rank_pos: p.rank_pos,
          rank_overall: 9000 + (p.rank_pos || 99), bye: p.bye ?? null, line: null, comps: [],
        });
      }
    }
  }
  let rows = [...STATE.proj.players, ...kdstRows];
  if (DRAFT.pos !== 'ALL') rows = rows.filter(p => p.pos === DRAFT.pos);
  else rows = rows.sort((a, b) => b.proj - a.proj);
  const body = rows.map((p, i) =>
    `<tr><td class="vg-num">${DRAFT.pos === 'ALL' ? (p.rank_overall < 9000 ? p.rank_overall : i + 1) : p.rank_pos}</td>`
    + `<td>${nameCell(p)}</td>`
    + `<td><span class="vg-pos ${p.pos}">${p.pos}</span> <span class="vg-slot">${p.pos}${p.rank_pos || ''}</span></td>`
    + `<td><span class="vg-tier">T${p.tier || '—'}</span></td>`
    + `<td class="vg-num">${p.proj.toFixed(1)}</td>`
    + `<td class="vg-num">${adpText(p)}</td>`
    + `<td>${valueBadge(p)}</td>`
    + `<td class="vg-num">${byeText(p)}</td>`
    + `<td class="vg-num" style="text-align:left;font-size:11px">${p.line ? statLine(p.line) : (p.pos === 'K' || p.pos === 'DST' ? 'recency-weighted ppg' : '')}</td></tr>`).join('');
  const src = STATE.adp && STATE.adp.sources ? STATE.adp.sources.join(', ') : 'none';
  $('#draft-body').innerHTML = table(['#', 'Player', 'Pos', 'Tier', 'Proj/g', 'ADP', 'Value', 'Bye', 'Projected line'], body)
    + `<p class="vg-note" style="margin-top:8px">${STATE.proj.proj_season} projected PPR points/game (MTNN) + K/DST from kdst.json vs
       consensus <b>ADP</b> (${src}). <b>Value</b> = model's positional rank minus the market's — <b>VALUE</b> means
       the model likes him more than the room at his position; <b>REACH</b> the opposite. ${rows.length} players.</p>`;
  bindNameLinks();
}

/* ================================================================
   NEXT GAME — weekly MTNN with real matchup / weather / Vegas
   ================================================================ */
const NEXTG = { pos: 'ALL' };
function initNextGame() {
  if (!STATE.next) { $('#nextgame-body').innerHTML = '<div class="vg-empty">No upcoming-week model available.</div>'; return; }
  $$('#nextgame-posfilter button').forEach(b => b.addEventListener('click', () => {
    NEXTG.pos = b.dataset.pos;
    $$('#nextgame-posfilter button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    renderNextGame();
  }));
  renderNextGame();
}
function renderNextGame() {
  if (!STATE.next) return;
  // Season-rate K/DST (no matchup model) appended so the board isn't skill-only.
  const kdstRows = [];
  if (STATE.kdst) {
    for (const [arr, pos] of [[STATE.kdst.kickers, 'K'], [STATE.kdst.dst, 'DST']]) {
      for (const p of (arr || [])) {
        kdstRows.push({
          name: p.name, pos, team: p.team, opp: '—',
          proj: p.proj, floor: Math.max(0, +(p.proj - 3).toFixed(1)), ceil: +(p.proj + 3).toFixed(1),
          conditions: { is_home: true, roof: '—', temp: 0, wind: 0, team_implied: '—', primetime: false },
          line: null, _kdst: true,
        });
      }
    }
  }
  let rows = [...STATE.next.players, ...kdstRows];
  if (NEXTG.pos !== 'ALL') rows = rows.filter(p => p.pos === NEXTG.pos);
  else rows = rows.sort((a, b) => b.proj - a.proj);
  const body = rows.slice(0, 180).map(p => {
    const c = p.conditions || {};
    const matchup = p._kdst ? '<span class="vg-unmatched">season rate</span>'
      : ((c.is_home ? 'vs ' : '@ ') + escapeHtml(p.opp));
    const cond = p._kdst ? 'no matchup model'
      : ((c.roof === 'dome' || c.roof === 'closed') ? 'indoor' : `${Math.round(c.temp)}°/${Math.round(c.wind)}mph`);
    return `<tr><td>${nameCell(p)}</td><td><span class="vg-pos ${p.pos}">${p.pos}</span></td>`
      + `<td class="vg-num">${matchup}</td>`
      + `<td class="vg-num" style="font-size:11px">${cond}${p._kdst ? '' : ` · impl ${c.team_implied}${c.primetime ? ' · PT' : ''}`}</td>`
      + `<td class="vg-num">${p.proj.toFixed(1)}</td>`
      + `<td class="vg-num">${p.floor.toFixed(0)}–${p.ceil.toFixed(0)}</td>`
      + `<td class="vg-num" style="text-align:left;font-size:11px">${p.line ? statLine(p.line) : (p._kdst ? 'recency-weighted ppg' : '')}</td></tr>`;
  }).join('');
  $('#nextgame-body').innerHTML = table(['Player', 'Pos', 'Matchup', 'Conditions', 'Proj', 'Fl–Ce', 'Projected line'], body)
    + `<p class="vg-note" style="margin-top:8px">Multi-task MTNN for ${STATE.next.season} Week ${STATE.next.week}:
       ${STATE.latestSeason} form × each team's real opponent, roof/weather, and Vegas implied total.
       Floor–ceiling = per-pos conformal |residual| q80 (~80% held-out coverage). K/DST use season-rate projections.
       Re-run the pipeline weekly to roll forward.</p>`;
  bindNameLinks();
}

/* ================================================================
   EXPLORER — search + profile + comps
   ================================================================ */
const eInput = $('#explore-input'), eSuggest = $('#explore-suggest');
eInput.addEventListener('input', () => {
  const q = eInput.value.toLowerCase().trim();
  if (!q) { eSuggest.hidden = true; return; }
  const hits = STATE.names.filter(n => n.name.toLowerCase().includes(q)).slice(0, 8);
  eSuggest.innerHTML = hits.map(h =>
    `<li role="option" data-name="${escapeAttr(h.name)}" data-pos="${h.pos}">
      <span class="vg-pos ${h.pos}">${h.pos}</span> ${escapeHtml(h.name)}
      <span class="vg-slot" style="margin-left:auto">${escapeHtml(h.team || '')}</span></li>`).join('');
  eSuggest.hidden = !hits.length;
});
eSuggest.addEventListener('click', e => {
  const li = e.target.closest('li'); if (!li) return;
  eInput.value = li.dataset.name; eSuggest.hidden = true;
  showProfile(li.dataset.name, li.dataset.pos);
});
document.addEventListener('click', e => { if (!e.target.closest('.vg-search')) eSuggest.hidden = true; });

function statLine(line) {
  if (!line) return '';
  const parts = [];
  if (line.pass_yds >= 15) parts.push(`${Math.round(line.pass_yds)} pass`);
  if (line.rush_yds >= 5) parts.push(`${Math.round(line.rush_yds)} rush`);
  if (line.rec_yds >= 5) parts.push(`${Math.round(line.rec_yds)} rec (${line.rec.toFixed(1)} ct)`);
  parts.push(`${line.td.toFixed(1)} TD`);
  return parts.join(' · ');
}
function towerContribHtml(src) {
  const rows = src?.tower_contrib;
  if (!rows?.length) return '';
  const max = Math.max(...rows.map(r => r.w), 0.01);
  const bars = rows.map(r => {
    const pct = Math.round(100 * r.w / max);
    return `<div class="vg-tcontrib"><span class="vg-tcontrib__l">${escapeHtml(r.family)}</span>
      <span class="vg-tcontrib__track"><span class="vg-tcontrib__fill" style="width:${pct}%"></span></span>
      <span class="vg-tcontrib__v">${(r.w * 100).toFixed(0)}%</span></div>`;
  }).join('');
  return `<div class="vg-card__title" style="margin-top:12px">What the model weighed
      <span class="vg-slot">tower mix</span></div>
    <div class="vg-tcontribs">${bars}</div>
    <p class="vg-note">Gated fusion attention × gate, renormalized — top families for this projection.</p>`;
}

function nextGameBlock(name, pos) {
  const n = nextFor(name, pos);
  if (!n) return '';
  const c = n.conditions || {};
  const cond = [];
  cond.push(c.is_home ? `vs ${escapeHtml(n.opp)}` : `@ ${escapeHtml(n.opp)}`);
  cond.push(c.roof === 'dome' || c.roof === 'closed' ? 'indoor'
    : `${Math.round(c.temp)}°/${Math.round(c.wind)}mph`);
  cond.push(`impl ${c.team_implied}`);
  if (c.primetime) cond.push('primetime');
  return `<div class="vg-card__title" style="margin-top:16px">Next game — ${STATE.next.season} Week ${STATE.next.week}
      <span class="vg-slot">(MTNN · this matchup)</span></div>
    <div class="vg-lineup-total">
      <div class="vg-metric"><div class="vg-metric__n">${n.proj.toFixed(1)}</div><div class="vg-metric__l">proj pts</div></div>
      <div class="vg-metric"><div class="vg-metric__n">${n.floor.toFixed(0)}–${n.ceil.toFixed(0)}</div><div class="vg-metric__l">floor–ceiling</div></div>
    </div>
    <p class="vg-note">${cond.join(' · ')}</p>
    <p class="vg-note"><b>Projected line:</b> ${statLine(n.line)}</p>
    ${towerContribHtml(n)}`;
}

function showProfile(name, pos) {
  const v = vecFor(name, pos);
  const p = projFor(name, pos);
  const box = $('#explore-profile');
  const disp = v || (p ? { name: p.name, pos: p.pos, team: p.team, headshot: p.headshot } : null);
  if (!disp) { box.innerHTML = `<div class="vg-empty">No data for ${escapeHtml(name)}.</div>`; return; }

  const comps = p && p.comps && p.comps.length
    ? `<div class="vg-comp">` + p.comps.map(c =>
      `<span data-name="${escapeAttr(c.name)}" data-pos="${c.pos}" title="${c.note ? escapeAttr(c.note) : ''}" style="cursor:pointer">${escapeHtml(c.name)}${c.sim ? ` · ${(c.sim * 100).toFixed(0)}%` : ''}</span>`).join('') + `</div>`
    : '<span class="vg-unmatched">—</span>';

  const projBlock = p
    ? `<div class="vg-lineup-total">
         <div class="vg-metric"><div class="vg-metric__n">${p.proj.toFixed(1)}</div><div class="vg-metric__l">proj ${STATE.proj.proj_season} pts/g</div></div>
         <div class="vg-metric"><div class="vg-metric__n">${p.floor.toFixed(0)}–${p.ceil.toFixed(0)}</div><div class="vg-metric__l">floor–ceiling</div></div>
         <div class="vg-metric"><div class="vg-metric__n">${lastPpg(name, pos)?.toFixed(1) ?? '—'}</div><div class="vg-metric__l">${STATE.scoring} /g '${String(STATE.latestSeason).slice(2)}</div></div>
       </div>
       ${p.line ? `<p class="vg-note"><b>Projected ${STATE.proj.proj_season} line/g:</b> ${statLine(p.line)}${p.tier ? ` · tier ${p.tier} ${p.pos}${p.rank_pos}` : ''}</p>` : ''}
       ${adpFor(name, pos) ? `<p class="vg-note"><b>ADP:</b> ${adpFor(name, pos).adp.toFixed(1)} overall (${adpFor(name, pos).pos}${adpFor(name, pos).pos_rank}, ${adpFor(name, pos).n}-source consensus) · ${valueBadge(p)}</p>` : ''}
       ${towerContribHtml(p)}` : '';

  // σ-profile + career arc only when we have a season vector
  const feats = STATE.vectors.features, labels = STATE.vectors.featureLabels;
  const bars = v ? feats.map((f, i) => {
    const z = v.v[i]; const pct = Math.max(0, Math.min(100, 50 + z / 4 * 50));
    const col = z >= 0 ? 'var(--accent-2)' : 'var(--sit)';
    const left = z >= 0 ? 50 : pct, w = Math.abs(pct - 50);
    return `<div class="vg-bar"><span class="vg-bar__label">${labels[f]}</span>
      <span class="vg-bar__track"><span class="vg-bar__mid"></span>
      <span class="vg-bar__fill" style="left:${left}%;width:${w}%;background:${col}"></span></span>
      <span class="vg-bar__val">${z >= 0 ? '+' : ''}${z.toFixed(1)}</span></div>`;
  }).join('') : '<span class="vg-unmatched">no σ-profile (limited sample)</span>';

  const career = CAREER.get(normKey(name, pos)) || [];
  const maxpp = Math.max(1, ...career.map(c => ppgOf(c)));
  const arcHtml = career.length > 1
    ? `<div class="vg-card__title" style="margin-top:16px">Career arc — ${STATE.scoring.toUpperCase()} pts/game by season</div>
       <div class="vg-arc">` + career.map(c =>
        `<div class="vg-arc__col" title="${c.season}: ${ppgOf(c).toFixed(1)} ${STATE.scoring} ppg · ${STATE.vectors.clusters[c.c]}">
           <div class="vg-arc__pp">${ppgOf(c).toFixed(0)}</div>
           <div class="vg-arc__bar" style="height:${Math.round(6 + ppgOf(c) / maxpp * 64)}px;background:${POS_COLOR[c.pos] || '#888'}${v && c.season === v.season ? ';outline:2px solid var(--ink)' : ''}"></div>
           <div class="vg-arc__yr">'${String(c.season).slice(2)}</div></div>`).join('') + `</div>`
    : '';
  box.innerHTML =
    `<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
       ${disp.headshot ? `<img class="vg-headshot" style="width:48px;height:48px" src="${disp.headshot}" alt="">` : ''}
       <div><div style="font-size:20px;font-weight:800">${isMine(name, pos) ? '<span class="vg-mine">★</span> ' : ''}${escapeHtml(disp.name)}${p && p.rookie ? ' <span class="vg-rookie">R</span>' : ''}${p ? availBadge(p) + movedBadge(p) : ''}</div>
       <div class="vg-note"><span class="vg-pos ${disp.pos}">${disp.pos}</span> ${escapeHtml(disp.team)}${p && p.bye ? ` · bye ${p.bye}` : ''}${p && ownerOf(name, pos) ? ` · rostered by ${escapeHtml(ownerOf(name, pos))}` : ''}${v ? ` · ${v.season} · ${STATE.vectors.clusters[v.c]}`
         : p && p.rookie ? ` · rookie · Rd ${p.draft.round}, Pick ${p.draft.pick}` : ''}</div></div>
     </div>
     ${p && p.rookie ? `<p class="vg-note">Rookie — no NFL sample yet; ${STATE.proj.proj_season} projection from the draft-capital model. Comps = similar draft pedigree.</p>` : ''}
     ${projBlock}
     ${nextGameBlock(name, pos)}
     <div class="vg-profile">
       <div><div class="vg-card__title">Profile — σ vs season</div><div class="vg-bars">${bars}</div></div>
       <div><div class="vg-card__title">Plays like (learned embedding)</div>${comps}</div>
     </div>
     ${arcHtml}`;
  $$('.vg-comp span', box).forEach(s => s.addEventListener('click', () => {
    eInput.value = s.dataset.name; showProfile(s.dataset.name, s.dataset.pos);
  }));
  if (v) focusMapOn(v.id);
}

/* ================================================================
   3D VECTOR MAP
   ================================================================ */
let MAP = null;
function initMap(vec, latest) {
  const cv = $('#map'), ctx = cv.getContext('2d');
  MAP = { cv, ctx, all: vec.players, latest, latestOnly: true, pts: [], rot: 0, ry: 0.5,
    paused: false, drag: null, zoom: 1, focus: -1 };
  buildMapPts();
  $('#map-pause').addEventListener('click', () => {
    MAP.paused = !MAP.paused; $('#map-pause').textContent = MAP.paused ? 'Play' : 'Pause';
    if (!MAP.paused) loopMap();
  });
  $('#map-season').addEventListener('click', () => {
    MAP.latestOnly = !MAP.latestOnly;
    $('#map-season').textContent = MAP.latestOnly ? 'Latest yr' : 'All yrs';
    buildMapPts();
  });
  cv.addEventListener('pointerdown', e => { MAP.drag = { x: e.offsetX, y: e.offsetY, r: MAP.rot, ry: MAP.ry }; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => {
    if (!MAP.drag) return;
    MAP.rot = MAP.drag.r + (e.offsetX - MAP.drag.x) * 0.01;
    MAP.ry = Math.max(-1.4, Math.min(1.4, MAP.drag.ry + (e.offsetY - MAP.drag.y) * 0.008));
    if (MAP.paused) drawMap();
  });
  cv.addEventListener('pointerup', e => {
    if (MAP.drag && Math.hypot(e.offsetX - MAP.drag.x, e.offsetY - MAP.drag.y) < 4) pickPoint(e.offsetX, e.offsetY);
    MAP.drag = null;
  });
  cv.addEventListener('wheel', e => { e.preventDefault(); MAP.zoom = Math.max(0.5, Math.min(3, MAP.zoom * (e.deltaY < 0 ? 1.1 : 0.9))); if (MAP.paused) drawMap(); }, { passive: false });
  loopMap();
}
function buildMapPts() {
  MAP.pts = MAP.all
    .filter(p => !MAP.latestOnly || p.season === MAP.latest)
    .map(p => ({ id: p.id, name: p.name, pos: p.pos, season: p.season, c: POS_COLOR[p.pos] || '#888',
      mine: isMine(p.name, p.pos),
      px: p.x - 0.5, py: p.y - 0.5, pz: p.z - 0.5 }));
}
function loopMap() {
  if (!MAP || MAP.paused) return;
  MAP.rot += 0.0035;
  drawMap();
  requestAnimationFrame(loopMap);
}
function drawMap() {
  const { ctx, cv } = MAP, W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, scale = Math.min(W, H) * 0.62 * MAP.zoom;
  const cosR = Math.cos(MAP.rot), sinR = Math.sin(MAP.rot);
  const cosP = Math.cos(MAP.ry), sinP = Math.sin(MAP.ry);
  const proj = MAP.pts.map(p => {
    let x = p.px * cosR - p.pz * sinR;
    let z = p.px * sinR + p.pz * cosR;
    let y = p.py * cosP - z * sinP;
    z = p.py * sinP + z * cosP;
    const persp = 1 / (1.9 + z);
    return { p, sx: cx + x * scale * persp * 1.9, sy: cy + y * scale * persp * 1.9, depth: z, persp };
  }).sort((a, b) => a.depth - b.depth);
  for (const q of proj) {
    const foc = q.p.id === MAP.focus, mine = q.p.mine;
    const r = (foc ? 5.5 : mine ? 4 : 2.2) * (0.6 + q.persp);
    ctx.globalAlpha = foc ? 1 : mine ? 0.95 : 0.35 + q.persp * 0.9;
    ctx.beginPath(); ctx.arc(q.sx, q.sy, r, 0, 7); ctx.fillStyle = q.p.c; ctx.fill();
    if (mine && !foc) { ctx.globalAlpha = 1; ctx.strokeStyle = '#f0873c'; ctx.lineWidth = 1.5; ctx.stroke(); }
    if (foc) {
      ctx.globalAlpha = 1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = '600 13px ui-monospace,monospace';
      // when >1 season is on the map, tag which year this dot is
      const tag = MAP.latestOnly ? '' : ` '${String(q.p.season).slice(2)}`;
      ctx.fillText(q.p.name + tag, q.sx + 8, q.sy - 8);
    }
  }
  ctx.globalAlpha = 1;
  MAP._proj = proj;
}
function pickPoint(mx, my) {
  const dpr = MAP.cv.width / MAP.cv.getBoundingClientRect().width;
  const x = mx * dpr, y = my * dpr;
  let best = null, bd = 1e9;
  for (const q of (MAP._proj || [])) {
    const d = Math.hypot(q.sx - x, q.sy - y);
    if (d < bd) { bd = d; best = q; }
  }
  if (best && bd < 16) {
    MAP.focus = best.p.id;
    selectView('explorer');
    showProfile(best.p.name, best.p.pos);
    $('#explore-input').value = best.p.name;
    if (MAP.paused) drawMap();
  }
}
function focusMapOn(id) { if (MAP) { MAP.focus = id; if (MAP.paused) drawMap(); } }

/* ================================================================
   SESSION PERSISTENCE
   ================================================================ */
function persistSession() {
  const lg = STATE.league;
  if (!lg || lg.platform === 'demo') return;
  const creds = {
    platform: lg.platform, leagueId: lg.leagueId, year: lg.year, myId: lg.myId,
    espn_s2: $('#espn-s2')?.value.trim() || STATE.espnCreds?.s2 || '',
    swid: $('#espn-swid')?.value.trim() || STATE.espnCreds?.swid || '',
  };
  try { localStorage.setItem('vg_session', JSON.stringify(creds)); } catch (_) {}
}
async function restoreSession() {
  let c; try { c = JSON.parse(localStorage.getItem('vg_session') || 'null'); } catch (_) { c = null; }
  if (!c) return;
  try {
    if (c.platform === 'espn') {
      if (c.espn_s2) $('#espn-s2').value = c.espn_s2;
      if (c.swid) $('#espn-swid').value = c.swid;
      if (c.leagueId) $('#espn-league').value = c.leagueId;
      if (c.year) $('#espn-year').value = c.year;
    }
    const lg = c.platform === 'espn'
      ? await loadEspn(c.leagueId, c.year, c.espn_s2, c.swid)
      : await loadSleeper(c.leagueId);
    lg.myId = c.myId || null;
    STATE.league = lg; STATE.scoring = lg.scoring;
    populateTeamSelect(lg);
    afterLeagueLoaded();
  } catch (e) { console.warn('session restore failed', e); }
}

/* ---------- Demo hook: add a button into the connect modal ---------- */
(function addDemo() {
  const holder = $('#connect-go').parentElement;
  const b = document.createElement('button');
  b.className = 'vg-btn vg-btn--ghost'; b.type = 'button'; b.textContent = 'Try demo';
  b.title = 'Load a synthetic 10-team PPR league built from the real projection board';
  b.addEventListener('click', loadDemo);
  holder.appendChild(b);
})();

/* ================================================================
   NUX — first-run onboarding
   ================================================================ */
const NUX = { step: 0, N: 4 };
const nuxEl = $('#nux');
function nuxRender() {
  $$('.vg-nux__slide').forEach(s => s.classList.toggle('is-active', +s.dataset.step === NUX.step));
  $('#nux-back').disabled = NUX.step === 0;
  const next = $('#nux-next');
  next.textContent = NUX.step === NUX.N - 1 ? 'Maybe later' : 'Next';
  $$('#nux-dotsnav button').forEach((d, i) => d.setAttribute('aria-current', String(i === NUX.step)));
}
function nuxGo(s) { NUX.step = Math.max(0, Math.min(NUX.N - 1, s)); nuxRender(); }
function nuxOpen() { NUX.step = 0; nuxEl.hidden = false; nuxRender(); }
function nuxClose(done) {
  nuxEl.hidden = true;
  if (done) { try { localStorage.setItem('vg_onboarded', '1'); } catch (_) {} }
}
(function nuxInit() {
  const nav = $('#nux-dotsnav');
  nav.innerHTML = Array.from({ length: NUX.N }, (_, i) =>
    `<button type="button" aria-label="Step ${i + 1}"></button>`).join('');
  $$('#nux-dotsnav button').forEach((b, i) => b.addEventListener('click', () => nuxGo(i)));
  $('#nux-next').addEventListener('click', () => NUX.step === NUX.N - 1 ? nuxClose(true) : nuxGo(NUX.step + 1));
  $('#nux-back').addEventListener('click', () => nuxGo(NUX.step - 1));
  $('#nux-skip').addEventListener('click', () => nuxClose(true));
  $('#tour-btn').addEventListener('click', nuxOpen);
  $('#nux-connect').addEventListener('click', () => { nuxClose(true); openConnect(); });
  $('#nux-demo').addEventListener('click', () => { nuxClose(true); loadDemo(); });
  nuxEl.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight') nuxGo(NUX.step + 1);
    else if (e.key === 'ArrowLeft') nuxGo(NUX.step - 1);
    else if (e.key === 'Escape') nuxClose(true);
  });
})();
function maybeOnboard() {
  const onboarded = localStorage.getItem('vg_onboarded');
  const hasSession = localStorage.getItem('vg_session');
  if (!onboarded && !hasSession) nuxOpen();
}

/* ---------- utils ---------- */
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function escapeAttr(s) { return escapeHtml(s).replace(/'/g, '&#39;'); }

boot();

/* ================================================================
   DRAFT ROOM — live 2026 assistant + what-if replay on the vector map
   Self-contained: lazy-inits on first view, reads STATE + kdst.json.
   ================================================================ */
const DR = {
  ready: false, kdst: null, started: false, mode2: 'live',
  teams: 12, myTeam: 0, slots: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1 },
  rounds: 15, order: [], pickIndex: 0, rosters: [], drafted: new Set(), pool: [],
  valueLabel: 'value', wiSeason: null,
};
const DR_FLEX = new Set(['RB', 'WR', 'TE']);
const drCanvas = () => $('#dr-map');

async function drInit() {
  if (DR.ready || !STATE.proj) return;
  DR.ready = true;
  try { DR.kdst = await fetch('assets/kdst.json').then(r => r.json()); } catch (_) { DR.kdst = { kickers: [], dst: [] }; }
  const seasons = [...STATE.vectors.seasons].sort((a, b) => b - a);
  $('#dr-mode').innerHTML = `<option value="live">Live ${STATE.proj.proj_season} draft</option>`
    + seasons.map(s => `<option value="wi:${s}">What-if replay · ${s}</option>`).join('');
  const n = STATE.league ? STATE.league.teams.length : 12;
  $('#dr-teams').innerHTML = [8, 10, 12, 14].map(t => `<option${t === n ? ' selected' : ''}>${t}</option>`).join('');
  if (STATE.league && STATE.league.slots) DR.slots = { ...DR.slots, ...STATE.league.slots };
  drSlotOpts();
  $('#dr-teams').addEventListener('change', drSlotOpts);
  $('#dr-mode').addEventListener('change', () => { DR.started = false; drStatus('Mode set — hit Start.'); });
  $('#dr-start').addEventListener('click', drStart);
  $('#dr-auto').addEventListener('click', () => drStep(true));
  $('#dr-sync')?.addEventListener('click', drSyncLeague);
  drCanvas().addEventListener('click', drBoardClick);
  drStatus('Pick teams + your slot, then <b>Start</b>. CPU teams draft by value; you pick on the clock. Live = 2026 projections + ADP; What-if = a prior season’s real results (hindsight redraft).');
}
function drSlotOpts() {
  DR.teams = +$('#dr-teams').value;
  $('#dr-slot').innerHTML = Array.from({ length: DR.teams }, (_, i) => `<option value="${i}">${i + 1}</option>`).join('');
}
function drStatus(msg) { $('#dr-status').innerHTML = msg; }

function drBuildPool() {
  const mode = $('#dr-mode').value; DR.mode2 = mode;
  const pool = [];
  const push = (name, pos, team, value, x, y, ref) => { if (value != null) pool.push({ key: normKey(name, pos), name, pos, team, value, x, y, ref }); };
  if (mode === 'live') {
    for (const p of STATE.proj.players) { const v = vecFor(p.name, p.pos); push(p.name, p.pos, p.team, p.proj, v ? v.x : null, v ? v.y : null, p); }
    (DR.kdst?.kickers || []).forEach(k => push(k.name, 'K', k.team, k.proj, null, null, k));
    (DR.kdst?.dst || []).forEach(d => push(d.name, 'DST', d.team, d.proj, null, null, d));
    DR.valueLabel = `proj ${STATE.proj.proj_season}`;
  } else {
    const s = +mode.split(':')[1]; DR.wiSeason = s;
    for (const p of STATE.vectors.players) if (p.season === s) push(p.name, p.pos, p.team, ppgOf(p), p.x, p.y, p);
    (DR.kdst?.kickers || []).forEach(k => k.history[s] != null && push(k.name, 'K', k.team, k.history[s], null, null, k));
    (DR.kdst?.dst || []).forEach(d => d.history[s] != null && push(d.name, 'DST', d.team, d.history[s], null, null, d));
    DR.valueLabel = `${s} ppg`;
  }
  for (const pos of ['K', 'DST']) {
    const arr = pool.filter(p => p.pos === pos).sort((a, b) => b.value - a.value);
    arr.forEach((p, i) => { p.x = 0.04 + 0.92 * (i / Math.max(1, arr.length - 1)); p.y = pos === 'K' ? 0.965 : 0.02; });
  }
  pool.sort((a, b) => b.value - a.value);
  DR.pool = pool;
}

function drStart() {
  DR.myTeam = +$('#dr-slot').value; DR.teams = +$('#dr-teams').value;
  drBuildPool();
  DR.started = true; DR.imported = false; DR.drafted = new Set();
  DR.rosters = Array.from({ length: DR.teams }, () => []);
  DR.order = [];
  for (let r = 0; r < DR.rounds; r++) {
    const seq = r % 2 === 0 ? [...Array(DR.teams).keys()] : [...Array(DR.teams).keys()].reverse();
    DR.order.push(...seq);
  }
  DR.pickIndex = 0;
  drStep(false);
}
// Live sync: import the connected league's REAL rosters as a completed draft,
// then grade it. Works post-draft (rosters = the draft) or mid-draft (re-Sync
// pulls the latest picks). No id mapping needed — rosters already carry names.
function drSyncLeague() {
  if (!DR.ready) return;
  const lg = STATE.league;
  if (!lg) { drStatus('Connect your league first (top-right <b>Connect league</b>), then hit Sync.'); return; }
  DR.teams = lg.teams.length;
  DR.slots = { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1, ...lg.slots };
  drBuildPool();
  const byKey = new Map(DR.pool.map(p => [p.key, p]));
  DR.rosters = lg.teams.map(t => (t.roster || []).map(pl => {
    const k = normKey(pl.name, pl.pos), hit = byKey.get(k);
    if (hit) return hit;
    const pr = projFor(pl.name, pl.pos);
    return { key: k, name: pl.name, pos: pl.pos, team: pl.team || '', value: pr ? pr.proj : 0, x: null, y: null, ref: pr };
  }));
  DR.drafted = new Set(DR.rosters.flat().map(p => p.key));
  DR.myTeam = Math.max(0, lg.teams.findIndex(t => t.id === lg.myId));
  DR.order = []; DR.pickIndex = 0; DR.started = true; DR.imported = lg.name;
  const to = $('#dr-teams'); if ([...to.options].some(o => +o.value === DR.teams)) to.value = String(DR.teams);
  drSlotOpts(); $('#dr-slot').value = String(DR.myTeam);
  drRender();
}

const drAvailable = () => DR.pool.filter(p => !DR.drafted.has(p.key));
function drNeeds(roster) {
  const have = {}; roster.forEach(p => have[p.pos] = (have[p.pos] || 0) + 1);
  const need = {};
  for (const pos of ['QB', 'RB', 'WR', 'TE', 'K', 'DST']) need[pos] = Math.max(0, (DR.slots[pos] || 0) - (have[pos] || 0));
  return { need };
}
function drCpuPick(team) {
  const avail = drAvailable();
  const { need } = drNeeds(DR.rosters[team]);
  return avail.find(p => need[p.pos] > 0) || avail.find(p => DR_FLEX.has(p.pos) || p.pos === 'QB') || avail[0];
}
function drDoPick(pick, team) {
  if (!pick) { DR.pickIndex++; return; }
  DR.drafted.add(pick.key); DR.rosters[team].push(pick); DR.pickIndex++;
}
function drStep(auto) {
  if (!DR.started) return;
  let guard = 0;
  while (DR.pickIndex < DR.order.length && guard++ < 1000) {
    const team = DR.order[DR.pickIndex];
    if (team === DR.myTeam) { if (auto) { drDoPick(drCpuPick(team), team); continue; } break; }
    drDoPick(drCpuPick(team), team);
  }
  drRender();
}
function drBoardClick(e) {
  if (!DR.started || DR.order[DR.pickIndex] !== DR.myTeam) { drStatus('Not your pick — hit <b>Sim to my pick</b>.'); return; }
  const cv = drCanvas(), rect = cv.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width, y = (e.clientY - rect.top) / rect.height;
  let best = null, bd = 1e9;
  for (const p of drAvailable()) { if (p.x == null) continue; const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p; } }
  if (best && bd < 0.06) drPick(best.key);
}
function drPick(key) {
  if (DR.order[DR.pickIndex] !== DR.myTeam) return;
  drDoPick(drAvailable().find(p => p.key === key), DR.myTeam);
  drStep(false);
}
window.drPick = drPick;

function drRender() {
  drBoard();
  const done = DR.pickIndex >= DR.order.length;
  const onClock = DR.order[DR.pickIndex];
  const rnd = Math.floor(DR.pickIndex / DR.teams) + 1;
  if (!done) drStatus(onClock === DR.myTeam
    ? `<b>You're on the clock</b> — round ${rnd}. Click a lit dot on the map or a <b>+</b> below.`
    : `Round ${rnd}, pick ${DR.pickIndex + 1} — Team ${onClock + 1} (CPU). Hit <b>Sim to my pick</b>.`);
  const { need } = drNeeds(DR.rosters[DR.myTeam] || []);
  const myTurn = onClock === DR.myTeam && !done;
  const ranked = [...drAvailable()].sort((a, b) => (need[b.pos] > 0) - (need[a.pos] > 0) || b.value - a.value).slice(0, 14);
  $('#dr-avail-title').textContent = `Best available — ${DR.valueLabel}`;
  $('#dr-avail').innerHTML = table(['Player', 'Pos', 'Val', ''],
    ranked.map(p => `<tr><td>${escapeHtml(p.name)}${p.ref && p.ref.rookie ? ' <span class="vg-rookie">R</span>' : ''}${need[p.pos] > 0 ? ' <span class="vg-slot">need</span>' : ''}</td>`
      + `<td><span class="vg-pos ${p.pos}">${p.pos}</span></td><td class="vg-num">${p.value.toFixed(1)}</td>`
      + `<td>${myTurn ? `<button class="vg-btn vg-btn--ghost" onclick="drPick('${escapeAttr(p.key)}')">+</button>` : ''}</td></tr>`).join(''));

  const mine = DR.rosters[DR.myTeam] || [];
  const fill = greedyFill(mine.map(p => ({ player: p, p: p.value })), DR.slots);
  const kS = drExtra(mine, 'K'), dS = drExtra(mine, 'DST');
  const total = fill.total + (kS ? kS.value : 0) + (dS ? dS.value : 0);
  const starters = [...fill.starters.filter(s => s.player).map(s => ({ slot: s.slot, p: s.player, v: s.p })),
    kS ? { slot: 'K', p: kS, v: kS.value } : null, dS ? { slot: 'DST', p: dS, v: dS.value } : null].filter(Boolean);
  const stKeys = new Set(starters.map(s => s.p.key));
  $('#dr-myteam-title').textContent = `Your draft — ${mine.length} picks · ${total.toFixed(1)} starter/wk`;
  $('#dr-myteam').innerHTML = table(['Slot', 'Player', 'Val'],
    starters.map(s => `<tr class="vg-row--mine"><td><span class="vg-slot">${s.slot}</span></td><td>${escapeHtml(s.p.name)} <span class="vg-pos ${s.p.pos}">${s.p.pos}</span></td><td class="vg-num">${s.v.toFixed(1)}</td></tr>`)
      .concat(mine.filter(p => !stKeys.has(p.key)).map(p => `<tr><td><span class="vg-slot">BN</span></td><td>${escapeHtml(p.name)} <span class="vg-pos ${p.pos}">${p.pos}</span></td><td class="vg-num">${p.value.toFixed(1)}</td></tr>`)).join(''));

  $('#dr-log').innerHTML = table(['#', 'Team', 'Pick'],
    drPickLog().slice(-18).reverse().map(e => `<tr><td class="vg-num">${e.n}</td><td>${e.team === DR.myTeam ? '<span class="vg-mine">★</span> ' : ''}Team ${e.team + 1}</td><td>${escapeHtml(e.name)} <span class="vg-pos ${e.pos}">${e.pos}</span></td></tr>`).join(''));

  if (done) drGrade(total);
}
function drExtra(roster, pos) {
  if (!(DR.slots[pos] > 0)) return null;
  return roster.filter(p => p.pos === pos).sort((a, b) => b.value - a.value)[0] || null;
}
function drPickLog() {
  const counts = {}, out = [];
  for (let i = 0; i < DR.pickIndex; i++) {
    const team = DR.order[i]; counts[team] = counts[team] || 0;
    const p = DR.rosters[team][counts[team]]; counts[team]++;
    if (p) out.push({ n: i + 1, team, name: p.name, pos: p.pos });
  }
  return out;
}
function drGrade(total) {
  const totals = DR.rosters.map(r => {
    const f = greedyFill(r.map(p => ({ player: p, p: p.value })), DR.slots);
    const k = drExtra(r, 'K'), d = drExtra(r, 'DST');
    return f.total + (k ? k.value : 0) + (d ? d.value : 0);
  });
  const sorted = [...totals].sort((a, b) => b - a);
  const myRank = sorted.indexOf(total) + 1;
  const pct = 1 - (myRank - 1) / (DR.teams - 1 || 1);
  const grade = pct >= .9 ? 'A' : pct >= .75 ? 'B' : pct >= .5 ? 'C' : pct >= .25 ? 'D' : 'F';
  drStatus(`<b>Draft ${DR.imported ? 'synced' : 'complete'} — grade ${grade}.</b> Your starters: <b>${total.toFixed(1)}</b>/wk, <b>#${myRank}</b> of ${DR.teams}. `
    + (DR.imported ? `Your real league <b>${escapeHtml(DR.imported)}</b>, graded on ${STATE.proj.proj_season} projections.`
      : DR.mode2 !== 'live' ? `What-if replay of ${DR.wiSeason}: could you beat the field with hindsight?`
        : `Live ${STATE.proj.proj_season} value captured.`));
}
function drBoard() {
  const cv = drCanvas(); if (!cv) return;
  const ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const onClock = DR.order[DR.pickIndex], myTurn = onClock === DR.myTeam;
  const { need } = drNeeds(DR.rosters[DR.myTeam] || []);
  const availSet = new Set(drAvailable().map(p => p.key));
  const bestKeys = new Set(drAvailable().sort((a, b) => (need[b.pos] > 0) - (need[a.pos] > 0) || b.value - a.value).slice(0, 8).map(p => p.key));
  const mineKeys = new Set((DR.rosters[DR.myTeam] || []).map(p => p.key));
  for (const p of DR.pool) {
    if (p.x == null) continue;
    const sx = 16 + p.x * (W - 32), sy = 16 + p.y * (H - 32);
    const isAvail = availSet.has(p.key), mine = mineKeys.has(p.key), best = myTurn && bestKeys.has(p.key);
    ctx.globalAlpha = isAvail ? (best ? 1 : 0.8) : 0.15;
    ctx.beginPath(); ctx.arc(sx, sy, mine ? 4.5 : best ? 4 : 2.4, 0, 7);
    ctx.fillStyle = mine ? '#f0873c' : (POS_COLOR[p.pos] || '#888'); ctx.fill();
    if (best || mine) { ctx.globalAlpha = 1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.4; ctx.stroke(); }
  }
  ctx.globalAlpha = 1;
}
$$('.vg-tab').forEach(t => t.dataset.view === 'draftroom' && t.addEventListener('click', () => {
  const tryInit = () => { if (STATE.proj) drInit(); else setTimeout(tryInit, 150); };
  tryInit();
}));
