// Verify K/DST join into lookback VOR + ESPN draft history.
import fs from 'fs';

const vec = JSON.parse(fs.readFileSync('assets/vectors.json', 'utf8'));
const kdst = JSON.parse(fs.readFileSync('assets/kdst.json', 'utf8'));
const ALL = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
const ESPN_POS = { 1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST' };
const ESPN_TEAM = {
  1: 'ATL', 2: 'BUF', 3: 'CHI', 4: 'CIN', 5: 'CLE', 6: 'DAL', 7: 'DEN', 8: 'DET',
  9: 'GB', 10: 'TEN', 11: 'IND', 12: 'KC', 13: 'LV', 14: 'LA', 15: 'MIA', 16: 'MIN',
  17: 'NE', 18: 'NO', 19: 'NYG', 20: 'NYJ', 21: 'PHI', 22: 'ARI', 23: 'PIT', 24: 'LAC',
  25: 'SF', 26: 'SEA', 27: 'TB', 28: 'WAS', 29: 'CAR', 30: 'JAX', 33: 'BAL', 34: 'HOU',
};
function normKey(name, pos) {
  let s = (name || '').toLowerCase();
  s = s.replace(/[.'`]/g, '');
  s = s.replace(/\s+(jr|sr|ii|iii|iv|v)$/i, '');
  s = s.replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim();
  return `${s}|${pos}`;
}
function espnPlayerName(pl, pos) {
  if (pos === 'DST') {
    const a = ESPN_TEAM[pl.proTeamId] || ESPN_TEAM[Math.abs(pl.id) - 16000];
    return a ? `${a} DST` : (pl.fullName || 'DST');
  }
  return pl.fullName || '';
}

const year = 2020;
const LB_REPL = { QB: 12, RB: 30, WR: 36, TE: 12, K: 12, DST: 12 };
const pool = [];
for (const p of vec.players.filter(p => p.season === year)) {
  pool.push({ key: normKey(p.name, p.pos), pos: p.pos, ppg: p.ppg.ppr, games: p.games });
}
for (const [arr, pos] of [[kdst.kickers, 'K'], [kdst.dst, 'DST']]) {
  for (const p of arr) {
    const h = p.history || {};
    const ppg = h[year] ?? h[String(year)];
    if (ppg == null) continue;
    pool.push({ key: normKey(p.name, pos), pos, ppg, games: 16 });
  }
}
const repl = {};
for (const pos of ALL) {
  const ppgs = pool.filter(p => p.pos === pos && p.games >= 4).map(p => p.ppg).sort((a, b) => b - a);
  const r = LB_REPL[pos];
  repl[pos] = ppgs.length > r ? ppgs[r] : (ppgs.at(-1) || 0);
}
const vor = new Map();
for (const p of pool) {
  if (p.games < 3) continue;
  vor.set(p.key, +(p.ppg - repl[p.pos]).toFixed(2));
}

const data = await fetch(
  `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/segments/0/leagues/899513?view=mDraftDetail&view=mTeam`,
  { headers: { 'User-Agent': 'vector-gridiron/1.0' } }
).then(r => r.json());
const picks = data.draftDetail.picks;
const ids = [...new Set(picks.map(p => p.playerId))];
const resolved = new Map();
for (let i = 0; i < ids.length; i += 80) {
  const chunk = ids.slice(i, i + 80);
  const rows = await fetch(
    `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/players?scoringPeriodId=0&view=players_wl`,
    { headers: { 'User-Agent': 'vector-gridiron/1.0', 'X-Fantasy-Filter': JSON.stringify({ filterIds: { value: chunk } }) } }
  ).then(r => r.json());
  for (const p of rows) {
    const pos = ESPN_POS[p.defaultPositionId] || (p.id < 0 ? 'DST' : '?');
    resolved.set(p.id, { name: espnPlayerName(p, pos), pos });
  }
}

let k = 0, d = 0, km = 0, dm = 0;
for (const pk of picks) {
  const pl = resolved.get(pk.playerId);
  if (!pl) continue;
  if (pl.pos === 'K') { k++; if (vor.has(normKey(pl.name, pl.pos))) km++; }
  if (pl.pos === 'DST') { d++; if (vor.has(normKey(pl.name, pl.pos))) dm++; }
}

const lb = JSON.parse(fs.readFileSync('assets/lookback_seasons.json', 'utf8'));
const s2020 = lb.seasons.find(s => s.season === 2020);
const seedPos = new Set();
for (const t of s2020.teams) for (const p of (t.roster || [])) if (p) seedPos.add(p.pos);

let fails = 0;
const check = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
check(vor.size > 400, `VOR pool ${vor.size} (skill+K/DST)`);
check([...vor.keys()].filter(k => k.endsWith('|K')).length > 20, `K in VOR map`);
check([...vor.keys()].filter(k => k.endsWith('|DST')).length === 32, `32 DST in VOR map`);
check(k >= 4 && km === k, `ESPN K join ${km}/${k}`);
check(d >= 4 && dm === d, `ESPN DST join ${dm}/${d}`);
check(seedPos.has('K') && seedPos.has('DST'), `seeded lookback drafts K+DST (${[...seedPos]})`);
console.log(fails ? `FAILED ${fails}` : 'ALL PASS');
process.exit(fails ? 1 : 0);
