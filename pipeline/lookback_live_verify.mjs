import fs from 'fs';
// Pull pure helpers by eval'ing the lookback grading section is hard;
// instead re-implement the same normKey/VOR path the app uses and grade ESPN 899513/2020.

const vec = JSON.parse(fs.readFileSync('assets/vectors.json','utf8'));
const POS = ['QB','RB','WR','TE'];
const GRADE_TIERS = [[.95,'A+'],[.88,'A'],[.80,'A-'],[.72,'B+'],[.62,'B'],[.52,'B-'],[.42,'C+'],[.32,'C'],[.22,'C-'],[.12,'D'],[0,'F']];
const letterGrade = pct => (GRADE_TIERS.find(([t]) => pct >= t) || [0,'F'])[1];
const LB_REPL = { QB:12, RB:30, WR:36, TE:12 };
function normKey(name, pos) {
  let s = (name || '').toLowerCase();
  s = s.replace(/[.'`]/g, '');
  s = s.replace(/\s+(jr|sr|ii|iii|iv|v)$/i, '');
  s = s.replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim();
  return `${s}|${pos}`;
}
const ppgOf = p => p.ppg?.ppr ?? 0;
function actualVorMap(season) {
  const pool = vec.players.filter(p => p.season === season && POS.includes(p.pos));
  const repl = {};
  for (const pos of POS) {
    const ppgs = pool.filter(p => p.pos === pos && p.games >= 4).map(ppgOf).sort((a,b)=>b-a);
    const r = LB_REPL[pos];
    repl[pos] = ppgs.length > r ? ppgs[r] : (ppgs.at(-1) || 0);
  }
  const m = new Map();
  for (const p of pool) {
    if (p.games < 3) continue;
    m.set(normKey(p.name, p.pos), {
      vor: +(ppgOf(p) - repl[p.pos]).toFixed(2),
      ppg: +ppgOf(p).toFixed(1),
      total: +(ppgOf(p) * p.games).toFixed(1),
      name: p.name, pos: p.pos, team: p.team, headshot: p.headshot || '',
    });
  }
  return m;
}

const ESPN_POS = {1:'QB',2:'RB',3:'WR',4:'TE',5:'K',16:'DST'};
const year = 2020, leagueId = 899513;
const data = await fetch(
  `https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${year}/segments/0/leagues/${leagueId}?view=mDraftDetail&view=mTeam&view=mSettings`,
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
  for (const p of rows) resolved.set(p.id, { name: p.fullName, pos: ESPN_POS[p.defaultPositionId] || (p.id < 0 ? 'DST' : '?') });
}

const teamsById = {};
for (const t of data.teams) {
  const rec = t.record?.overall || {};
  teamsById[t.id] = {
    name: t.name || `Team ${t.id}`,
    wins: rec.wins || 0, losses: rec.losses || 0, ties: rec.ties || 0,
    points_for: +(t.points || 0).toFixed(1),
    seed: t.playoffSeed || 99,
    champion: t.rankCalculatedFinal === 1,
    picks: [],
  };
}
for (const pk of picks) {
  const tm = teamsById[pk.teamId]; if (!tm) continue;
  const pl = resolved.get(pk.playerId); if (!pl || !POS.includes(pl.pos)) continue;
  tm.picks.push({ pick: pk.overallPickNumber, name: pl.name, pos: pl.pos, key: normKey(pl.name, pl.pos) });
}
const teams = Object.values(teamsById).filter(t => t.picks.length);
if (!teams.some(t => t.champion)) {
  [...teams].sort((a,b) => a.seed - b.seed || b.wins - a.wins || b.points_for - a.points_for)[0].champion = true;
}

const vor = actualVorMap(year);
const scored = teams.map(tm => {
  const value = tm.picks.reduce((s, pk) => s + Math.max(0, vor.get(pk.key)?.vor || 0), 0);
  const matched = tm.picks.filter(p => vor.has(p.key)).length;
  return { name: tm.name, draft_value: +value.toFixed(1), champion: tm.champion, matched, n: tm.picks.length, record: `${tm.wins}-${tm.losses}` };
});
scored.sort((a,b) => b.draft_value - a.draft_value);
scored.forEach((t,i) => { t.grade = letterGrade(scored.length > 1 ? 1 - i / (scored.length - 1) : 1); });

let fails = 0;
const check = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };
check(vor.size > 300, `VOR map has ${vor.size} players for ${year}`);
check(teams.length === 10, `10 teams graded (${teams.length})`);
check(scored.every(t => t.matched / t.n > 0.7), `name join >70% on every team (worst ${Math.min(...scored.map(t=>t.matched/t.n)).toFixed(2)})`);
check(scored[0].draft_value > scored.at(-1).draft_value, `draft values spread ${scored[0].draft_value} .. ${scored.at(-1).draft_value}`);
check(scored.filter(t => t.champion).length === 1, 'exactly one champion');
check(/^A/.test(scored[0].grade) && /[CDF]/.test(scored.at(-1).grade), `grades ${scored[0].grade} .. ${scored.at(-1).grade}`);
console.log('--- top 3 ---');
scored.slice(0,3).forEach(t => console.log(`  ${t.grade} ${t.name} VOR ${t.draft_value} (${t.matched}/${t.n}) ${t.champion?'CHAMP':''}`));
console.log(fails ? `FAILED ${fails}` : 'ALL PASS');
process.exit(fails ? 1 : 0);

