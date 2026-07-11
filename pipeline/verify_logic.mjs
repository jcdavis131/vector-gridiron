// Headless verification of the cockpit's core logic against real artifacts.
// Mirrors the pure functions in assets/app.js (normKey, optimalLineup, demo
// draft, waivers, comps) so we can prove the data joins + algorithms work
// without a browser. Run: node pipeline/verify_logic.mjs
import fs from 'fs';

const vec = JSON.parse(fs.readFileSync('assets/vectors.json', 'utf8'));
const proj = JSON.parse(fs.readFileSync('assets/projections.json', 'utf8'));
const POS = ['QB', 'RB', 'WR', 'TE'];
const FLEX_ELIG = new Set(['RB', 'WR', 'TE']);
let fails = 0;
const check = (cond, msg) => { console.log((cond ? 'PASS ' : 'FAIL ') + msg); if (!cond) fails++; };

function normKey(name, pos) {
  let s = (name || '').toLowerCase().replace(/[.'`]/g, '');
  s = s.replace(/\s+(jr|sr|ii|iii|iv|v)$/i, '').replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim();
  return `${s}|${pos}`;
}
const projByKey = new Map(proj.players.map(p => [p.key, p]));
const latest = vec.seasons.at(-1);
const vecByKey = new Map(vec.players.filter(p => p.season === latest).map(p => [normKey(p.name, p.pos), p]));

const next = JSON.parse(fs.readFileSync('assets/nextgame.json', 'utf8'));
const emb = JSON.parse(fs.readFileSync('assets/embedding.json', 'utf8'));
const rep = proj.model.report;

// 1) artifacts sane
check(vec.count === vec.players.length && vec.count > 4000, `vectors.json has ${vec.count} player-seasons`);
check(vec.seasons.at(-1) === 2025, `vectors include 2025 (latest = ${vec.seasons.at(-1)})`);
check(proj.count > 300 && proj.proj_season === 2026, `projections: ${proj.count} players, season ${proj.proj_season}`);
check(rep.model_fpts_mae < rep.baseline_last4_mae,
  `MTNN MAE ${rep.model_fpts_mae} beats last-4 baseline ${rep.baseline_last4_mae} (R² ${rep.model_fpts_r2})`);
check(next.count > 300 && next.week >= 1, `nextgame.json: ${next.count} players, ${next.season} wk ${next.week}`);
check(emb.count > 300 && emb.dims >= 16, `embedding.json: ${emb.count} players, ${emb.dims}-d learned space`);

// rookies: draft-capital model added drafted players with no NFL stats
const rookies = proj.players.filter(p => p.rookie);
check(rookies.length >= 1 && proj.rookie_count === rookies.length,
  `${rookies.length} rookies on the board (draft-capital model)`);
check(rookies.every(r => r.draft && r.draft.pick > 0 && r.line), 'rookies carry draft capital + a stat line');
check(rookies.every(r => r.comps.every(c => c.name !== r.name)), 'rookie comps never include the player himself');
check(proj.rookie_model.val_fpts_mae < proj.rookie_model.baseline_pos_mae,
  `rookie model MAE ${proj.rookie_model.val_fpts_mae} beats positional baseline ${proj.rookie_model.baseline_pos_mae}`);

// ADP: consensus from a variety of sources, joins to the board, flags value
if (fs.existsSync('assets/adp.json')) {
  const adp = JSON.parse(fs.readFileSync('assets/adp.json', 'utf8'));
  const aByKey = new Map(adp.players.map(p => [p.key, p]));
  check(adp.sources.length >= 2 && adp.count > 100, `ADP: ${adp.count} players from ${adp.sources.length} sources [${adp.sources}]`);
  check(adp.players.every(p => p.adp > 0 && p.adp_rank >= 1 && p.n >= 1), 'ADP rows have consensus adp + rank + source count');
  const joined = proj.players.filter(p => aByKey.has(p.key));
  check(joined.length > 100, `${joined.length} projections joined to ADP`);
  // positional value edge = ADP pos rank - model pos rank; both signs should exist
  const edges = joined.filter(p => aByKey.get(p.key).pos_rank).map(p => aByKey.get(p.key).pos_rank - p.rank_pos);
  check(edges.some(e => e >= 8) && edges.some(e => e <= -8), `value edges span VALUE and REACH (max +${Math.max(...edges)}, min ${Math.min(...edges)})`);
}

// 2) skill projections carry comps (learned embedding) + a stat line
//    K/DST are season-rate (source=kdst) — no embedding comps / vector join.
const skill = proj.players.filter(p => p.source !== 'kdst' && p.pos !== 'K' && p.pos !== 'DST');
const kdstN = proj.count - skill.length;
const withComps = skill.filter(p => p.comps && p.comps.length >= 3).length;
check(withComps === skill.length, `all ${skill.length} skill projections have comps (${withComps}; +${kdstN} K/DST)`);
check(proj.players.every(p => p.line && 'rush_yds' in p.line && 'rec_yds' in p.line), 'every projection has a stat line');
let matched = 0; for (const p of skill) if (vecByKey.has(p.key)) matched++;
check(matched / skill.length > 0.8, `proj->2025-vector join ${(100 * matched / skill.length).toFixed(0)}% of skill (rest low-sample/rookies)`);

// availability: byes (schedule), team-changes (latest_team), roster status
check(proj.players.every(p => p.bye == null || (p.bye >= 1 && p.bye <= 18)), 'projections carry valid bye weeks');
check(proj.players.filter(p => p.moved).length >= 10, `${proj.players.filter(p => p.moved).length} players flagged as offseason team-movers`);
check(proj.players.some(p => p.avail), 'some players carry an availability flag (RES/CUT/IR)');
check(next.players.every(p => 'avail' in p && 'bye' in p), 'next-game rows carry availability + bye');

// seeded lookback league: graded drafts + narratives for every prior season
if (fs.existsSync('assets/lookback_seasons.json')) {
  const lb = JSON.parse(fs.readFileSync('assets/lookback_seasons.json', 'utf8'));
  check(lb.seasons.length >= 9, `lookback seed covers ${lb.seasons.length} seasons`);
  const s = lb.seasons.find(x => x.season === 2023) || lb.seasons[0];
  check(s.teams.length === 12 && s.teams.every(t => /^[A-F]/.test(t.grade)), 'every team has a draft letter grade');
  check(s.teams.filter(t => t.champion).length === 1, 'exactly one champion per season');
  check(s.narratives.draft.length > 80 && s.narratives.season.length > 80, 'draft + season narratives written');
  check(s.narratives.weeks.length >= 14 && s.narratives.weeks.every(w => w.text.includes('Week')), 'weekly recaps written');
  check(s.teams.every(t => t.best_pick && t.worst_pick), 'each team has a best steal + biggest bust');
}

// 3) known players resolve with sane projections + next-game predictions
for (const [nm, pos] of [['Bijan Robinson', 'RB'], ['Ja\'Marr Chase', 'WR'], ['Josh Allen', 'QB']]) {
  const p = projByKey.get(normKey(nm, pos));
  check(p && p.proj > 8, `${nm} proj = ${p ? p.proj : 'MISSING'} pts/g`);
  check(p && p.comps.length >= 3 && p.comps.every(c => c.sim > 0), `${nm} has ${p ? p.comps.length : 0} comps`);
}
// next-game predictions carry matchup + conditions
const ng0 = next.players[0];
check(ng0.opp && ng0.conditions && 'temp' in ng0.conditions && 'team_implied' in ng0.conditions,
  `top next-game (${ng0.name}) has opponent + conditions`);
check(ng0.line && ng0.line.td != null, 'next-game predictions include a stat line');

// 4) demo draft + optimizer (ported from app.js loadDemo/optimalLineup)
function optimalLineup(roster, slots) {
  const wp = roster.map(pl => ({ player: pl, p: (projByKey.get(normKey(pl.name, pl.pos)) || {}).proj }))
    .filter(x => x.p != null && POS.includes(x.player.pos)).sort((a, b) => b.p - a.p);
  const slotList = [];
  for (const p of POS) for (let i = 0; i < (slots[p] || 0); i++) slotList.push(p);
  for (let i = 0; i < (slots.FLEX || 0); i++) slotList.push('FLEX');
  const used = new Set(), starters = [];
  for (const slot of slotList.filter(s => s !== 'FLEX')) {
    const c = wp.find(x => !used.has(x) && x.player.pos === slot);
    if (c) { used.add(c); starters.push({ slot, ...c }); } else starters.push({ slot, player: null, p: 0 });
  }
  for (const _ of slotList.filter(s => s === 'FLEX')) {
    const c = wp.find(x => !used.has(x) && FLEX_ELIG.has(x.player.pos));
    if (c) { used.add(c); starters.push({ slot: 'FLEX', ...c }); } else starters.push({ slot: 'FLEX', player: null, p: 0 });
  }
  const bench = wp.filter(x => !used.has(x));
  return { starters, bench, total: starters.reduce((s, x) => s + (x.p || 0), 0) };
}

const board = [...projByKey.values()].sort((a, b) => b.proj - a.proj);
const pools = Object.fromEntries(POS.map(p => [p, board.filter(x => x.pos === p)]));
const T = 10, need = { QB: 2, RB: 5, WR: 6, TE: 2 }, cursor = { QB: 0, RB: 0, WR: 0, TE: 0 };
const teams = Array.from({ length: T }, (_, i) => ({ id: String(i + 1), roster: [] }));
for (let r = 0; r < Math.max(...Object.values(need)); r++)
  for (const pos of POS) {
    if (r >= need[pos]) continue;
    const seq = r % 2 === 0 ? teams : [...teams].reverse();
    for (const t of seq) { const pk = pools[pos][cursor[pos]++]; if (pk) t.roster.push({ name: pk.name, pos: pk.pos }); }
  }
const slots = { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1 };
const rosterSizes = teams.map(t => t.roster.length);
check(rosterSizes.every(s => s === 15), `all demo teams drafted 15 players (${rosterSizes.join(',')})`);
const opt = optimalLineup(teams[0].roster, slots);
check(opt.starters.filter(s => s.player).length === 7, `optimizer fills 7 starters (${opt.starters.filter(s => s.player).length})`);
check(opt.total > 90, `team 1 optimal projected total = ${opt.total.toFixed(1)} pts`);
// no player double-started
const startNames = opt.starters.filter(s => s.player).map(s => s.player.name);
check(new Set(startNames).size === startNames.length, 'no player started twice');
// flex holds an RB/WR/TE
const flex = opt.starters.find(s => s.slot === 'FLEX');
check(flex && flex.player && FLEX_ELIG.has(flex.player.pos), `FLEX = ${flex?.player?.name} (${flex?.player?.pos})`);

// 5) waivers: available = projections not rostered in demo league
const rostered = new Set(teams.flatMap(t => t.roster.map(pl => normKey(pl.name, pl.pos))));
const avail = board.filter(p => !rostered.has(p.key));
check(avail.length > 100, `${avail.length} players on the waiver value board`);
check(avail[0].proj <= board[0].proj, 'top waiver <= top overall (rostered players removed)');

// 6) ESPN scoring/slot parsing (mock mSettings)
function detectEspnScoring(s) {
  const it = (s.scoringSettings?.scoringItems || []).find(i => i.statId === 53);
  const pts = it ? (it.points ?? 0) : 0;
  return pts >= 1 ? 'ppr' : pts >= 0.5 ? 'half' : 'std';
}
function espnSlots(s) {
  const c = s.rosterSettings?.lineupSlotCounts || {};
  return { QB: c[0] || 0, RB: c[2] || 0, WR: c[4] || 0, TE: c[6] || 0, FLEX: (c[23] || 0) + (c[7] || 0) };
}
const mock = { scoringSettings: { scoringItems: [{ statId: 53, points: 1 }] },
  rosterSettings: { lineupSlotCounts: { 0: 1, 2: 2, 4: 2, 6: 1, 23: 1, 20: 6 } } };
check(detectEspnScoring(mock) === 'ppr', `ESPN scoring parsed = ${detectEspnScoring(mock)}`);
const es = espnSlots(mock);
check(es.QB === 1 && es.RB === 2 && es.WR === 2 && es.TE === 1 && es.FLEX === 1, `ESPN slots parsed = ${JSON.stringify(es)}`);

// 7) Sleeper slot counting (mock roster_positions)
function countSlots(arr) {
  const s = { QB: 0, RB: 0, WR: 0, TE: 0, FLEX: 0 };
  for (const p of arr) { if (p === 'FLEX') s.FLEX++; else if (s[p] !== undefined) s[p]++; }
  return s;
}
const ss = countSlots(['QB', 'RB', 'RB', 'WR', 'WR', 'TE', 'FLEX', 'K', 'DEF', 'BN', 'BN']);
check(ss.QB === 1 && ss.RB === 2 && ss.WR === 2 && ss.TE === 1 && ss.FLEX === 1, `Sleeper slots parsed = ${JSON.stringify(ss)}`);

console.log(fails === 0 ? '\nALL CHECKS PASSED' : `\n${fails} CHECK(S) FAILED`);
process.exit(fails ? 1 : 0);
