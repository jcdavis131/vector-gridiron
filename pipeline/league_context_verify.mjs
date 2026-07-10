// League-context unit checks: owner career rollup keyed by manager id.
// Mirrors helpers in assets/app.js (kept small / dependency-free).
import fs from 'fs';

let fails = 0;
const check = (cond, msg) => {
  console.log((cond ? 'PASS ' : 'FAIL ') + msg);
  if (!cond) fails++;
};

const GRADE_ORD = { 'A+': 12, A: 11, 'A-': 10, 'B+': 9, B: 8, 'B-': 7, 'C+': 6, C: 5, 'C-': 4, D: 3, F: 1 };
const ORD_GRADE = Object.entries(GRADE_ORD).sort((a, b) => b[1] - a[1]);
function gradeFromOrd(mean) {
  let best = 'F', dist = Infinity;
  for (const [g, o] of ORD_GRADE) {
    const d = Math.abs(o - mean);
    if (d < dist) { dist = d; best = g; }
  }
  return best;
}
const ownerNorm = name => String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
function ownerCareerKey(t) {
  if (!t) return null;
  if (t.ownerId) return 'oid:' + String(t.ownerId);
  if (t.ownerName) return 'oname:' + ownerNorm(t.ownerName);
  if (t.name) return 'tname:' + ownerNorm(t.name);
  return null;
}

function ownerCareerMap(seasons) {
  const by = new Map();
  for (const rec of seasons) {
    for (const t of rec.teams) {
      const k = ownerCareerKey(t);
      if (!k) continue;
      let row = by.get(k);
      if (!row) {
        row = {
          name: t.ownerName || t.name, ownerId: t.ownerId || null,
          teamNames: [], seasons: 0, titles: 0, pf: 0, gradeOrds: [],
        };
        by.set(k, row);
      }
      row.seasons += 1;
      if (t.champion) row.titles += 1;
      row.pf += t.points_for || 0;
      if (GRADE_ORD[t.grade] != null) row.gradeOrds.push(GRADE_ORD[t.grade]);
      if (t.name && !row.teamNames.includes(t.name)) row.teamNames.push(t.name);
      if (t.ownerName) row.name = t.ownerName;
    }
  }
  for (const row of by.values()) {
    const mean = row.gradeOrds.reduce((a, b) => a + b, 0) / row.gradeOrds.length;
    row.avgGrade = gradeFromOrd(mean);
    row.avgPf = row.pf / row.seasons;
  }
  return by;
}

check(gradeFromOrd(12) === 'A+', 'A+ ordinal maps back');
check(gradeFromOrd(8) === 'B', 'B ordinal maps back');
check(ownerNorm('Team  Awesome!!') === 'team awesome', 'ownerNorm strips punctuation');
check(ownerCareerKey({ ownerId: '{ABC}', name: 'Wheat Plate' }) === 'oid:{ABC}', 'ownerCareerKey prefers ownerId');

// Same manager, different team names across seasons → one career row.
const seasons = [
  {
    season: 2023, source: 'league',
    teams: [
      { name: 'Booze Cruise', ownerId: 'mgr-1', ownerName: 'Cam', grade: 'A', champion: true, points_for: 1600 },
      { name: 'Sad Squad', ownerId: 'mgr-2', ownerName: 'Pat', grade: 'D', champion: false, points_for: 1200 },
    ],
  },
  {
    season: 2024, source: 'league',
    teams: [
      { name: 'Wheat Plate', ownerId: 'mgr-1', ownerName: 'Cam', grade: 'B+', champion: false, points_for: 1500 },
      { name: 'Sad Squad Reloaded', ownerId: 'mgr-2', ownerName: 'Pat', grade: 'C', champion: true, points_for: 1550 },
    ],
  },
];
const careers = ownerCareerMap(seasons);
const cam = careers.get('oid:mgr-1');
const pat = careers.get('oid:mgr-2');
check(careers.size === 2, `rename seasons collapse to 2 owners (got ${careers.size})`);
check(cam.seasons === 2 && cam.titles === 1, 'Cam: 2 seasons, 1 title across renames');
check(cam.teamNames.includes('Booze Cruise') && cam.teamNames.includes('Wheat Plate'),
  'Cam teamNames tracks both aliases');
check(pat.titles === 1, 'Pat: 1 title');
check(cam.avgGrade === 'A-' || cam.avgGrade === 'A' || cam.avgGrade === 'B+',
  `Cam avg grade sensible (${cam.avgGrade})`);
check(Math.abs(cam.avgPf - 1550) < 1, `Cam avg PF ${cam.avgPf}`);

// Name-only fallback still works when no ownerId (seed / legacy).
const byName = ownerCareerMap([
  { season: 2022, teams: [{ name: 'Solo Act', grade: 'B', champion: false, points_for: 1400 }] },
]);
check(byName.has('tname:solo act'), 'falls back to team-name key without ownerId');

const html = fs.readFileSync('index.html', 'utf8');
check(html.includes('id="ll-owners"'), 'index.html has #ll-owners');
check(html.includes('career draft grades'), 'Lookback owners card copy present');
check(html.includes('manager'), 'owners card copy mentions manager identity');

const app = fs.readFileSync('assets/app.js', 'utf8');
check(app.includes('function ownerCareerMap'), 'app.js defines ownerCareerMap');
check(app.includes('function ownerCareerKey'), 'app.js defines ownerCareerKey');
check(app.includes('function espnMembersById'), 'app.js maps ESPN members');
check(app.includes('primaryOwner'), 'app.js reads ESPN primaryOwner');
check(app.includes('ownerId: r.owner_id') || app.includes('ownerId: r.owner_id != null'), 'Sleeper history stores ownerId');
check(app.includes('renderTrades()') && app.includes('applyLiveLookback'), 'lookback ready refreshes trades');
check(app.includes('partnerLabel') || app.includes('vg-owner-chip'), 'trades show owner career chips');
check(app.includes('careers.get(ownerCareerKey'), 'trades look up partners by ownerCareerKey');
check(
  app.includes("rest.mode !== 'players'") && app.includes("rest.mode !== 'playerPool'"),
  'espnProxy preserves players/playerPool arrays (no data[0] collapse)',
);

console.log(fails ? `\n${fails} failure(s)` : '\nALL LEAGUE-CONTEXT CHECKS PASSED');
process.exit(fails ? 1 : 0);
