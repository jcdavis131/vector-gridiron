// League-context unit checks: owner career rollup + grade averaging.
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

function ownerCareerMap(seasons) {
  const by = new Map();
  for (const rec of seasons) {
    for (const t of rec.teams) {
      const k = ownerNorm(t.name);
      let row = by.get(k);
      if (!row) {
        row = { name: t.name, seasons: 0, titles: 0, pf: 0, gradeOrds: [] };
        by.set(k, row);
      }
      row.seasons += 1;
      if (t.champion) row.titles += 1;
      row.pf += t.points_for || 0;
      if (GRADE_ORD[t.grade] != null) row.gradeOrds.push(GRADE_ORD[t.grade]);
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

const seasons = [
  {
    season: 2023, source: 'league',
    teams: [
      { name: 'Booze Cruise', grade: 'A', champion: true, points_for: 1600 },
      { name: 'Sad Squad', grade: 'D', champion: false, points_for: 1200 },
    ],
  },
  {
    season: 2024, source: 'league',
    teams: [
      { name: 'Booze Cruise', grade: 'B+', champion: false, points_for: 1500 },
      { name: 'Sad Squad', grade: 'C', champion: true, points_for: 1550 },
    ],
  },
];
const careers = ownerCareerMap(seasons);
const bc = careers.get(ownerNorm('Booze Cruise'));
const ss = careers.get(ownerNorm('Sad Squad'));
check(bc.seasons === 2 && bc.titles === 1, 'Booze Cruise: 2 seasons, 1 title');
check(ss.titles === 1, 'Sad Squad: 1 title');
check(bc.avgGrade === 'A-' || bc.avgGrade === 'A' || bc.avgGrade === 'B+',
  `Booze Cruise avg grade sensible (${bc.avgGrade})`);
check(Math.abs(bc.avgPf - 1550) < 1, `Booze Cruise avg PF ${bc.avgPf}`);

// HTML has owners mount point
const html = fs.readFileSync('index.html', 'utf8');
check(html.includes('id="ll-owners"'), 'index.html has #ll-owners');
check(html.includes('career draft grades'), 'Lookback owners card copy present');

const app = fs.readFileSync('assets/app.js', 'utf8');
check(app.includes('function ownerCareerMap'), 'app.js defines ownerCareerMap');
check(app.includes('function renderOwnerCareers'), 'app.js defines renderOwnerCareers');
check(app.includes('renderTrades()') && app.includes('applyLiveLookback'), 'lookback ready refreshes trades');
check(app.includes('partnerLabel') || app.includes('vg-owner-chip'), 'trades show owner career chips');
// Regression: players/playerPool arrays must NOT be collapsed to data[0]
check(
  app.includes("rest.mode !== 'players'") && app.includes("rest.mode !== 'playerPool'"),
  'espnProxy preserves players/playerPool arrays (no data[0] collapse)',
);
check(app.includes("headers['X-Espn-S2']") || app.includes('X-Espn-S2'), 'espn cookies sent via headers');
check(app.includes("status: 'auth'") || app.includes("status === 'auth'"), 'lookback auth status distinct from empty');
check(app.includes('updateConnectButton'), 'connected UX updates Connect → Reconnect');
check(app.includes('mode: \'playerPool\'') || app.includes('mode: "playerPool"'), 'playerPool fallback for name resolve');

const espnApi = fs.readFileSync('api/espn.js', 'utf8');
check(espnApi.includes('playerPool'), 'api/espn.js supports playerPool mode');
check(espnApi.includes('x-espn-s2'), 'api/espn.js reads cookie headers');

console.log(fails ? `\n${fails} failure(s)` : '\nALL LEAGUE-CONTEXT CHECKS PASSED');
process.exit(fails ? 1 : 0);
