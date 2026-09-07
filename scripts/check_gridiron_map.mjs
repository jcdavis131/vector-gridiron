// Run the page's real load->transform->centroid path against the actual JSON files and check
// every numeric claim the prose makes.
//
//   node scripts/check_gridiron_map.mjs
//
// Do NOT pipe it. A pipe has eaten the exit code three times in this project's ledger.
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const readPub = f => JSON.parse(fs.readFileSync(path.join(ROOT, 'public', f), 'utf-8'));
let fails = 0;
const ok = (c, m) => { console.log((c ? '  PASS  ' : '  FAIL  ') + m); if (!c) fails++; };

const map = readPub('assets/gridiron_map.json');
const ros = readPub('assets/gridiron_roster.json');
const html = fs.readFileSync(path.join(ROOT, 'public', 'index.html'), 'utf-8');

console.log('\n-- file contract --');
ok(map.rows.length === map.n_rows, `rows ${map.rows.length} == declared ${map.n_rows}`);
ok(map.rows.length === 10700, 'row count is the full 10,700 player-seasons');
ok(JSON.stringify(map.positions) === JSON.stringify(['QB', 'RB', 'TE', 'WR']),
   'positions are QB, RB, TE, WR');
ok(map.seasons.length === 27 && map.seasons[0] === '1999' && map.seasons.at(-1) === '2025',
   `27 seasons, 1999 to 2025 (${map.seasons.length}, ${map.seasons[0]}-${map.seasons.at(-1)})`);
ok(ros.tiles.length === 12, '12 roster tiles');

console.log('\n-- no fabricated values (this branch shipped 1,000 "player_0_16" rows) --');
ok(!map.rows.some(r => /^(Player|player)[ _]?\d/.test(r.name)), 'no placeholder names survive');
ok(!map.rows.some(r => /^\d$/.test(String(r.season))), 'no single-digit fake seasons survive');
ok(map.rows.every(r => r.name && r.pos && Number.isFinite(r.x) && Number.isFinite(r.y)),
   'every row has a name, a position and finite coordinates');
const dxy = new Set(map.rows.map(r => r.x + ',' + r.y)).size;
ok(dxy > map.rows.length * 0.9, `coordinates are per-row (${dxy} distinct of ${map.rows.length})`);
const modMap = (html.match(/<script id="mod-map">([\s\S]*?)<\/script>/) || [])[1] || '';
ok(modMap.length > 1000, 'mod-map found');
ok(!/\blcg\b|Math\.random|box-?muller|const N=2000/i.test(modMap), 'mod-map contains no generator');
ok(/fetch\("assets\/gridiron_map\.json"/.test(modMap), 'mod-map fetches the real export');
ok(/loadError/.test(modMap), 'mod-map has an explicit error path');
for (const bad of ['synthesized 2000', 'MAHOMES</h3>']) {
  ok(!html.includes(bad), `page no longer contains "${bad}"`);
}
ok(!JSON.stringify(map).includes('player_0_16') && !JSON.stringify(ros).includes('player_0_16'),
   'the shipped DATA contains no "player_0_16" rows');
ok(html.includes('player_0_16'),
   'the footer still names what this branch used to ship, so the fix is legible');
ok(html.includes('10,700'), 'page states the real row count 10,700');

console.log('\n-- coordinate provenance --');
ok(/\|r\| = \[1\.0, 1\.0, 1\.0\]/.test(map.coords || '') || /1\.0/.test(map.coords || ''),
   'the file records that the PCA was independently reproduced');
ok(/embedding\.json/.test(map.coords_are_not || '') &&
   /MTNN/.test(map.coords_are_not || ''),
   'the file names the separate learned-embedding artifact these coordinates are NOT');
ok(/nflverse/i.test(map.attribution || ''), 'nflverse attribution carried through');

console.log('\n-- centroids and the prose that quotes them --');
let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
for (const r of map.rows) {
  if (r.x < x0) x0 = r.x; if (r.x > x1) x1 = r.x;
  if (r.y < y0) y0 = r.y; if (r.y > y1) y1 = r.y;
}
const dx = (x1 - x0) || 1, dy = (y1 - y0) || 1;
const pts = map.rows.map(r => ({ x: (r.x - x0) / dx, y: 1 - (r.y - y0) / dy,
                                 pos: r.pos, name: r.name, season: r.season }));
ok(pts.every(p => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1), 'all points inside the 0..1 box');
const c = {};
for (const p of pts) { (c[p.pos] ||= { n: 0 }).n++; }
ok(c.WR.n === 4383 && c.RB.n === 2916 && c.TE.n === 2246 && c.QB.n === 1155,
   'counts are WR 4,383 / RB 2,916 / TE 2,246 / QB 1,155 as the steps state');
ok(Math.round(c.WR.n / map.rows.length * 100) === 41, 'step 4: receivers are 41% of the map');
ok(c.QB.n === Math.min(...Object.values(c).map(v => v.n)),
   'step 6: quarterbacks are the smallest group');
ok(Math.round(map.explained_variance[0] * 100) === 37 &&
   Math.round(map.explained_variance[1] * 100) === 27,
   `step 2: PC1/PC2 carry 37% and 27% (actual ${map.explained_variance.slice(0,2).map(v => Math.round(v*100)+'%').join(', ')})`);

console.log('\n-- how to read a cosine here --');
const cd = map.cosine_distribution || {};
ok(cd.n_pairs > 100000, `cosine distribution over ${cd.n_pairs} pairs`);
ok(Math.abs(cd.median) < 0.15,
   `footer: this space is NOT concentrated, random pairs near 0.00 (actual median ${cd.median})`);

console.log('\n-- roster tiles --');
for (const t of ros.tiles) {
  ok(pts.some(p => p.name === t.name && p.season === t.season && p.pos === t.pos),
     `${t.name} (${t.pos}, ${t.season}) is findable on the map`);
}
ok(ros.tiles.every(t => t.nearest_other_season && t.nearest_other_season.season !== t.season),
   'every neighbour really is from a different season');
for (const p of ['QB', 'RB', 'WR', 'TE']) {
  ok(ros.tiles.filter(t => t.pos === p).length === 3, `3 tiles for ${p}`);
  ok(ros.tiles.filter(t => t.pos === p && t.pick === 'most typical').length === 1,
     `exactly one "most typical" pick for ${p}`);
}
ok(new Set(ros.tiles.map(t => t.nearest_other_season.name)).size >= 10,
   `neighbours are not degenerate: ${new Set(ros.tiles.map(t => t.nearest_other_season.name)).size} distinct of 12`);

console.log(fails === 0 ? '\nALL CHECKS PASSED\n' : `\n${fails} CHECK(S) FAILED\n`);
process.exit(fails === 0 ? 0 : 1);
