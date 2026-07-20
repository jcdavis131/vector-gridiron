# Vector Gridiron — your fantasy football cockpit

> Solo personal project, no connection to employer, built with public/free-tier only.
> **Built in raw WebGPU / WebGL / Canvas — no Unity/Unreal, just browser graphics APIs straight.**

MTNN next-game MAE 4.268 (R² 0.39) · 2025 nflverse data · holistic features (usage · snaps · age · weather · Vegas · rest · def-vs-pos) · no tracking.

Live: https://gridiron.dumbmodel.com/ → redirects from gridiron.jcamd.com

**Rendering:** raw WebGPU/WebGL path — `<canvas>` map with custom shaders, no engine, static hosting, free-tier.



## Mobile-first responsive fix (2026-07-10)
Mirrors vector-hoops 35af415 + vector-pitch 739e2ab + jcamd e8f5447:

- `assets/responsive.css` loads AFTER `gridiron.css + shell.css + nux.css`
- fluid `--page-gutter: clamp(12px,4vw,20px)` + `env(safe-area-inset-*)`
- `--touch-min:44px`, inputs 16px to avoid iOS zoom
- tabs `.vg-tabs` horizontal rail with mask fade + momentum scroll
- canvas `.vg-mapwrap` + `canvas#dr-map` fluid `aspect-ratio:1`, `max-height:min(70vh,600px)`, `touch-action:none`
- tables `.vg-tablewrap` momentum scroll
- dialogs `.vg-dialog` 92vw safe-area
- scoped only — no blanket svg/canvas rules

Ultimate target: Everything responsive with mobile friendly UX.

## Share flow (fantasy Wordle)
- URL `?l=BOOZEBOIS` league code stored in `vectorGridiron.v1` + `vectorGridiron.league.<CODE>.board`
- Result card copy/paste + Web Share API
- OG SVG 1200x630 dark gridiron
- Zero backend, device-only board

## Deploy
Vercel: `cleanUrls:true` + redirect gridiron.jcamd.com → gridiron.dumbmodel.com

Link new repo in Vercel dashboard if current deploy is not from GitHub.