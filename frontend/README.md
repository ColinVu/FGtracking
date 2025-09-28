# PrizePicks-styled Pixel Minigame

Small Vite + React prototype. This repository contains a compact minigame UI where players guess how far back a made field goal would still be good.

Game rules (demo):
- Opening screen has a 10s countdown to press PLAY (auto-starts when it expires).
- When the question appears you have 7 seconds to type a yards number and submit.
- For this demo the 'actual' physics value is randomized. A guess counts as correct if it is >= actual and <= actual + 3.

Files:
- `index.html` - entry point
- `src/` - React source files (App, main, styles)

Getting started:

1. Install dependencies

```powershell
cd prizepicksgame
npm install
```

2. Run dev server

```powershell
npm run dev
# open http://localhost:5173
```

3. Build for production

```powershell
npm run build
npm run preview
```

Notes:
- This is a small prototype; for production you may want to optimize assets and create proper build and deployment steps.

License: MIT (see LICENSE)
