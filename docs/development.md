# Development and verification

Run all commands from the repository root unless noted. Setup installs API/test dependencies into `.venv` and JavaScript dependencies into `web/node_modules`.

```sh
python scripts/setup.py
python scripts/test.py
python scripts/doctor.py
```

The test runner checks portable arrays, sparse expressions, FK/IK, pose encoding/interpolation, synthetic-rig retargeting, source-photo access, generation admission with mocked workers, package compaction, IDOL math helpers and the production frontend build. It never launches neural generation. The legacy Python/C# experiments are separate from this maintained browser contract.

## Frontend development

Keep `python scripts/serve.py` running on port 8765. In a second terminal:

```sh
npm --prefix web run dev
```

Open http://127.0.0.1:5173. Vite proxies API/WebSocket requests to 8765. Use `npm --prefix web run build` after changes to update the server's built UI. Do not expose Vite to the Internet.

## Browser smoke tests

Install the test browser once, then start the server before testing:

```sh
cd web
npx playwright install chromium
cd ..
python scripts/serve.py
```

In another terminal:

```sh
npm --prefix web run test:browser
```

These tests require generation **disabled**, a synthetic demo and a WebGL-capable browser. They check loading, animation, recording and a receiver window. Set `AVATAR_LAB_URL` if using another port. On Linux CI, `npx playwright install --with-deps chromium` installs browser system dependencies too. Traces and screenshots remain in ignored `web/test-results/`; review them for private data before sharing.

## Optional camera and motion assets

Review [third-party terms](../THIRD_PARTY_NOTICES.md) before explicit downloads:

```sh
node scripts/setup-optional-assets.mjs --camera
node scripts/setup-optional-assets.mjs --mixamo
npm --prefix web run build
```

The camera option downloads three checksum-pinned MediaPipe task files and copies the package WASM runtime. The motion option downloads the pinned Three.js r185 Mixamo example. Neither runs by default; both stay under ignored `web/public/`. Importing your own licensed FBX does not require downloading the example.

MediaPipe inference and FBX parsing run in the browser. Camera permission is requested only when enabled. Webcam retargeting is experimental; it is not a Meta SDK integration, global root tracker or a calibrated semantic expression mapper.

## Native parity and research reproduction

The native test skips unless `GSAVATAR_NATIVE_FIXTURE` names a folder containing a dense v1 LHM export with seven native reference poses. Use an export from your own authorized inputs. Compact-v2 native references are checked by the in-app reference action and lab validators. Never use the synthetic avatar to claim native parity.

LHM++ and IDOL lab runs require separate environments, weights and evidence folders. See [generation setup](generation.md) and [evidence boundaries](evidence.md). Performance measurements need the exact hardware, asset, viewport, warm-up and sample window; an FPS label is not a benchmark.

## Website and publication checks

`docs/index.html` is the canonical, self-contained architecture page. Keep inline SVG diagrams and source links accessible. Do not embed participant photos or native evidence binaries.

```sh
python scripts/check-publication.py
python scripts/build-pages.py
```

The audit checks the proposed Git source set and site links. The Pages build emits only `_site/index.html` and `_site/.nojekyll`; never point an upload step at the entire repository. CI runs the audit on every push and pull request, plus Windows/Linux unit/build jobs and a Linux browser smoke test. GPU inference and licensed-data tests are intentionally not part of public CI.
