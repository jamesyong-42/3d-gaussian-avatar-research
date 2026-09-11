# Publishing documentation and sharing a preview

## GitHub Pages: static architecture only

The project site is [jamesyong-42.github.io/3d-gaussian-avatar-research](https://jamesyong-42.github.io/3d-gaussian-avatar-research/). The source is `docs/index.html`; `.github/workflows/pages.yml` audits source, builds an explicit `_site` allowlist, and deploys it using GitHub's Pages environment.

For a fork, enable **Settings → Pages → Source: GitHub Actions**, then run **Architecture Pages** or push a documentation change to `main`. Update repository/source URLs in the README, architecture and publication checker when renaming or moving the repo. The original project's Pages configuration is not automatically inherited by forks.

Only `index.html` and `.nojekyll` are uploaded. Pages does not run the API, execute CUDA, accept photos, hold the local avatar library or publish a live preview URL. Do not replace the workflow artifact path with `.` or the research workspace. See [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Local application

```sh
python scripts/serve.py
```

The launcher binds only to `127.0.0.1:8765`, serves the built frontend and runs one API worker. Do not launch multiple application processes against the same data directory. Keep the generation environment separate, back up data deliberately, and define your own retention policy before using personal photos.

The bundled implementation is not a production service: no per-user library, formal access roles, abuse controls, storage quotas, queue durability guarantees or operational monitoring. GPU inference and hostile-image handling require a separate deployment/security review.

## Optional password-protected Tailscale preview

This is an opt-in shared-password preview for trusted testers. Install and authorize Tailscale yourself, review its Funnel/public-access settings, and identify the exact DNS hostname assigned to your device. No real hostname or credential is stored in this repository.

With the app already running on 8765, start the separate gateway from the repository root, substituting your actual device hostname:

```sh
node web/scripts/run_public_gateway.mjs YOUR-EXACT-DEVICE-DNS.ts.net
```

The placeholder is not a working hostname. The launcher generates a strong random password and prints it **once to your terminal**, alongside the username and intended public URL. Do not paste this output into an issue, transcript, CI log or commit. Restarting rotates the password and invalidates sessions. Keep the terminal running.

The gateway listens on loopback **8766** and forwards to the app on **8765**. After verifying it is listening and denying unauthenticated requests, use your Tailscale Funnel configuration to expose **8766 only**. Inspect existing Tailscale serve/funnel status first so you do not replace another service's routing. Never route Funnel directly to the unauthenticated API or the Vite development server.

The gateway requires HTTPS at the public edge and validates the exact public origin. Its Secure/HttpOnly/SameSite cookie and WebSocket checks are intended for that HTTPS hostname. A request reaching `http://127.0.0.1:8766` without credentials should return 401. In the gateway terminal, `check-local` and `check-public` run the optional verification helper; that helper requires the optional browser/model fixtures documented in its code and is not part of the asset-free quick start.

All admitted testers can see the saved library and, if enabled, submit generation jobs. A shared password is not per-user isolation. Use a separate data directory containing only demo material approved for those testers. Camera permission is local to their browser, while uploaded photos are stored on your server.

Stop the gateway with Ctrl+C, then disable only the Funnel mapping you created using the Tailscale CLI/UI. Confirm the public endpoint no longer reaches the app. This repository does not change or disclose any existing private Tailscale deployment.
