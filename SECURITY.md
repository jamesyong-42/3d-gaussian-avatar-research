# Security and private data

This is a local, single-user research prototype. The main API has loopback host/origin checks and bounded uploads, but no per-user authentication, tenant isolation, storage quota, comprehensive rate limiting or automatic data-retention policy. Do not deploy it as a public production API.

For a trusted preview, put the separate password gateway in front of it and tunnel **only the gateway**. The gateway authenticates assets, API requests and WebSocket upgrades; all authorized testers still share the same avatar library. See [deployment](docs/deployment.md).

Process only authorized photos and trusted model/code assets. Native models and FBX parsers are not a security sandbox. Keep model environments separate from the lightweight API. Do not place secrets or personal captures in source, HTML, GitHub issues, CI logs or screenshots. `.gitignore` is a safety net, not a permissions system; `git add -f` can bypass it.

For a suspected vulnerability, use the repository's private vulnerability reporting feature under **Security → Report a vulnerability**, when available. Do not include real participant media or credentials. If private reporting is unavailable, open a minimal issue requesting a private contact without disclosing exploit details or personal data.

If credentials were published, revoke/rotate them first. Removing a file in a later commit does not remove Git history or copies. For accidentally published personal data, stop distribution and coordinate removal; do not assume a history rewrite recalls existing downloads.
