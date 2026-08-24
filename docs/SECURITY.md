# Security Considerations

## Transport

- All server communication uses HTTPS (`server.verify_tls: true` by default)
- TLS certificate verification is **never** disabled in production
- Connection timeouts prevent hung connections

## Credentials

- Auth tokens stored in `/etc/idcam/config.yaml` (mode 600) or `IDCAM_AUTH_TOKEN` environment variable
- Tokens are **never** logged
- Service runs as unprivileged `idcam` user
- systemd unit uses `ProtectSystem=strict`, `NoNewPrivileges=true`

## Input Validation

- Configuration validated with Pydantic on startup
- Server responses parsed as JSON with malformed-response handling
- Image file size bounded by `server.max_upload_bytes`
- File paths use `pathlib`; no shell interpolation

## Upload Semantics

- Unique `capture_id` and `Idempotency-Key` per transaction
- No blind POST retries after body sent (prevents duplicate processing)
- One safe connection retry on pre-send failures only

## Filesystem

- Captures written to temp files, atomic rename
- Temp directory owned by service user
- Log rotation prevents disk exhaustion

## Attack Surface

- No admin HTTP server exposed by default
- Status file is local JSON (readable by service user group)
- No arbitrary command execution
- No shell=True in subprocess calls

## Recommendations

1. Use short-lived API tokens, rotate regularly
2. Restrict `/etc/idcam/` permissions to root:idcam 750
3. Keep Raspberry Pi OS and Arducam drivers updated
4. Use firewall rules to limit outbound connections to API host only
5. Enable full-disk encryption if documents are sensitive
