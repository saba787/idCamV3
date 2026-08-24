# idCam

Production document capture for **Raspberry Pi 5** + **Arducam 64MP Hawkeye** autofocus camera.

Continuously monitors the capture area, detects documents, stabilizes, autofocuses, captures, validates, and uploads to a remote server. Designed for 24/7 unattended operation.

## Features

- Explicit deterministic state machine (no callback spaghetti)
- Single active capture transaction (no queue)
- Arducam 64MP / libcamera / Picamera2 integration
- Autofocus with sharpness verification
- Document detection and stability gating
- Atomic capture files with validation
- HTTPS upload with idempotency key
- Structured JSON logging with rotation
- systemd autostart with recovery
- Mock camera mode for development/testing

## Requirements

- Raspberry Pi 5
- Arducam 64MP Hawkeye (CSI, CAM1 port on Pi 5)
- Raspberry Pi OS Bookworm or Trixie (64-bit)
- Arducam driver stack (see Installation)

## Production installation on Raspberry Pi

On a fresh Raspberry Pi OS (Bookworm/Trixie, 64-bit) the operator does **not** create
`/var/log/idcam`, `/var/lib/idcam`, or `/run/idcam` by hand. The installer and systemd own those paths.

```bash
git clone https://github.com/shipittofar/idCamV3.git
cd idCamV3
sudo ./scripts/install.sh
```

The installer creates the `idcam` system user and group, installs the app into
`/opt/idcam/venv`, writes a first-boot config **only if** `/etc/idcam/config.yaml`
does not already exist, installs `/etc/systemd/system/idcam.service`, reloads
systemd, and enables the unit. It does **not** start the service until
`device_id` and `server.url` are real values (the shipped template uses
`CHANGE_ME` / `https://unconfigured.invalid` and will fail validation on purpose).

Configure the device:

```bash
sudo nano /etc/idcam/config.yaml
```

Set at least:

- `device_id` — unique id for this Pi (not `CHANGE_ME`)
- `server.url` — real HTTPS API base URL (not `example.com` / `unconfigured.invalid`)
- `server.path` — upload path (default `/api/document`)

Set the auth token in the environment file (preferred; never commit secrets):

```bash
sudo nano /etc/idcam/environment
```

```bash
IDCAM_AUTH_TOKEN=your-token-here
```

Optional overrides in the same file: `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`.

Then enable and start:

```bash
sudo systemctl enable --now idcam
```

Verify:

```bash
sudo systemctl status idcam
journalctl -u idcam -f
cat /run/idcam/status.json
```

### Upgrade

Existing config, token, captured data, and logs are preserved.

```bash
cd idCamV3
git pull
sudo ./scripts/install.sh
sudo systemctl restart idcam   # if the service was not already running
```

### Uninstall

```bash
sudo ./scripts/uninstall.sh              # keeps /etc/idcam, /var/lib/idcam, /var/log/idcam
sudo ./scripts/uninstall.sh --purge      # type DELETE to also remove config/data/logs
```

## Development (off-device)

```bash
pip install -e ".[dev]"
IDCAM_MOCK_CAMERA=1 idcam --config config.example.yaml
```

## Configuration

See `config.example.yaml`. Key settings:

| Setting | Description |
|---------|-------------|
| `device_id` | Unique device identifier |
| `server.url` | HTTPS API base URL |
| `server.path` | Upload endpoint path |
| `auth.token` | Bearer token (prefer `IDCAM_AUTH_TOKEN` env) |
| `camera.capture_width/height` | Full-res capture (default 9152×6944) |
| `quality.min_sharpness` | Laplacian variance threshold |

Environment overrides: `IDCAM_CONFIG`, `IDCAM_AUTH_TOKEN`, `IDCAM_DEVICE_ID`, `IDCAM_SERVER_URL`, `IDCAM_ENV`, `IDCAM_MOCK_CAMERA`.

## User Status

The application writes real-time status to `/run/idcam/status.json`:

```json
{
  "state": "IDLE",
  "message": "Ready — Place document",
  "camera_healthy": true,
  "network_healthy": true
}
```

## Server API

See [API Protocol](docs/API_PROTOCOL.md) for full idempotency and timeout semantics.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ -m hardware  # on Pi with camera
```

Current test suite: **133 unit tests passing**, 3 hardware tests skipped off-device.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Security](docs/SECURITY.md)

## License

Proprietary
