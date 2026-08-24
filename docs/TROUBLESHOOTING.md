# Troubleshooting Guide

## Camera

### Camera not detected

```bash
rpicam-hello --list-cameras
libcamera-hello --list-cameras
```

- Verify CSI cable on **CAM1** port (Pi 5)
- Check `dtoverlay=arducam-64mp` in `/boot/firmware/config.txt`
- Reinstall driver: `./install_pivariety_pkgs.sh -p 64mp_pi_hawk_eye_kernel_driver`
- Reboot after driver install

### Autofocus controls not advertised

Some libcamera versions do not expose `AfMode`/`AfTrigger`. idCam probes at startup and falls back to `autofocus_cycle()` or sharpness-only verification.

Try:
- Update Arducam libcamera packages to latest
- Set `LIBCAMERA_NOTPARSED_EMBEDDED_DATA=1` in systemd unit
- Downgrade to libcamera 0.0.10 if AF broken on newer version (see Arducam forum)

### Slow capture (>10s)

64MP JPEG capture is inherently slow. Expected 10–20s at full resolution. Preview runs at lower resolution.

### Capture fails / camera stuck

```bash
sudo systemctl restart idcam
journalctl -u idcam --since "10 min ago"
```

idCam auto-recovers up to `camera.recovery_max_attempts` times.

## Network

### Upload failures

```bash
curl -v https://your-api.example.com/
journalctl -u idcam | grep upload
```

- Check Wi-Fi: `nmcli dev wifi list`
- Verify TLS cert is valid (do not disable `verify_tls`)
- Check `IDCAM_AUTH_TOKEN` is set

### Ambiguous upload (connection dropped)

If upload may have reached server but response was lost, server must deduplicate via `Idempotency-Key`. Client does not auto-retry POST after send.

## Application

### Stuck state

Watchdog triggers recovery after `watchdog.state_timeout_sec` (default 300s).

### Status file

```bash
cat /run/idcam/status.json | python3 -m json.tool
```

### Logs

```bash
tail -f /var/log/idcam/idcam.log
journalctl -u idcam -f
```

### Invalid configuration

Application exits with code 1. Fix `/etc/idcam/config.yaml` and restart.

### Disk full

```bash
df -h
du -sh /var/log/idcam /var/lib/idcam/tmp
```

Temp files auto-cleaned after 1 hour. Logs rotate at 10MB × 5 files.

## Repeated captures of same document

After success, system waits in `WAITING_FOR_REMOVAL` until scene returns to baseline. If triggered repeatedly, increase `detection.removal_frames_required` or `motion_threshold`.

## Service won't start

```bash
systemctl status idcam
/opt/idcam/venv/bin/idcam --config /etc/idcam/config.yaml  # run manually
```

Check StartLimit: `systemctl reset-failed idcam`
