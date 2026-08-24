#!/bin/bash
# idCam installation script for Raspberry Pi 5 + Arducam 64MP Hawkeye
# Run as root on a fresh Raspberry Pi OS Bookworm/Trixie system.

set -euo pipefail

IDC_VERSION="1.0.0"
INSTALL_DIR="/opt/idcam"
CONFIG_DIR="/etc/idcam"
VENV_DIR="${INSTALL_DIR}/venv"
SERVICE_USER="idcam"

echo "=== idCam ${IDC_VERSION} Installer ==="

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo)" >&2
    exit 1
fi

# --- 1. System packages ---
echo "--- Installing system packages ---"
apt-get update
apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    libatlas-base-dev \
    libcamera-dev libcamera-apps \
    v4l-utils \
    wget curl

# --- 2. Arducam driver (if not already installed) ---
if ! grep -q "arducam-64mp" /boot/firmware/config.txt 2>/dev/null; then
    echo "--- Installing Arducam 64MP driver ---"
  echo "Ensure dtoverlay=arducam-64mp is in /boot/firmware/config.txt"
    wget -O /tmp/install_pivariety_pkgs.sh \
        https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
    chmod +x /tmp/install_pivariety_pkgs.sh
    /tmp/install_pivariety_pkgs.sh -p libcamera_dev
    /tmp/install_pivariety_pkgs.sh -p libcamera_apps
    /tmp/install_pivariety_pkgs.sh -p 64mp_pi_hawk_eye_kernel_driver
    echo ""
    echo "IMPORTANT: Add 'dtoverlay=arducam-64mp' to /boot/firmware/config.txt and reboot"
fi

# --- 3. Picamera2 ---
echo "--- Installing Picamera2 ---"
apt-get install -y python3-kms++ python3-libcamera python3-picamera2 || true

# --- 4. Service user ---
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir /var/lib/idcam "$SERVICE_USER"
fi

# --- 5. Application install ---
echo "--- Installing idCam application ---"
mkdir -p "$INSTALL_DIR" "$CONFIG_DIR" /var/lib/idcam/tmp /var/log/idcam /run/idcam

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cp -r "${PROJECT_DIR}/src" "${INSTALL_DIR}/"
cp "${PROJECT_DIR}/pyproject.toml" "${INSTALL_DIR}/"

if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
    cp "${PROJECT_DIR}/config.example.yaml" "${CONFIG_DIR}/config.yaml"
    chmod 600 "${CONFIG_DIR}/config.yaml"
    echo "Created ${CONFIG_DIR}/config.yaml — EDIT BEFORE STARTING"
fi

# --- 6. Python venv ---
python3 -m venv "$VENV_DIR"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -e "${INSTALL_DIR}[pi]"

# --- 7. Permissions ---
chown -R "${SERVICE_USER}:${SERVICE_USER}" /var/lib/idcam /var/log/idcam /run/idcam
chown -R root:"${SERVICE_USER}" "$CONFIG_DIR"
chmod 750 "$CONFIG_DIR"

# --- 8. systemd ---
cp "${SCRIPT_DIR}/idcam.service" /etc/systemd/system/idcam.service
systemctl daemon-reload
systemctl enable idcam.service

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit ${CONFIG_DIR}/config.yaml (device_id, server URL, auth)"
echo "  2. Set auth token: echo 'IDCAM_AUTH_TOKEN=...' >> /etc/idcam/environment"
echo "  3. Verify camera: rpicam-hello --list-cameras"
echo "  4. Start service: systemctl start idcam"
echo "  5. Check status: systemctl status idcam"
echo "  6. View logs: journalctl -u idcam -f"
echo "  7. UI status: cat /run/idcam/status.json"
