#!/usr/bin/env bash
# idCam production installer for Raspberry Pi OS (Bookworm/Trixie).
# Creates the idcam user, directories, venv, config templates, and systemd unit.
# Does NOT create /run/idcam — systemd RuntimeDirectory=idcam owns that path.
#
# Usage:
#   sudo ./scripts/install.sh
#   sudo ./scripts/install.sh --start
#   sudo IDC_DEVICE_ID=pi-01 IDC_SERVER_URL=https://api.example.net \
#        IDC_AUTH_TOKEN=secret ./scripts/install.sh --start
#   ./scripts/install.sh --staging /tmp/idcam-staging --skip-python

set -euo pipefail

IDC_VERSION="1.0.0"
SERVICE_USER="idcam"
SERVICE_GROUP="idcam"

STAGING_ROOT=""
START_SERVICE=0
SKIP_APT=0
SKIP_CAMERA=0
SKIP_PYTHON=0
SKIP_SYSTEMD=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_DIR="${PROJECT_DIR}/deploy"

usage() {
    cat <<'EOF'
idCam installer

Usage: sudo ./scripts/install.sh [options]

Options:
  --start           Start/restart the service if configuration validates
  --skip-apt        Skip apt-get
  --skip-camera     Skip Arducam driver installation
  --skip-python     Skip venv/pip (layout-only / tests)
  --skip-systemd    Skip enabling systemd (unit file is still installed)
  --staging DIR     Install under DIR (no root, apt, useradd, or systemctl)
  -h, --help        Show this help

Optional environment variables (applied only when creating new files, and
never overwriting an existing non-empty IDCAM_AUTH_TOKEN):
  IDC_DEVICE_ID     device_id / IDCAM_DEVICE_ID
  IDC_SERVER_URL    server.url / IDCAM_SERVER_URL
  IDC_AUTH_TOKEN    IDCAM_AUTH_TOKEN in /etc/idcam/environment
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start) START_SERVICE=1; shift ;;
        --skip-apt) SKIP_APT=1; shift ;;
        --skip-camera) SKIP_CAMERA=1; shift ;;
        --skip-python) SKIP_PYTHON=1; shift ;;
        --skip-systemd) SKIP_SYSTEMD=1; shift ;;
        --staging)
            [[ -n "${2:-}" ]] || { echo "ERROR: --staging requires a directory" >&2; exit 2; }
            STAGING_ROOT="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$STAGING_ROOT" ]]; then
    mkdir -p "$STAGING_ROOT"
    STAGING_ROOT="$(cd "$STAGING_ROOT" && pwd)"
    SKIP_APT=1
    SKIP_CAMERA=1
    SKIP_SYSTEMD=1
fi

prefix_path() {
    if [[ -n "$STAGING_ROOT" ]]; then
        printf '%s\n' "${STAGING_ROOT}$1"
    else
        printf '%s\n' "$1"
    fi
}

INSTALL_DIR="$(prefix_path /opt/idcam)"
VENV_DIR="${INSTALL_DIR}/venv"
CONFIG_DIR="$(prefix_path /etc/idcam)"
STATE_DIR="$(prefix_path /var/lib/idcam)"
TEMP_DIR="${STATE_DIR}/tmp"
LOG_DIR="$(prefix_path /var/log/idcam)"
SYSTEMD_DIR="$(prefix_path /etc/systemd/system)"
CONFIG_FILE="${CONFIG_DIR}/config.yaml"
ENV_FILE="${CONFIG_DIR}/environment"
UNIT_FILE="${SYSTEMD_DIR}/idcam.service"
DROPIN_DIR="${SYSTEMD_DIR}/idcam.service.d"

log() { echo "=== $* ==="; }
warn() { echo "WARNING: $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

ensure_dir() {
    mkdir -p "$2"
    chmod "$1" "$2" 2>/dev/null || true
}

copy_file() {
    cp "$2" "$3"
    chmod "$1" "$3" 2>/dev/null || true
}

require_root() {
    if [[ -n "$STAGING_ROOT" ]]; then
        return 0
    fi
    if [[ "${EUID}" -ne 0 ]]; then
        die "Run as root: sudo ./scripts/install.sh"
    fi
}

python_ok() {
    python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

check_prerequisites() {
    log "Checking prerequisites"
    command -v python3 >/dev/null || die "python3 is required"
    python_ok || die "Python 3.11 or newer is required"
    if [[ -z "$STAGING_ROOT" && "$SKIP_SYSTEMD" -eq 0 ]]; then
        command -v systemctl >/dev/null || die "systemd (systemctl) is required"
    fi
    [[ -f "${PROJECT_DIR}/pyproject.toml" ]] || die "pyproject.toml not found in ${PROJECT_DIR}"
    [[ -f "${DEPLOY_DIR}/idcam.service" ]] || die "Missing ${DEPLOY_DIR}/idcam.service"
    [[ -f "${DEPLOY_DIR}/config.yaml.template" ]] || die "Missing ${DEPLOY_DIR}/config.yaml.template"
    [[ -f "${DEPLOY_DIR}/environment" ]] || die "Missing ${DEPLOY_DIR}/environment"
}

install_apt_packages() {
    if [[ "$SKIP_APT" -eq 1 ]]; then
        return 0
    fi
    log "Installing system packages"
    apt-get update
    apt-get install -y \
        python3 python3-pip python3-venv python3-dev \
        python3-kms++ python3-libcamera python3-picamera2 \
        libatlas-base-dev \
        libcamera-dev libcamera-apps \
        v4l-utils \
        wget curl
}

install_arducam() {
    if [[ "$SKIP_CAMERA" -eq 1 ]]; then
        return 0
    fi
    local bootcfg=""
    if [[ -f /boot/firmware/config.txt ]]; then
        bootcfg=/boot/firmware/config.txt
    elif [[ -f /boot/config.txt ]]; then
        bootcfg=/boot/config.txt
    fi
    if [[ -n "$bootcfg" ]] && grep -q "arducam-64mp" "$bootcfg"; then
        echo "Arducam overlay already present in ${bootcfg}"
        return 0
    fi
    log "Installing Arducam 64MP packages (best-effort)"
    echo "Ensure dtoverlay=arducam-64mp is in /boot/firmware/config.txt, then reboot"
    if wget -O /tmp/install_pivariety_pkgs.sh \
        https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh; then
        chmod +x /tmp/install_pivariety_pkgs.sh
        /tmp/install_pivariety_pkgs.sh -p libcamera_dev || true
        /tmp/install_pivariety_pkgs.sh -p libcamera_apps || true
        /tmp/install_pivariety_pkgs.sh -p 64mp_pi_hawk_eye_kernel_driver || true
    else
        warn "Could not download Arducam installer; install the driver manually"
    fi
}

create_user() {
    if [[ -n "$STAGING_ROOT" ]]; then
        return 0
    fi
    log "Ensuring ${SERVICE_USER} user and group"
    if ! getent group "$SERVICE_GROUP" >/dev/null; then
        groupadd --system "$SERVICE_GROUP"
    fi
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
        useradd --system --gid "$SERVICE_GROUP" \
            --home-dir /var/lib/idcam --shell /usr/sbin/nologin \
            --no-create-home "$SERVICE_USER"
    fi
    local extra=""
    for g in video render plugdev; do
        if getent group "$g" >/dev/null; then
            usermod -aG "$g" "$SERVICE_USER"
            extra="${extra} ${g}"
        fi
    done
    if [[ -n "$extra" ]]; then
        echo "Added ${SERVICE_USER} to groups:${extra}"
    fi
}

create_directories() {
    log "Creating directories (not /run/idcam)"
    ensure_dir 0755 "$INSTALL_DIR"
    ensure_dir 0750 "$CONFIG_DIR"
    ensure_dir 0750 "$STATE_DIR"
    ensure_dir 0750 "$TEMP_DIR"
    ensure_dir 0750 "$LOG_DIR"
    if [[ -z "$STAGING_ROOT" ]]; then
        chown root:root "$INSTALL_DIR"
        chown root:"$SERVICE_GROUP" "$CONFIG_DIR"
        chown "$SERVICE_USER":"$SERVICE_GROUP" "$STATE_DIR" "$TEMP_DIR" "$LOG_DIR"
    fi
}

quote_shell() {
    python3 -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$1"
}

append_env() {
    local key="$1"
    local value="$2"
    if grep -qE "^${key}=.+" "$ENV_FILE"; then
        echo "Keeping existing ${key}"
        return 0
    fi
    if [[ -f "$ENV_FILE" ]]; then
        grep -vE "^${key}=" "$ENV_FILE" > "${ENV_FILE}.tmp" || true
        mv "${ENV_FILE}.tmp" "$ENV_FILE"
    fi
    printf '%s=%s\n' "$key" "$(quote_shell "$value")" >> "$ENV_FILE"
    chmod 0640 "$ENV_FILE"
    echo "Set ${key} in ${ENV_FILE}"
}

install_config() {
    log "Installing configuration (existing files are never overwritten)"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        copy_file 0640 "${DEPLOY_DIR}/config.yaml.template" "$CONFIG_FILE"
        if [[ -n "${IDC_DEVICE_ID:-}" ]]; then
            python3 - "$CONFIG_FILE" "${IDC_DEVICE_ID}" <<'PY'
import sys
from pathlib import Path
path, device_id = Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8").replace('device_id: "CHANGE_ME"', f"device_id: {device_id!r}")
path.write_text(text, encoding="utf-8")
PY
        fi
        if [[ -n "${IDC_SERVER_URL:-}" ]]; then
            python3 - "$CONFIG_FILE" "${IDC_SERVER_URL}" <<'PY'
import sys
from pathlib import Path
path, url = Path(sys.argv[1]), sys.argv[2]
text = path.read_text(encoding="utf-8").replace("url: https://unconfigured.invalid", f"url: {url}")
path.write_text(text, encoding="utf-8")
PY
        fi
        echo "Created ${CONFIG_FILE}"
    else
        echo "Keeping existing ${CONFIG_FILE}"
    fi
    copy_file 0640 "${DEPLOY_DIR}/config.yaml.template" "${CONFIG_DIR}/config.yaml.template"

    if [[ ! -f "$ENV_FILE" ]]; then
        copy_file 0640 "${DEPLOY_DIR}/environment" "$ENV_FILE"
    else
        echo "Keeping existing ${ENV_FILE}"
        chmod 0640 "$ENV_FILE"
    fi

    if [[ -n "${IDC_DEVICE_ID:-}" ]]; then
        append_env IDCAM_DEVICE_ID "$IDC_DEVICE_ID"
    fi
    if [[ -n "${IDC_SERVER_URL:-}" ]]; then
        append_env IDCAM_SERVER_URL "$IDC_SERVER_URL"
    fi
    if [[ -n "${IDC_AUTH_TOKEN:-}" ]]; then
        append_env IDCAM_AUTH_TOKEN "$IDC_AUTH_TOKEN"
    fi

    if [[ -z "$STAGING_ROOT" ]]; then
        chown root:"$SERVICE_GROUP" "$CONFIG_FILE" "${CONFIG_DIR}/config.yaml.template" "$ENV_FILE"
        chmod 0640 "$CONFIG_FILE" "$ENV_FILE" "${CONFIG_DIR}/config.yaml.template"
        chmod 0750 "$CONFIG_DIR"
    fi
}

install_application() {
    if [[ "$SKIP_PYTHON" -eq 1 ]]; then
        warn "Skipping Python install (--skip-python)"
        return 0
    fi
    log "Installing application into ${VENV_DIR}"
    python3 -m venv --system-site-packages "$VENV_DIR"
    "${VENV_DIR}/bin/pip" install --upgrade pip wheel
    if "${VENV_DIR}/bin/python" -c "import picamera2" 2>/dev/null; then
        "${VENV_DIR}/bin/pip" install "${PROJECT_DIR}"
    else
        "${VENV_DIR}/bin/pip" install "${PROJECT_DIR}[pi]" || "${VENV_DIR}/bin/pip" install "${PROJECT_DIR}"
    fi
    printf '%s\n' "$IDC_VERSION" > "${INSTALL_DIR}/VERSION"
    if [[ -z "$STAGING_ROOT" ]]; then
        chown -R root:root "$INSTALL_DIR"
        chmod 0755 "$INSTALL_DIR"
    fi
}

write_systemd_dropin() {
    local groups=()
    if [[ -n "$STAGING_ROOT" ]]; then
        groups=(video render)
    else
        for g in video render; do
            if getent group "$g" >/dev/null; then
                groups+=("$g")
            fi
        done
    fi
    if [[ ${#groups[@]} -eq 0 ]]; then
        return 0
    fi
    ensure_dir 0755 "$DROPIN_DIR"
    cat > "${DROPIN_DIR}/hardware.conf" <<EOF
[Service]
SupplementaryGroups=${groups[*]}
EOF
    chmod 0644 "${DROPIN_DIR}/hardware.conf"
}

install_systemd_unit() {
    log "Installing systemd unit"
    ensure_dir 0755 "$SYSTEMD_DIR"
    copy_file 0644 "${DEPLOY_DIR}/idcam.service" "$UNIT_FILE"
    write_systemd_dropin
    if [[ "$SKIP_SYSTEMD" -eq 1 ]]; then
        echo "Wrote ${UNIT_FILE} (not enabled)"
        return 0
    fi
    systemctl daemon-reload
    systemctl enable idcam.service
}

config_is_valid() {
    if [[ "$SKIP_PYTHON" -eq 1 || ! -x "${VENV_DIR}/bin/idcam-admin" ]]; then
        return 1
    fi
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        set +a
    fi
    export IDCAM_ENV=production
    export IDCAM_CONFIG="$CONFIG_FILE"
    "${VENV_DIR}/bin/idcam-admin" --config "$CONFIG_FILE" validate >/dev/null 2>&1
}

maybe_start() {
    if [[ "$SKIP_SYSTEMD" -eq 1 ]]; then
        return 0
    fi
    local was_active=0
    if systemctl is-active --quiet idcam.service; then
        was_active=1
    fi
    if [[ "$START_SERVICE" -eq 1 || "$was_active" -eq 1 ]]; then
        if config_is_valid; then
            log "Restarting idcam.service"
            systemctl restart idcam.service
        else
            warn "Configuration is not valid yet; not starting idcam.service"
            warn "Edit ${CONFIG_FILE} and ${ENV_FILE}, then: sudo systemctl start idcam"
        fi
    elif config_is_valid; then
        echo "Configuration is valid. Start with: sudo systemctl start idcam"
    else
        echo "Service is enabled but not started (configure device_id, server.url, and token first)."
    fi
}

print_next_steps() {
    cat <<EOF

=== idCam ${IDC_VERSION} installation complete ===

Service user/group: ${SERVICE_USER}:${SERVICE_GROUP}
Application:        ${VENV_DIR}
Config:             ${CONFIG_FILE}   (root:${SERVICE_GROUP} 0640; never overwritten on upgrade)
Environment:        ${ENV_FILE}      (root:${SERVICE_GROUP} 0640)
State/temp:         ${STATE_DIR} and ${TEMP_DIR}  (${SERVICE_USER}:${SERVICE_GROUP} 0750)
Logs:               ${LOG_DIR}       (${SERVICE_USER}:${SERVICE_GROUP} 0750)
Runtime:            /run/idcam via systemd RuntimeDirectory=idcam (mode 0755)

Configure, then start:
  sudo nano ${CONFIG_FILE}          # device_id, server.url
  sudo nano ${ENV_FILE}             # IDCAM_AUTH_TOKEN=...
  sudo systemctl enable --now idcam
  sudo systemctl status idcam
  journalctl -u idcam -f
  cat /run/idcam/status.json

Upgrade (preserves config, credentials, captured data, and logs):
  cd ${PROJECT_DIR} && git pull && sudo ./scripts/install.sh
  sudo systemctl restart idcam     # if it was not already running

Uninstall (keeps config/data/logs unless you pass --purge):
  sudo ./scripts/uninstall.sh
EOF
}

require_root
check_prerequisites
install_apt_packages
install_arducam
create_user
create_directories
install_application
install_config
install_systemd_unit
maybe_start
print_next_steps
