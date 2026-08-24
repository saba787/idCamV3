"""User-visible status messages."""

from idcam.app.state_machine import AppState

USER_MESSAGES: dict[AppState, str] = {
    AppState.BOOT: "Starting...",
    AppState.INITIALIZING: "Initializing...",
    AppState.CAMERA_INITIALIZING: "Camera initializing...",
    AppState.CAMERA_READY: "Preparing...",
    AppState.IDLE: "Ready — Place document",
    AppState.DOCUMENT_DETECTED: "Document detected",
    AppState.WAITING_FOR_STABILITY: "Hold still...",
    AppState.FOCUSING: "Focusing...",
    AppState.FOCUS_VERIFICATION: "Verifying focus...",
    AppState.CAPTURING: "Capturing...",
    AppState.IMAGE_VALIDATION: "Checking image...",
    AppState.UPLOADING: "Sending...",
    AppState.WAITING_FOR_SERVER: "Processing...",
    AppState.PROCESSING_RESPONSE: "Processing...",
    AppState.SUCCESS: "Done",
    AppState.WAITING_FOR_REMOVAL: "Please remove document",
    AppState.RECOVERY: "Recovering...",
    AppState.ERROR: "Please contact support",
    AppState.SHUTDOWN: "Shutting down...",
}

FAILURE_MESSAGES = {
    "focus_failed": "Could not focus — try again",
    "capture_failed": "Capture failed — try again",
    "upload_failed": "Connection problem — try again",
    "server_rejected": "Could not process — contact support",
    "image_invalid": "Image not usable — try again",
    "camera_unavailable": "Camera problem — recovering...",
    "network_unavailable": "No network connection",
}
