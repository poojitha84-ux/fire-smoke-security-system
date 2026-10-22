"""
Fire and Smoke Real-Time Security Monitoring System
Flask Web Application and REST API.

Key features:
- Live camera stream with detection overlays (/video_feed)
- Real-time telemetry and state monitoring (/api/status)
- Security incident lifecycle management (acknowledge/resolve)
- Alarm silence and audio control
- Configuration management
"""

import logging
import os
from pathlib import Path
from typing import Optional
from flask import Flask, Response, jsonify, render_template, request

from config import AppConfig
from detector.model_loader import ModelLoader, ModelLoadError
from detector.screen_context import ScreenContextFilter
from detector.temporal_filter import TemporalFilter
from detector.alert_manager import AlertManager, AlertEvent
from detector.inference import DetectionPipeline
from services.incident_service import IncidentService, IncidentReport, NotificationDeliveryStatus
from services.notification_service import NotificationService, NotificationResult
from services.camera_service import CameraService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("fire_smoke_monitor")

def create_app(config: Optional[AppConfig] = None) -> Flask:
    """Application factory for Flask dashboard."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    if config is None:
        config = AppConfig.from_env()

    app.config["MONITOR_CONFIG"] = config
    app.config["SECRET_KEY"] = config.SECRET_KEY

    # 1. Initialize incident and notification services
    log_file = config.BASE_DIR / "logs" / "incidents.jsonl"
    incident_service = IncidentService(log_file=log_file)
    notification_service = NotificationService(config=config)

    # 2. Callback when hazard is confirmed by AlertManager
    def handle_confirmed_incident(alert_event: AlertEvent):
        """Called automatically when an environmental hazard is temporally confirmed or sustained uncertain."""
        confirmation_status = getattr(alert_event, "confirmation_status", "CONFIRMED")
        incident = incident_service.create_incident(
            camera_id=alert_event.camera_id,
            camera_location=alert_event.camera_location,
            hazard_type=alert_event.hazard_type,
            confidence=alert_event.max_confidence,
            severity=alert_event.severity.value,
            confirmation_status=confirmation_status,
            metadata=alert_event.metadata,
            incident_id=alert_event.event_id,
        )

        # Dispatch external notification asynchronously only if confirmed
        if confirmation_status == "CONFIRMED":
            def on_notify_complete(inc: IncidentReport, result: NotificationResult):
                incident_service.update_notification_status(
                    incident_id=inc.incident_id,
                    status=result.status,
                    details=result.message,
                )

            notification_service.dispatch_async(incident, on_complete=on_notify_complete)

    # 3. Initialize detector components
    models_dir = config.BASE_DIR / "models"
    model_loader = ModelLoader(models_dir=models_dir)

    # Attempt loading models (logs warning if offline; tests can inject mock models)
    try:
        model_loader.load_fire_smoke_model(config.FIRE_SMOKE_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Fire/Smoke model not immediately loaded at startup: {e}")

    try:
        model_loader.load_screen_detector_model(config.SCREEN_DETECTOR_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Screen detector model not immediately loaded at startup: {e}")

    screen_filter = ScreenContextFilter(
        reject_threshold=config.SCREEN_OVERLAP_REJECT_THRESHOLD,
        uncertain_threshold=config.SCREEN_OVERLAP_UNCERTAIN_THRESHOLD,
        enabled=config.ENABLE_SCREEN_REJECTION,
        min_screen_conf=config.SCREEN_DETECTION_CONFIDENCE,
        tilted_min_conf=config.SCREEN_TILTED_MIN_CONFIDENCE,
        protrusion_margin_ratio=config.PROTRUSION_MARGIN_RATIO,
        fire_behind_screen_area_ratio=config.FIRE_BEHIND_SCREEN_AREA_RATIO,
        reject_smoke_on_screens=config.REJECT_SMOKE_ON_SCREENS,
        enable_face_smoke_filter=config.ENABLE_FACE_SMOKE_FILTER,
        min_person_conf=config.PERSON_DETECTION_CONFIDENCE,
        face_overlap_reject_threshold=config.FACE_OVERLAP_REJECT_THRESHOLD,
        face_overlap_uncertain_threshold=config.FACE_OVERLAP_UNCERTAIN_THRESHOLD,
        face_smoke_calibrated_threshold=config.FACE_SMOKE_CALIBRATED_THRESHOLD,
        face_uncertain_conf_floor=config.FACE_UNCERTAIN_CONF_FLOOR,
        smoke_plume_dispersion_threshold=config.SMOKE_PLUME_DISPERSION_THRESHOLD,
        enable_white_object_filter=config.ENABLE_WHITE_OBJECT_FILTER,
        enable_clothing_smoke_filter=config.ENABLE_CLOTHING_SMOKE_FILTER,
        white_object_reject_confidence=config.WHITE_OBJECT_REJECT_CONFIDENCE,
        white_object_uncertain_confidence=config.WHITE_OBJECT_UNCERTAIN_CONFIDENCE,
        glare_saturation_threshold=config.GLARE_SATURATION_THRESHOLD,
        uniform_wall_std_threshold=config.UNIFORM_WALL_STD_THRESHOLD,
    )

    temporal_filter = TemporalFilter(
        window_size=config.TEMPORAL_WINDOW_SIZE,
        confirm_count=config.TEMPORAL_CONFIRM_COUNT,
        clear_consecutive_frames=config.CLEAR_CONSECUTIVE_FRAMES,
    )

    alert_manager = AlertManager(
        camera_id=config.CAMERA_ID,
        camera_location=config.CAMERA_LOCATION,
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
        enable_audio=config.ENABLE_AUDIO_ALARM,
        audio_cooldown_seconds=config.AUDIO_COOLDOWN_SECONDS,
        on_confirmed_incident=handle_confirmed_incident,
    )

    pipeline = DetectionPipeline(
        model_loader=model_loader,
        screen_filter=screen_filter,
        temporal_filter=temporal_filter,
        alert_manager=alert_manager,
        fire_threshold=config.FIRE_CONFIDENCE_THRESHOLD,
        smoke_threshold=config.SMOKE_CONFIDENCE_THRESHOLD,
        enable_fire_detection=config.ENABLE_FIRE_DETECTION,
        enable_smoke_detection=config.ENABLE_SMOKE_DETECTION,
    )

    camera_service = CameraService(config=config, pipeline=pipeline)

    # Store services on app context
    app.camera_service = camera_service  # type: ignore
    app.incident_service = incident_service  # type: ignore
    app.notification_service = notification_service  # type: ignore
    app.alert_manager = alert_manager  # type: ignore
    app.pipeline = pipeline  # type: ignore

    # -------------------------------------------------------------
    # ROUTES
    # -------------------------------------------------------------

    @app.route("/")
    def index():
        """Main Security Monitoring Dashboard."""
        return render_template(
            "index.html",
            config=config,
            camera_status=camera_service.status.value,
        )

    @app.route("/video_feed")
    def video_feed():
        """Live MJPEG video stream with visual detection HUD."""
        return Response(
            camera_service.generate_mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        """Returns real-time camera telemetry and hazard states."""
        status = camera_service.get_status()
        active_incidents = [i.to_dict() for i in incident_service.get_active_incidents()]
        status["active_incidents_count"] = len(active_incidents)
        status["audio_silenced"] = alert_manager.audio_silenced
        status["audio_enabled"] = alert_manager.enable_audio
        return jsonify(status)

    @app.route("/api/monitoring/start", methods=["POST"])
    def start_monitoring():
        """Start live camera monitoring."""
        data = request.get_json(silent=True) or {}
        source = data.get("source")
        success = camera_service.start_monitoring(source=source)
        return jsonify({
            "success": success,
            "status": camera_service.status.value,
            "message": camera_service.status_message,
        })

    @app.route("/api/monitoring/stop", methods=["POST"])
    def stop_monitoring():
        """Stop camera monitoring and release hardware."""
        camera_service.stop_monitoring()
        return jsonify({
            "success": True,
            "status": camera_service.status.value,
            "message": "Monitoring stopped",
        })

    @app.route("/api/incidents", methods=["GET"])
    def get_incidents():
        """Fetch list of recorded security incidents."""
        limit = request.args.get("limit", default=50, type=int)
        incidents = [i.to_dict() for i in incident_service.get_all_incidents(limit=limit)]
        return jsonify({"incidents": incidents, "total": len(incidents)})

    @app.route("/api/incidents/<incident_id>/acknowledge", methods=["POST"])
    def acknowledge_incident(incident_id):
        """Security staff acknowledges an incident."""
        data = request.get_json(silent=True) or {}
        operator = data.get("operator_name", "Security Officer")
        notes = data.get("notes", "")
        updated = incident_service.acknowledge_incident(incident_id, operator_name=operator, notes=notes)
        if updated:
            # Silence active audio alarm if the acknowledged incident is the active one
            if alert_manager.last_incident_id == incident_id:
                alert_manager.silence_audio()
            return jsonify({"success": True, "incident": updated.to_dict()})
        return jsonify({"success": False, "error": "Incident not found"}), 404

    @app.route("/api/incidents/<incident_id>/resolve", methods=["POST"])
    def resolve_incident(incident_id):
        """Security staff resolves an incident."""
        data = request.get_json(silent=True) or {}
        operator = data.get("operator_name", "Security Officer")
        notes = data.get("notes", "Hazard resolved and area verified secure.")
        updated = incident_service.resolve_incident(incident_id, operator_name=operator, notes=notes)
        if updated:
            # Silence audio alarm if the resolved incident was the active one
            if alert_manager.last_incident_id == incident_id:
                alert_manager.silence_audio()
            return jsonify({"success": True, "incident": updated.to_dict()})
        return jsonify({"success": False, "error": "Incident not found"}), 404

    @app.route("/api/audio/silence", methods=["POST"])
    def silence_audio():
        """Operator silences currently sounding audio alarm."""
        alert_manager.silence_audio()
        return jsonify({"success": True, "message": "Audio alarm silenced."})

    @app.route("/api/audio/toggle", methods=["POST"])
    def toggle_audio():
        """Enable or disable audio alert sounding."""
        data = request.get_json(silent=True) or {}
        enabled = data.get("enabled", not alert_manager.enable_audio)
        alert_manager.enable_audio_alarm(enabled)
        return jsonify({"success": True, "audio_enabled": alert_manager.enable_audio})

    @app.route("/api/smoke/toggle", methods=["POST", "GET"])
    @app.route("/api/detection/smoke", methods=["POST", "GET"])
    @app.route("/api/smoke/start", methods=["POST", "GET"])
    def smoke_detection_handler():
        """Handle smoke detection endpoint safely without starting smoke inference."""
        return jsonify({
            "success": True,
            "status": "ready",
            "enable_smoke_detection": False,
        }), 200

    @app.route("/api/config", methods=["GET", "POST"])
    def manage_config():
        """View or update camera identification and thresholds."""
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            if "camera_id" in data:
                config.CAMERA_ID = str(data["camera_id"])
                camera_service.config.CAMERA_ID = config.CAMERA_ID
                alert_manager.camera_id = config.CAMERA_ID
            if "camera_location" in data:
                config.CAMERA_LOCATION = str(data["camera_location"])
                camera_service.config.CAMERA_LOCATION = config.CAMERA_LOCATION
                alert_manager.camera_location = config.CAMERA_LOCATION
            if "fire_threshold" in data:
                pipeline.fire_threshold = float(data["fire_threshold"])
            if "smoke_threshold" in data:
                pipeline.smoke_threshold = float(data["smoke_threshold"])
            return jsonify({"success": True, "message": "Settings updated"})

        return jsonify({
            "camera_id": config.CAMERA_ID,
            "camera_location": config.CAMERA_LOCATION,
            "camera_source": str(camera_service.source),
            "fire_threshold": pipeline.fire_threshold,
            "smoke_threshold": pipeline.smoke_threshold,
            "screen_rejection_enabled": screen_filter.enabled,
            "screen_reject_threshold": screen_filter.reject_threshold,
            "screen_uncertain_threshold": screen_filter.uncertain_threshold,
            "notifications_enabled": config.ENABLE_NOTIFICATIONS,
            "notification_type": config.NOTIFICATION_TYPE,
            "notifications_configured": notification_service.is_configured(),
        })

    return app


if __name__ == "__main__":
    app_instance = create_app()
    port = int(os.getenv("PORT", "5000"))
    logger.info(f"Starting Fire & Smoke Security Dashboard on http://localhost:{port}")
    app_instance.run(host="0.0.0.0", port=port, debug=False, threaded=True)
