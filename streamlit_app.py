"""
AI-Based Real-Time Fire and Smoke Security Monitoring System
Streamlit Cloud Dashboard Application.

Designed for free cloud hosting (Streamlit Community Cloud).
Features:
- Browser-side webcam capture via WebRTC (no server-side /dev/video0 needed)
- Fallback interactive camera input (st.camera_input) for maximum browser compatibility
- Deep learning inference using fine-tuned YOLOv8n models
- False-positive defense against digital screens/phones, human faces, and white surfaces
- Real-time telemetry, HUD rendering, and incident event logging
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("streamlit_fire_smoke")

# Import existing project architecture
from config import AppConfig
from detector.model_loader import ModelLoader
from detector.screen_context import ScreenContextFilter
from detector.temporal_filter import TemporalFilter, HazardState
from detector.alert_manager import AlertManager, AlertState, AlertEvent
from detector.inference import DetectionPipeline, FrameResult

# Try importing streamlit-webrtc and av gracefully
WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase,
    )
    import av
    WEBRTC_AVAILABLE = True
except Exception as e:
    logger.warning(f"streamlit-webrtc / av not available or blocked in current runtime: {e}")
    WEBRTC_AVAILABLE = False


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Fire & Smoke Security Monitor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Security Monitoring Dashboard
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: #151e32;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .status-normal {
        color: #10b981;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .status-incident {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .disclaimer-banner {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing AI detection pipeline and loading YOLO weights...")
def get_shared_pipeline():
    """
    Load models and instantiate the detection pipeline once per application lifecycle.
    Cached as a singleton resource across Streamlit reruns.
    """
    config = AppConfig.from_env()

    models_dir = config.BASE_DIR / "models"
    model_loader = ModelLoader(models_dir=models_dir)

    try:
        model_loader.load_fire_smoke_model(config.FIRE_SMOKE_MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load fire/smoke model: {e}")

    try:
        model_loader.load_screen_detector_model(config.SCREEN_DETECTOR_MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load screen detector model: {e}")

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
        enable_audio=False,  # Browser alerts handled via UI banners
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

    return pipeline, config


# Initialize shared pipeline and configuration
pipeline, config = get_shared_pipeline()

# Session State for incident log and statistics
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# -------------------------------------------------------------
# SIDEBAR CONTROLS & THRESHOLDS
# -------------------------------------------------------------
st.sidebar.title("🛡️ Security Controls")
st.sidebar.markdown(f"**Camera:** `{config.CAMERA_ID}`")
st.sidebar.markdown(f"**Location:** `{config.CAMERA_LOCATION}`")
st.sidebar.divider()

st.sidebar.subheader("Detection Sensitivities")
fire_thresh = st.sidebar.slider(
    "Fire Confidence Threshold",
    min_value=0.20,
    max_value=0.90,
    value=float(config.FIRE_CONFIDENCE_THRESHOLD),
    step=0.05,
    help="Minimum model confidence required to register a fire detection candidate.",
)
pipeline.fire_threshold = fire_thresh

enable_fire = st.sidebar.checkbox("Enable Fire Detection", value=config.ENABLE_FIRE_DETECTION)
pipeline.enable_fire_detection = enable_fire

enable_smoke = st.sidebar.checkbox(
    "Enable Smoke Detection",
    value=config.ENABLE_SMOKE_DETECTION,
    help="Disabled by default to prevent false positives on white objects, clothing, and bright walls.",
)
pipeline.enable_smoke_detection = enable_smoke

if enable_smoke:
    smoke_thresh = st.sidebar.slider(
        "Smoke Confidence Threshold",
        min_value=0.20,
        max_value=0.90,
        value=float(config.SMOKE_CONFIDENCE_THRESHOLD),
        step=0.05,
    )
    pipeline.smoke_threshold = smoke_thresh

st.sidebar.divider()
st.sidebar.subheader("False-Positive Defenses")
screen_defense = st.sidebar.checkbox(
    "Screen & Phone Defense",
    value=config.ENABLE_SCREEN_REJECTION,
    help="Filters fire displayed on smartphones, tablets, TVs, and computer monitors.",
)
pipeline.screen_filter.enabled = screen_defense

face_defense = st.sidebar.checkbox(
    "Face & Person Defense",
    value=config.ENABLE_FACE_SMOKE_FILTER,
    help="Prevents human faces and heads from triggering false smoke detections.",
)
pipeline.screen_filter.enable_face_smoke_filter = face_defense

white_defense = st.sidebar.checkbox(
    "White-Object & Glare Defense",
    value=config.ENABLE_WHITE_OBJECT_FILTER,
    help="Filters white walls, paper sheets, glare, and low-contrast indoor surfaces.",
)
pipeline.screen_filter.enable_white_object_filter = white_defense

st.sidebar.divider()
st.sidebar.info(
    "💡 **Cloud Deployment Note:**\n"
    "Video is captured directly in your browser and processed on the cloud server. "
    "Camera permissions must be allowed in your browser over HTTPS."
)


# -------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -------------------------------------------------------------
st.title("🔥 AI Real-Time Fire & Smoke Surveillance")
st.markdown(
    """
    <div class="disclaimer-banner">
    ⚠️ <b>AI Surveillance Prototype:</b> Designed for security monitoring and false-alarm defense research. 
    Not certified under UL 268 / EN 54 life-safety fire protection standards.
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Telemetry Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Hazard Threat Level</div>
            <div id="status-level" class="status-normal">NORMAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Fire Detection</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                {"ACTIVE" if pipeline.enable_fire_detection else "OFF"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">Smoke Detection</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">
                {"ACTIVE" if pipeline.enable_smoke_detection else "INACTIVE"}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase;">False Alarms Blocked</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #3b82f6;">
                ACTIVE DEFENSE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# -------------------------------------------------------------
# CAMERA FEED & WEBRTC INFERENCE PIPELINE
# -------------------------------------------------------------
tab_webrtc, tab_snapshot = st.tabs(["📹 Live Webcam Stream (WebRTC)", "📸 Snapshot / Interactive Camera"])

with tab_webrtc:
    if WEBRTC_AVAILABLE:
        st.write("Click **Start** below to request browser camera access and initiate live AI surveillance.")
        try:
            class StreamlitVideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.pipeline = pipeline

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    res: FrameResult = self.pipeline.process_frame(img)

                    # Record confirmed incidents in session history
                    if res.alert_state == AlertState.INCIDENT:
                        event_info = {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "fire_state": res.fire_state.value,
                            "smoke_state": res.smoke_state.value,
                            "alert_state": res.alert_state.value,
                        }
                        if not st.session_state.incident_log or st.session_state.incident_log[-1] != event_info:
                            st.session_state.incident_log.append(event_info)

                    # Return the annotated BGR frame with visual HUD
                    return av.VideoFrame.from_ndarray(res.annotated_frame, format="bgr24")

            # Configure WebRTC with Google Public STUN server for NAT traversal
            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            webrtc_ctx = webrtc_streamer(
                key="fire-smoke-security-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=StreamlitVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx and webrtc_ctx.state.playing:
                st.success("🟢 Surveillance Active: Streaming live frames to YOLO detection pipeline.")
            else:
                st.info("⏸️ Camera Standby: Click **Start** to connect your laptop camera.")
        except Exception as webrtc_err:
            logger.warning(f"WebRTC custom component initialization notice: {webrtc_err}")
            st.warning(
                f"ℹ️ **WebRTC Streaming Notice:** Continuous WebRTC video is configured for Streamlit Cloud (HTTPS). "
                f"In this local environment, please use the **Snapshot / Interactive Camera** tab to capture and test frames."
            )
    else:
        st.warning(
            "⚠️ WebRTC live streaming library (`av` / `streamlit-webrtc`) is unavailable in this local runtime "
            "environment. Use the **Snapshot / Interactive Camera** tab below to monitor via browser camera."
        )

with tab_snapshot:
    st.write("Capture a frame from your webcam or phone camera to run complete AI fire/smoke analysis.")
    captured_img = st.camera_input("Take a security scan photo")

    if captured_img is not None:
        file_bytes = np.asarray(bytearray(captured_img.read()), dtype=np.uint8)
        bgr_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if bgr_frame is not None:
            with st.spinner("Processing frame through YOLOv8n & Context Filters..."):
                result: FrameResult = pipeline.process_frame(bgr_frame)

            rgb_annotated = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, caption="AI Detection Result & Security HUD", use_container_width=True)

            # Detail Metrics
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                st.metric("Threat Status", result.alert_state.value.upper())
            with res_col2:
                st.metric("Environmental Fire", result.environmental_fire_count)
            with res_col3:
                st.metric("Environmental Smoke", result.environmental_smoke_count)
            with res_col4:
                blocked = (
                    result.screen_rejected_count
                    + result.face_rejected_count
                    + getattr(result, "white_rejected_count", 0)
                )
                st.metric("False Positives Filtered", blocked)

            if result.alert_state == AlertState.INCIDENT:
                st.error("🚨 CRITICAL HAZARD CONFIRMED: Immediate operator action required!")
            elif result.alert_state == AlertState.WARNING:
                st.warning("⚠️ SUSPECTED HAZARD: Under temporal persistence verification.")
            else:
                st.success("✅ Scene Clear: No environmental fire hazards detected.")


# -------------------------------------------------------------
# AUDIT LOG & RECENT INCIDENTS
# -------------------------------------------------------------
st.divider()
st.subheader("📋 Session Incident Audit Trail")

if st.session_state.incident_log:
    st.table(st.session_state.incident_log[-10:])
else:
    st.caption("No hazard incidents registered during this session. Normal surveillance ongoing.")
