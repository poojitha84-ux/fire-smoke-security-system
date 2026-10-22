# Real-Time Fire & Smoke Monitoring System with False-Positive Screen Defense

A production-grade, real-time computer vision security system designed for college and commercial building surveillance demonstrations. The system continuously monitors live camera feeds (Webcam or RTSP/IP CCTV), performs real-time fire and smoke object detection, prevents mobile phone and digital screen false alarms, tracks incidents through an operator dashboard, and dispatches external security notifications.

> **CRITICAL ARCHITECTURAL CONSTRAINTS:**
> - **Input Source:** Strictly live camera feeds (Webcam or RTSP/IP stream).
> - **No File Uploads:** Uploading pre-recorded videos or images is intentionally omitted and prohibited by system design.
> - **Life-Safety Disclaimer:** This application is an AI demonstration prototype. It is **not** certified under UL 268 / EN 54 or local fire regulatory standards for life-safety fire protection. Do **not** use as a replacement for certified smoke/heat detectors or life-safety fire alarm panels.

---

## Architecture Overview

```
                          +------------------------------------------+
                          |   Live Camera (Webcam or RTSP Stream)    |
                          +--------------------+---------------------+
                                               |
                                               v
                          +--------------------+---------------------+
                          |        Camera Capture Worker             |
                          |   (Thread-safe OpenCV VideoCapture)      |
                          +--------------------+---------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
     +---------------+---------------+                   +---------------+---------------+
     |  Fire & Smoke YOLOv8n Model   |                   |   Screen Context Detector     |
     |  (D-Fire Dataset Fine-Tuned)  |                   |   (COCO: Phone, TV, Laptop)   |
     +---------------+---------------+                   +---------------+---------------+
                     |                                                   |
                     +-------------------------+-------------------------+
                                               |
                                               v
                          +--------------------+---------------------+
                          |      Screen-Context Filter Engine        |
                          |  - Evaluates bounding-box containment    |
                          |  - Rejects displayed fire on screens     |
                          |  - Flags ambiguous proximity as UNCERTAIN|
                          |  - Retains genuine environmental fire    |
                          +--------------------+---------------------+
                                               |
                                               v
                          +--------------------+---------------------+
                          |      Temporal Confirmation Filter        |
                          |  - Sliding window (M of N frames)        |
                          |  - Rejects single-frame transient glitches|
                          |  - Smooth recovery when hazard clears    |
                          +--------------------+---------------------+
                                               |
                                               v
                          +--------------------+---------------------+
                          |         Alert State Manager              |
                          |  - NORMAL -> SUSPECTED -> CONFIRMED      |
                          |  - Severity: MEDIUM / HIGH / CRITICAL    |
                          |  - Duplicate suppression & Cooldown      |
                          |  - Web Audio alarm triggering & silence  |
                          +--------------------+---------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
     +---------------+---------------+                   +---------------+---------------+
     |   Security Incident Service   |                   |  External Notification Engine |
     | - Audit log persistence       |                   | - Async non-blocking thread   |
     | - Operator Ack / Resolve      |                   | - SMTP Email & Webhook alerts |
     | - Local JSONL audit trail     |                   | - Delivery status tracking    |
     +---------------+---------------+                   +---------------+---------------+
                     |
                     v
     +---------------+---------------+
     |     Flask Security Dashboard  |
     | - Live MJPEG stream with HUD  |
     | - Incident response controls  |
     | - Real-time REST telemetry    |
     +-------------------------------+
```

---

## 1. Verified Models & Provenance

| Model | Provenance / Repository | Dataset | Classes | Licence | Local Path |
|---|---|---|---|---|---|
| **Fire & Smoke Detector** | [`rabahdev/fire-smoke-yolov8n`](https://huggingface.co/rabahdev/fire-smoke-yolov8n) | D-Fire Dataset (4,306 test split images) | `0: smoke`<br>`1: fire` | AGPL-3.0 | `models/fire_smoke_yolov8n.pt` |
| **Screen Context Detector** | [Ultralytics Official Releases](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt) | COCO Dataset | `62: tv`<br>`63: laptop`<br>`67: cell phone` | AGPL-3.0 | `models/yolov8n.pt` |

The models are verified on startup and automatically downloaded to `models/` if absent.

---

## 2. Mobile Phone and Screen False-Positive Prevention

When a person holds a smartphone in front of the camera displaying a fire video or image:
1. **Screen Detection:** The system detects digital displays (`cell phone`, `tv`, `laptop`) via YOLOv8n object detection augmented by geometric display contour boundary analysis.
2. **Containment Calculation:** For each candidate flame box $B_{\text{hazard}}$ and screen box $B_{\text{screen}}$, the intersection ratio is computed:
   $$\text{Overlap Ratio} = \frac{\text{Area}(B_{\text{hazard}} \cap B_{\text{screen}})}{\text{Area}(B_{\text{hazard}})}$$
3. **Classification Policy:**
   - **`SCREEN_REJECTED`** ($\ge 65\%$ overlap): Fire is predominantly contained inside the display surface. Classified as a screen-displayed false positive; highlighted in cyan/amber on the HUD; recorded in the screen false-positives counter; **never** triggers an environmental fire alarm.
   - **`SCREEN_UNCERTAIN`** ($25\% - 65\%$ overlap): Partial overlap or ambiguous boundary. Flagged for security operator review without sounding high-priority alarms.
   - **`CONFIRMED_ENVIRONMENTAL`** ($< 25\%$ overlap): Hazard is environmental.
4. **Multi-Hazard Scene Independence:**
   - **Crucial Rule:** The system does **not** discard the entire frame just because a phone is present.
   - If a person holds a phone showing fire while a real fire is burning elsewhere in the room, the phone fire is rejected while the real fire continues to be monitored and confirmed.
   - Genuine fire near a monitor or TV is **not** discarded unless significant containment inside the screen bezel is proven.

---

## 3. Installation & Setup (Windows PowerShell)

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14 (Tested on Python 3.14.6 64-bit on Windows).
- A connected USB/built-in webcam, or an accessible RTSP/IP CCTV stream URL.
- Windows PowerShell.

### Step 1: Open PowerShell and Navigate to Project
```powershell
cd c:\Users\poojitha\OneDrive\Attachments\Desktop\minor_project\fire_smoke_security_system
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)
Copy the example environment configuration:
```powershell
Copy-Item .env.example .env
```
Edit `.env` to configure camera source (default `0` for webcam, or an RTSP URL like `rtsp://admin:pass@192.168.1.50:554/stream1`), camera ID, camera location, or external notification credentials.

---

## 4. Running the Application

### Start the Dashboard
```powershell
python app.py
```
Output:
```
Starting Fire & Smoke Security Dashboard on http://localhost:5000
```

### Accessing the Web Interface
Open your web browser and navigate to:
**`http://localhost:5000`**

### Dashboard Operations:
1. **Start Monitoring:** Click the **▶ Start Monitoring** button in the header. The camera hardware connects and begins streaming live video with detection HUD overlays.
2. **Stop Monitoring:** Click **⏹ Stop Monitoring** to immediately release camera hardware resources and halt inference.
3. **Audio Alarm:**
   - When a hazard is confirmed, the browser triggers an audible siren buzzer via the Web Audio API.
   - Click **🔕 Silence Alarm** to mute the current alarm.
   - Click **🔔 Sound: ON/OFF** to toggle audio alarms.
4. **Security Incidents:**
   - Click **Ack** on an active incident to acknowledge receipt.
   - Click **Resolve** once the area is confirmed safe to transition the incident to `RESOLVED`.
5. **Settings:** Click **⚙️ Settings** to modify camera name, location, and confidence thresholds on the fly.

---

## 5. Security Notifications Configuration

The system supports free external alert integrations via SMTP Email or Webhooks:

### Option A: SMTP Email Alerts
In your `.env` file:
```env
ENABLE_NOTIFICATIONS=True
NOTIFICATION_TYPE=smtp
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=True
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=security-team@domain.com
```

### Option B: Webhook Alerts (Slack / MS Teams / Custom Gateway)
```env
ENABLE_NOTIFICATIONS=True
NOTIFICATION_TYPE=webhook
WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/XXXXX
WEBHOOK_TIMEOUT_SECONDS=5.0
WEBHOOK_MAX_RETRIES=2
```

> **Note on Delivery Status:** If notification credentials are not configured, the local dashboard and incident logging continue to operate normally. Incidents will explicitly display `NOT_CONFIGURED` or `FAILED`. The system never claims security was notified unless the delivery is confirmed.

---

## 6. Running Automated Tests

A comprehensive automated test suite covers all pipeline stages:
- Fire and smoke class mapping
- Confidence threshold filtering
- Temporal confirmation and sliding window logic
- Intermittent detection rejection
- Screen-context rejection (phone screens rejected, outside fires confirmed, near-screen fires preserved)
- Alert cooldown and duplicate suppression
- Incident management lifecycle (create, acknowledge, resolve)
- Notification delivery success and failure
- Camera unavailable, disconnection, and resource release
- Missing model weights and invalid configurations

Run the test suite with:
```powershell
pytest tests -v
```

Expected output:
```
============================= 42 passed in 6.33s ==============================
```

The 42 tests include:
- Unit & integration tests for screen context geometry, containment, protrusion, and occlusion
- Alert manager state machine, cooldown logic, and audio silencing
- Incident logging and thread-safe persistence to `logs/incidents.jsonl`
- Full-workflow end-to-end integration tests (`test_full_workflow.py`):
  - Confirmed environmental fire (RED ALERT, pulsing banner, continuous repeating buzzer, persistence)
  - Confirmed environmental smoke (MEDIUM severity, buzzer, persistence)
  - Mobile phone fire video rejection (0 alarms, 0 audio, 0 incidents)
  - Normal background scene (clean baseline)
  - Sustained uncertain screen proximity (AMBER WARNING, UNCONFIRMED incident)
  - Staff alarm silencing, incident acknowledgment, resolution, and clean recovery
  - Real disk model weight validation (loads actual `.pt` files without mocks)

---

## 7. Real-Camera Empirical Validation Plan

To evaluate the system under real-world operating conditions, execute the following 9-stage validation protocol. Record detections, false positives, and latencies.

| Test Scenario | Setup Procedure | Expected Behavior | Evaluation Metric |
|---|---|---|---|
| **1. Ordinary Room** | Point camera at empty hallway or classroom with normal activity (people walking, lights on/off). | Zero fire or smoke detections. Status remains `NORMAL`. | False-Positive Rate ($FPR = 0\%$) |
| **2. Mobile Phone Fire** | A tester holds a smartphone playing high-definition campfire or explosion footage at 1m distance. | Phone detected (`cell phone`). Fire bounding box overlaps phone screen $\ge 65\%$. Overlay displays cyan box `REJECTED: FIRE ON SCREEN`. Alarm is **suppressed**. | Phone Screen Rejection Rate |
| **3. Television / Monitor** | Display YouTube fire video on a laptop screen or wall monitor in view of camera. | Display detected (`tv`/`laptop`). Fire classified as `SCREEN_REJECTED`. No incident triggered. | Screen Rejection Accuracy |
| **4. Genuine Flame** | Safely introduce controlled flame (e.g. candle or lighter, following strict laboratory fire safety protocols with extinguisher present). | Bounding box detected with label `FIRE: X% [ENVIRONMENTAL]`. After 5 of 8 frames, status transitions to `CONFIRMED HAZARD`, incident created, continuous audio sounds. | Confirmation Latency & Precision |
| **5. Smoke Simulation** | Use safe theatrical aerosol smoke spray or mist diffuser in camera view. | Bounding box detected with label `SMOKE: X% [ENVIRONMENTAL]`. Severity classified as `MEDIUM`. | Smoke Detection Recall |
| **6. Fire Near a Screen** | Place a safe controlled candle 20cm beside an active computer screen displaying text. | Candle flame detected. Overlap with monitor is $< 25\%$. System correctly confirms environmental fire despite proximity to screen. | Near-Screen Preservation Rate |
| **7. Sustained Screen Occlusion** | Partially obscure flame behind phone bezel or tilted display ($25\% - 65\%$ overlap) for sustained period ($\ge 5$ frames). | System triggers `UNCERTAIN WARNING` amber banner and UNCONFIRMED incident so potential obscured real hazard is not ignored. | Occlusion Safety Recall |
| **8. Alarm Silence & Ack** | When confirmed fire sounds the alarm, click `[🔕 Silence Alarm]` or `[✓ Acknowledge]`. | Buzzer stops immediately. Active RED ALERT banner remains visible until flame physically cleared. | Response & Control Integrity |
| **9. Camera Disconnection** | Unplug USB camera cable while live monitoring is active. | System detects consecutive read failures within 5 frames, transitions state to `DISCONNECTED`, displays warning on HUD, attempts clean recovery upon reconnect. | Fault Tolerance & Recovery Time |

---

## 8. Limitations & Engineering Disclaimers

1. **Environmental Screen Angles:** Extreme oblique angles (> 60 degrees tilt) of a smartphone screen may reduce the bounding-box aspect ratio or cause YOLO to miss the phone boundary. In such cases, the system relies on contour display analysis or flags the detection as uncertain.
2. **Reflective Surfaces:** Shiny metallic or glass partitions reflecting distant lights may occasionally produce brief candidate detections; the temporal filter requires persistent multi-frame confirmation before elevating alerts.
3. **Model Fine-Tuning Scope:** The primary fire/smoke model is trained on the public D-Fire dataset. For specialized industrial environments (chemical plants, oil refineries), additional fine-tuning with domain-specific imagery is recommended.
4. **Safety Verification:** This system is an AI-assisted visual surveillance demonstration tool and must never replace certified building fire alarm systems (e.g., NFPA / EN 54 / UL 268 standards).
