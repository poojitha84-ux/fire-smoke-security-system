/**
 * Fire & Smoke Security Monitor - Front-End Controller
 * Real-time telemetry polling, alarm synthesis, and incident handling.
 */

let audioContext = null;
let alarmInterval = null;
let isAlarmSounding = false;
let audioEnabled = true;

let selectedIncidentId = null;
let activeIncidentIdForAck = null;
let lastKnownAlertState = "normal";
let lastKnownCamStatus = "STOPPED";

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    initControls();
    startPolling();
    fetchIncidents();
    appendEventLog("System dashboard initialized.");
});

function initControls() {
    // Start / Stop monitoring
    document.getElementById("btnStartMonitoring").addEventListener("click", () => {
        apiCall("/api/monitoring/start", "POST", {})
            .then(res => {
                appendEventLog(`Start monitoring command sent: ${res.message}`);
                fetchStatus();
            });
    });

    document.getElementById("btnStopMonitoring").addEventListener("click", () => {
        apiCall("/api/monitoring/stop", "POST", {})
            .then(res => {
                appendEventLog("Monitoring stopped by operator.");
                stopAudioAlarm();
                fetchStatus();
            });
    });

    // Audio controls
    document.getElementById("btnSilenceAlarm").addEventListener("click", () => {
        apiCall("/api/audio/silence", "POST", {})
            .then(() => {
                stopAudioAlarm();
                document.getElementById("btnSilenceAlarm").style.display = "none";
                const bSil = document.getElementById("btnBannerSilence");
                if (bSil) bSil.style.display = "none";
                appendEventLog("Alarm audio silenced by operator.");
            });
    });

    const btnBannerSilence = document.getElementById("btnBannerSilence");
    if (btnBannerSilence) {
        btnBannerSilence.addEventListener("click", () => {
            apiCall("/api/audio/silence", "POST", {})
                .then(() => {
                    stopAudioAlarm();
                    document.getElementById("btnSilenceAlarm").style.display = "none";
                    btnBannerSilence.style.display = "none";
                    appendEventLog("Alarm audio silenced from alert banner.");
                });
        });
    }

    const btnBannerAck = document.getElementById("btnBannerAck");
    if (btnBannerAck) {
        btnBannerAck.addEventListener("click", () => {
            if (activeIncidentIdForAck) {
                openIncidentModal(activeIncidentIdForAck, 'ack');
            } else {
                apiCall("/api/incidents?limit=1", "GET").then(res => {
                    if (res.incidents && res.incidents.length > 0) {
                        openIncidentModal(res.incidents[0].incident_id, 'ack');
                    }
                });
            }
        });
    }

    document.getElementById("btnToggleAudio").addEventListener("click", () => {
        audioEnabled = !audioEnabled;
        apiCall("/api/audio/toggle", "POST", { enabled: audioEnabled })
            .then(res => {
                const btn = document.getElementById("btnToggleAudio");
                btn.textContent = res.audio_enabled ? "🔔 Sound: ON" : "🔕 Sound: OFF";
                if (!res.audio_enabled) {
                    stopAudioAlarm();
                }
                appendEventLog(`Alarm sound toggled: ${res.audio_enabled ? "ENABLED" : "MUTED"}`);
            });
    });

    // Settings modal
    document.getElementById("btnOpenSettings").addEventListener("click", () => {
        document.getElementById("settingsModal").style.display = "flex";
    });
}

function startPolling() {
    fetchStatus();
    setInterval(fetchStatus, 1000);
    setInterval(fetchIncidents, 3000);
}

function fetchStatus() {
    apiCall("/api/status", "GET")
        .then(data => {
            updateHUD(data);
            handleAlarmState(data);
        })
        .catch(err => {
            console.error("Telemetry error:", err);
        });
}

function updateHUD(data) {
    // 1. Camera info and status
    const camStatusElem = document.getElementById("hudCamStatus");
    camStatusElem.textContent = data.status;
    camStatusElem.className = `hud-value status-badge badge-pill badge-${data.status.toLowerCase()}`;

    document.getElementById("hudFps").textContent = (data.fps || 0).toFixed(1);
    document.getElementById("hudFrame").textContent = data.frame_index || 0;
    const totalBlocked = (data.total_false_positives_blocked !== undefined) ? data.total_false_positives_blocked : (data.total_screen_false_positives_blocked || 0);
    document.getElementById("hudRejections").textContent = totalBlocked;

    // Log camera state transitions
    if (data.status !== lastKnownCamStatus) {
        appendEventLog(`Camera state transition: ${lastKnownCamStatus} -> ${data.status}`);
        lastKnownCamStatus = data.status;
    }

    // 2. Hazard tiles
    updateTile("tileFire", "tileFireStatus", "tileFireCount", data.fire_state, data.environmental_fire_count);
    updateTile("tileSmoke", "tileSmokeStatus", "tileSmokeCount", data.smoke_state, data.environmental_smoke_count);

    // Uncertain screen proximity tile
    const uncertainTile = document.getElementById("tileUncertain");
    const uncertainStatus = document.getElementById("tileUncertainStatus");
    const uncertainCount = document.getElementById("tileUncertainCount");
    uncertainCount.textContent = data.screen_uncertain_count || 0;

    if (data.screen_uncertain_count > 0) {
        uncertainTile.className = "hazard-tile tile-warning";
        uncertainStatus.textContent = "REVIEW REQ.";
    } else {
        uncertainTile.className = "hazard-tile tile-normal";
        uncertainStatus.textContent = "CLEAR";
    }

    // Screen & Face False-Positive Shield Tile
    const blockedCount = (data.total_false_positives_blocked !== undefined) ? data.total_false_positives_blocked : (data.screen_rejected_count || 0);
    document.getElementById("tilePhoneBlocked").textContent = blockedCount;

    // 3. System state badge and prominent alert banners
    const badge = document.getElementById("systemStateBadge");
    const text = document.getElementById("systemStateText");
    const confirmedBanner = document.getElementById("confirmedAlertBanner");
    const uncertainBanner = document.getElementById("uncertainAlertBanner");

    if (data.alert_state === "confirmed_alert" || data.alert_state === "cooldown") {
        badge.className = "status-badge badge-critical";
        text.textContent = "HAZARD CONFIRMED";
        if (confirmedBanner) confirmedBanner.style.display = "flex";
        if (uncertainBanner) uncertainBanner.style.display = "none";

        const titleText = `CONFIRMED ${data.fire_state === "confirmed" ? "FIRE " : ""}${data.smoke_state === "confirmed" ? "SMOKE" : ""} DETECTED`;
        const bannerTitleElem = document.getElementById("bannerTitle");
        if (bannerTitleElem) bannerTitleElem.textContent = titleText;

        if (data.active_alert) {
            activeIncidentIdForAck = data.active_alert.event_id;
            const hType = (data.active_alert.hazard_type || "").replace(/_/g, " ").toUpperCase();
            const elHazard = document.getElementById("bannerHazardType");
            const elSeverity = document.getElementById("bannerSeverity");
            const elConf = document.getElementById("bannerConfidence");
            const elCam = document.getElementById("bannerCameraId");
            const elLoc = document.getElementById("bannerCameraLoc");
            const elTime = document.getElementById("bannerTimestamp");

            if (elHazard) elHazard.textContent = hType;
            if (elSeverity) elSeverity.textContent = data.active_alert.severity || "HIGH";
            if (elConf) elConf.textContent = `${((data.active_alert.max_confidence || 0) * 100).toFixed(1)}%`;
            if (elCam) elCam.textContent = data.active_alert.camera_id || data.camera_id;
            if (elLoc) elLoc.textContent = data.active_alert.camera_location || data.camera_location;
            if (elTime && data.active_alert.timestamp) {
                elTime.textContent = new Date(data.active_alert.timestamp).toLocaleTimeString();
            }
        }
    } else if (data.alert_state === "uncertain_warning") {
        badge.className = "status-badge badge-warning";
        text.textContent = "UNCERTAIN WARNING";
        if (confirmedBanner) confirmedBanner.style.display = "none";
        if (uncertainBanner) {
            uncertainBanner.style.display = "flex";
            if (data.active_alert) {
                activeIncidentIdForAck = data.active_alert.event_id;
                const elCam = document.getElementById("bannerUncertainCam");
                const elLoc = document.getElementById("bannerUncertainLoc");
                if (elCam) elCam.textContent = data.active_alert.camera_id || data.camera_id;
                if (elLoc) elLoc.textContent = data.active_alert.camera_location || data.camera_location;
            }
        }
    } else if (data.alert_state === "suspected") {
        badge.className = "status-badge badge-warning";
        text.textContent = "SUSPECTED HAZARD";
        if (confirmedBanner) confirmedBanner.style.display = "none";
        if (uncertainBanner) uncertainBanner.style.display = "none";
    } else {
        badge.className = "status-badge badge-normal";
        text.textContent = "SYSTEM NORMAL";
        if (confirmedBanner) confirmedBanner.style.display = "none";
        if (uncertainBanner) uncertainBanner.style.display = "none";
    }

    if (data.alert_state !== lastKnownAlertState) {
        appendEventLog(`Alert Manager state changed: ${lastKnownAlertState.toUpperCase()} -> ${data.alert_state.toUpperCase()}`);
        lastKnownAlertState = data.alert_state;
    }
}

function updateTile(tileId, statusId, countId, state, count) {
    const tile = document.getElementById(tileId);
    const statusElem = document.getElementById(statusId);
    const countElem = document.getElementById(countId);

    countElem.textContent = count || 0;
    statusElem.textContent = (state || "NORMAL").toUpperCase();

    if (state === "confirmed") {
        tile.className = "hazard-tile tile-hazard";
    } else if (state === "suspected") {
        tile.className = "hazard-tile tile-warning";
    } else {
        tile.className = "hazard-tile tile-normal";
    }
}

function handleAlarmState(data) {
    const silenceBtn = document.getElementById("btnSilenceAlarm");
    const bannerSilenceBtn = document.getElementById("btnBannerSilence");

    if (data.audio_alarm_active && audioEnabled && !data.audio_silenced) {
        startAudioAlarm();
        silenceBtn.style.display = "inline-flex";
        if (bannerSilenceBtn) bannerSilenceBtn.style.display = "inline-flex";
    } else {
        stopAudioAlarm();
        if (data.audio_silenced || !data.audio_alarm_active) {
            silenceBtn.style.display = "none";
            if (bannerSilenceBtn) bannerSilenceBtn.style.display = "none";
        }
    }
}

/**
 * Local Web Audio API Emergency Siren Synthesizer
 * Plays a continuous, repeated alternating pulse alarm until acknowledged or silenced.
 */
function playAlarmTonePulse() {
    if (!audioEnabled || !isAlarmSounding) return;
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioContext.state === "suspended") {
            audioContext.resume();
        }

        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();

        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(880, audioContext.currentTime);
        osc.frequency.exponentialRampToValueAtTime(587.33, audioContext.currentTime + 0.28);

        gain.gain.setValueAtTime(0.3, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.32);

        osc.connect(gain);
        gain.connect(audioContext.destination);

        osc.start();
        osc.stop(audioContext.currentTime + 0.35);
    } catch (e) {
        console.warn("Web Audio API tone error:", e);
    }
}

function startAudioAlarm() {
    if (isAlarmSounding) return;
    isAlarmSounding = true;
    playAlarmTonePulse();
    if (alarmInterval) clearInterval(alarmInterval);
    alarmInterval = setInterval(playAlarmTonePulse, 550);
}

function stopAudioAlarm() {
    isAlarmSounding = false;
    if (alarmInterval) {
        clearInterval(alarmInterval);
        alarmInterval = null;
    }
}

function fetchIncidents() {
    apiCall("/api/incidents?limit=25", "GET")
        .then(data => {
            renderIncidents(data.incidents || []);
        })
        .catch(err => console.error("Error fetching incidents:", err));
}

function renderIncidents(incidents) {
    const tbody = document.getElementById("incidentTableBody");
    const countBadge = document.getElementById("activeIncidentCountBadge");

    const activeCount = incidents.filter(i => i.current_status !== "RESOLVED").length;
    countBadge.textContent = `${activeCount} Active`;

    if (!incidents || incidents.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="7">No security incidents recorded. System monitoring normal.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = incidents.map(inc => {
        const hazardTag = `<span class="badge-tag tag-${inc.hazard_type}">${inc.hazard_type.toUpperCase()}</span>`;
        const timeShort = new Date(inc.timestamp).toLocaleTimeString();
        
        // Status formatting
        let statusBadge = `<span class="status-badge badge-pill badge-stopped">${inc.current_status}</span>`;
        if (inc.current_status === "CONFIRMED") {
            statusBadge = `<span class="status-badge badge-pill badge-critical">CONFIRMED</span>`;
        } else if (inc.current_status === "ACKNOWLEDGED") {
            statusBadge = `<span class="status-badge badge-pill badge-warning">ACKNOWLEDGED</span>`;
        } else if (inc.current_status === "RESOLVED") {
            statusBadge = `<span class="status-badge badge-pill badge-normal">RESOLVED</span>`;
        }

        // Notification formatting
        let notifBadge = `<span class="camera-sub">${inc.notification_status}</span>`;
        if (inc.notification_status === "DELIVERED") {
            notifBadge = `<span style="color: var(--accent-green); font-size: 0.72rem;">✓ Delivered</span>`;
        } else if (inc.notification_status === "FAILED") {
            notifBadge = `<span style="color: var(--accent-red); font-size: 0.72rem;">✗ Delivery Failed</span>`;
        }

        // Action buttons
        let actions = ``;
        if (inc.current_status === "CONFIRMED") {
            actions = `<button class="btn btn-warning btn-pill" onclick="openIncidentModal('${inc.incident_id}', 'ack')">Ack</button>`;
        } else if (inc.current_status === "ACKNOWLEDGED") {
            actions = `<button class="btn btn-primary btn-pill" onclick="openIncidentModal('${inc.incident_id}', 'resolve')">Resolve</button>`;
        } else {
            actions = `<span class="camera-sub">Resolved</span>`;
        }

        return `
            <tr>
                <td><strong>${inc.incident_id}</strong></td>
                <td>${timeShort}</td>
                <td>${hazardTag}</td>
                <td><strong>${inc.severity}</strong></td>
                <td>${statusBadge}</td>
                <td>${notifBadge}</td>
                <td>${actions}</td>
            </tr>
        `;
    }).join("");
}

function openIncidentModal(incidentId, defaultAction) {
    selectedIncidentId = incidentId;
    document.getElementById("modalIncidentTitle").textContent = `Security Action: Incident ${incidentId}`;
    document.getElementById("modalIncidentDetails").textContent = `Update status for confirmed incident ${incidentId}. Enter operator name and action notes below:`;
    
    const btnAck = document.getElementById("btnModalAck");
    const btnResolve = document.getElementById("btnModalResolve");

    if (defaultAction === "ack") {
        btnAck.style.display = "inline-flex";
        btnResolve.style.display = "none";
    } else {
        btnAck.style.display = "none";
        btnResolve.style.display = "inline-flex";
    }

    document.getElementById("incidentModal").style.display = "flex";
}

function closeIncidentModal() {
    document.getElementById("incidentModal").style.display = "none";
    selectedIncidentId = null;
}

function submitIncidentAction(action) {
    if (!selectedIncidentId) return;

    const operator = document.getElementById("operatorName").value.trim() || "Security Officer";
    const notes = document.getElementById("actionNotes").value.trim();

    const endpoint = `/api/incidents/${selectedIncidentId}/${action}`;
    apiCall(endpoint, "POST", { operator_name: operator, notes: notes })
        .then(res => {
            appendEventLog(`Incident ${selectedIncidentId} marked ${action.toUpperCase()} by ${operator}`);
            stopAudioAlarm();
            const silBtn = document.getElementById("btnSilenceAlarm");
            if (silBtn) silBtn.style.display = "none";
            const bSil = document.getElementById("btnBannerSilence");
            if (bSil) bSil.style.display = "none";
            closeIncidentModal();
            fetchIncidents();
            fetchStatus();
        })
        .catch(err => {
            alert(`Error updating incident: ${err.message}`);
        });
}

function closeSettingsModal() {
    document.getElementById("settingsModal").style.display = "none";
}

function saveSettings() {
    const camId = document.getElementById("settingCamId").value.trim();
    const camLoc = document.getElementById("settingCamLoc").value.trim();
    const fireThresh = parseFloat(document.getElementById("settingFireThresh").value);
    const smokeThresh = parseFloat(document.getElementById("settingSmokeThresh").value);

    apiCall("/api/config", "POST", {
        camera_id: camId,
        camera_location: camLoc,
        fire_threshold: fireThresh,
        smoke_threshold: smokeThresh,
    }).then(() => {
        document.getElementById("cameraInfoText").textContent = `${camId} • ${camLoc}`;
        appendEventLog(`Camera settings updated: ${camId} (${camLoc})`);
        closeSettingsModal();
    });
}

function appendEventLog(message) {
    const logElem = document.getElementById("eventStreamLog");
    const item = document.createElement("div");
    item.className = "log-item";
    const time = new Date().toLocaleTimeString();
    item.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-msg">${escapeHtml(message)}</span>`;
    logElem.prepend(item);

    // Limit log rows
    while (logElem.children.length > 50) {
        logElem.removeChild(logElem.lastChild);
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function apiCall(url, method = "GET", body = null) {
    const options = {
        method: method,
        headers: { "Content-Type": "application/json" }
    };
    if (body && (method === "POST" || method === "PUT")) {
        options.body = JSON.stringify(body);
    }
    return fetch(url, options).then(res => {
        if (!res.ok) {
            return res.json().then(data => { throw new Error(data.error || "Request failed"); });
        }
        return res.json();
    });
}
