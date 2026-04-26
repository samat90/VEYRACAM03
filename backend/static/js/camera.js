function getCookie(name) {
    const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[1]) : null;
}

const EMOTION_MAP = {
    happy: 'Радость',
    angry: 'Злость',
    surprised: 'Удивление',
    sad: 'Грусть',
    neutral: 'Нейтрально',
};

const POSE_CONNECTIONS = [
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [23, 25], [25, 27], [24, 26], [26, 28],
    [27, 29], [28, 30],
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
];

const TARGET_FPS = 15;
const FRAME_INTERVAL_MS = Math.round(1000 / TARGET_FPS);
const FRAME_TIMEOUT_MS = 4000;
const CALIBRATION_HELP_AFTER_MS = 15000;

class VeyraCamera {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('overlay');
        this.octx = this.overlay ? this.overlay.getContext('2d') : null;
        this.calibrationOverlay = document.getElementById('calibration-overlay');
        this.calibrationBar = document.getElementById('calibration-bar');
        this.calibrationHint = document.getElementById('calibration-hint');
        this.pausedOverlay = document.getElementById('paused-overlay');
        this.errorBanner = document.getElementById('error-banner');

        this.isStreaming = false;
        this.isPaused = false;
        this.stream = null;
        this.frameCount = 0;
        this.inFlight = false;
        this.consecutiveErrors = 0;
        this.loopTimer = null;
        this.calibrationStartedAt = null;
        this.radarChart = null;
        this.bindEvents();
        this.initRadar();
    }

    bindEvents() {
        document.getElementById('start-btn').addEventListener('click', () => this.startCamera());
        document.getElementById('pause-btn').addEventListener('click', () => this.togglePause());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopCamera());
        const reportBtn = document.getElementById('self-report-btn');
        if (reportBtn) reportBtn.addEventListener('click', () => this.openReport());
        document.querySelectorAll('.report-btn').forEach(b => {
            b.addEventListener('click', () => this.submitReport(parseInt(b.dataset.feeling, 10)));
        });
    }

    initRadar() {
        const canvas = document.getElementById('radar');
        if (!canvas || !window.Chart) return;
        const styles = getComputedStyle(document.documentElement);
        const text = styles.getPropertyValue('--card-text') || '#222';
        this.radarChart = new Chart(canvas.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Поза', 'Бодрость', 'Расслабленность', 'Сосредоточенность', 'Дыхание', 'Эмоция'],
                datasets: [{
                    label: 'Состояние',
                    data: [null, null, null, null, null, null],
                    backgroundColor: 'rgba(0, 210, 255, 0.2)',
                    borderColor: 'rgba(0, 210, 255, 0.9)',
                    pointBackgroundColor: 'rgba(0, 210, 255, 1)',
                    spanGaps: false,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    r: {
                        suggestedMin: 0, suggestedMax: 100,
                        ticks: { stepSize: 25, color: text, backdropColor: 'transparent' },
                        pointLabels: { color: text, font: { size: 11 } },
                        grid: { color: 'rgba(128,128,128,0.3)' },
                        angleLines: { color: 'rgba(128,128,128,0.3)' },
                    },
                },
            },
        });
    }

    updateRadar(data) {
        if (!this.radarChart) return;
        const posture = data.posture || {};
        const blink = data.blink || {};
        const breath = data.respiration || {};
        const hint = document.getElementById('radar-hint');

        const faceDetected = data.head_pose !== null && data.head_pose !== undefined;
        const poseDetected = data.pose_landmarks !== null && data.pose_landmarks !== undefined;
        const postureReady = poseDetected && posture.calibration_complete;
        const blinkReady = faceDetected && blink.calibration_complete;

        const postureMap = {
            'норма': 90, 'небольшой наклон': 55,
            'сильный наклон': 20, 'плечи неровно': 25,
        };
        const emotionMap = {
            happy: 90, neutral: 70, surprised: 60, sad: 30, angry: 20,
        };

        const postureScore = postureReady && postureMap[posture.status] !== undefined
            ? postureMap[posture.status] : null;
        const energy = blinkReady ? Math.max(0, 100 - (data.fatigue || 0)) : null;
        const calm = blinkReady && data.heart_rate > 0
            ? Math.max(0, 100 - (data.stress || 0)) : null;
        const focus = faceDetected && data.attention && data.attention !== 'unknown'
            ? (data.attention === 'сосредоточен' ? 90 : 35) : null;
        const breathScore = breath.rate > 0
            ? Math.max(0, 100 - Math.abs(breath.rate - 14) * 5) : null;
        const emotionScore = faceDetected && emotionMap[data.emotion] !== undefined
            ? emotionMap[data.emotion] : null;

        const values = [postureScore, energy, calm, focus, breathScore, emotionScore];
        this.radarChart.data.datasets[0].data = values;
        this.radarChart.update('none');

        const filled = values.filter(v => v !== null).length;
        if (hint) {
            if (filled === 0) hint.textContent = 'нет данных';
            else if (filled < 6) hint.textContent = `${filled}/6 осей готово`;
            else hint.textContent = '';
        }
    }

    openReport() {
        const m = document.getElementById('report-modal');
        if (m && window.bootstrap) bootstrap.Modal.getOrCreateInstance(m).show();
    }

    async submitReport(feeling) {
        try {
            await fetch('/self-report/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({ feeling }),
            });
        } catch (_) {}
        const m = document.getElementById('report-modal');
        if (m && window.bootstrap) bootstrap.Modal.getOrCreateInstance(m).hide();
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false,
            });
            this.video.srcObject = this.stream;
            await new Promise(resolve => { this.video.onloadedmetadata = () => resolve(); });
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            if (this.overlay) {
                this.overlay.width = this.video.videoWidth;
                this.overlay.height = this.video.videoHeight;
            }

            document.getElementById('start-btn').disabled = true;
            document.getElementById('pause-btn').disabled = false;
            document.getElementById('pause-btn').textContent = 'Пауза';
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('camera-status').className = 'badge bg-success';
            document.getElementById('camera-status').textContent = 'Активна';
            document.getElementById('resolution').textContent =
                `${this.video.videoWidth}x${this.video.videoHeight}`;

            if (this.calibrationOverlay) this.calibrationOverlay.style.display = '';
            this.calibrationStartedAt = Date.now();
            this.hideError();

            this.isStreaming = true;
            this.isPaused = false;
            this.startProcessing();
        } catch (err) {
            console.error('Camera access error:', err);
            this.showError('Не удалось получить доступ к камере. Проверьте разрешения.');
        }
    }

    togglePause() {
        if (!this.isStreaming) return;
        this.isPaused = !this.isPaused;
        const btn = document.getElementById('pause-btn');
        btn.textContent = this.isPaused ? 'Продолжить' : 'Пауза';
        btn.className = this.isPaused ? 'btn btn-success' : 'btn btn-warning';
        if (this.pausedOverlay) {
            this.pausedOverlay.style.display = this.isPaused ? '' : 'none';
        }
        if (this.isPaused) {
            document.getElementById('fps').textContent = '0';
            fetch('/pause-session/', {
                method: 'POST',
                headers: { 'X-CSRFToken': getCookie('csrftoken') },
            }).catch(() => {});
        }
    }

    async stopCamera() {
        this.isStreaming = false;
        this.isPaused = false;
        if (this.loopTimer) {
            clearTimeout(this.loopTimer);
            this.loopTimer = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
        document.getElementById('pause-btn').textContent = 'Пауза';
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('camera-status').className = 'badge bg-secondary';
        document.getElementById('camera-status').textContent = 'Выключена';
        document.getElementById('face-status').className = 'badge bg-secondary';
        document.getElementById('face-status').textContent = '—';
        document.getElementById('pose-status').className = 'badge bg-secondary';
        document.getElementById('pose-status').textContent = '—';

        if (this.calibrationOverlay) this.calibrationOverlay.style.display = 'none';
        if (this.pausedOverlay) this.pausedOverlay.style.display = 'none';
        document.getElementById('pause-btn').className = 'btn btn-warning';
        if (this.octx) {
            this.octx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        }

        try {
            const resp = await fetch('/stop-session/', {
                method: 'POST',
                headers: { 'X-CSRFToken': getCookie('csrftoken') },
            });
            const data = await resp.json();
            if (data.summary) this.showSummary(data.summary);
        } catch (e) { console.error('stop-session:', e); }
    }

    startProcessing() {
        let lastTime = Date.now();
        let frames = 0;

        const tick = async () => {
            if (!this.isStreaming) return;
            if (!this.isPaused) {
                frames++;
                const now = Date.now();
                if (now - lastTime >= 1000) {
                    document.getElementById('fps').textContent =
                        Math.round((frames * 1000) / (now - lastTime));
                    frames = 0;
                    lastTime = now;
                }
                if (!this.inFlight) {
                    await this.processSingleFrame();
                }
            }
            this.loopTimer = setTimeout(tick, FRAME_INTERVAL_MS);
        };
        tick();
    }

    async processSingleFrame() {
        this.inFlight = true;
        this.frameCount++;
        const needAdvice = this.frameCount % 30 === 0;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const imageData = this.canvas.toDataURL('image/jpeg', 0.7);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), FRAME_TIMEOUT_MS);

        try {
            const response = await fetch('/process-frame/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken'),
                },
                body: JSON.stringify({
                    image: imageData,
                    timestamp: Date.now(),
                    need_advice: needAdvice,
                }),
                signal: controller.signal,
            });
            clearTimeout(timeoutId);
            const data = await response.json();
            if (data.success) {
                this.consecutiveErrors = 0;
                this.hideError();
                this.drawOverlay(data.pose_landmarks, data.rppg_roi);
                this.renderMetrics(data);
                if (data.advice) {
                    document.getElementById('advice-text').textContent = data.advice;
                    const card = document.getElementById('advice-card');
                    if (card) {
                        card.classList.remove('severity-0','severity-1','severity-2','severity-3');
                        card.classList.add('severity-' + (data.advice_severity || 0));
                    }
                }
            } else {
                this.handleFrameError(data.error || 'Ошибка обработки');
            }
        } catch (err) {
            clearTimeout(timeoutId);
            const msg = err.name === 'AbortError'
                ? 'Сервер не ответил за 4 секунды'
                : 'Сетевая ошибка: ' + err.message;
            console.error('Frame error:', err);
            this.handleFrameError(msg);
        } finally {
            this.inFlight = false;
        }
    }

    handleFrameError(msg) {
        this.consecutiveErrors++;
        if (this.consecutiveErrors >= 3) {
            this.showError(msg);
        }
    }

    showError(text) {
        if (this.errorBanner) {
            this.errorBanner.textContent = text;
            this.errorBanner.style.display = '';
        }
    }

    hideError() {
        if (this.errorBanner) this.errorBanner.style.display = 'none';
    }

    drawOverlay(landmarks, rppgRoi) {
        if (!this.octx) return;
        const w = this.overlay.width;
        const h = this.overlay.height;
        this.octx.clearRect(0, 0, w, h);

        if (rppgRoi) {
            this.octx.strokeStyle = 'rgba(0, 210, 255, 0.7)';
            this.octx.lineWidth = 2;
            this.octx.setLineDash([6, 4]);
            this.octx.strokeRect(rppgRoi.x * w, rppgRoi.y * h, rppgRoi.w * w, rppgRoi.h * h);
            this.octx.setLineDash([]);
        }

        if (landmarks) {
            this.octx.lineWidth = 2;
            this.octx.strokeStyle = 'rgba(255, 60, 60, 0.9)';
            this.octx.beginPath();
            for (const [a, b] of POSE_CONNECTIONS) {
                const la = landmarks[a], lb = landmarks[b];
                if (!la || !lb || la.v < 0.5 || lb.v < 0.5) continue;
                this.octx.moveTo(la.x * w, la.y * h);
                this.octx.lineTo(lb.x * w, lb.y * h);
            }
            this.octx.stroke();
            this.octx.fillStyle = 'rgba(80, 255, 80, 0.95)';
            for (const lm of landmarks) {
                if (lm.v < 0.5) continue;
                this.octx.beginPath();
                this.octx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
                this.octx.fill();
            }
        }
    }

    updateCalibration(data) {
        const posture = data.posture || {};
        const blink = data.blink || {};
        const postureProg = posture.calibration_progress || 0;
        const blinkProg = blink.calibration_progress || 0;
        const minProg = Math.min(postureProg, blinkProg);
        const bothDone = posture.calibration_complete && blink.calibration_complete;

        if (this.calibrationBar) {
            this.calibrationBar.style.width = Math.round(minProg * 100) + '%';
        }
        if (this.calibrationOverlay) {
            this.calibrationOverlay.style.display = bothDone ? 'none' : '';
        }
        if (bothDone) {
            this.calibrationStartedAt = null;
        } else if (this.calibrationHint && this.calibrationStartedAt) {
            const elapsed = Date.now() - this.calibrationStartedAt;
            if (elapsed > CALIBRATION_HELP_AFTER_MS) {
                this.calibrationHint.textContent =
                    'Долго не получается? Включите свет, отсядьте дальше, чтобы плечи попадали в кадр.';
            }
        }
    }

    renderMetrics(data) {
        this.updateCalibration(data);

        const postureEl = document.getElementById('posture-value');
        const postureStatus = document.getElementById('posture-status');
        const posture = data.posture || {};

        if (posture.calibrating) {
            postureEl.textContent = '…';
            postureStatus.textContent = `калибровка ${Math.round((posture.calibration_progress||0)*100)}%`;
            postureEl.className = 'metric-value warning';
            postureStatus.className = 'warning';
        } else if (posture.angle !== null && posture.angle !== undefined) {
            postureEl.textContent = `${Number(posture.angle).toFixed(1)}°`;
            postureStatus.textContent = posture.status || '';
            const s = posture.status;
            if (s === 'сильный наклон' || s === 'плечи неровно') {
                postureEl.className = 'metric-value danger';
                postureStatus.className = 'danger';
            } else if (s === 'небольшой наклон') {
                postureEl.className = 'metric-value warning';
                postureStatus.className = 'warning';
            } else {
                postureEl.className = 'metric-value good';
                postureStatus.className = 'good';
            }
        } else {
            postureEl.textContent = '—';
            postureStatus.textContent = 'нет данных';
        }

        const blink = data.blink || {};
        const blinkEl = document.getElementById('blink-value');
        if (blink.calibrating) {
            blinkEl.textContent = '…';
        } else if (blink.rate > 0) {
            blinkEl.textContent = Math.round(blink.rate);
        } else {
            blinkEl.textContent = '—';
        }
        const perclosEl = document.getElementById('perclos-value');
        if (perclosEl) {
            perclosEl.textContent = blink.calibration_complete && blink.perclos !== undefined
                ? `${blink.perclos}%` : '—';
        }

        const breath = data.respiration || {};
        document.getElementById('breath-value').textContent =
            breath.rate > 0 ? Math.round(breath.rate) : '—';

        const hrvEl = document.getElementById('hrv-value');
        const hrvDetail = document.getElementById('hrv-detail');
        if (hrvEl) {
            if (data.hrv_rmssd_ms > 0) {
                hrvEl.textContent = Math.round(data.hrv_rmssd_ms);
                if (hrvDetail) hrvDetail.textContent = `RMSSD · SDNN ${Math.round(data.hrv_sdnn_ms || 0)}`;
            } else {
                hrvEl.textContent = '—';
                if (hrvDetail) hrvDetail.textContent = 'RMSSD мс';
            }
        }

        const stabEl = document.getElementById('stability-value');
        if (stabEl) {
            stabEl.textContent = (data.stability_std && data.stability_std > 0)
                ? Number(data.stability_std).toFixed(1) : '—';
        }

        const yawnEl = document.getElementById('yawn-value');
        if (yawnEl) {
            const yc = data.yawn && data.yawn.count;
            yawnEl.textContent = yc > 0 ? yc : (data.yawn ? '0' : '—');
        }

        const hrEl = document.getElementById('hr-value');
        const hrDetail = document.getElementById('hr-detail');
        if (data.heart_rate && data.heart_rate > 0) {
            hrEl.textContent = Math.round(data.heart_rate);
            const hrConf = Math.round((data.heart_rate_confidence || 0) * 100);
            hrDetail.textContent = `уд/мин · уверенность ${hrConf}%`;
        } else {
            hrEl.textContent = '—';
            hrDetail.textContent = 'прогрев ~10 сек';
        }

        const faceDetected = data.head_pose !== null && data.head_pose !== undefined;
        const emoEl = document.getElementById('emotion-value');
        const emoDetail = document.getElementById('emotion-detail');
        if (faceDetected && data.emotion) {
            emoEl.textContent = EMOTION_MAP[data.emotion] || data.emotion;
            const conf = Math.round((data.emotion_confidence || 0) * 100);
            if (emoDetail) emoDetail.textContent = `${conf}%`;
        } else {
            emoEl.textContent = '—';
            if (emoDetail) emoDetail.textContent = '—';
        }

        const attEl = document.getElementById('attention-value');
        const attention = data.attention || 'unknown';
        attEl.textContent = attention === 'unknown' ? '—' : attention;
        if (attention === 'сосредоточен') {
            attEl.className = 'metric-value good';
        } else if (attention === 'unknown') {
            attEl.className = 'metric-value';
        } else {
            attEl.className = 'metric-value warning';
        }
        attEl.style.fontSize = '1.1rem';
        const hp = data.head_pose;
        const hpDetail = document.getElementById('head-pose-detail');
        if (hp && hpDetail) {
            hpDetail.textContent = `pitch ${hp.pitch}° · yaw ${hp.yaw}°`;
        } else if (hpDetail) {
            hpDetail.textContent = 'pitch — · yaw —';
        }

        const blinkReady = (data.blink && data.blink.calibration_complete);
        const fVal = document.getElementById('fatigue-value');
        const fBar = document.getElementById('fatigue-bar');
        if (blinkReady) {
            const fatigue = data.fatigue || 0;
            fVal.textContent = fatigue;
            fBar.style.width = fatigue + '%';
            fBar.className = 'progress-bar ' + (
                fatigue >= 70 ? 'bg-danger' :
                fatigue >= 45 ? 'bg-warning' : 'bg-success'
            );
        } else {
            fVal.textContent = '—';
            fBar.style.width = '0%';
            fBar.className = 'progress-bar bg-secondary';
        }

        const sVal = document.getElementById('stress-value');
        const sBar = document.getElementById('stress-bar');
        if (sVal && sBar) {
            if (blinkReady && data.heart_rate > 0) {
                const stress = data.stress || 0;
                sVal.textContent = stress;
                sBar.style.width = stress + '%';
                sBar.className = 'progress-bar ' + (
                    stress >= 70 ? 'bg-danger' :
                    stress >= 45 ? 'bg-warning' : 'bg-success'
                );
            } else {
                sVal.textContent = '—';
                sBar.style.width = '0%';
                sBar.className = 'progress-bar bg-secondary';
            }
        }

        this.updateRadar(data);

        const poseDetected = data.pose_landmarks !== null && data.pose_landmarks !== undefined;
        const faceEl = document.getElementById('face-status');
        const poseEl = document.getElementById('pose-status');
        faceEl.className = 'badge ' + (faceDetected ? 'bg-success' : 'bg-warning');
        faceEl.textContent = faceDetected ? 'Активно' : 'Нет лица';
        poseEl.className = 'badge ' + (poseDetected ? 'bg-success' : 'bg-warning');
        poseEl.textContent = poseDetected ? 'Активно' : 'Нет позы';
    }

    showSummary(s) {
        const body = document.getElementById('summary-body');
        if (!body) return;
        if (!s || s.sample_count === 0) {
            body.innerHTML = '<p>Сессия была слишком короткой для итогов.</p>';
        } else {
            const rows = [
                ['Длительность', s.duration_sec != null ? `${s.duration_sec} сек` : '—'],
                ['Замеров', s.sample_count],
                ['Средний угол осанки', s.avg_posture_angle != null ? `${s.avg_posture_angle}°` : '—'],
                ['Средние моргания', s.avg_blink_rate != null ? `${s.avg_blink_rate}/мин` : '—'],
                ['Средний PERCLOS', s.avg_perclos != null ? `${s.avg_perclos}%` : '—'],
                ['Среднее дыхание', s.avg_breath_rate != null ? `${s.avg_breath_rate}/мин` : '—'],
                ['Доминирующая эмоция', EMOTION_MAP[s.dominant_emotion] || s.dominant_emotion || '—'],
            ];
            const dist = Object.entries(s.posture_distribution || {})
                .map(([k, v]) => `<li>${k}: <b>${v}%</b></li>`).join('');
            body.innerHTML =
                '<ul class="list-unstyled mb-3">' +
                rows.map(([k, v]) => `<li><b>${k}:</b> ${v}</li>`).join('') +
                '</ul>' +
                (dist ? `<div><b>Распределение осанки:</b><ul>${dist}</ul></div>` : '');
            const exportLink = document.getElementById('summary-export-csv');
            if (exportLink && s.session_id) {
                exportLink.href = `/api/export/${s.session_id}/?format=csv`;
            }
        }
        const modalEl = document.getElementById('summary-modal');
        if (modalEl && window.bootstrap) {
            bootstrap.Modal.getOrCreateInstance(modalEl).show();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (!localStorage.getItem('veyra_tos_accepted')) {
        const tosEl = document.getElementById('tos-modal');
        if (tosEl && window.bootstrap) {
            bootstrap.Modal.getOrCreateInstance(tosEl).show();
        }
        const acc = document.getElementById('tos-accept');
        if (acc) acc.addEventListener('click', () => {
            localStorage.setItem('veyra_tos_accepted', '1');
        });
    }
    window.veyraCamera = new VeyraCamera();
});
