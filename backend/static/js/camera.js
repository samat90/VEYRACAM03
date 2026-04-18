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

class VeyraCamera {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlay = document.getElementById('overlay');
        this.octx = this.overlay ? this.overlay.getContext('2d') : null;
        this.calibrationOverlay = document.getElementById('calibration-overlay');
        this.calibrationBar = document.getElementById('calibration-bar');

        this.isStreaming = false;
        this.stream = null;
        this.frameCount = 0;
        this.inFlight = false;
        this.bindEvents();
    }

    bindEvents() {
        document.getElementById('start-btn').addEventListener('click', () => this.startCamera());
        document.getElementById('stop-btn').addEventListener('click', () => this.stopCamera());
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
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('camera-status').className = 'badge bg-success';
            document.getElementById('camera-status').textContent = 'Активна';
            document.getElementById('resolution').textContent =
                `${this.video.videoWidth}x${this.video.videoHeight}`;

            if (this.calibrationOverlay) this.calibrationOverlay.style.display = '';

            this.isStreaming = true;
            this.startProcessing();
        } catch (err) {
            console.error('Camera access error:', err);
            alert('Не удалось получить доступ к камере. Проверьте разрешения.');
        }
    }

    async stopCamera() {
        this.isStreaming = false;
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('camera-status').className = 'badge bg-secondary';
        document.getElementById('camera-status').textContent = 'Выключена';
        document.getElementById('face-status').className = 'badge bg-secondary';
        document.getElementById('face-status').textContent = '—';
        document.getElementById('pose-status').className = 'badge bg-secondary';
        document.getElementById('pose-status').textContent = '—';

        if (this.calibrationOverlay) this.calibrationOverlay.style.display = 'none';

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

        const loop = async () => {
            if (!this.isStreaming) return;
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
            requestAnimationFrame(loop);
        };
        loop();
    }

    async processSingleFrame() {
        this.inFlight = true;
        this.frameCount++;
        const needAdvice = this.frameCount % 120 === 0;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const imageData = this.canvas.toDataURL('image/jpeg', 0.75);

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
            });
            const data = await response.json();
            if (data.success) {
                this.renderProcessedImage(data.processed_image);
                this.renderMetrics(data);
                if (data.advice) {
                    document.getElementById('advice-text').textContent = data.advice;
                    const card = document.getElementById('advice-card');
                    if (card) {
                        card.classList.remove('severity-0','severity-1','severity-2','severity-3');
                        card.classList.add('severity-' + (data.advice_severity || 0));
                    }
                }
            }
        } catch (err) {
            console.error('Frame processing error:', err);
        } finally {
            this.inFlight = false;
        }
    }

    renderProcessedImage(b64) {
        if (!b64 || !this.octx) return;
        const img = new Image();
        img.onload = () => {
            this.octx.clearRect(0, 0, this.overlay.width, this.overlay.height);
            this.octx.drawImage(img, 0, 0, this.overlay.width, this.overlay.height);
        };
        img.src = 'data:image/jpeg;base64,' + b64;
    }

    updateCalibration(data) {
        const posture = data.posture || {};
        const blink = data.blink || {};
        const postureProg = posture.calibration_progress || 0;
        const blinkProg = blink.calibration_progress || 0;
        const minProg = Math.min(postureProg, blinkProg);
        const stillCalibrating = posture.calibrating || blink.calibrating;

        if (this.calibrationBar) {
            this.calibrationBar.style.width = Math.round(minProg * 100) + '%';
        }
        if (this.calibrationOverlay) {
            this.calibrationOverlay.style.display = stillCalibrating ? '' : 'none';
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
        } else {
            blinkEl.textContent = Math.round(blink.rate || 0);
        }
        const perclosEl = document.getElementById('perclos-value');
        if (perclosEl) perclosEl.textContent = `${blink.perclos || 0}%`;

        const breath = data.respiration || {};
        document.getElementById('breath-value').textContent =
            breath.rate > 0 ? Math.round(breath.rate) : '—';

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

        const emotion = data.emotion || 'neutral';
        document.getElementById('emotion-value').textContent = EMOTION_MAP[emotion] || emotion;
        const conf = Math.round((data.emotion_confidence || 0) * 100);
        const emoDetail = document.getElementById('emotion-detail');
        if (emoDetail) emoDetail.textContent = `уверенность ${conf}%`;

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

        const fatigue = data.fatigue || 0;
        const fVal = document.getElementById('fatigue-value');
        const fBar = document.getElementById('fatigue-bar');
        fVal.textContent = fatigue;
        fBar.style.width = fatigue + '%';
        fBar.className = 'progress-bar ' + (
            fatigue >= 70 ? 'bg-danger' :
            fatigue >= 45 ? 'bg-warning' : 'bg-success'
        );

        document.getElementById('face-status').className = 'badge bg-success';
        document.getElementById('face-status').textContent = 'Активно';
        document.getElementById('pose-status').className = 'badge bg-success';
        document.getElementById('pose-status').textContent = 'Активно';
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
        }
        const modalEl = document.getElementById('summary-modal');
        if (modalEl && window.bootstrap) {
            bootstrap.Modal.getOrCreateInstance(modalEl).show();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.veyraCamera = new VeyraCamera();
});
