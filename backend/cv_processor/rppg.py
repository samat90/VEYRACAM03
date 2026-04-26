"""Оценка пульса по ROI лба методом CHROM."""

import logging
import time
import numpy as np
import cv2
from collections import deque
from scipy import signal

logger = logging.getLogger(__name__)

FOREHEAD_LANDMARKS = [10, 67, 109, 338, 297, 151]

HR_LOW_HZ = 0.7
HR_HIGH_HZ = 3.0

BUFFER_SEC = 12.0
MIN_SAMPLES_FOR_HR = 6.0


class RPPGDetector:
    def __init__(self):
        self.r_buffer = deque()
        self.g_buffer = deque()
        self.b_buffer = deque()
        self.t_buffer = deque()
        self.heart_rate = 0.0
        self.confidence = 0.0
        self.last_analysis = 0.0
        self.sdnn_ms = 0.0
        self.rmssd_ms = 0.0

    def pause_reset(self):
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.t_buffer.clear()
        self.last_analysis = 0.0
        self.sdnn_ms = 0.0
        self.rmssd_ms = 0.0

    def _trim(self, now):
        cutoff = now - BUFFER_SEC
        while self.t_buffer and self.t_buffer[0] < cutoff:
            self.t_buffer.popleft()
            self.r_buffer.popleft()
            self.g_buffer.popleft()
            self.b_buffer.popleft()

    def update(self, bgr_image, face_landmarks):
        data = {
            'heart_rate': self.heart_rate,
            'confidence': self.confidence,
            'ready': False,
            'roi': None,
        }
        if face_landmarks is None:
            return data

        h, w = bgr_image.shape[:2]
        xs = [face_landmarks[i].x * w for i in FOREHEAD_LANDMARKS]
        ys = [face_landmarks[i].y * h for i in FOREHEAD_LANDMARKS]

        x1 = max(0, int(min(xs)))
        x2 = min(w, int(max(xs)))
        y1 = max(0, int(min(ys)) - int(0.05 * h))
        y2 = min(h, int(max(ys)))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return data

        data['roi'] = {
            'x': round(x1 / w, 4),
            'y': round(y1 / h, 4),
            'w': round((x2 - x1) / w, 4),
            'h': round((y2 - y1) / h, 4),
        }

        roi = bgr_image[y1:y2, x1:x2]
        if roi.size == 0:
            return data

        b_mean, g_mean, r_mean = cv2.mean(roi)[:3]

        now = time.monotonic()
        self.r_buffer.append(r_mean)
        self.g_buffer.append(g_mean)
        self.b_buffer.append(b_mean)
        self.t_buffer.append(now)
        self._trim(now)

        span = (self.t_buffer[-1] - self.t_buffer[0]) if len(self.t_buffer) > 1 else 0
        if span >= MIN_SAMPLES_FOR_HR and now - self.last_analysis > 1.5:
            self._analyze()
            self.last_analysis = now

        data['heart_rate'] = self.heart_rate
        data['confidence'] = self.confidence
        data['ready'] = self.heart_rate > 0
        data['sdnn_ms'] = self.sdnn_ms
        data['rmssd_ms'] = self.rmssd_ms
        return data

    def _analyze(self):
        t = np.array(self.t_buffer)
        r = np.array(self.r_buffer)
        g = np.array(self.g_buffer)
        b = np.array(self.b_buffer)

        if len(t) < 30:
            return

        dt = np.diff(t)
        if len(dt) == 0:
            return
        fs = 1.0 / max(np.median(dt), 1e-3)
        if fs < 5 or fs > 120:
            return

        def norm(x):
            m = np.mean(x)
            return (x - m) / m if m > 1 else x

        rn, gn, bn = norm(r), norm(g), norm(b)
        x = 3 * rn - 2 * gn
        y = 1.5 * rn + gn - 1.5 * bn

        nyq = fs / 2.0
        low = HR_LOW_HZ / nyq
        high = HR_HIGH_HZ / nyq
        if not (0 < low < high < 1):
            return
        try:
            b_coef, a_coef = signal.butter(3, [low, high], btype='band')
            if len(x) < 3 * max(len(a_coef), len(b_coef)):
                return
            xf = signal.filtfilt(b_coef, a_coef, x)
            yf = signal.filtfilt(b_coef, a_coef, y)
        except (ValueError, RuntimeError) as exc:
            logger.debug('rPPG filter failed (fs=%.1f, n=%d): %s', fs, len(x), exc)
            return

        alpha = np.std(xf) / (np.std(yf) + 1e-8)
        s = xf - alpha * yf

        n = len(s)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(s * np.hanning(n)))
        mask = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
        if not np.any(mask):
            return

        band = spectrum[mask]
        band_freqs = freqs[mask]
        peak = int(np.argmax(band))
        peak_power = float(band[peak])
        mean_power = float(np.mean(band))
        snr = peak_power / (mean_power + 1e-8)

        hr_bpm = float(band_freqs[peak]) * 60.0

        if 42 <= hr_bpm <= 180:
            if self.heart_rate == 0:
                self.heart_rate = round(hr_bpm, 0)
            else:
                self.heart_rate = round(0.7 * self.heart_rate + 0.3 * hr_bpm, 0)
            self.confidence = round(min(1.0, max(0.0, (snr - 1.5) / 4.0)), 2)
            self._estimate_hrv(s, fs)

    def _estimate_hrv(self, signal_1d, fs):
        """Inter-beat intervals из пиков отфильтрованного сигнала → SDNN, RMSSD."""
        nyq = fs / 2.0
        if nyq <= 0:
            return
        # Грубая оценка ожидаемого периода удара по уже найденному HR
        if self.heart_rate <= 0:
            return
        period_sec = 60.0 / self.heart_rate
        min_distance = max(1, int(period_sec * fs * 0.6))
        try:
            peaks, _ = signal.find_peaks(signal_1d, distance=min_distance)
        except (ValueError, RuntimeError):
            return
        if len(peaks) < 4:
            return
        ibi_sec = np.diff(peaks) / fs
        ibi_sec = ibi_sec[(ibi_sec > 0.33) & (ibi_sec < 1.5)]  # 40–180 BPM физиологично
        if len(ibi_sec) < 3:
            return
        ibi_ms = ibi_sec * 1000.0
        sdnn = float(np.std(ibi_ms))
        diffs = np.diff(ibi_ms)
        rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0
        self.sdnn_ms = round(sdnn, 1)
        self.rmssd_ms = round(rmssd, 1)
