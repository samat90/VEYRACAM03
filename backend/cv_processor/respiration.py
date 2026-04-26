import time
import numpy as np
from collections import deque
from scipy import signal

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0

BREATH_LOW_HZ = 0.1
BREATH_HIGH_HZ = 0.5

BUFFER_MAX = 225
MIN_SAMPLES = 45
ANALYSIS_INTERVAL = 2.0


class RespirationDetector:
    """Считает частоту дыхания по ландмаркам плеч.

    Ландмарки приходят извне (от PoseDetector) — сам MediaPipe не запускаем,
    чтобы не делать вторую pose-детекцию на кадр.
    """

    def __init__(self):
        self.signal_buffer = deque(maxlen=BUFFER_MAX)
        self.time_buffer = deque(maxlen=BUFFER_MAX)
        self.breathing_rate = 0.0
        self.confidence = 0.0
        self.last_analysis = 0.0

    def pause_reset(self):
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.last_analysis = 0.0

    def update(self, landmarks):
        data = {
            'breathing_rate': self.breathing_rate,
            'phase': 'unknown',
            'confidence': self.confidence,
        }
        if landmarks is None:
            return data

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        nose = landmarks[NOSE]

        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return data

        shoulder_y = (ls.y + rs.y) / 2.0
        if nose.visibility > 0.5:
            shoulder_y = shoulder_y - nose.y

        ts_sec = time.monotonic()
        self.signal_buffer.append(shoulder_y)
        self.time_buffer.append(ts_sec)

        if len(self.signal_buffer) >= 6:
            recent = list(self.signal_buffer)[-6:]
            data['phase'] = 'inhale' if recent[-1] < recent[0] else 'exhale'

        if ts_sec - self.last_analysis > ANALYSIS_INTERVAL:
            self._analyze()
            self.last_analysis = ts_sec

        data['breathing_rate'] = self.breathing_rate
        data['confidence'] = self.confidence
        return data

    def _analyze(self):
        if len(self.signal_buffer) < MIN_SAMPLES:
            return

        times = np.array(self.time_buffer)
        values = np.array(self.signal_buffer)

        dt = np.diff(times)
        fs = 1.0 / np.median(dt) if len(dt) > 0 and np.median(dt) > 0 else 15.0

        detrended = signal.detrend(values)

        nyq = fs / 2.0
        filtered = detrended
        low = BREATH_LOW_HZ / nyq
        high = BREATH_HIGH_HZ / nyq
        if 0 < low < high < 1:
            try:
                b, a = signal.butter(2, [low, high], btype='band')
                if len(detrended) > 3 * max(len(a), len(b)):
                    filtered = signal.filtfilt(b, a, detrended)
            except (ValueError, RuntimeError):
                filtered = detrended

        std = float(np.std(filtered)) or 1e-6
        peaks, _ = signal.find_peaks(
            filtered,
            distance=max(1, int(fs * (60.0 / 30.0))),
            prominence=0.3 * std,
        )

        if len(peaks) >= 2:
            intervals = np.diff(times[peaks])
            if len(intervals) > 0:
                avg = float(np.mean(intervals))
                if 2.0 <= avg <= 10.0:
                    self.breathing_rate = round(60.0 / avg, 1)
                    rel_std = float(np.std(intervals) / avg) if avg > 0 else 1.0
                    self.confidence = round(max(0.0, min(1.0, 1.0 - rel_std)), 2)
