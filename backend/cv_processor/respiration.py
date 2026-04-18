import cv2
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import signal

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0

BREATH_LOW_HZ = 0.1
BREATH_HIGH_HZ = 0.5

BUFFER_MAX = 225
MIN_SAMPLES = 45
ANALYSIS_INTERVAL = 2.0


class RespirationDetector:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, 'pose_landmarker_full.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                'Модели MediaPipe не найдены. Запустите: '
                'python backend/cv_processor/download_models.py'
            )
        with open(model_path, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self._start_ms = int(time.monotonic() * 1000)

        self.signal_buffer = deque(maxlen=BUFFER_MAX)
        self.time_buffer = deque(maxlen=BUFFER_MAX)
        self.breathing_rate = 0.0
        self.confidence = 0.0
        self.last_analysis = 0.0

    def _timestamp_ms(self):
        return max(1, int(time.monotonic() * 1000) - self._start_ms)

    def detect_respiration(self, image):
        data = {
            'breathing_rate': self.breathing_rate,
            'phase': 'unknown',
            'confidence': self.confidence,
        }

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        ts_ms = self._timestamp_ms()
        ts_sec = ts_ms / 1000.0
        results = self.detector.detect_for_video(mp_image, ts_ms)

        if not results.pose_landmarks:
            return data
        landmarks = results.pose_landmarks[0]

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        nose = landmarks[NOSE]

        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return data

        shoulder_y = (ls.y + rs.y) / 2.0
        # Вычитаем движение головы, чтобы покачивания не давали ложный сигнал
        if nose.visibility > 0.5:
            shoulder_y = shoulder_y - nose.y

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
