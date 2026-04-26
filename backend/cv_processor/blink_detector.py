import cv2
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .head_pose import estimate_head_pose, classify_attention
from .rppg import RPPGDetector

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# MediaPipe FaceLandmarker возвращает 478 точек: 0–467 — лицо, 468–477 — радужки
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Точки щёк для измерения цвета кожи
LEFT_CHEEK = [50, 101, 36, 205]
RIGHT_CHEEK = [280, 330, 266, 425]

SACCADE_VELOCITY_THRESHOLD = 0.025  # нормированных единиц / кадр

MIN_BLINK_MS = 80
MAX_BLINK_MS = 400
LONG_BLINK_MS = 600

CALIBRATION_SAMPLES = 120
CALIBRATION_MIN = 30

PERCLOS_WINDOW_SEC = 60.0
EMOTION_SMOOTHING_WINDOW = 30
EMOTION_CNN_EVERY_N = 5
YAWN_THRESHOLD = 0.55
YAWN_MIN_DURATION_MS = 800


class BlinkDetector:
    def __init__(self, emotion_classifier=None):
        model_path = os.path.join(MODELS_DIR, 'face_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                'Модели MediaPipe не найдены. Запустите: '
                'python backend/cv_processor/download_models.py'
            )
        with open(model_path, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = 0

        self.blink_count = 0
        self.blink_timestamps = deque(maxlen=120)
        self.long_blink_count = 0

        self.ear_open_samples = deque(maxlen=CALIBRATION_SAMPLES)
        self.ear_threshold = None

        self.eye_closed_start = None

        self.closed_history = deque()

        self.emotion_history = deque(maxlen=EMOTION_SMOOTHING_WINDOW)

        self.yawn_count = 0
        self.yawn_timestamps = deque(maxlen=30)
        self.mouth_open_start = None

        self.emotion_classifier = emotion_classifier
        self.rppg = RPPGDetector()
        self._frame_idx = 0
        self._last_raw_emotion = ('neutral', 0.0)
        self._iris_history = deque(maxlen=60)
        self._saccade_buffer = deque(maxlen=300)
        self._cheek_redness_baseline = None
        self._cheek_redness_samples = deque(maxlen=30)

    def _timestamp_ms(self):
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        return ts

    def pause_reset(self):
        """Сбрасывает time-series буферы при паузе. Калибровка сохраняется."""
        self.blink_timestamps.clear()
        self.closed_history.clear()
        self.eye_closed_start = None
        self.mouth_open_start = None
        self.emotion_history.clear()
        self._iris_history.clear()
        self._saccade_buffer.clear()
        self._cheek_redness_samples.clear()
        self.rppg.pause_reset()

    def _update_saccades(self, landmarks, ts_sec):
        if len(landmarks) <= LEFT_IRIS_CENTER:
            return 0.0
        rx = landmarks[RIGHT_IRIS_CENTER].x
        ry = landmarks[RIGHT_IRIS_CENTER].y
        lx = landmarks[LEFT_IRIS_CENTER].x
        ly = landmarks[LEFT_IRIS_CENTER].y
        cx = (rx + lx) / 2.0
        cy = (ry + ly) / 2.0
        if self._iris_history:
            px, py, pt = self._iris_history[-1]
            dt = max(ts_sec - pt, 1e-3)
            v = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / dt
            if v > SACCADE_VELOCITY_THRESHOLD * 30:  # порог в норм. ед./сек
                self._saccade_buffer.append(ts_sec)
        self._iris_history.append((cx, cy, ts_sec))
        cutoff = ts_sec - 30.0
        while self._saccade_buffer and self._saccade_buffer[0] < cutoff:
            self._saccade_buffer.popleft()
        return len(self._saccade_buffer) * 2.0  # саккад в минуту (окно 30 сек)

    def _update_skin_color(self, image, landmarks):
        if landmarks is None:
            return 0.0
        h, w = image.shape[:2]
        samples = []
        for idx in LEFT_CHEEK + RIGHT_CHEEK:
            if idx >= len(landmarks):
                continue
            cx = int(landmarks[idx].x * w)
            cy = int(landmarks[idx].y * h)
            x1, y1 = max(0, cx - 4), max(0, cy - 4)
            x2, y2 = min(w, cx + 4), min(h, cy + 4)
            patch = image[y1:y2, x1:x2]
            if patch.size > 0:
                samples.append(patch.reshape(-1, 3).mean(axis=0))
        if not samples:
            return 0.0
        mean_bgr = np.mean(samples, axis=0)
        b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
        # Простой индекс «покраснения»: R относительно (G+B)/2
        redness = (r - (g + b) / 2.0) / max(1.0, (r + g + b) / 3.0)
        self._cheek_redness_samples.append(redness)
        if self._cheek_redness_baseline is None and len(self._cheek_redness_samples) >= 20:
            self._cheek_redness_baseline = float(
                np.median(list(self._cheek_redness_samples))
            )
        if self._cheek_redness_baseline is None:
            return 0.0
        delta = redness - self._cheek_redness_baseline
        return round(float(delta), 3)

    @staticmethod
    def _ear(eye_indices, landmarks):
        pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
        p1, p2, p3, p4, p5, p6 = pts
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        h = np.linalg.norm(p1 - p4)
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    @staticmethod
    def _mar(landmarks):
        p_left = np.array([landmarks[78].x, landmarks[78].y])
        p_right = np.array([landmarks[308].x, landmarks[308].y])
        p_top = np.array([landmarks[13].x, landmarks[13].y])
        p_bot = np.array([landmarks[14].x, landmarks[14].y])
        horiz = np.linalg.norm(p_right - p_left)
        vert = np.linalg.norm(p_top - p_bot)
        return vert / horiz if horiz > 0 else 0.0

    def _smooth_emotion(self, emo, conf):
        self.emotion_history.append((emo, conf))
        if len(self.emotion_history) < 5:
            return emo, conf
        counter = Counter(e for e, _ in self.emotion_history)
        most_common, _count = counter.most_common(1)[0]
        confs = [c for e, c in self.emotion_history if e == most_common]
        avg_conf = float(np.mean(confs)) if confs else conf
        return most_common, avg_conf

    def detect_blink(self, image):
        calibrated = self.ear_threshold is not None
        blink_data = {
            'ear': 0.0,
            'mar': 0.0,
            'blink_detected': False,
            'blink_rate': 0.0,
            'perclos': 0.0,
            'long_blink_count': self.long_blink_count,
            'emotion': 'neutral',
            'emotion_confidence': 0.0,
            'calibrating': not calibrated,
            'calibration_complete': calibrated,
            'calibration_progress': 1.0 if calibrated else 0.0,
            'threshold': None,
            'yawn_count': self.yawn_count,
            'yawn_rate': 0.0,
            'yawn_detected': False,
            'head_pose': None,
            'attention': 'unknown',
            'heart_rate': 0.0,
            'heart_rate_confidence': 0.0,
            'rppg_roi': None,
            'hrv_sdnn_ms': 0.0,
            'hrv_rmssd_ms': 0.0,
            'blink_asymmetry': 0.0,
            'cognitive_load': 0.0,
            'skin_redness': 0.0,
        }

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        ts_ms = self._timestamp_ms()
        ts_sec = ts_ms / 1000.0
        results = self.detector.detect_for_video(mp_image, ts_ms)

        if not results.face_landmarks:
            return blink_data

        landmarks = results.face_landmarks[0]

        ear_left = self._ear(LEFT_EYE_EAR, landmarks)
        ear_right = self._ear(RIGHT_EYE_EAR, landmarks)
        ear = (ear_left + ear_right) / 2.0
        blink_data['ear'] = round(ear, 3)
        blink_data['blink_asymmetry'] = round(
            abs(ear_left - ear_right) / max(ear_left, ear_right, 1e-3), 3
        )

        mar = self._mar(landmarks)
        blink_data['mar'] = round(mar, 3)

        h, w = image.shape[:2]
        pose = estimate_head_pose(landmarks, (w, h))
        if pose is not None:
            blink_data['head_pose'] = {
                'pitch': round(pose['pitch'], 1),
                'yaw': round(pose['yaw'], 1),
                'roll': round(pose['roll'], 1),
            }
            blink_data['attention'] = classify_attention(pose)

        rppg_data = self.rppg.update(image, landmarks)
        blink_data['heart_rate'] = rppg_data['heart_rate']
        blink_data['heart_rate_confidence'] = rppg_data['confidence']
        blink_data['rppg_roi'] = rppg_data.get('roi')
        blink_data['hrv_sdnn_ms'] = rppg_data.get('sdnn_ms', 0.0)
        blink_data['hrv_rmssd_ms'] = rppg_data.get('rmssd_ms', 0.0)

        blink_data['cognitive_load'] = round(
            self._update_saccades(landmarks, ts_sec), 1
        )
        blink_data['skin_redness'] = self._update_skin_color(image, landmarks)

        if self.ear_threshold is None:
            self.ear_open_samples.append(ear)
            progress = len(self.ear_open_samples) / CALIBRATION_SAMPLES
            blink_data['calibrating'] = True
            blink_data['calibration_complete'] = False
            blink_data['calibration_progress'] = round(min(1.0, progress), 2)
            if len(self.ear_open_samples) >= CALIBRATION_MIN:
                arr = np.array(self.ear_open_samples)
                open_vals = arr[arr >= np.percentile(arr, 30)]
                mean_open = float(np.mean(open_vals))
                std_open = float(np.std(open_vals)) if len(open_vals) > 1 else 0.02
                self.ear_threshold = max(0.12, mean_open - 3.5 * std_open)
                blink_data['calibration_complete'] = True
            blink_data['threshold'] = self.ear_threshold
            return blink_data

        blink_data['threshold'] = round(self.ear_threshold, 3)
        blink_data['calibrating'] = False
        blink_data['calibration_complete'] = True
        blink_data['calibration_progress'] = 1.0
        is_closed = ear < self.ear_threshold

        if is_closed and self.eye_closed_start is None:
            self.eye_closed_start = ts_ms
        elif not is_closed and self.eye_closed_start is not None:
            duration = ts_ms - self.eye_closed_start
            self.eye_closed_start = None
            if MIN_BLINK_MS <= duration <= MAX_BLINK_MS:
                self.blink_count += 1
                self.blink_timestamps.append(ts_sec)
                blink_data['blink_detected'] = True
            elif duration > LONG_BLINK_MS:
                self.long_blink_count += 1

        if len(self.blink_timestamps) >= 2:
            span = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if span > 0:
                blink_data['blink_rate'] = round(len(self.blink_timestamps) / span * 60, 1)

        self.closed_history.append((ts_sec, is_closed))
        cutoff = ts_sec - PERCLOS_WINDOW_SEC
        while self.closed_history and self.closed_history[0][0] < cutoff:
            self.closed_history.popleft()
        if len(self.closed_history) > 1:
            closed_count = sum(1 for _, c in self.closed_history if c)
            blink_data['perclos'] = round(closed_count / len(self.closed_history) * 100, 1)

        if mar > YAWN_THRESHOLD and self.mouth_open_start is None:
            self.mouth_open_start = ts_ms
        elif mar <= YAWN_THRESHOLD and self.mouth_open_start is not None:
            duration = ts_ms - self.mouth_open_start
            self.mouth_open_start = None
            if duration >= YAWN_MIN_DURATION_MS:
                self.yawn_count += 1
                self.yawn_timestamps.append(ts_sec)
                blink_data['yawn_detected'] = True

        if self.yawn_timestamps:
            recent = [t for t in self.yawn_timestamps if ts_sec - t < 600]
            blink_data['yawn_rate'] = len(recent) * 6

        blink_data['long_blink_count'] = self.long_blink_count
        blink_data['yawn_count'] = self.yawn_count

        self._frame_idx += 1
        if self.emotion_classifier is not None and self.emotion_classifier.available:
            if self._frame_idx % EMOTION_CNN_EVERY_N == 0 or self._last_raw_emotion[1] == 0.0:
                self._last_raw_emotion = self.emotion_classifier.classify(image, landmarks)
            raw_emo, raw_conf = self._last_raw_emotion
        elif results.face_blendshapes:
            raw_emo, raw_conf = self._classify_emotion_blendshapes(results.face_blendshapes[0])
        else:
            raw_emo, raw_conf = 'neutral', 0.0

        smooth_emo, smooth_conf = self._smooth_emotion(raw_emo, raw_conf)
        blink_data['emotion'] = smooth_emo
        blink_data['emotion_confidence'] = round(float(smooth_conf), 3)

        return blink_data

    @staticmethod
    def _classify_emotion_blendshapes(blendshapes):
        bs = {b.category_name: b.score for b in blendshapes}
        smile = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
        brow_down = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2
        brow_up = (
            bs.get('browInnerUp', 0)
            + bs.get('browOuterUpLeft', 0)
            + bs.get('browOuterUpRight', 0)
        ) / 3
        mouth_open = bs.get('jawOpen', 0)
        mouth_frown = (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0)) / 2

        scores = {
            'happy': smile,
            'surprised': brow_up * 0.6 + mouth_open * 0.4,
            'angry': brow_down * 0.6 + mouth_frown * 0.4,
            'sad': mouth_frown * 0.7 + brow_down * 0.3,
        }
        best = max(scores, key=scores.get)
        if scores[best] > 0.15:
            return best, float(scores[best])
        return 'neutral', 0.5
