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

    def _timestamp_ms(self):
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        return ts

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
        }

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        ts_ms = self._timestamp_ms()
        ts_sec = ts_ms / 1000.0
        results = self.detector.detect_for_video(mp_image, ts_ms)

        if not results.face_landmarks:
            return blink_data

        landmarks = results.face_landmarks[0]

        ear = (self._ear(LEFT_EYE_EAR, landmarks) + self._ear(RIGHT_EYE_EAR, landmarks)) / 2.0
        blink_data['ear'] = round(ear, 3)

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
