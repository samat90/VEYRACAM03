import cv2
import numpy as np
import mediapipe as mp
import os
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .filters import Vec2Filter

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
]

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0

SPINE_WARN_IN, SPINE_WARN_OUT = 12.0, 9.0
SPINE_BAD_IN, SPINE_BAD_OUT = 22.0, 18.0
SHOULDER_WARN_IN, SHOULDER_WARN_OUT = 8.0, 6.0
SHOULDER_BAD_IN, SHOULDER_BAD_OUT = 16.0, 13.0

CALIBRATION_FRAMES = 30


class PoseDetector:
    def __init__(self, model_variant='heavy'):
        candidate = f'pose_landmarker_{model_variant}.task'
        model_path = os.path.join(MODELS_DIR, candidate)
        if not os.path.exists(model_path):
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
        self._last_ts = 0

        self._filters = {
            i: Vec2Filter(freq=15.0, mincutoff=1.2, beta=0.02)
            for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE)
        }

        self.calibration_samples = deque(maxlen=CALIBRATION_FRAMES)
        self.baseline_spine_angle = None
        self.baseline_shoulder_angle = None

        self.current_status = 'норма'

    def _calibration_complete(self):
        return (
            self.baseline_spine_angle is not None
            or self.baseline_shoulder_angle is not None
        )

    def _timestamp_ms(self):
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        return ts

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        ts = self._timestamp_ms()
        results = self.detector.detect_for_video(mp_image, ts)
        if not results.pose_landmarks:
            return None, ts
        landmarks = results.pose_landmarks[0]

        for idx, flt in self._filters.items():
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                lm.x, lm.y = flt(lm.x, lm.y, ts / 1000.0)
        return landmarks, ts

    def detect_posture(self, image):
        posture_data = {
            'angle': None,
            'status': 'unknown',
            'landmarks': None,
            'mode': None,
            'confidence': 0.0,
            'calibrating': not self._calibration_complete(),
            'calibration_progress': 0.0,
            'calibration_complete': self._calibration_complete(),
        }

        landmarks, _ = self.detect(image)
        if landmarks is None:
            return posture_data
        posture_data['landmarks'] = landmarks

        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]

        if ls.visibility < 0.5 or rs.visibility < 0.5:
            return posture_data

        posture_data['confidence'] = float(min(ls.visibility, rs.visibility))

        # Бёдра должны быть ниже плеч хотя бы на 10% кадра — иначе в кадре только верх
        hips_ok = (
            lh.visibility > 0.5 and rh.visibility > 0.5
            and lh.y > ls.y + 0.10
            and rh.y > rs.y + 0.10
        )

        shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])

        if hips_ok:
            hip_mid = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
            spine = hip_mid - shoulder_mid
            n = np.linalg.norm(spine)
            if n == 0:
                return posture_data
            cos_a = np.clip(spine[1] / n, -1.0, 1.0)
            raw_angle = float(np.degrees(np.arccos(cos_a)))
            mode = 'spine'
        else:
            dx = rs.x - ls.x
            dy = rs.y - ls.y
            if abs(dx) < 0.001:
                return posture_data
            raw_angle = float(abs(np.degrees(np.arctan2(dy, dx))))
            mode = 'shoulders'

        if not self._calibration_complete():
            self.calibration_samples.append((mode, raw_angle))
            posture_data['calibrating'] = True
            posture_data['calibration_progress'] = round(
                len(self.calibration_samples) / CALIBRATION_FRAMES, 2
            )
            posture_data['angle'] = round(raw_angle, 1)
            posture_data['mode'] = mode
            posture_data['status'] = 'калибровка'
            if len(self.calibration_samples) >= CALIBRATION_FRAMES:
                self._finalize_calibration()
                posture_data['calibration_complete'] = self._calibration_complete()
            return posture_data

        posture_data['calibrating'] = False
        posture_data['calibration_complete'] = True
        posture_data['calibration_progress'] = 1.0

        baseline = (
            self.baseline_spine_angle if mode == 'spine'
            else self.baseline_shoulder_angle
        ) or 0.0
        angle = abs(raw_angle - baseline)

        posture_data['angle'] = round(angle, 1)
        posture_data['mode'] = mode
        posture_data['status'] = self._apply_hysteresis(mode, angle)
        return posture_data

    def _finalize_calibration(self):
        spine = [a for m, a in self.calibration_samples if m == 'spine']
        shoulder = [a for m, a in self.calibration_samples if m == 'shoulders']
        if spine:
            self.baseline_spine_angle = float(np.median(spine))
        if shoulder:
            self.baseline_shoulder_angle = float(np.median(shoulder))

    def _apply_hysteresis(self, mode, angle):
        if mode == 'spine':
            warn_in, warn_out = SPINE_WARN_IN, SPINE_WARN_OUT
            bad_in, bad_out = SPINE_BAD_IN, SPINE_BAD_OUT
            bad_label, warn_label = 'сильный наклон', 'небольшой наклон'
        else:
            warn_in, warn_out = SHOULDER_WARN_IN, SHOULDER_WARN_OUT
            bad_in, bad_out = SHOULDER_BAD_IN, SHOULDER_BAD_OUT
            bad_label, warn_label = 'плечи неровно', 'небольшой наклон'

        current = self.current_status

        if current == bad_label:
            if angle < bad_out:
                current = warn_label if angle >= warn_out else 'норма'
        elif current == warn_label:
            if angle >= bad_in:
                current = bad_label
            elif angle < warn_out:
                current = 'норма'
        else:
            if angle >= bad_in:
                current = bad_label
            elif angle >= warn_in:
                current = warn_label

        self.current_status = current
        return current

    def draw_landmarks(self, image, landmarks):
        h, w = image.shape[:2]
        for start, end in POSE_CONNECTIONS:
            s, e = landmarks[start], landmarks[end]
            if s.visibility > 0.5 and e.visibility > 0.5:
                cv2.line(
                    image,
                    (int(s.x * w), int(s.y * h)),
                    (int(e.x * w), int(e.y * h)),
                    (0, 0, 255), 2,
                )
        for lm in landmarks:
            if lm.visibility > 0.5:
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        return image
