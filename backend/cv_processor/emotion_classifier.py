"""Классификация эмоций через hsemotion-onnx."""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_HSEMOTION_TO_UI = {
    'Happiness': 'happy',
    'Surprise': 'surprised',
    'Anger': 'angry',
    'Sadness': 'sad',
    'Fear': 'sad',
    'Disgust': 'angry',
    'Neutral': 'neutral',
    'Contempt': 'angry',
}


class EmotionClassifier:
    def __init__(self):
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            self._recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
            self._available = True
        except Exception as exc:
            logger.warning('HSEmotion unavailable (%s), falling back to blendshapes', exc)
            self._recognizer = None
            self._available = False

    @property
    def available(self):
        return self._available

    def classify(self, bgr_image, face_landmarks):
        if not self._available:
            return 'neutral', 0.0

        try:
            import cv2
            h, w = bgr_image.shape[:2]
            xs = [lm.x for lm in face_landmarks]
            ys = [lm.y for lm in face_landmarks]
            x1 = max(0, int(min(xs) * w) - 10)
            y1 = max(0, int(min(ys) * h) - 10)
            x2 = min(w, int(max(xs) * w) + 10)
            y2 = min(h, int(max(ys) * h) + 10)

            if x2 <= x1 or y2 <= y1:
                return 'neutral', 0.0

            face = bgr_image[y1:y2, x1:x2]
            if face.size == 0:
                return 'neutral', 0.0

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            label, scores = self._recognizer.predict_emotions(face_rgb, logits=False)

            ui = _HSEMOTION_TO_UI.get(label, 'neutral')
            confidence = float(np.max(scores)) if hasattr(scores, '__len__') else 0.5
            return ui, confidence
        except Exception as exc:
            logger.exception('Emotion classify failed: %s', exc)
            return 'neutral', 0.0
