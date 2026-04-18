"""Изоляция детекторов по сессиям клиента."""

import threading
import time
import logging

logger = logging.getLogger(__name__)

SESSION_TTL_SEC = 10 * 60
GC_INTERVAL_SEC = 60

_lock = threading.Lock()
_sessions = {}
_shared_emotion = None


def _get_emotion_classifier():
    global _shared_emotion
    if _shared_emotion is None:
        from cv_processor.emotion_classifier import EmotionClassifier
        _shared_emotion = EmotionClassifier()
    return _shared_emotion


def _new_detectors():
    from cv_processor.pose_detector import PoseDetector
    from cv_processor.blink_detector import BlinkDetector
    from cv_processor.respiration import RespirationDetector

    emotion = _get_emotion_classifier()
    return {
        'pose': PoseDetector(),
        'blink': BlinkDetector(emotion_classifier=emotion),
        'respiration': RespirationDetector(),
    }


def get_detectors(session_key):
    now = time.time()
    with _lock:
        _gc(now)
        entry = _sessions.get(session_key)
        if entry is None:
            entry = {'detectors': _new_detectors(), 'last_seen': now}
            _sessions[session_key] = entry
        else:
            entry['last_seen'] = now
        return entry['detectors']


def reset_session(session_key):
    with _lock:
        _sessions.pop(session_key, None)


def _gc(now):
    stale = [k for k, v in _sessions.items() if now - v['last_seen'] > SESSION_TTL_SEC]
    for k in stale:
        _sessions.pop(k, None)
