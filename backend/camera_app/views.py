import base64
import csv
import io
import json
import logging
import time
from collections import Counter

import cv2
import numpy as np
from django.db.models import Avg, Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .advisor import HistoryBuffer, advice_from_metrics, fatigue_score
from .models import AnalysisSession, MetricSample
from .session_manager import get_detectors, reset_session

logger = logging.getLogger(__name__)

METRIC_SAMPLE_INTERVAL_SEC = 5.0
_last_sample_ts = {}
_active_session_obj = {}
_history_buffers = {}


def _serialize_landmarks(landmarks):
    if landmarks is None:
        return None
    return [
        {'x': round(lm.x, 4), 'y': round(lm.y, 4), 'v': round(lm.visibility, 2)}
        for lm in landmarks
    ]


def _ensure_session(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _get_history(session_key):
    buf = _history_buffers.get(session_key)
    if buf is None:
        buf = HistoryBuffer(size=80)
        _history_buffers[session_key] = buf
    return buf


def _get_or_create_db_session(session_key):
    sid = _active_session_obj.get(session_key)
    if sid is None:
        session = AnalysisSession.objects.create(session_key=session_key)
        sid = session.id
        _active_session_obj[session_key] = sid
    return sid


def index(request):
    _ensure_session(request)
    return render(request, 'index.html')


def history_page(request):
    _ensure_session(request)
    return render(request, 'history.html')


@require_POST
def process_frame(request):
    session_key = _ensure_session(request)
    try:
        data = json.loads(request.body)
        raw = data.get('image', '')
        if ',' not in raw:
            return JsonResponse({'success': False, 'error': 'Invalid image payload'}, status=400)
        nparr = np.frombuffer(base64.b64decode(raw.split(',', 1)[1]), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JsonResponse({'success': False, 'error': 'Cannot decode image'}, status=400)

        detectors = get_detectors(session_key)
        pose = detectors['pose']
        blink = detectors['blink']
        resp = detectors['respiration']

        posture_data = pose.detect_posture(image)
        blink_data = blink.detect_blink(image)
        respiration_data = resp.update(posture_data.get('landmarks'))

        pose_landmarks = _serialize_landmarks(posture_data.get('landmarks'))

        metrics = {
            'posture_angle': posture_data.get('angle'),
            'posture_status': posture_data.get('status', 'unknown'),
            'posture_mode': posture_data.get('mode'),
            'posture_calibrating': posture_data.get('calibrating', False),
            'posture_calibration_progress': posture_data.get('calibration_progress', 0.0),
            'posture_confidence': posture_data.get('confidence', 0.0),
            'blink_rate': blink_data.get('blink_rate', 0),
            'blink_count': blink.blink_count,
            'blink_calibrating': blink_data.get('calibrating', False),
            'blink_calibration_progress': blink_data.get('calibration_progress', 0.0),
            'perclos': blink_data.get('perclos', 0),
            'ear': blink_data.get('ear', 0),
            'long_blink_count': blink_data.get('long_blink_count', 0),
            'yawn_count': blink_data.get('yawn_count', 0),
            'yawn_rate': blink_data.get('yawn_rate', 0),
            'breath_rate': respiration_data.get('breathing_rate', 0),
            'breath_phase': respiration_data.get('phase', 'unknown'),
            'breath_confidence': respiration_data.get('confidence', 0),
            'emotion': blink_data.get('emotion', 'neutral'),
            'emotion_confidence': blink_data.get('emotion_confidence', 0),
            'head_pose': blink_data.get('head_pose'),
            'attention': blink_data.get('attention', 'unknown'),
            'heart_rate': blink_data.get('heart_rate', 0),
            'heart_rate_confidence': blink_data.get('heart_rate_confidence', 0),
        }

        history_buf = _get_history(session_key)
        history_buf.push(metrics)
        fatigue = fatigue_score(metrics)

        _persist_sample(session_key, metrics)

        advice_text = None
        advice_severity = 0
        if data.get('need_advice'):
            advice_text, _f, advice_severity = advice_from_metrics(metrics, history_buf)

        return JsonResponse({
            'success': True,
            'pose_landmarks': pose_landmarks,
            'rppg_roi': blink_data.get('rppg_roi'),
            'posture': {
                'angle': metrics['posture_angle'],
                'status': metrics['posture_status'],
                'mode': metrics['posture_mode'],
                'calibrating': metrics['posture_calibrating'],
                'calibration_complete': posture_data.get('calibration_complete', False),
                'calibration_progress': metrics['posture_calibration_progress'],
                'confidence': round(float(metrics['posture_confidence']), 2),
            },
            'blink': {
                'rate': metrics['blink_rate'],
                'count': metrics['blink_count'],
                'perclos': metrics['perclos'],
                'ear': metrics['ear'],
                'long_blinks': metrics['long_blink_count'],
                'calibrating': metrics['blink_calibrating'],
                'calibration_complete': blink_data.get('calibration_complete', False),
                'calibration_progress': metrics['blink_calibration_progress'],
                'detected': blink_data.get('blink_detected', False),
            },
            'respiration': {
                'rate': metrics['breath_rate'],
                'phase': metrics['breath_phase'],
                'confidence': metrics['breath_confidence'],
            },
            'yawn': {
                'count': metrics['yawn_count'],
                'rate_per_hour': metrics['yawn_rate'],
                'detected': blink_data.get('yawn_detected', False),
            },
            'emotion': metrics['emotion'],
            'emotion_confidence': metrics['emotion_confidence'],
            'head_pose': metrics['head_pose'],
            'attention': metrics['attention'],
            'heart_rate': metrics['heart_rate'],
            'heart_rate_confidence': metrics['heart_rate_confidence'],
            'fatigue': fatigue,
            'advice': advice_text,
            'advice_severity': advice_severity,
            'timestamp': time.time(),
        })

    except Exception as exc:
        logger.exception('process_frame failed: %s', exc)
        return JsonResponse({'success': False, 'error': str(exc)}, status=500)


@require_POST
def pause_session(request):
    """Не сбрасывает детекторы — калибровка и счётчики сохраняются."""
    return JsonResponse({'success': True})


@require_POST
def stop_session(request):
    session_key = _ensure_session(request)
    reset_session(session_key)
    _last_sample_ts.pop(session_key, None)
    _history_buffers.pop(session_key, None)
    sid = _active_session_obj.pop(session_key, None)
    summary = None
    if sid is not None:
        try:
            AnalysisSession.objects.filter(id=sid).update(ended_at=timezone.now())
            summary = _build_summary(sid)
        except Exception as exc:
            logger.warning('Failed to close session %s: %s', sid, exc)
    return JsonResponse({'success': True, 'summary': summary})


def _build_summary(session_id):
    session = AnalysisSession.objects.filter(id=session_id).first()
    if session is None:
        return None
    samples = MetricSample.objects.filter(session_id=session_id)
    if not samples.exists():
        return {'session_id': session_id, 'sample_count': 0}

    agg = samples.aggregate(
        avg_angle=Avg('posture_angle'),
        avg_blink=Avg('blink_rate'),
        avg_perclos=Avg('perclos'),
        avg_breath=Avg('breath_rate'),
        count=Count('id'),
    )

    emotion_counts = Counter(s.emotion for s in samples if s.emotion)
    dominant_emotion, _cnt = emotion_counts.most_common(1)[0] if emotion_counts else ('neutral', 0)

    status_counts = Counter(s.posture_status for s in samples if s.posture_status)
    total = sum(status_counts.values()) or 1
    posture_dist = {k: round(v / total * 100, 1) for k, v in status_counts.items()}

    duration = None
    if session.ended_at:
        duration = int((session.ended_at - session.started_at).total_seconds())

    return {
        'session_id': session_id,
        'started_at': session.started_at.isoformat(),
        'ended_at': session.ended_at.isoformat() if session.ended_at else None,
        'duration_sec': duration,
        'sample_count': agg['count'],
        'avg_posture_angle': round(agg['avg_angle'], 1) if agg['avg_angle'] is not None else None,
        'avg_blink_rate': round(agg['avg_blink'], 1) if agg['avg_blink'] is not None else None,
        'avg_perclos': round(agg['avg_perclos'], 1) if agg['avg_perclos'] is not None else None,
        'avg_breath_rate': round(agg['avg_breath'], 1) if agg['avg_breath'] is not None else None,
        'dominant_emotion': dominant_emotion,
        'posture_distribution': posture_dist,
    }


@require_GET
def session_history(request):
    session_key = _ensure_session(request)
    sid = request.GET.get('session_id')
    if sid is None:
        sid = _active_session_obj.get(session_key)
    samples = []
    if sid is not None:
        qs = MetricSample.objects.filter(session_id=sid).order_by('created_at')[:600]
        samples = [_sample_to_dict(s) for s in qs]
    return JsonResponse({'success': True, 'samples': samples, 'session_id': sid})


@require_GET
def session_list(request):
    sessions = AnalysisSession.objects.all().order_by('-started_at')[:30]
    return JsonResponse({
        'success': True,
        'sessions': [
            {
                'id': s.id,
                'started_at': s.started_at.isoformat(),
                'ended_at': s.ended_at.isoformat() if s.ended_at else None,
                'samples': s.samples.count(),
            }
            for s in sessions
        ],
    })


@require_GET
def export_session(request, session_id):
    fmt = request.GET.get('format', 'csv').lower()
    samples = MetricSample.objects.filter(session_id=session_id).order_by('created_at')
    if not samples.exists():
        return JsonResponse({'success': False, 'error': 'No data'}, status=404)

    rows = [_sample_to_dict(s) for s in samples]

    if fmt == 'json':
        resp = JsonResponse({'session_id': int(session_id), 'samples': rows}, json_dumps_params={'ensure_ascii': False})
        resp['Content-Disposition'] = f'attachment; filename="session_{session_id}.json"'
        return resp

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    resp = HttpResponse(buf.getvalue(), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = f'attachment; filename="session_{session_id}.csv"'
    return resp


def _sample_to_dict(s):
    return {
        'ts': s.created_at.isoformat(),
        'posture_angle': s.posture_angle,
        'posture_status': s.posture_status,
        'blink_rate': s.blink_rate,
        'perclos': s.perclos,
        'breath_rate': s.breath_rate,
        'emotion': s.emotion,
    }


def _persist_sample(session_key, m):
    now = time.time()
    last = _last_sample_ts.get(session_key)
    sid = _active_session_obj.get(session_key)

    if last is None and sid is None:
        # Первый кадр сессии — фиксируем started_at сейчас, а не через 5 секунд
        sid = _get_or_create_db_session(session_key)
        _last_sample_ts[session_key] = now
        _save_sample_row(sid, m)
        return

    if last is not None and now - last < METRIC_SAMPLE_INTERVAL_SEC:
        return

    _last_sample_ts[session_key] = now
    if sid is None:
        sid = _get_or_create_db_session(session_key)
    _save_sample_row(sid, m)


def _save_sample_row(sid, m):
    try:
        MetricSample.objects.create(
            session_id=sid,
            posture_angle=m.get('posture_angle'),
            posture_status=m.get('posture_status', '')[:32],
            posture_mode=m.get('posture_mode') or '',
            blink_rate=m.get('blink_rate') or 0,
            blink_count=m.get('blink_count') or 0,
            perclos=m.get('perclos') or 0,
            breath_rate=m.get('breath_rate') or 0,
            breath_phase=m.get('breath_phase') or '',
            emotion=m.get('emotion', ''),
            emotion_confidence=m.get('emotion_confidence') or 0,
        )
    except Exception as exc:
        logger.exception('persist sample failed: %s', exc)
