"""
Замер реальных метрик детекторов на живом видео из веб-камеры.

Запуск:
    .venv\\Scripts\\python.exe scripts\\benchmark.py

Сядь перед камерой на 15 секунд, смотри в объектив.
Скрипт прогонит все детекторы и выдаст усреднённые результаты.
"""

import os
import sys
import time
import statistics

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

import django
django.setup()

from cv_processor.pose_detector import PoseDetector
from cv_processor.blink_detector import BlinkDetector
from cv_processor.respiration import RespirationDetector
from cv_processor.emotion_classifier import EmotionClassifier

DURATION_SEC = 15


def main():
    print(f'Смотри в камеру {DURATION_SEC} секунд...')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Камера недоступна')
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    time.sleep(1.0)
    for _ in range(5):
        cap.read()

    emo = EmotionClassifier()
    print(f'HSEmotion доступен: {emo.available}')

    pose = PoseDetector()
    blink = BlinkDetector(emotion_classifier=emo)
    resp = RespirationDetector()

    records = {
        'pose_times_ms': [],
        'blink_times_ms': [],
        'resp_times_ms': [],
        'pose_angles': [],
        'ear_values': [],
        'blink_rates': [],
        'perclos_values': [],
        'head_pitches': [],
        'head_yaws': [],
        'head_rolls': [],
        'emotions': [],
        'emotion_confs': [],
        'heart_rates': [],
        'breath_rates': [],
        'breath_confs': [],
    }

    faces_detected = 0
    poses_detected = 0
    frames_total = 0
    start = time.monotonic()

    while time.monotonic() - start < DURATION_SEC:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames_total += 1

        t0 = time.perf_counter()
        p = pose.detect_posture(frame)
        records['pose_times_ms'].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        b = blink.detect_blink(frame)
        records['blink_times_ms'].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        r = resp.detect_respiration(frame)
        records['resp_times_ms'].append((time.perf_counter() - t0) * 1000)

        if p.get('landmarks') is not None:
            poses_detected += 1
        if p.get('angle') is not None and not p.get('calibrating'):
            records['pose_angles'].append(p['angle'])

        if b.get('ear', 0) > 0:
            faces_detected += 1
            records['ear_values'].append(b['ear'])
        if b.get('blink_rate', 0) > 0:
            records['blink_rates'].append(b['blink_rate'])
        if b.get('perclos') is not None:
            records['perclos_values'].append(b['perclos'])
        hp = b.get('head_pose')
        if hp:
            records['head_pitches'].append(hp['pitch'])
            records['head_yaws'].append(hp['yaw'])
            records['head_rolls'].append(hp['roll'])
        if b.get('emotion'):
            records['emotions'].append(b['emotion'])
            records['emotion_confs'].append(b.get('emotion_confidence', 0))
        if b.get('heart_rate', 0) > 0:
            records['heart_rates'].append(b['heart_rate'])

        if r.get('breathing_rate', 0) > 0:
            records['breath_rates'].append(r['breathing_rate'])
            records['breath_confs'].append(r.get('confidence', 0))

    cap.release()

    elapsed = time.monotonic() - start
    fps = frames_total / elapsed if elapsed > 0 else 0

    def fmt(vals, prec=2):
        if not vals:
            return '—'
        return f'mean={statistics.mean(vals):.{prec}f} min={min(vals):.{prec}f} max={max(vals):.{prec}f} n={len(vals)}'

    print()
    print('=' * 60)
    print(f'ОБЩЕЕ: {frames_total} кадров за {elapsed:.1f} с → {fps:.1f} FPS')
    print(f'  Поза обнаружена:  {poses_detected}/{frames_total}')
    print(f'  Лицо обнаружено:  {faces_detected}/{frames_total}')
    print()
    print('ПРОИЗВОДИТЕЛЬНОСТЬ (мс/кадр):')
    print(f'  Pose:        {fmt(records["pose_times_ms"], 1)}')
    print(f'  Face/Blink:  {fmt(records["blink_times_ms"], 1)}')
    print(f'  Respiration: {fmt(records["resp_times_ms"], 1)}')
    print()
    print('МЕТРИКИ:')
    print(f'  Угол осанки (°):    {fmt(records["pose_angles"])}')
    print(f'  EAR:                {fmt(records["ear_values"], 3)}')
    print(f'  Частота морганий:   {fmt(records["blink_rates"], 1)}')
    print(f'  PERCLOS (%):        {fmt(records["perclos_values"], 1)}')
    print(f'  Моргания всего:     {blink.blink_count}')
    print(f'  Зевков:             {blink.yawn_count}')
    print(f'  Head pitch (°):     {fmt(records["head_pitches"], 1)}')
    print(f'  Head yaw (°):       {fmt(records["head_yaws"], 1)}')
    print(f'  Head roll (°):      {fmt(records["head_rolls"], 1)}')
    print(f'  Пульс (уд/мин):     {fmt(records["heart_rates"], 0)}')
    print(f'  Дыхание (ц/мин):    {fmt(records["breath_rates"], 1)}')
    print(f'  Confidence дыхания: {fmt(records["breath_confs"], 2)}')
    if records['emotions']:
        from collections import Counter
        dist = Counter(records['emotions']).most_common()
        total = len(records['emotions'])
        print('  Распределение эмоций:')
        for emo_name, cnt in dist:
            avg_conf = statistics.mean(
                c for e, c in zip(records['emotions'], records['emotion_confs'])
                if e == emo_name
            )
            print(f'    {emo_name}: {cnt / total * 100:.1f}% (conf={avg_conf:.2f})')
    print()
    print('КАЛИБРОВКА:')
    print(f'  Осанка: baseline_spine={pose.baseline_spine_angle} '
          f'baseline_shoulder={pose.baseline_shoulder_angle}')
    print(f'  Моргания: ear_threshold={blink.ear_threshold}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
