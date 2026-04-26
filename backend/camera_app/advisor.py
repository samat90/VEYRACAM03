"""Формирование рекомендаций на основе метрик и их трендов."""

from collections import deque


def stress_score(metrics):
    """
    Композит стресса 0..100. Параллелен fatigue, но измеряет другое:
    HRV низкая, дыхание учащённое, негативная эмоция, поза нестабильна — всё про стресс.
    """
    rmssd = metrics.get('hrv_rmssd_ms') or 0
    breath_rate = metrics.get('breath_rate') or 0
    emotion = metrics.get('emotion', 'neutral')
    hr = metrics.get('heart_rate') or 0
    stability = metrics.get('stability_std') or 0
    redness = metrics.get('skin_redness') or 0

    score = 0.0
    # Низкая HRV (RMSSD < 20 мс) — сильный сигнал стресса
    if 0 < rmssd < 15:
        score += 35
    elif 0 < rmssd < 25:
        score += 20
    elif 0 < rmssd < 40:
        score += 10

    # Учащённое дыхание
    if breath_rate > 24:
        score += 20
    elif breath_rate > 18:
        score += 10

    # Повышенный пульс
    if hr > 100:
        score += 15
    elif hr > 85:
        score += 8

    # Негативная эмоция
    if emotion == 'angry':
        score += 15
    elif emotion == 'sad':
        score += 8

    # Нестабильная поза
    if stability > 4.0:
        score += 10
    elif stability > 2.5:
        score += 5

    # Покраснение кожи
    if redness > 0.05:
        score += 7

    return int(min(100, score))


def fatigue_score(metrics):
    perclos = metrics.get('perclos') or 0
    yawn_rate = metrics.get('yawn_rate') or 0
    long_blinks = metrics.get('long_blink_count') or 0
    head_pose = metrics.get('head_pose') or {}
    pitch = abs(head_pose.get('pitch', 0)) if head_pose else 0
    emotion = metrics.get('emotion', 'neutral')
    hr = metrics.get('heart_rate') or 0

    score = 0.0
    if perclos > 30:
        score += 40
    elif perclos > 20:
        score += 25
    elif perclos > 15:
        score += 12

    if yawn_rate > 6:
        score += 20
    elif yawn_rate > 3:
        score += 10

    if long_blinks > 5:
        score += 15
    elif long_blinks > 2:
        score += 8

    if pitch > 30:
        score += 15
    elif pitch > 20:
        score += 7

    if emotion == 'sad':
        score += 8

    if 40 < hr < 55:
        score += 5

    return int(min(100, score))


class HistoryBuffer:
    def __init__(self, size=60):
        self.buf = deque(maxlen=size)

    def push(self, metrics):
        self.buf.append(metrics)

    def last_n(self, n):
        return list(self.buf)[-n:]

    def __len__(self):
        return len(self.buf)


def _trend(history, key):
    vals = [h.get(key) for h in history if h.get(key) is not None]
    if len(vals) < 6:
        return 'stable'
    first = sum(vals[: len(vals) // 2]) / (len(vals) // 2)
    second = sum(vals[len(vals) // 2:]) / (len(vals) - len(vals) // 2)
    diff = second - first
    threshold = 0.15 * max(1, abs(first))
    if diff > threshold:
        return 'rising'
    if diff < -threshold:
        return 'falling'
    return 'stable'


def _duration_of_status(history, key, value):
    count = 0
    for h in reversed(history):
        if h.get(key) == value:
            count += 1
        else:
            break
    return count


def advice_from_metrics(metrics, history_buf=None):
    """Возвращает (text, fatigue, severity 0..3) на основе метрик и истории."""
    posture_status = metrics.get('posture_status', '')
    blink_rate = metrics.get('blink_rate') or 0
    perclos = metrics.get('perclos') or 0
    breath_rate = metrics.get('breath_rate') or 0
    emotion = metrics.get('emotion', 'neutral')
    hr = metrics.get('heart_rate') or 0
    attention = metrics.get('attention', '')
    yawn_rate = metrics.get('yawn_rate') or 0
    fatigue = fatigue_score(metrics)

    history = history_buf.last_n(40) if history_buf else []
    issues = []

    if fatigue >= 70:
        issues.append((3, f'Сильная усталость ({fatigue}%). Сделайте перерыв на 10–15 минут.'))
    elif fatigue >= 45:
        issues.append((2, f'Признаки утомления ({fatigue}%). Разомните глаза и плечи.'))

    if perclos > 30:
        issues.append((3, 'Глаза закрываются надолго — признак сонливости.'))
    elif perclos > 18 and _trend(history, 'perclos') == 'rising':
        issues.append((2, 'PERCLOS растёт — усталость накапливается.'))

    bad_duration = _duration_of_status(history, 'posture_status', 'сильный наклон')
    warn_duration = _duration_of_status(history, 'posture_status', 'небольшой наклон')

    if posture_status == 'сильный наклон':
        if bad_duration > 10:
            issues.append((3, 'Сильный наклон уже давно — выпрямитесь.'))
        else:
            issues.append((2, 'Выпрямите спину, плечи расслаблены.'))
    elif posture_status == 'плечи неровно':
        issues.append((2, 'Плечи перекошены — выровняйте.'))
    elif posture_status == 'небольшой наклон' and warn_duration > 15:
        issues.append((1, 'Небольшой наклон затянулся — подправьте позу.'))

    if blink_rate > 0:
        if blink_rate < 6:
            issues.append((2, 'Очень мало морганий — посмотрите вдаль 20 секунд.'))
        elif blink_rate > 35:
            issues.append((1, 'Частое моргание — возможно, глаза устали.'))

    if yawn_rate > 6:
        issues.append((2, 'Много зевков — нужен воздух и движение.'))

    if breath_rate > 22:
        issues.append((2, 'Дыхание учащённое — сделайте 3 глубоких вдоха.'))
    elif 0 < breath_rate < 8:
        issues.append((1, 'Дыхание редкое — проверьте самочувствие.'))

    if hr > 100:
        issues.append((2, f'Пульс повышен ({int(hr)} уд/мин).'))
    elif 0 < hr < 50:
        issues.append((1, f'Низкий пульс ({int(hr)} уд/мин).'))

    if attention in ('отвлёкся', 'смотрит в сторону'):
        att_duration = _duration_of_status(history, 'attention', attention)
        if att_duration > 10:
            issues.append((2, 'Внимание рассеяно — вернитесь к задаче.'))

    if emotion == 'angry':
        issues.append((1, 'Напряжённое выражение — сделайте паузу.'))
    elif emotion == 'sad':
        issues.append((1, 'Выглядите уставшим — возможно, пора отдохнуть.'))

    stress = stress_score(metrics)
    if stress >= 70:
        issues.append((3, f'Высокий уровень стресса ({stress}%) — сделайте дыхательную практику.'))
    elif stress >= 50:
        issues.append((2, f'Уровень стресса повышен ({stress}%). Расслабьтесь, потянитесь.'))

    if not issues:
        good_posture_streak = _duration_of_status(history, 'posture_status', 'норма')
        if good_posture_streak > 30:
            return ('Отличная осанка держится уже долго — продолжайте.', fatigue, 0)
        return ('Всё в норме — можно продолжать работу.', fatigue, 0)

    issues.sort(key=lambda x: -x[0])
    texts = [t for _, t in issues[:2]]
    return (' '.join(texts), fatigue, issues[0][0])
