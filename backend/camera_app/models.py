from django.db import models


class AnalysisSession(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'Session {self.id} ({self.session_key[:8]})'


class MetricSample(models.Model):
    session = models.ForeignKey(
        AnalysisSession, related_name='samples', on_delete=models.CASCADE
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    posture_angle = models.FloatField(null=True, blank=True)
    posture_status = models.CharField(max_length=32, blank=True)
    posture_mode = models.CharField(max_length=16, blank=True)

    blink_rate = models.FloatField(default=0)
    blink_count = models.IntegerField(default=0)
    perclos = models.FloatField(default=0)

    breath_rate = models.FloatField(default=0)
    breath_phase = models.CharField(max_length=16, blank=True)

    emotion = models.CharField(max_length=16, blank=True)
    emotion_confidence = models.FloatField(default=0)

    class Meta:
        ordering = ['-created_at']


class SelfReport(models.Model):
    """Самооценка пользователя 1..5 для валидации измерений."""

    session = models.ForeignKey(
        AnalysisSession, related_name='reports', on_delete=models.CASCADE
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    feeling = models.IntegerField()
    fatigue_at_report = models.FloatField(null=True, blank=True)
    stress_at_report = models.FloatField(null=True, blank=True)
    note = models.CharField(max_length=120, blank=True)
