from django.test import TestCase

from .advisor import HistoryBuffer, advice_from_metrics, fatigue_score


class FatigueScoreTests(TestCase):
    def test_fresh_state_is_zero(self):
        self.assertEqual(fatigue_score({}), 0)

    def test_high_perclos_dominates(self):
        score = fatigue_score({'perclos': 35})
        self.assertGreaterEqual(score, 40)

    def test_components_compose(self):
        score = fatigue_score({
            'perclos': 25,
            'yawn_rate': 7,
            'long_blink_count': 6,
            'head_pose': {'pitch': 35},
        })
        self.assertEqual(score, 25 + 20 + 15 + 15)

    def test_clamped_to_100(self):
        score = fatigue_score({
            'perclos': 90, 'yawn_rate': 30, 'long_blink_count': 50,
            'head_pose': {'pitch': 60}, 'emotion': 'sad', 'heart_rate': 45,
        })
        self.assertEqual(score, 100)


class AdvisorTests(TestCase):
    def test_normal_state_is_severity_zero(self):
        text, _f, sev = advice_from_metrics(
            {'posture_status': 'норма', 'blink_rate': 18, 'perclos': 5},
            HistoryBuffer(),
        )
        self.assertEqual(sev, 0)
        self.assertIn('норм', text.lower())

    def test_severe_posture_returns_severity_three(self):
        buf = HistoryBuffer()
        for _ in range(20):
            buf.push({'posture_status': 'сильный наклон'})
        text, _f, sev = advice_from_metrics(
            {'posture_status': 'сильный наклон', 'blink_rate': 18, 'perclos': 5},
            buf,
        )
        self.assertEqual(sev, 3)
        self.assertIn('давно', text)

    def test_high_fatigue_triggers_break(self):
        text, _f, sev = advice_from_metrics(
            {'perclos': 35, 'yawn_rate': 7, 'long_blink_count': 6,
             'head_pose': {'pitch': 35}},
            HistoryBuffer(),
        )
        self.assertEqual(sev, 3)
        self.assertIn('перерыв', text.lower())


class HistoryBufferTests(TestCase):
    def test_keeps_last_n(self):
        buf = HistoryBuffer(size=3)
        for i in range(10):
            buf.push({'i': i})
        self.assertEqual(len(buf), 3)
        self.assertEqual([h['i'] for h in buf.last_n(3)], [7, 8, 9])
