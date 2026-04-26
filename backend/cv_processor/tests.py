from types import SimpleNamespace

from django.test import TestCase

from .head_pose import classify_attention, estimate_head_pose, LANDMARK_IDX, MODEL_3D


def _frontal_landmarks(width=640, height=480):
    cx, cy = width / 2.0, height / 2.0
    focal = float(width)
    points = []
    for i in range(478):
        if i in LANDMARK_IDX:
            xyz = MODEL_3D[LANDMARK_IDX.index(i)]
            px = (xyz[0] * focal) / focal + cx / width
            py = (xyz[1] * focal) / focal + cy / height
            points.append(SimpleNamespace(x=px, y=py, visibility=1.0))
        else:
            points.append(SimpleNamespace(x=0.5, y=0.5, visibility=0.0))
    return points


class HeadPoseTests(TestCase):
    def test_returns_pitch_yaw_roll(self):
        landmarks = _frontal_landmarks()
        pose = estimate_head_pose(landmarks, (640, 480))
        self.assertIsNotNone(pose)
        self.assertIn('pitch', pose)
        self.assertIn('yaw', pose)
        self.assertIn('roll', pose)


class AttentionTests(TestCase):
    def test_focus_when_centered(self):
        self.assertEqual(
            classify_attention({'pitch': 0, 'yaw': 0, 'roll': 0}),
            'сосредоточен',
        )

    def test_distracted_on_large_yaw(self):
        self.assertEqual(
            classify_attention({'pitch': 0, 'yaw': 50, 'roll': 0}),
            'отвлёкся',
        )

    def test_head_down(self):
        self.assertEqual(
            classify_attention({'pitch': 30, 'yaw': 0, 'roll': 0}),
            'голова опущена',
        )

    def test_unknown_when_pose_missing(self):
        self.assertEqual(classify_attention(None), 'unknown')
