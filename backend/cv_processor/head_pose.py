"""Оценка ориентации головы (pitch, yaw, roll) по ландмаркам лица."""

import numpy as np
import cv2

MODEL_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -63.6,  -12.5],
    [-43.3,  32.7,  -26.0],
    [43.3,   32.7,  -26.0],
    [-28.9, -28.9,  -24.1],
    [28.9,  -28.9,  -24.1],
], dtype=np.float64)

LANDMARK_IDX = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(landmarks, image_size):
    w, h = image_size
    pts_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in LANDMARK_IDX],
        dtype=np.float64,
    )

    focal = float(w)
    camera_matrix = np.array(
        [[focal, 0, w / 2.0],
         [0, focal, h / 2.0],
         [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros(4)

    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_3D, pts_2d, camera_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw = np.arctan2(-rot_mat[2, 0], sy)
        roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        yaw = np.arctan2(-rot_mat[2, 0], sy)
        roll = 0.0

    return {
        'pitch': float(np.degrees(pitch)),
        'yaw': float(np.degrees(yaw)),
        'roll': float(np.degrees(roll)),
    }


def classify_attention(pose, pitch_thresh=25.0, yaw_thresh=30.0):
    if pose is None:
        return 'unknown'
    if abs(pose['pitch']) > 40 or abs(pose['yaw']) > 45:
        return 'отвлёкся'
    if pose['pitch'] > pitch_thresh:
        return 'голова опущена'
    if abs(pose['yaw']) > yaw_thresh:
        return 'смотрит в сторону'
    return 'сосредоточен'
