"""
Скачивает модели MediaPipe Tasks API.
Запустить: python backend/cv_processor/download_models.py
"""
import urllib.request
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

MODELS = {
    'pose_landmarker_full.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
    ),
    'pose_landmarker_heavy.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
    ),
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/'
        'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    ),
}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f'[skip] {filename} already present')
            continue
        print(f'[download] {filename} ...')
        urllib.request.urlretrieve(url, dest)
        print(f'[ok] {filename}')


if __name__ == '__main__':
    download_models()
