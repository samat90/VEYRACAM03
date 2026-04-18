"""Скачивание моделей MediaPipe.

Вызывается вручную (`python backend/cv_processor/download_models.py`) либо
автоматически при старте Django.
"""

import os
import sys
import urllib.request

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

def missing_models():
    return [name for name in MODELS if not os.path.exists(os.path.join(MODELS_DIR, name))]


def ensure_models(verbose=True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            continue
        if verbose:
            print(f'[MediaPipe] downloading {filename} ...', file=sys.stderr, flush=True)
        tmp = dest + '.part'
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
        except Exception as exc:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise RuntimeError(
                f'Не удалось скачать модель MediaPipe {filename} из {url}. '
                f'Проверьте интернет и запустите вручную: '
                f'python backend/cv_processor/download_models.py. Ошибка: {exc}'
            ) from exc
        if verbose:
            print(f'[MediaPipe] ok {filename}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    ensure_models(verbose=True)
