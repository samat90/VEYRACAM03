from django.apps import AppConfig


class CvProcessorConfig(AppConfig):
    name = 'cv_processor'

    def ready(self):
        from .download_models import ensure_models, missing_models
        if missing_models():
            ensure_models(verbose=True)
