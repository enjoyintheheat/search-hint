from celery import Celery


CELERY_IMPORTS = ('search_hint.tasks',)


celery = Celery(include=['search_hint.tasks'])
celery.conf.broker_url = "redis://localhost:6379"
celery.conf.result_backend = "redis://localhost:6379"
