from fastapi import FastAPI, Request
from search_hint.common.websocket import asgi
from search_hint.celery import celery


app = FastAPI()

celery_app = celery

app.mount('/', asgi)


@app.get('/')
async def index(request: Request):
    return {'check': True}
