from fastapi import FastAPI, Request
from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from search_hint.common.websocket import asgi
from search_hint.celery import celery
from search_hint.common.settings import static_dir, templates_dir


app = FastAPI()

celery_app = celery

app.mount('/ws', asgi)

app.mount(str(static_dir), StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=str(templates_dir))

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})
