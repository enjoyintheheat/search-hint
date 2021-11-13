import asyncio
import search_hint.common.websocket
from search_hint.celery import celery


@celery.task()
def recognize(text: str, event: str):
    loop = asyncio.get_event_loop()

    coroutine = search_hint.common.websocket.sio.emit(
        "message",
        data={'text': list(text)},
        to=event,
        namespace="/Hint",
    )

    loop.run_until_complete(coroutine)
