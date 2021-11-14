import asyncio
import search_hint.common.websocket
from search_hint.celery import celery
# from search_hint.modules.hint.process import text_processor_factory


@celery.task()
def recognize(text: str, event: str):
    # processor = text_processor_factory()

    # output = processor.recognize(text)
    output = ''

    loop = asyncio.get_event_loop()

    coroutine = search_hint.common.websocket.sio.emit(
        "message",
        data={'text': output},
        to=event,
        namespace="/Hint",
    )

    loop.run_until_complete(coroutine)
