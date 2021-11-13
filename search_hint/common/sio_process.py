from typing import List
from socketio.asyncio_namespace import AsyncNamespace
from pydantic import BaseModel, ValidationError
from search_hint.tasks.recognize import recognize


class InputPacketModel(BaseModel):
    event: str
    text: str


class OutputPacketModel(BaseModel):
    event: str
    text: List[str]


class HintNameSpace(AsyncNamespace):
    async def on_connect(self, sid, *args, **kwargs):
        pass

    async def on_disconnect(self, sid):
        pass

    async def on_message(self, environ, *args, **kwargs):
        try:
            data = args[0]
            packet = InputPacketModel(**data)

            recognize.delay(packet.text, environ)
        except ValidationError:
            return OutputPacketModel(
                event='error', text=['ValidationError']
            ).dict()
