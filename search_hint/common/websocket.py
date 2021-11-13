import socketio
from search_hint.common.sio_process import HintNameSpace


mgr = socketio.AsyncRedisManager(
    "redis://localhost/1"
)
sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*", client_manager=mgr
)
sio.register_namespace(HintNameSpace(
    "/Hint"))
asgi = socketio.ASGIApp(sio)
