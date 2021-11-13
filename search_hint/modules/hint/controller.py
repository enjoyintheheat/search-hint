from fastapi import APIRouter


router = APIRouter(prefix='hints', tags=['hints'])


@router.get()
async def get_hint(text: str):
    return {'hints': ['text']}
