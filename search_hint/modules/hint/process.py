from search_hint.common.settings import model


class TextProcessor:
    model = model

    def __init__(self):
        pass

    def recognize(self, text: str) -> List[str]:
        raise Exception('Not implemented')
