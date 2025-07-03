class Scorable:
    def __init__(self, text: str, id: str = "",  target_type: str = "custom"):
        self._id = id
        self._text = text
        self._target_type = target_type

    @property
    def text(self) -> str:
        return self._text

    @property
    def id(self) -> str:
        return self._id

    @property
    def target_type(self) -> str:
        return self._target_type
