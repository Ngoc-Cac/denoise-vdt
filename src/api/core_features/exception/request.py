from typing import Optional


class ApiError(Exception):

    def __init__(self, status_code: int, message: str):
        super().__init__()

        self.status_code = status_code
        self.message = message


class ValidationError(ApiError):

    def __init__(
        self,
        message,
        status_code: Optional[int] = 400,
        payload=None,
    ):
        super().__init__(status_code, message)
        self.payload = payload

    def to_dict(self) -> dict:
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv
