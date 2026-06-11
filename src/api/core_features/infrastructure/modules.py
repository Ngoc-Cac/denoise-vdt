from typing import List, Any

from .app import AppModule
from .models import ModelModule


def create_modules() -> List[Any]:
    modules = [
        AppModule(),
        ModelModule(),
    ]

    return modules
