from enum import Enum
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


class ServerType(Enum):
    VLLM = "vllm"


class ModelServerConfig(BaseModel):
    model_id: str
    server_type: ServerType = ServerType.VLLM
    port: int = 8000
    host: str = "0.0.0.0"
