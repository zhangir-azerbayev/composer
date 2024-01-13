from enum import Enum
from dataclasses import dataclass
from typing import Callable


class ServerType(Enum):
    VLLM = "vllm"


@dataclass
class ModelServerConfig:
    model_id: str
    server_type: ServerType = ServerType.VLLM
    port: int = 8000
    host: str = "0.0.0.0"
