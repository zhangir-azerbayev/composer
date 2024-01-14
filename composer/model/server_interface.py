from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import *
from enum import Enum
import requests

from pydantic import BaseModel


class ServerType(Enum):
    VLLM = "vllm"
    OPENAI = "openai"


class ModelServerConfig(BaseModel):
    model_id: Optional[str]
    server_type: ServerType = ServerType.VLLM
    port: int = 8000
    host: str = "0.0.0.0"

class GenerateRequest(BaseModel):
    """
    openai-style interface (RIP completion api). 
    """
    model_id: Optional[str]
    """
    Some backends require passing model_id on server startup (e.g vLLM), others
        require doing so on each request (e.g OpenAI)
    """
    prompt: Union[str, List[str]]
    n: int = 1
    best_of: Optional[int]
    echo: bool = True
    """
    Echo back the prompt in addition to the completion
    """
    max_new_tokens: int = 16
    seed: Optional[int]
    """
    Request-level determinism.
    """
    stop: Union[str, List[str]]
    temperature: float = 0
    top_p: float = 1

class GenerateResponse(BaseModel):
    text: List[str]
    request_id: str
    """
    A unique identifier
    """ 

class Endpoints(ABC):
    """
    Methods for communicating with an inference server.

    Note than an `Endpoints` object simply stores an address to the inference
        methods for communicating with the inference endpoint. The serve
        process runs independently of `Endpoints` objects, and you are free to
        copy `Endpoints objects`.
    """
    @abstractmethod
    def generate(self, **kwargs) -> GenerateResponse:
        """
        kwargs have same schema as GenerateRequest
        """
        pass

    @abstractmethod
    def health(self) -> requests.Response:
        pass

    @abstractmethod
    def wait_for_health(self, timeout=60):
        pass
