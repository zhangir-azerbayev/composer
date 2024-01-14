from pydantic import BaseModel
from typing import Union, Dict, List, Optional

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

    
