import pytest

from huggingface_hub import snapshot_download
from pydantic import BaseModel

from composer.model import ServerType, ModelServerConfig, start_model_server
import composer.model as model

class FakeModelServerConfig(BaseModel):
    model_id: str
    server_type: ServerType = ServerType.VLLM
    port: int = 8000
    host: str = "0.0.0.0"


@pytest.mark.parametrize("server_type", ServerType)
def test_server_and_request(server_type):
    EXAMPLE_MODEL_ID = {
        ServerType.VLLM: "codellama/CodeLlama-7b-hf",
    }

    if server_type == ServerType.VLLM:
        snapshot_download(EXAMPLE_MODEL_ID[server_type])

    server_config = ModelServerConfig(
        model_id=EXAMPLE_MODEL_ID[server_type],
        server_type=server_type,
        port=8000,
    )
    endpoints = start_model_server(server_config)
    endpoints.wait_for_health(timeout=120)

    response = endpoints.health()
    response.raise_for_status()

    response = endpoints.generate(
        prompt="1958 - John McCarthy and Paul Graham invent LISP. Due to high costs caused by",
        temperature=0,
    )

    print(response)

    endpoints.terminate()
