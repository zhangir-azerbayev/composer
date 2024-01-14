import pytest

from huggingface_hub import snapshot_download
from pydantic import BaseModel

from composer.model import *

@pytest.mark.parametrize("server_type", ServerType)
def test_completion_endpoints(server_type):
    SERVER_MODEL_ID = {
        ServerType.VLLM: "codellama/CodeLlama-7b-hf",
        ServerType.OPENAI: None,
    }

    if server_type == ServerType.VLLM:
        snapshot_download(SERVER_MODEL_ID[server_type])

    server_config = ModelServerConfig(
        model_id=SERVER_MODEL_ID[server_type],
        server_type=server_type,
        port=8000,
    )
    endpoints = start_model_server(server_config)

    # try-except block shuts down API server if main thread raises exception
    try:
        endpoints.wait_for_health(timeout=120)

        response = endpoints.health()
        response.raise_for_status()

        match server_type:
            case ServerType.VLLM:
                prompt = "1958 - John McCarthy and Paul Graham invent LISP. Due to high costs caused by"
                request_model_id = None
            case ServerType.OPENAI:
                prompt = "Implement a LISP Interpreter."
                request_model_id = endpoints.model_ids[0]

        response = endpoints.generate(
            model_id = request_model_id if request_model_id else None,
            prompt=prompt,
            n=4,
            echo=True,
            max_new_tokens=24,
            stop=".",
            temperature=0.6,
            top_p=0.95
        )

        print(f"{server_type}: {response}")

        endpoints.terminate()

    except Exception as e:
        endpoints.terminate()
        raise e
