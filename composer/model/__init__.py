from typing import *
from dataclasses import dataclass
from enum import Enum
import multiprocessing
import requests
import time
from openai import OpenAI

from typeguard import typechecked

from vllm.sampling_params import SamplingParams

from composer.model.vllm import start_vllm_server
from composer.model.server_interface import *


@typechecked
def start_model_server(config: ModelServerConfig):
    """
    Args:
        config ModelServerConfig
    Returns:
        endpoints Endpoints
    """
    match config.server_type:
        case ServerType.VLLM:
            server_process = multiprocessing.Process(
                target=start_vllm_server, args=(config,)
            )
            server_process.start()

            endpoints = LocalEndpoints(
                generate_endpoint=f"http://localhost:{config.port}/generate",
                health_endpoint=f"http://localhost:{config.port}/health",
                terminate_method=server_process.terminate,
            )

        case ServerType.OPENAI:

            endpoints = OpenAICompletionEndpoints()

        case _:
            raise ValueError(
                f"Invalid ModelServerConfig.server_type {config.server_type}"
            )

    return endpoints


class LocalEndpoints(Endpoints):
    @typechecked
    def __init__(
        self, generate_endpoint: str, health_endpoint: str, terminate_method: Callable
    ):
        self.generate_endpoint = generate_endpoint
        self.health_endpoint = health_endpoint
        self.terminate = terminate_method

    def generate(self, **kwargs):
        payload = GenerateRequest(**kwargs)

        response = requests.post(self.generate_endpoint, json=payload.dict())
        response.raise_for_status()

        return GenerateResponse(**response.json())

    def health(self):
        response = requests.get(self.health_endpoint)

        return response

    def wait_for_health(self, timeout=60):
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.health()
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass

            time.sleep(0.5)

        raise RuntimeError(f"{self.health_endpoint} not ready in time")

    def terminate(self):
        self.terminate()


class OpenAICompletionEndpoints(Endpoints):
    def __init__(self, *args, **kwargs):
        self.model_ids = ["gpt-3.5-turbo-instruct"]
        self.client = OpenAI()

    def generate(self, **kwargs):
        payload = GenerateRequest(**kwargs).dict()

        payload["model"] = payload["model_id"]
        payload.pop("model_id")

        payload["max_tokens"] = payload["max_new_tokens"]
        payload.pop("max_new_tokens")

        if payload["best_of"] is None or payload["best_of"]==1:
            payload.pop("best_of")

        if payload["model"] not in self.model_ids:
            raise ValueError(f"Invalid model_id {payload.model_id}. Valid IDs are {self.model_ids}")


        response = self.client.completions.create(**payload)

        ret = {"text": [x.text for x in response.choices], "request_id": response.id}
        return GenerateResponse(**ret)

    def health(self):
        resp = requests.Response()
        resp.status_code = 200

        return resp

    def wait_for_health(self, timeout=None):
        pass

    def terminate(self):
        pass
