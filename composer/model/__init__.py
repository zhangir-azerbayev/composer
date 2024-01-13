from typing import Literal, Callable
from dataclasses import dataclass
from enum import Enum
import multiprocessing
import requests
import time

import uvicorn
from vllm.sampling_params import SamplingParams

from composer.model.vllm import start_vllm_server
from composer.model.data import ServerType, ModelServerConfig


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

            endpoints = Endpoints(
                generate_endpoint=f"http://localhost:{config.port}/generate",
                health_endpoint=f"http://localhost:{config.port}/health",
                terminate_method=server_process.terminate,
            )

        case _:
            raise ValueError(
                f"Invalid ModelServerConfig.server_type {config.server_type}"
            )

    return endpoints


class Endpoints:
    """
    Methods for communicating with an inference server.

    Note than an `Endpoints` object simply stores an address to the inference
        methods for communicating with the inference endpoint. The serve
        process runs independently of `Endpoints` objects, and you are free to
        copy `Endpoints objects`.

    Todo:
    - define a standard set of endpoints and a interface for each one.
    - The two most important things to support are vllm and openai.
    """

    def __init__(
        self, generate_endpoint: str, health_endpoint: str, terminate_method: Callable
    ):
        self.generate_endpoint = generate_endpoint
        self.health_endpoint = health_endpoint
        self.terminate = terminate_method

    def generate(self, prompt, temperature=0.3, top_p=0.95):
        payload = {"prompt": prompt, "temperature": temperature, "top_p": top_p}

        response = requests.post(self.generate_endpoint, json=payload)
        response.raise_for_status()

        return response.json()

    def health(self):
        response = requests.get(self.health_endpoint)

        return response

    def wait_for_ready(self, timeout=60):
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
