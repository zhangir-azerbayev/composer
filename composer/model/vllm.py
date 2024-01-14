import argparse
import json
from typing import AsyncGenerator
from dataclasses import asdict

from typeguard import typechecked

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from composer.model.data import ModelServerConfig
from composer.model.server_interface import GenerateRequest, GenerateResponse

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the composer.model.server_interface.GenerateRequest
    """
    request_dict = await request.json()
    request_obj = GenerateRequest(**request_dict)

    prompt = request_obj.prompt

    if request_obj.model_id:
        return Response(status_code=400, detail=f"vLLM server does not accept GenerateRequest.model_id")
    if request_obj.seed:
        return Response(status_code=400, detail=f"random seed not yet implemented")

    sampling_params = SamplingParams(
            n=request_obj.n,
            best_of=request_obj.best_of,
            temperature=request_obj.n,
            top_p=request_obj.top_p,
            stop=request_obj.stop,
    )
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    if request_obj.echo:
        text_outputs = [prompt + output.text for output in final_output.outputs]
    else:
        text_outputs = [output.text for output in final_output.outputs]
    ret = GenerateResponse(text=text_outputs, request_id=request_id).dict()
    return JSONResponse(ret)


@typechecked
def start_vllm_server(config: ModelServerConfig):
    global engine

    engine_args = AsyncEngineArgs(model=config.model_id)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
