#!/usr/bin/env python3
"""
vLLM OpenAI-compatible server extended with /perturb and /restore endpoints
for seed-based weight perturbation via WorkerExtension.

  POST /perturb  {"seed": int, "sigma": float, "negate": bool}
  POST /restore  {"seed": int, "sigma": float, "negate": bool}

Both endpoints call collective_rpc on all GPU workers and block until done.
All standard vLLM API server arguments are accepted and forwarded unchanged.

Example:
  python3 randopt_server.py \\
    --model /dev/shm/qwen25-32b-instruct \\
    --worker-extension-cls utils.worker_extn.WorkerExtension \\
    --tensor-parallel-size 4 \\
    --max-model-len 16384 \\
    --max-num-seqs 64
"""

import uvloop
from fastapi import FastAPI, Request
from pydantic import BaseModel

import vllm.entrypoints.openai.api_server as _api_server
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser


class PerturbRequest(BaseModel):
    seed: int
    sigma: float
    negate: bool = False


class RestoreRequest(BaseModel):
    seed: int
    sigma: float
    negate: bool = False


_original_build_app = _api_server.build_app


def _build_app_with_randopt_routes(args) -> FastAPI:
    app = _original_build_app(args)

    @app.post("/perturb")
    async def perturb(body: PerturbRequest, raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc(
            "perturb_self_weights",
            args=(body.seed, body.sigma, body.negate),
        )
        return {"status": "ok", "seed": body.seed, "sigma": body.sigma}

    @app.post("/restore")
    async def restore(body: RestoreRequest, raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc(
            "restore_self_weights",
            args=(body.seed, body.sigma, body.negate),
        )
        return {"status": "ok", "seed": body.seed, "sigma": body.sigma}

    return app


_api_server.build_app = _build_app_with_randopt_routes


if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible server with /perturb and /restore endpoints."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))