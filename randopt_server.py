#!/usr/bin/env python3
"""
vLLM OpenAI-compatible server extended with weight perturbation endpoints
via WorkerExtension.

  POST /perturb       {"seed": int, "sigma": float, "negate": bool}
  POST /restore       {"seed": int, "sigma": float, "negate": bool}
  POST /store_base    {}   — snapshot current weights as the reset target
  POST /reset         {}   — restore weights to the last /store_base snapshot

/store_base + /reset is the preferred alternative to /restore: no seed needed,
always exact (no floating-point add/subtract), safe even if perturbation params
are not remembered.  Call /store_base once after the server is ready, then use
/perturb … /reset for each evaluation cycle.

All standard vLLM API server arguments are accepted and forwarded unchanged.

Example:
  python3 randopt_server.py \\
    --model /dev/shm/qwen25-32b-instruct \\
    --worker-extension-cls utils.worker_extn.WorkerExtension \\
    --tensor-parallel-size 4 \\
    --max-model-len 16384 \\
    --max-num-seqs 64
"""

import os
import sys

# vLLM (>=~0.15 / nightly) forces 'spawn' for worker processes. Unlike 'fork',
# a spawned worker re-execs Python and does NOT inherit the launcher's
# sys.path[0] (this script's directory). It only sees PYTHONPATH. Without this,
# resolving --worker-extension-cls utils.worker_extn.WorkerExtension fails in
# the workers with ModuleNotFoundError: No module named 'utils.worker_extn'.
# Propagate our own directory via PYTHONPATH so spawned workers can import it.
_RANDOPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in (_RANDOPT_DIR, os.environ.get("PYTHONPATH", "")) if path
)
if _RANDOPT_DIR not in sys.path:
    sys.path.insert(0, _RANDOPT_DIR)

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

    @app.post("/store_base")
    async def store_base(raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc("store_base_weights", args=())
        return {"status": "ok"}

    @app.post("/reset")
    async def reset(raw_request: Request):
        ec = raw_request.app.state.engine_client
        await ec.collective_rpc("reset_to_base_weights", args=())
        return {"status": "ok"}

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