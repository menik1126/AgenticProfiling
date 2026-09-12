"""HTTP frontend for the PrefillSplit TensorRT-LLM graph."""

import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from dynamo import sdk
from dynamo.sdk import depends, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

from prefillsplit.worker import PrefillSplitTensorRTLLMWorker

logger = logging.getLogger(__name__)


def get_dynamo_run_binary():
    binary = Path(sdk.__file__).parent / "cli/bin/dynamo-run"
    return str(binary) if binary.exists() else "dynamo-run"


class FrontendConfig(BaseModel):
    served_model_name: str
    endpoint: str
    port: int = 8000
    router: str = "round-robin"
    block_size: int = 32


@service(
    dynamo={"namespace": "dynamo"},
    workers=1,
    image=DYNAMO_IMAGE,
    app=FastAPI(title="TensorRT-LLM PrefillSplit"),
)
class PrefillSplitFrontend:
    worker = depends(PrefillSplitTensorRTLLMWorker)

    def __init__(self):
        self.frontend_config = FrontendConfig(
            **ServiceConfig.get_parsed_config("PrefillSplitFrontend")
        )
        self.process = subprocess.Popen(
            [
                get_dynamo_run_binary(),
                "in=http",
                "out=dyn",
                "--http-port",
                str(self.frontend_config.port),
                "--router-mode",
                self.frontend_config.router,
            ]
        )

    def close(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process = None

    def __del__(self):
        self.close()

