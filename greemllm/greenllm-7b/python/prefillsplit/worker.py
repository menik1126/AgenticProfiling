"""TensorRT-LLM decode worker with deterministic prompt-length prefill routing."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from pathlib import Path

from common.base_engine import (
    BaseTensorrtLLMEngine,
    DisaggRequestType,
)
from common.parser import parse_tensorrt_llm_args
from common.protocol import TRTLLMWorkerRequest
from common.utils import ServerType
from components.prefill_worker import TensorRTLLMPrefillWorker
from tensorrt_llm.serve.openai_protocol import DisaggregatedParams

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


def _load_routing_config() -> tuple[int, int]:
    path = Path(
        os.environ.get(
            "PREFILL_SPLIT_ROUTING_CONFIG",
            "/experiment/config/routing.json",
        )
    )
    data = json.loads(path.read_text())
    threshold = int(data["threshold_tokens"])
    expected_workers = int(data["expected_prefill_workers"])
    if threshold < 1:
        raise ValueError("threshold_tokens must be positive")
    if expected_workers != 2:
        raise ValueError("PrefillSplit requires exactly two prefill workers")
    return threshold, expected_workers


@service(
    dynamo={"namespace": "dynamo"},
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class PrefillSplitTensorRTLLMWorker(BaseTensorrtLLMEngine):
    """Decode worker that pins short and long prompts to separate queues."""

    prefill_worker = depends(TensorRTLLMPrefillWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        config_args = config.as_args(class_name, prefix="")
        args, engine_config = parse_tensorrt_llm_args(config_args)
        self.served_model_name = args.served_model_name
        self._threshold_tokens, self._expected_prefill_workers = _load_routing_config()
        self._prefill_instance_ids: tuple[int, int] | None = None
        self._route_counts = {"short_medium": 0, "long": 0}

        worker_id = dynamo_context["endpoints"][0].lease_id()
        namespace, _ = PrefillSplitTensorRTLLMWorker.dynamo_address()  # type: ignore
        self._min_prefill_workers = args.min_prefill_workers
        if self._min_prefill_workers != self._expected_prefill_workers:
            raise ValueError(
                "min-prefill-workers must match expected_prefill_workers in routing.json"
            )

        super().__init__(
            namespace_str=namespace,
            component_str=class_name,
            worker_id=worker_id,
            engine_config=engine_config,
            remote_prefill=args.remote_prefill,
            min_workers=args.min_workers,
            disagg_config_file=args.llmapi_disaggregated_config,
            block_size=args.block_size,
            router=args.router,
            server_type=ServerType.GEN,
        )

    @async_on_start
    async def async_init(self):
        self._init_engine()

        runtime = dynamo_context["runtime"]
        namespace, component = PrefillSplitTensorRTLLMWorker.dynamo_address()  # type: ignore
        endpoint_handle = runtime.namespace(namespace).component(component).endpoint(
            "generate"
        )
        await register_llm(
            ModelType.Backend,
            endpoint_handle,
            self._engine_config.model_name,
            self.served_model_name,
            kv_cache_block_size=self._kv_block_size,
        )

        if not self._remote_prefill:
            raise ValueError("PrefillSplit requires remote-prefill=true")

        prefill_namespace, prefill_component = (
            TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
        )
        self._prefill_client = (
            await runtime.namespace(prefill_namespace)
            .component(prefill_component)
            .endpoint("generate")
            .client()
        )
        while len(self._prefill_client.instance_ids()) < self._expected_prefill_workers:
            logger.info(
                "Waiting for two prefill workers; currently discovered %d",
                len(self._prefill_client.instance_ids()),
            )
            await asyncio.sleep(5)

        instance_ids = tuple(sorted(self._prefill_client.instance_ids()))
        if len(instance_ids) != self._expected_prefill_workers:
            raise RuntimeError(
                f"Expected exactly two prefill workers, discovered {instance_ids}"
            )
        self._prefill_instance_ids = instance_ids
        logger.warning(
            "PrefillSplit routing active: <=%d tokens -> instance %d; >%d tokens -> instance %d",
            self._threshold_tokens,
            instance_ids[0],
            self._threshold_tokens,
            instance_ids[1],
        )

        if self._kv_metrics_publisher is not None:
            task = asyncio.create_task(self.create_metrics_publisher_endpoint())
            task.add_done_callback(
                lambda _: logger.info("metrics publisher endpoint created")
            )

    async def create_metrics_publisher_endpoint(self):
        component = dynamo_context["component"]
        await self._kv_metrics_publisher.create_endpoint(component)

    def _checked_prefill_instances(self) -> tuple[int, int]:
        if self._prefill_client is None or self._prefill_instance_ids is None:
            raise RuntimeError("Prefill client is not initialized")
        current = tuple(sorted(self._prefill_client.instance_ids()))
        if current != self._prefill_instance_ids:
            raise RuntimeError(
                "Prefill worker membership changed during the run: "
                f"started with {self._prefill_instance_ids}, now {current}"
            )
        return self._prefill_instance_ids

    async def _get_remote_prefill_response(self, request):
        prefill_request = copy.deepcopy(request)
        prefill_request.stop_conditions.max_tokens = 1
        prefill_request.stop_conditions.min_tokens = 1
        prefill_request.stop_conditions.ignore_eos = False
        prefill_request.disaggregated_params = DisaggregatedParams(
            request_type=DisaggRequestType.CONTEXT_ONLY.value
        )

        instance_ids = self._checked_prefill_instances()
        input_tokens = len(prefill_request.token_ids)
        route = "short_medium" if input_tokens <= self._threshold_tokens else "long"
        instance_id = instance_ids[0] if route == "short_medium" else instance_ids[1]
        self._route_counts[route] += 1
        route_total = sum(self._route_counts.values())
        if route_total == 1 or route_total % 1000 == 0:
            logger.info(
                "PrefillSplit routes total=%d short_medium=%d long=%d",
                route_total,
                self._route_counts["short_medium"],
                self._route_counts["long"],
            )

        responses = [
            response
            async for response in await self._prefill_client.direct(
                prefill_request.model_dump_json(), instance_id
            )
        ]
        if len(responses) != 1:
            raise RuntimeError(
                f"Prefill worker returned {len(responses)} responses; expected one"
            )
        return responses[0]

    @endpoint()
    async def generate(self, request: TRTLLMWorkerRequest):
        async for response in super().generate(request):
            yield response

