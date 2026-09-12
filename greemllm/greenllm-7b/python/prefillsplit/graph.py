"""Dynamo service graph for the PrefillSplit ablation."""

from components.prefill_worker import TensorRTLLMPrefillWorker

from prefillsplit.frontend import PrefillSplitFrontend
from prefillsplit.worker import PrefillSplitTensorRTLLMWorker

PrefillSplitFrontend.link(PrefillSplitTensorRTLLMWorker).link(
    TensorRTLLMPrefillWorker
)

