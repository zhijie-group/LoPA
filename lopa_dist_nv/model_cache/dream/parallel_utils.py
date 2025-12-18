"""Utilities for integrating Dream models with TP/PP helpers.

This module intentionally keeps the public surface small so that
callers can depend on lightweight configuration objects even when the
full tensor/pipeline parallel runtime is not available (e.g. on CPU or
single-GPU hosts). When the specialised runtime is absent we fall back
to the standard forward pass and emit a warning once.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING, Union

try:  # pragma: no cover - optional dependency during CPU-only tests
	import torch.distributed as dist
except Exception:  # noqa: BLE001
	dist = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
	from .model_dream import DreamBaseModel, DreamModel


logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
	"""Configuration metadata for Dream tensor/pipeline (branch) parallel modes."""

	tensor_parallel_size: int = 1
	branch_parallel_size: int = 1
	pipeline_parallel_size: int = 1  # legacy field; kept for compatibility
	pipeline_chunks: int = 1
	devices: Optional[Sequence[Union[str, int]]] = None
	description: str = "stub"

	tensor_parallel_rank: int = 0
	branch_parallel_rank: int = 0
	tensor_parallel_group: Optional["dist.ProcessGroup"] = None
	branch_parallel_group: Optional["dist.ProcessGroup"] = None
	_groups_initialized: bool = False


def _init_tensor_parallel_groups(config: ParallelConfig) -> None:
	if config._groups_initialized:
		return
	if dist is None or not dist.is_available() or not dist.is_initialized():
		config.tensor_parallel_size = 1
		config.branch_parallel_size = 1
		config.tensor_parallel_rank = 0
		config.branch_parallel_rank = 0
		config.tensor_parallel_group = None
		config.branch_parallel_group = None
		config._groups_initialized = True
		return

	world_size = dist.get_world_size()
	rank = dist.get_rank()
	tp_size = max(1, int(config.tensor_parallel_size or 1))
	if world_size % tp_size != 0:
		raise RuntimeError(
			f"World size {world_size} must be divisible by tensor_parallel_size {tp_size}"
		)

	branch_size = world_size // tp_size
	branch_rank = rank // tp_size
	tp_rank = rank % tp_size

	tp_group_ranks = [branch_rank * tp_size + i for i in range(tp_size)]
	tp_group = dist.new_group(ranks=tp_group_ranks)

	config.tensor_parallel_size = tp_size
	config.branch_parallel_size = branch_size
	config.tensor_parallel_rank = tp_rank
	config.branch_parallel_rank = branch_rank
	config.tensor_parallel_group = tp_group

	# Branch-parallel group contains peers that share the same tensor rank.
	branch_group_ranks = list(range(tp_rank, world_size, tp_size))
	config.branch_parallel_group = dist.new_group(ranks=branch_group_ranks)
	config._groups_initialized = True


def _attach_parallel_stub(base_model: "DreamBaseModel") -> None:
	"""Bind a fallback parallel forward that delegates to the standard path."""

	def _forward_parallel_stub(self, *args, **kwargs):
		if not getattr(self, "_parallel_warning_emitted", False):
			logger.warning(
				"Dream parallel runtime is not initialised; falling back to standard forward()."
			)
			self._parallel_warning_emitted = True
		return self._forward_standard(*args, **kwargs)

	base_model._forward_parallel = types.MethodType(_forward_parallel_stub, base_model)


def configure_model_for_tp_pp(
	model: "DreamModel",
	parallel_config: Optional[ParallelConfig] = None,
) -> "DreamModel":
	"""Attach TP/PP configuration metadata to a Dream model.

	The helper keeps the API stable for legacy entry points that still
	expect to call into :mod:`model_cache.dream.model_dream` directly.
	When the TP/PP runtime is not present the function installs a stub
	forward so that the call path remains functional while emitting a
	diagnostic message.
	"""

	base_model = getattr(model, "model", model)
	if parallel_config is None:
		parallel_config = ParallelConfig()

	if not hasattr(base_model, "parallel_config"):
		raise AttributeError("Model does not expose a 'parallel_config' slot")

	base_model.parallel_config = parallel_config
	_init_tensor_parallel_groups(parallel_config)
	if getattr(parallel_config, "tensor_parallel_size", 1) not in (None, 1):
		enable_fn = getattr(base_model, "enable_tensor_parallel", None)
		if callable(enable_fn):
			enable_fn()

	if not hasattr(base_model, "_forward_parallel"):
		_attach_parallel_stub(base_model)

	return model


def prepare_parallel_config(parallel_config: ParallelConfig) -> ParallelConfig:
	_init_tensor_parallel_groups(parallel_config)
	return parallel_config


__all__ = ["ParallelConfig", "configure_model_for_tp_pp", "prepare_parallel_config"]

