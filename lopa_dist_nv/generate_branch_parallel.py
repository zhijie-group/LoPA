import logging
import gc
import json
import time
import sys
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import List, Optional, Tuple, Type, TypeVar, Union, Dict, Set, Any
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.distributions as dists
import torch.distributed as dist
import transformers
from transformers.cache_utils import DynamicCache
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, DistributedSampler
from datasets import Dataset
from packaging import version
from tqdm import tqdm
from peft import PeftConfig, PeftModel
import numpy as np
import os
import jinja2

# Import code evaluation helpers (pass_at_1 calculation)
try:
    import evaluate as hf_evaluate
    from sanitize import sanitize
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    CODE_EVAL_AVAILABLE = True
except ImportError:
    CODE_EVAL_AVAILABLE = False
    # Note: eval_logger is not defined yet; fall back to print
    print("Warning: Code evaluation dependencies not available. pass_at_1 will not be calculated.")

# Dream model components
from model_cache.dream.model_dream_bp import DreamModel, _sdpa_kernel_context, CacheUpdate, SharedStaticCache
from model_cache.dream.configuration_dream import DreamConfig
from utils.compile_control import configure_torch_compile

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.api import registry as model_registry
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate, setup_parser

# Optional stats module (skip if unavailable)
try:
    from eval_dream_stat import GenerationStatsCollector
    STATS_AVAILABLE = True
except ImportError:
    try:
        from utils.eval_dream_stat import GenerationStatsCollector
        STATS_AVAILABLE = True
    except ImportError:
        STATS_AVAILABLE = False
        GenerationStatsCollector = None

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="TemplateLM")

import random
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# [MODIFIED] Branch adds tracking fields for precise statistics
class Branch:
    """Represents the state of a generation branch."""
    def __init__(self, branch_id: int, x_t: torch.Tensor, block_states: Dict,
                 confidence: float = 1.0, past_key_values: Optional[Tuple] = None,
                 prompt_length: int = 0, is_base: bool = False,
                 creation_token_confidence: float = 1.0):
        self.branch_id = branch_id
        self.x_t = x_t.clone()  # current sequence state
        self.block_states = {k: v.copy() for k, v in block_states.items()}  # block state snapshots
        self.confidence = confidence  # overall branch confidence
        self.step_confidences = []  # per-step confidence
        self.is_active = True  # whether the branch is still active
        self.eos_detected = False  # whether EOS has been detected
        self.past_key_values = past_key_values
        self.max_length_reached = False  # flag if the branch hit the max length

        # Stats fields
        self.prompt_length = prompt_length
        self.steps_completed = -1  # parallel step index when finished (-1 means unfinished)
        self.is_base = is_base
        # Confidence of the sampled token used when the branch was forked
        self.creation_token_confidence = creation_token_confidence

    @property
    def generated_token_count(self) -> int:
        """Compute generated token count dynamically."""
        return self.x_t.shape[1] - self.prompt_length



    def copy(self):
        """Create a memory-efficient copy of the branch."""
        new_branch = Branch(
            self.branch_id,
            self.x_t.clone(),
            {k: v.copy() for k, v in self.block_states.items()},
            self.confidence,
            None,
            self.prompt_length,
            self.is_base,
            creation_token_confidence=self.creation_token_confidence,
        )
        new_branch.step_confidences = self.step_confidences.copy()
        new_branch.is_active = self.is_active
        new_branch.eos_detected = self.eos_detected
        new_branch.steps_completed = self.steps_completed
        new_branch.max_length_reached = self.max_length_reached
        return new_branch


@dataclass
class BranchTask:
    branch_index: int
    branch_id: int
    tokens: torch.Tensor
    input_start_pos: int
    shared_prefix_end: int
    update_kvcache_len: int


@dataclass
class BranchWorkerResult:
    branch_index: int
    logits: Optional[torch.Tensor]
    cache_updates: Optional[List[Optional[CacheUpdate]]]
    source_rank: int
    error: Optional[str] = None


@dataclass
class BranchWorkerResultMetadata:
    branch_index: int
    has_logits: bool
    logits_shape: Tuple[int, ...]
    logits_dtype: Optional[torch.dtype]
    cache_updates: Optional[List[Optional[CacheUpdate]]]
    source_rank: int
    error: Optional[str]


class BranchBatchDataset(TorchDataset):
    def __init__(self, tasks: List[BranchTask]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> BranchTask:
        return self.tasks[idx]


class CodeProfiler:
    """Lightweight section profiler that logs per-block durations."""

    def __init__(self, logger: logging.Logger, prefix: str = "", enabled: bool = True) -> None:
        self.logger = logger
        self.prefix = prefix
        self.enabled = enabled

    def _sync_cuda(self, enabled: bool) -> None:
        if not enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def section(
        self,
        name: str,
        *,
        extra: Optional[Union[int, str]] = None,
        sync_cuda: bool = False,
    ) -> Any:
        if not self.enabled:
            yield
            return

        label = f"{self.prefix}{name}"
        if extra is not None:
            label = f"{label}[{extra}]"
        self._sync_cuda(sync_cuda)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync_cuda(sync_cuda)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.logger.info("profile:%s %.3f ms", label, duration_ms)

    def tic(self, sync_cuda: bool = False) -> float:
        self._sync_cuda(sync_cuda)
        return time.perf_counter()

    def toc(
        self,
        start: float,
        name: str,
        *,
        extra: Optional[Union[int, str]] = None,
        sync_cuda: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self._sync_cuda(sync_cuda)
        label = f"{self.prefix}{name}"
        if extra is not None:
            label = f"{label}[{extra}]"
        duration_ms = (time.perf_counter() - start) * 1000.0
        self.logger.info("profile:%s %.3f ms", label, duration_ms)


def _coerce_bool_flag(value: Optional[Union[str, bool, int, float]]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "no", "off"}:
            return False
        if lowered in {"1", "true", "yes", "on"}:
            return True
        return bool(lowered)
    return bool(value)


def _suppress_lm_eval_logging_for_workers() -> None:
    """Silence lm_eval logging/output for non-main ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_rank() == 0:
        return
    try:
        from lm_eval.loggers import evaluation_tracker  # type: ignore
    except Exception:  # noqa: BLE001
        return

    tracker_logger = getattr(evaluation_tracker, "logger", None)
    if tracker_logger is None:
        tracker_logger = getattr(evaluation_tracker, "eval_logger", None)
    if tracker_logger is not None:
        tracker_logger.disabled = True

    def _noop(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN001
        return

    for attr in [
        "save_results",
        "save_per_sample_results",
        "save_bootstrap_results",
        "dump_results_table",
    ]:
        if hasattr(evaluation_tracker.EvaluationTracker, attr):
            setattr(evaluation_tracker.EvaluationTracker, attr, _noop)


class _AcceleratorCompat:
    """Minimal accelerator-compatible shim backed by torch.distributed."""

    def __init__(self, model_ref: "DreamLoRA") -> None:
        self._model_ref = model_ref

    @property
    def device(self) -> torch.device:
        return self._model_ref.device

    @property
    def num_processes(self) -> int:
        return self._model_ref._dist_world_size

    @property
    def process_index(self) -> int:
        return self._model_ref._dist_rank

    @property
    def local_process_index(self) -> int:
        return self._model_ref._local_rank

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            raise ValueError("tensor must not be None")
        if self.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            return tensor.detach().unsqueeze(0)

        tensor = tensor.detach().clone()
        gather_list = [torch.zeros_like(tensor) for _ in range(self.num_processes)]
        dist.all_gather(gather_list, tensor)
        return torch.stack(gather_list, dim=0)

    def wait_for_everyone(self) -> None:
        if self.num_processes > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Alias used by some callers
    def barrier(self) -> None:
        self.wait_for_everyone()


## Removed unused branch pruning/position helpers to reduce maintenance overhead
def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask


def create_full_attention_mask(max_length, device=None, dtype=None):
    """Create a full attention mask (all zeros)."""
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.zeros((1, 1, max_length, max_length), device=device, dtype=dtype)
    return attention_mask


def extract_attention_mask(full_mask, start_pos, input_length, cache_length, use_full_attention=False):
    end_pos = start_pos + input_length
    total_length = cache_length + input_length

    if use_full_attention:
        return None
    
    if full_mask is None:
        raise ValueError("full_mask must be provided when block attention is enabled")

    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf,
                                device=full_mask.device, dtype=full_mask.dtype)
    if cache_length > 0:
        extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]

    return extracted_mask


def find_shared_prefix_end(branches: List[Branch]) -> int:
    """Find the end of the longest shared prefix across branches (by completed blocks)."""
    if not branches:
        return 0

    # Find the smallest number of completed blocks across all branches
    min_completed_blocks = float('inf')
    for branch in branches:
        completed_blocks = 0
        for block_id in sorted(branch.block_states.keys()):
            if (branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
                branch.block_states[block_id]['mask_count'] == 0):
                completed_blocks += 1
            else:
                break
        min_completed_blocks = min(min_completed_blocks, completed_blocks)

    if min_completed_blocks == float('inf') or min_completed_blocks == 0:
        # Fallback: use the prompt length as the minimal shared prefix
        return branches[0].prompt_length

    # Locate the end position of the min_completed_blocks-th completed block
    first_branch = branches[0]
    completed_count = 0
    for block_id in sorted(first_branch.block_states.keys()):
        if (first_branch.block_states[block_id]['state'] in ['in_cache', 'to_cache'] and
            first_branch.block_states[block_id]['mask_count'] == 0):
            completed_count += 1
            if completed_count == min_completed_blocks:
                return first_branch.block_states[block_id]['end_pos']
        else:
            break

    return branches[0].prompt_length


def _create_base_block_causal_mask(
    seq_length: int, 
    prompt_length: int, 
    block_size: int, 
    device, 
    dtype
) -> torch.Tensor:
    """
    Creates a base block-causal attention mask for a single, contiguous sequence.
    This is a helper function to establish the correct local attention pattern.

    Rules:
    1. All tokens can attend to the entire prompt.
    2. A token after the prompt can attend to all tokens within its own block and all preceding blocks.
    """
    # 1. Initialize mask, blocking all attention by default. Shape is [query_len, key_len].
    mask = torch.full((seq_length, seq_length), -torch.inf, device=device, dtype=dtype)

    # 2. Rule 1: Allow all query tokens (rows) to attend to the full prompt (columns).
    mask[:, :prompt_length] = 0

    # 3. Rule 2: Process the block-causal attention for tokens after the prompt.
    for q_pos in range(prompt_length, seq_length):
        # Determine the block ID for the current query token.
        # Blocks are numbered starting from 0 *after* the prompt.
        pos_after_prompt = q_pos - prompt_length
        q_block_id = pos_after_prompt // block_size

        # A query can attend to all keys from block 0 up to its own block.
        for b_id in range(q_block_id + 1):
            # Calculate the absolute start and end positions of this key block.
            block_start = prompt_length + b_id * block_size
            block_end = min(prompt_length + (b_id + 1) * block_size, seq_length)
            
            # Allow the current query at q_pos to attend to all keys in this block.
            mask[q_pos, block_start:block_end] = 0
            
    return mask


## Removed unused multi-branch attention mask builders
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, sampling_strategy="default"):
    """
    Unified token sampling and confidence computation.
    
    Args:
        logits: input logits tensor
        temperature: sampling temperature (0 means argmax)
        top_p: nucleus sampling parameter
        top_k: top-k sampling parameter
        sampling_strategy: confidence calculation strategy
            - "default": use token probability as confidence
            - "margin": use the gap between top1 and top2
            - "neg_entropy": use negative entropy as confidence
    
    Returns:
        confidence: confidence value computed by the chosen strategy
        x0: sampled token
        initial_confidence: raw token probability (kept for backward compatibility)
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)

    # Compute final confidence based on sampling_strategy
    if sampling_strategy == "margin":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        confidence = top1_probs - top2_probs
    elif sampling_strategy == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    else:  # "default"
        confidence = initial_confidence.clone()

    return confidence, x0, initial_confidence

def _patch_dream_attention_forward() -> None:
    """
    Patch DreamAttention and DreamFastAttention forward to support merged qkv_proj
    via monkey patching without touching the original source files.
    """
    from model_cache.dream.model_dream_bp import DreamAttention, DreamFastAttention
    import types
    
    def _merged_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        update_kvcache: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """Forward implementation that uses the merged qkv_proj."""
        bsz, q_len, _ = hidden_states.size()
        
        # Use the merged qkv_proj for a single projection pass
        if hasattr(self, 'qkv_proj') and hasattr(self, '_qkv_merged') and self._qkv_merged:
            qkv_states = self.qkv_proj(hidden_states)  # [bsz, q_len, qkv_out_dim]
            
            # Split in Q, K, V order
            q_out_dim = self._q_out_dim
            k_out_dim = self._k_out_dim
            v_out_dim = self._v_out_dim
            
            query_states = qkv_states[:, :, :q_out_dim]
            key_states = qkv_states[:, :, q_out_dim:q_out_dim + k_out_dim]
            value_states = qkv_states[:, :, q_out_dim + k_out_dim:q_out_dim + k_out_dim + v_out_dim]
        else:
            # Fallback to the original projection layers for compatibility
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # From here, mirror the original attention logic inline.
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        from model_cache.dream.model_dream_bp import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        from model_cache.dream.model_dream_bp import repeat_kv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_mask = attn_mask.to(dtype=query_states.dtype, device=query_states.device)
            attn_weights = attn_weights + attn_mask
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _merged_fast_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_kvcache: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """DreamFastAttention forward that works with merged qkv_proj."""
        # If QKV is already merged, split once here
        if hasattr(self, 'qkv_proj') and hasattr(self, '_qkv_merged') and self._qkv_merged:
            qkv_states = self.qkv_proj(hidden_states)
            q_out_dim = self._q_out_dim
            k_out_dim = self._k_out_dim
            v_out_dim = self._v_out_dim
            
            # Proxy q/k/v projections to reuse precomputed tensors
            class _QKVProxy:
                def __init__(self, precomputed):
                    self._precomputed = precomputed
                def __call__(self, x):
                    return self._precomputed
            
            # Temporarily restore projection attributes so the original forward can run
            self.q_proj = _QKVProxy(qkv_states[:, :, :q_out_dim])
            self.k_proj = _QKVProxy(qkv_states[:, :, q_out_dim:q_out_dim + k_out_dim])
            self.v_proj = _QKVProxy(qkv_states[:, :, q_out_dim + k_out_dim:q_out_dim + k_out_dim + v_out_dim])
        
        # Call the saved DreamFastAttention.forward, preferring an instance-level override
        original_forward = getattr(self, '_original_forward', None)
        if original_forward is None:
            # Use the class-level _original_forward but ensure it belongs to the subclass
            original_forward = DreamFastAttention._original_forward
            if original_forward == DreamAttention._original_forward:
                # Fetch from the instance's class to avoid falling back to the parent method
                original_forward = type(self).__dict__.get('forward', DreamFastAttention._original_forward)
        
        # Check whether the original forward accepts update_kvcache
        import inspect
        import types
        try:
            sig = inspect.signature(original_forward)
            has_update_kvcache = 'update_kvcache' in sig.parameters
        except (ValueError, TypeError):
            # If signature inspection fails, assume it supports update_kvcache
            has_update_kvcache = True
        
        # Pass everything as keywords to avoid positional/keyword conflicts
        kwargs = {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_value': past_key_value,
            'output_attentions': output_attentions,
            'use_cache': use_cache,
            'cache_position': cache_position,
            'position_embeddings': position_embeddings,
        }
        
        if has_update_kvcache:
            kwargs['update_kvcache'] = update_kvcache
        
        # Handle bound vs unbound methods when invoking original_forward
        if isinstance(original_forward, types.MethodType):
            return original_forward(**kwargs)
        else:
            return original_forward(self, **kwargs)
    
    # Cache the original forward methods
    if not hasattr(DreamAttention, '_original_forward'):
        DreamAttention._original_forward = DreamAttention.forward
    
    # Swap in the merged-QKV versions
    DreamAttention.forward = _merged_forward
    
    # Apply the same patch to DreamFastAttention, keeping subclass methods intact
    if not hasattr(DreamFastAttention, '_original_forward'):
        if 'forward' in DreamFastAttention.__dict__:
            DreamFastAttention._original_forward = DreamFastAttention.__dict__['forward']
        else:
            DreamFastAttention._original_forward = DreamFastAttention.forward
    DreamFastAttention.forward = _merged_fast_forward


def merge_qkv_projections(model: torch.nn.Module) -> None:
    """
    Merge every attention layer's q_proj/k_proj/v_proj into a single qkv_proj to
    reduce parameters and inference cost.
    
    Args:
        model: DreamModel instance
    """
    from model_cache.dream.model_dream_bp import DreamAttention
    
    def _merge_attention_qkv(attn_module: DreamAttention) -> None:
        """Merge QKV projections for a single attention layer."""
        if not hasattr(attn_module, 'q_proj') or not hasattr(attn_module, 'k_proj') or not hasattr(attn_module, 'v_proj'):
            return
        
        # Skip if already merged
        if hasattr(attn_module, 'qkv_proj'):
            return
        
        q_proj = attn_module.q_proj
        k_proj = attn_module.k_proj
        v_proj = attn_module.v_proj
        
        hidden_size = q_proj.in_features
        num_heads = attn_module.num_heads
        num_key_value_heads = attn_module.num_key_value_heads
        head_dim = attn_module.head_dim
        
        # Compute output dimensions: Q + K + V
        q_out_dim = num_heads * head_dim
        k_out_dim = num_key_value_heads * head_dim
        v_out_dim = num_key_value_heads * head_dim
        qkv_out_dim = q_out_dim + k_out_dim + v_out_dim
        
        # Create the merged Linear layer
        qkv_proj = torch.nn.Linear(hidden_size, qkv_out_dim, bias=True)
        
        # Concatenate weights in [Q, K, V] order
        with torch.no_grad():
            qkv_weight = torch.cat([
                q_proj.weight,  # [q_out_dim, hidden_size]
                k_proj.weight,  # [k_out_dim, hidden_size]
                v_proj.weight,  # [v_out_dim, hidden_size]
            ], dim=0)  # [qkv_out_dim, hidden_size]
            
            qkv_bias = torch.cat([
                q_proj.bias,  # [q_out_dim]
                k_proj.bias,  # [k_out_dim]
                v_proj.bias,  # [v_out_dim]
            ], dim=0)  # [qkv_out_dim]
            
            qkv_proj.weight.copy_(qkv_weight)
            qkv_proj.bias.copy_(qkv_bias)
        
        # Move qkv_proj to the correct device and dtype
        qkv_proj = qkv_proj.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        
        # Replace the original projection layers
        attn_module.qkv_proj = qkv_proj
        attn_module._qkv_merged = True
        
        # Store original dimensions for splitting during forward
        attn_module._q_out_dim = q_out_dim
        attn_module._k_out_dim = k_out_dim
        attn_module._v_out_dim = v_out_dim
        
        # Optionally remove the original projections (kept only for compatibility)
        del attn_module.q_proj
        del attn_module.k_proj
        del attn_module.v_proj
    
    # Walk through model layers to find and merge DreamAttention modules
    dream_model = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LoRA already merged
        dream_model = model.model
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'layers'):
        # LoRA not merged
        dream_model = model.base_model.model
    elif hasattr(model, 'layers'):
        # Plain DreamModel
        dream_model = model
    
    if dream_model is None:
        eval_logger.warning("Could not find DreamModel layers for QKV merging. Skipping.")
        return
    
    merged_count = 0
    for layer in dream_model.layers:
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            # Check for DreamAttention or its subclasses (including DreamFastAttention)
            if isinstance(attn, DreamAttention):
                _merge_attention_qkv(attn)
                merged_count += 1
    
    if merged_count > 0:
        eval_logger.info(f"Successfully merged QKV projections for {merged_count} attention layers.")


def evaluate_branch_confidence(
    logits: torch.Tensor,
    branch: Branch,
    branch_start_in_input: int,
    branch_length: int,
    shared_prefix_end: int,
    mask_token_id: int,
    sampling_strategy: str = "default",
    branch_topp: float = 0.5,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[float] = None,
    selection_conf_alpha: float = 0.5,
) -> float:
    """Evaluate branch confidence by combining creation-token and future-mask confidence.

    final_confidence = alpha * creation_token_confidence + (1 - alpha) * future_mask_confidence
    """
    # --- Future-mask confidence ---
    # Keep the existing behavior: if branch_length == 0, future_conf is 1.0 (no further eval)
    if branch_length == 0:
        future_conf = 1.0
    else:
        branch_logits = logits[0, branch_start_in_input:branch_start_in_input + branch_length, :]
        branch_sequence = branch.x_t[0, shared_prefix_end:]
        mask_positions = (branch_sequence == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            future_conf = 1.0
        else:
            mask_logits = branch_logits[mask_positions, :]
            confidences, _, _ = sample_tokens(
                mask_logits, temperature, top_p, top_k, sampling_strategy
            )
            num_positions = len(confidences)
            if num_positions == 1:
                future_conf = confidences.item()
            else:
                # Use the lowest-confidence bottom proportion to measure weak regions
                bottom_cnt = max(1, int(num_positions * branch_topp))
                sorted_confidences, _ = torch.sort(confidences, descending=False)
                future_conf = sorted_confidences[:bottom_cnt].mean().item()

    # --- Creation-time token confidence ---
    creation_conf = float(getattr(branch, "creation_token_confidence", 1.0))
    alpha = float(max(0.0, min(1.0, selection_conf_alpha)))
    combined_conf = alpha * creation_conf + (1.0 - alpha) * future_conf
    return combined_conf


@register_model("dream_lora_bp")
class DreamLoRA(TemplateLM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,
        mask_token_id: Optional[int] = 151666,
        block_add_threshold: Optional[float] = 0.5,
        decoded_token_threshold: Optional[float] = 0.9,
        skip_threshold: Optional[float] = 1.0,
        sampling_strategy: Optional[str] = "default",
        save_dir: Optional[str] = None,
        experiment_label: Optional[str] = None,
        show_speed: Optional[bool] = True,
        show_branch_details: Optional[bool] = True,
        profile_logging: Optional[bool] = None,
        use_uncertainty_logic: Optional[bool] = True,
        max_branches_kept: Optional[int] = 1,
        branching_factor: Optional[int] = 2,
        branch_confidence_decay: Optional[float] = 0.8,
        # [NEW PARAMETER] Control whether to keep the base branch
        branch_verification_mode: Optional[bool] = True,
        # [NEW PARAMETER] Control whether the base branch competes
        base_branch_competition: Optional[bool] = True,
        # [NEW PARAMETER] Validation toggle: force the base branch to win during verification
        verification_force_base_winner: Optional[bool] = False,
        branch_topp: Optional[float] = 0.5,
        selection_conf_alpha: Optional[float] = 0.5,
        # [NEW PARAMETER] Control whether to use full attention
        use_full_attention: Optional[bool] = True,
        use_sage_attention: Optional[bool] = False,
        attn_implementation: Optional[str] = None,
        torch_compile: Optional[bool] = False,
        torch_compile_mode: Optional[str] = "reduce-overhead",
        # [NEW PARAMETER] Control whether to merge LoRA weights at load time (default True, speeds inference)
        merge_lora_weights: Optional[bool] = True,
        # [NEW PARAMETER] Control whether to enable prefix caching (few-shot: avoid repeated prefix forward)
        enable_prefix_cache: Optional[bool] = False,
        # [NEW PARAMETER] Control whether to merge QKV projections (reduce params/compute)
        merge_qkv_projections: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self._ddp_model = None
        self._distributed = False
        self._local_rank = 0
        self._dist_rank = 0
        self._dist_world_size = 1
        self._init_distributed(device)
        self.accelerator = _AcceleratorCompat(self)

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)

        self.lora_path = lora_path
        self.block_size = block_size
        self.block_add_threshold = block_add_threshold
        self.skip_threshold = skip_threshold
        self.sampling_strategy = sampling_strategy
        self.decoded_token_threshold = decoded_token_threshold
        self.branch_topp = branch_topp
        self.selection_conf_alpha = selection_conf_alpha
        merge_lora_flag = _coerce_bool_flag(merge_lora_weights)
        self.merge_lora_weights = True if merge_lora_flag is None else merge_lora_flag
        self.target_dtype = get_dtype(dtype)
        self.attn_implementation = attn_implementation
        sage_attention_flag = _coerce_bool_flag(use_sage_attention)
        self.use_sage_attention = False if sage_attention_flag is None else sage_attention_flag
        # QKV merge flags (must be set before _create_model_and_tokenizer)
        self.merge_qkv_projections = _coerce_bool_flag(merge_qkv_projections)
        if self.merge_qkv_projections is None:
            self.merge_qkv_projections = False
        self._torch_compile_enabled = bool(torch_compile)
        self._torch_compile_mode = torch_compile_mode
        compile_kwargs = {"mode": torch_compile_mode} if torch_compile_mode else {}
        configure_torch_compile(enabled=self._torch_compile_enabled, kwargs=compile_kwargs)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)
        self._model_base = self.model
        self._wrap_model_for_distributed()

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps
        self.backend = "causal"
        self.truncation = False

        self.debug_print = kwargs.get("debug_print", False)

        self.save_dir = save_dir
        self.experiment_label = experiment_label
        show_speed_flag = _coerce_bool_flag(show_speed)
        if show_speed_flag is None:
            show_speed_flag = True
        self.show_speed = show_speed_flag

        show_branch_details_flag = _coerce_bool_flag(show_branch_details)
        if show_branch_details_flag is None:
            show_branch_details_flag = True
        self.show_branch_details = show_branch_details_flag
        self.use_uncertainty_logic = use_uncertainty_logic

        self.max_branches_kept = max_branches_kept
        self.branching_factor = branching_factor
        self.branch_confidence_decay = branch_confidence_decay
        self.branch_verification_mode = branch_verification_mode
        self.base_branch_competition = base_branch_competition
        self.verification_force_base_winner = verification_force_base_winner
        self.use_full_attention = use_full_attention
        # self.keep_base_branch = keep_base_branch
        # Shared KV cache (single copy)
        self._last_committed_logit = None
        self._committed_cache_length = 0
        self._static_cache_ready = False
        self._ensure_static_cache()
        self._dist_warmup_done = False
        self._torch_compile_warmup_done = False
        
        # Prefix caching settings
        self.enable_prefix_cache = _coerce_bool_flag(enable_prefix_cache)
        if self.enable_prefix_cache is None:
            self.enable_prefix_cache = False
        self._prefix_cache_length = 0  # cached prefix length (token count)
        self._prefix_tokens: Optional[torch.Tensor] = None  # cached prefix token IDs (for matching)
        
        # QKV merge fields are initialized before _create_model_and_tokenizer

        env_profile_flag = _coerce_bool_flag(os.environ.get("D2F_PROFILE"))
        if env_profile_flag is None:
            env_profile_flag = True
        override_profile_flag = _coerce_bool_flag(profile_logging)
        profile_enabled = override_profile_flag if override_profile_flag is not None else env_profile_flag
        profiler_prefix = "main."
        self._profiler = CodeProfiler(
            eval_logger,
            prefix=profiler_prefix,
            enabled=profile_enabled and self._dist_rank == 0,
        )
        self._rank_profile_enabled = profile_enabled and self._dist_rank == 0
        self._profile_logging_enabled = profile_enabled
        cuda_ratio = os.environ.get("D2F_CUDA_CLEANUP_FREE_RATIO", "0.20")
        gc_margin = os.environ.get("D2F_GC_CLEANUP_MARGIN", "2.5")
        try:
            self._cuda_cleanup_free_ratio = max(0.0, min(1.0, float(cuda_ratio)))
        except ValueError:
            self._cuda_cleanup_free_ratio = 0.20
        try:
            self._gc_cleanup_margin = max(1.0, float(gc_margin))
        except ValueError:
            self._gc_cleanup_margin = 2.5

        self._warmup_distributed_collectives()
        self._warmup_torch_compile()
        
        # Initialize code evaluator (for pass_at_1)
        self._pass_at_k_metric = None
        if CODE_EVAL_AVAILABLE:
            try:
                self._pass_at_k_metric = hf_evaluate.load("code_eval")
            except Exception as e:
                eval_logger.warning(f"Failed to load code_eval metric: {e}")
                self._pass_at_k_metric = None

    @property
    def batch_size(self): return self.batch_size_per_gpu
    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def device(self): return self._device
    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size
    @property
    def dist_rank(self) -> int: return self._dist_rank
    @property
    def dist_world_size(self) -> int: return self._dist_world_size
    @property
    def is_distributed(self) -> bool: return self._distributed and self._dist_world_size > 1
    @property
    def model_module(self): return self.model.module if isinstance(self.model, DDP) else self.model

    def _rank_profile(self, label: str, duration_s: float, extra: Optional[Union[int, str]] = None) -> None:
        if not self._rank_profile_enabled:
            return
        prefix = f"profile_rank{self._dist_rank}:{label}"
        if extra is not None:
            prefix = f"{prefix}[{extra}]"
        eval_logger.info("%s %.3f ms", prefix, duration_s * 1000.0)

    def _init_distributed(self, preferred_device: str) -> None:
        """Initialize distributed env, binding device before NCCL init to avoid deadlock."""
        self._rank = 0
        self._world_size = 1
        self._dist_rank = 0
        self._dist_world_size = 1
        self._device = torch.device("cpu")

        # =================================================================
        # [FIX] Bind device before init_process_group to prevent NCCL deadlocks from GPU contention
        # =================================================================
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            self._local_rank = int(local_rank_env)
            if torch.cuda.is_available():
                try:
                    # Explicitly set the GPU visible to this process
                    torch.cuda.set_device(self._local_rank)
                except Exception as e:
                    eval_logger.warning(f"Pre-init set_device failed for rank {self._local_rank}: {e}")

        # Configure distributed backend
        backend_env = os.environ.get("DIST_BACKEND")
        init_method = os.environ.get("DIST_INIT_METHOD", "env://")
        dist_timeout = timedelta(weeks=52)

        # Try initializing the process group
        if dist.is_available() and not dist.is_initialized():
            backend = backend_env
            if backend is None:
                if torch.cuda.is_available():
                    backend = "nccl"
                else:
                    backend = "gloo"
            try:
                # At this point the CUDA device is already bound (for NCCL)
                dist.init_process_group(backend=backend, init_method=init_method, timeout=dist_timeout)
            except Exception as exc:  # noqa: BLE001
                eval_logger.warning(f"Failed to initialize distributed backend '{backend}' via {init_method}: {exc}. Continuing in single-process mode.")

        # If distributed init succeeded
        if dist.is_available() and dist.is_initialized():
            self._distributed = True
            self._dist_rank = dist.get_rank()
            self._dist_world_size = dist.get_world_size()
            self._rank = 0
            self._world_size = 1
            
            # Double-check local_rank in case the env var was missing
            self._local_rank = int(os.environ.get("LOCAL_RANK", self._dist_rank))

            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                if self._local_rank >= n_gpus:
                    raise RuntimeError(
                        f"Local rank {self._local_rank} >= available CUDA devices ({n_gpus}).\n"
                        "Ensure NUM_BRANCHES/--nproc_per_node does not exceed the number of visible GPUs,\n"
                        "or set CUDA_VISIBLE_DEVICES appropriately (e.g. export CUDA_VISIBLE_DEVICES=0,1)."
                    )
                # Double-safety: ensure the device is set correctly
                torch.cuda.set_device(self._local_rank)
                self._device = torch.device(f"cuda:{self._local_rank}")
            else:
                self._device = torch.device("cpu")

            # Silence worker logs; keep only rank 0 output
            _suppress_lm_eval_logging_for_workers()
            return

        # =================================================================
        # Single-process / fallback
        # =================================================================
        gpus = torch.cuda.device_count()
        device_list: Set[str] = set(["cuda", "cpu"] + [f"cuda:{i}" for i in range(gpus)] + ["mps", "mps:0"] + [f"npu:{i}" for i in range(gpus)])

        chosen_device = preferred_device if preferred_device in device_list else None
        if chosen_device is None:
            # Only log auto-selection when not distributed and device unspecified to avoid spam
            eval_logger.info("Device not specified or unavailable, auto-selecting...")
            if torch.cuda.is_available():
                chosen_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                chosen_device = "mps"
            else:
                chosen_device = "cpu"

        self._device = torch.device(chosen_device)
        
        # Check MPS version compatibility
        if self._device.type == "mps" and version.parse(torch.__version__) < version.parse("2.1"):
            raise RuntimeError(f"mps requires torch >= 2.1. You have {torch.__version__}")

        # Check CUDA availability
        if self._device.type.startswith("cuda") and not torch.cuda.is_available():
            eval_logger.warning(f"Requested CUDA device '{chosen_device}' is unavailable. Falling back to CPU.")
            self._device = torch.device("cpu")
            
    def _wrap_model_for_distributed(self) -> None:
        if not getattr(self, "model", None):
            return

        if not self._distributed or self._dist_world_size <= 1:
            return

        if isinstance(self.model, DDP):
            self._ddp_model = self.model
            return

        has_trainable_params = any(param.requires_grad for param in self.model.parameters())
        if not has_trainable_params:
            eval_logger.info("Model parameters are frozen; skipping DDP wrapping and running in inference-only distributed mode.")
            self._ddp_model = None
            return

        ddp_kwargs: Dict[str, Union[List[int], bool, None]] = {"broadcast_buffers": False, "find_unused_parameters": False}
        if self.device.type == "cuda":
            if self.device.index is not None:
                ddp_kwargs["device_ids"] = [self.device.index]
            else:
                ddp_kwargs["device_ids"] = None

        self.model = DDP(self.model, **ddp_kwargs)  # type: ignore[arg-type]
        self._ddp_model = self.model

    def _get_dream_model(self) -> DreamModel:
        module = self.model.module if isinstance(self.model, DDP) else self.model
        base_model = getattr(module, "base_model", None)
        if base_model is not None:
            candidate = getattr(base_model, "model", None)
            if isinstance(candidate, DreamModel):
                return candidate
            if isinstance(base_model, DreamModel):
                return base_model
        candidate = getattr(module, "model", None)
        if isinstance(candidate, DreamModel):
            return candidate
        if isinstance(module, DreamModel):
            return module
        raise RuntimeError("DreamModel instance not found within wrapped module")

    def _ensure_static_cache(self) -> None:
        if self._static_cache_ready:
            return
        dream_model = self._get_dream_model()
        dtype = self.target_dtype if isinstance(self.target_dtype, torch.dtype) else None
        dream_model.allocate_shared_kv_cache(self.max_length, device=self.device, dtype=dtype)
        self._committed_cache_length = dream_model.get_shared_cache_length()
        self._static_cache_ready = True

    def _reset_shared_cache(self, preserve_prefix: bool = False) -> None:
        """Reset the shared KV cache.

        Args:
            preserve_prefix: if True, keep the prefix cache (only clear content after the prefix)
        """
        self._ensure_static_cache()
        dream_model = self._get_dream_model()
        decoder = dream_model.get_decoder()
        cache_impl = getattr(decoder, "shared_kv_cache", None)
        
        if preserve_prefix and self.enable_prefix_cache and self._prefix_cache_length > 0:
            # Clear only the portion after the prefix while keeping the prefix cache
            if isinstance(cache_impl, SharedStaticCache):
                # Set cache length to the prefix length so later KV entries overwrite past it
                cache_impl.set_seq_length(self._prefix_cache_length)
                self._committed_cache_length = self._prefix_cache_length
                self._last_committed_logit = None
            else:
                dream_model.reset_shared_kv_cache()
                self._last_committed_logit = None
                self._committed_cache_length = 0
        else:
            dream_model.reset_shared_kv_cache()
            self._last_committed_logit = None
            self._committed_cache_length = 0

    def _apply_committed_cache_length(self, length: int) -> None:
        if length is None:
            return
        length = int(length)
        if length < 0:
            return
        self._committed_cache_length = length
        dream_model = self._get_dream_model()
        decoder = dream_model.get_decoder()
        cache_impl = getattr(decoder, "shared_kv_cache", None)
        if isinstance(cache_impl, SharedStaticCache):
            cache_impl.set_seq_length(length)

    def _warmup_distributed_collectives(self) -> None:
        """Run a lightweight communication round to amortize NCCL/Gloo setup costs."""
        if not (self._distributed and self._dist_world_size > 1):
            return
        if getattr(self, "_dist_warmup_done", False):
            return

        try:
            dummy_tokens = self._estimate_warmup_token_count()
            vocab_size = max(2, self._infer_vocab_size())
            token_template = torch.arange(dummy_tokens, dtype=torch.long) % vocab_size
            if token_template.numel() == 0:
                token_template = torch.zeros(1, dtype=torch.long)
                dummy_tokens = 1

            self._reset_shared_cache()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            payload_entries: List[Dict[str, Any]] = []
            world_size = self._dist_world_size
            for peer_rank in range(world_size):
                warmup_task = BranchTask(
                    branch_index=peer_rank,
                    branch_id=-(peer_rank + 1),
                    tokens=token_template.clone(),
                    input_start_pos=0,
                    shared_prefix_end=0,
                    update_kvcache_len=dummy_tokens,
                )
                payload_entries.append(
                    {
                        "task": warmup_task,
                        "reset_cache": False,
                        "epoch": -1,
                        "prompt_length": 0,
                    }
                )

            if self._dist_rank == 0:
                self._distributed_forward(payload_entries)
            else:
                self._distributed_forward(None)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            dummy_result = self._create_dummy_gather_result(dummy_tokens)
            self._gather_branch_results([dummy_result])
        except Exception as exc:  # noqa: BLE001
            if self._dist_rank == 0:
                eval_logger.warning("Distributed warmup failed: %s", exc)
        finally:
            try:
                self._reset_shared_cache()
            except Exception:  # noqa: BLE001
                pass
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        self._dist_warmup_done = True

    def _estimate_warmup_token_count(self) -> int:
        token_count = 32
        try:
            if self.block_size is not None:
                token_count = int(self.block_size)
        except (TypeError, ValueError):
            token_count = 32

        try:
            if self.max_length is not None:
                max_length_val = int(self.max_length)
                token_count = min(token_count, max_length_val)
        except (TypeError, ValueError):
            pass

        token_count = max(1, token_count)
        return min(token_count, 128)

    def _infer_vocab_size(self) -> int:
        vocab_size = getattr(getattr(self, "tokenizer", None), "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return int(vocab_size)

        module = self.model_module
        config_vocab = getattr(getattr(module, "config", None), "vocab_size", None)
        if isinstance(config_vocab, int) and config_vocab > 0:
            return int(config_vocab)

        get_embeddings = getattr(module, "get_output_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None and getattr(weight, "shape", None):
                try:
                    return int(weight.shape[0])
                except (TypeError, ValueError):
                    pass

        return 32000

    def _create_dummy_gather_result(self, token_count: int) -> BranchWorkerResult:
        vocab_size = self._infer_vocab_size()
        dtype = torch.bfloat16
        logits_tensor: Optional[torch.Tensor]

        try:
            logits_tensor = torch.zeros((1, token_count, vocab_size), dtype=dtype, device=self.device)
        except RuntimeError:
            fallback_vocab = max(2, min(vocab_size, 2048))
            try:
                logits_tensor = torch.zeros((1, token_count, fallback_vocab), dtype=dtype, device=self.device)
            except RuntimeError:
                logits_tensor = None

        return BranchWorkerResult(
            branch_index=0,
            logits=logits_tensor,
            cache_updates=None,
            source_rank=self._dist_rank,
            error=None,
        )

    def _warmup_torch_compile(self) -> None:
        if not (self._torch_compile_enabled and hasattr(torch, "compile")):
            return
        if getattr(self, "_torch_compile_warmup_done", False):
            return

        token_count = self._estimate_warmup_token_count()
        vocab_size = self._infer_vocab_size()
        high = max(vocab_size, 2)

        dummy_tokens = torch.randint(
            low=0,
            high=high,
            size=(1, token_count),
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = None
        if not self.use_full_attention:
            dtype = self.target_dtype if isinstance(self.target_dtype, torch.dtype) else torch.bfloat16
            try:
                max_length_val = int(self.max_length) if self.max_length is not None else token_count
            except (TypeError, ValueError):
                max_length_val = token_count
            try:
                block_size_val = int(self.block_size) if self.block_size is not None else token_count
            except (TypeError, ValueError):
                block_size_val = token_count
            max_length_val = max(max_length_val, token_count)
            block_size_val = max(1, block_size_val)
            dtype_for_mask = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
            full_mask = create_full_block_attention_mask(
                prompt_length=0,
                max_length=max_length_val,
                block_size=block_size_val,
                device=self.device,
                dtype=dtype_for_mask,
            )
            attention_mask = extract_attention_mask(
                full_mask=full_mask,
                start_pos=0,
                input_length=token_count,
                cache_length=0,
                use_full_attention=False,
            )

        try:
            with torch.inference_mode():
                self.model(
                    dummy_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                    update_kvcache=token_count,
                )
        except Exception as exc:  # noqa: BLE001
            if self._dist_rank == 0:
                eval_logger.warning("torch.compile warmup failed: %s", exc)
        finally:
            try:
                self._reset_shared_cache()
            except Exception:  # noqa: BLE001
                pass
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        self._torch_compile_warmup_done = True

    def _broadcast_kv_update(
        self,
        start: int,
        end: int,
        layer_indices: List[int],
        src_rank: int,
    ) -> None:
        if not (self.is_distributed and self._dist_world_size > 1):
            return
        block_len = end - start
        if block_len <= 0 or not layer_indices:
            return

        dream_model = self._get_dream_model()
        base_model = dream_model.get_decoder()
        cache_impl = getattr(base_model, "shared_kv_cache", None)
        if cache_impl is None:
            raise RuntimeError("Shared KV cache is not initialised on this rank")

        num_kv_heads = cache_impl.key.shape[1]
        head_dim = cache_impl.key.shape[3]
        dtype = cache_impl.key.dtype
        block_len = end - start

        for layer_idx in layer_indices:
            key_buffer = torch.empty((1, num_kv_heads, block_len, head_dim), dtype=dtype, device=self.device)
            value_buffer = torch.empty_like(key_buffer)
            if self._dist_rank == src_rank:
                key_slice, value_slice = dream_model.get_shared_kv_block(layer_idx, start, end)
                key_buffer.copy_(key_slice)
                value_buffer.copy_(value_slice)
            dist.broadcast(key_buffer, src=src_rank)
            dist.broadcast(value_buffer, src=src_rank)
            dream_model.write_shared_kv_block(layer_idx, start, key_buffer, value_buffer)

    def _commit_shared_cache(
        self,
        cache_updates: Optional[List[Optional[CacheUpdate]]],
        *,
        source_rank: int,
    ) -> Tuple[int, int]:
        commit_timer = self._profiler.tic()
        start, end = self._get_dream_model().commit_shared_kv_chunks(cache_updates)
        self._committed_cache_length = end
        committed_span = max(0, end - start)
        self._profiler.toc(commit_timer, "kv_commit_local", extra=committed_span if committed_span > 0 else None)
        if (
            cache_updates
            and self.is_distributed
            and self._dist_world_size > 1
            and end > start
        ):
            layer_indices = sorted({update.layer_idx for update in cache_updates if update is not None})
            if layer_indices:
                if self._dist_rank == 0:
                    broadcast_timer = self._profiler.tic()
                    payload_list = [
                        {
                            "broadcast_update": {
                                "start": start,
                                "end": end,
                                "layers": layer_indices,
                                "src_rank": source_rank,
                            }
                        }
                        for _ in range(self._dist_world_size)
                    ]
                    marker, _ = self._distributed_forward(payload_list)
                    self._profiler.toc(
                        broadcast_timer,
                        "kv_commit_broadcast",
                        extra=f"{len(layer_indices)}x{committed_span}",
                    )
                    if marker != "broadcast":
                        eval_logger.warning(
                            "Unexpected marker '%s' from broadcast dispatch", marker
                        )
        return start, end

    def _broadcast_branch_tokens(self, tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if not (self._distributed and self._dist_world_size > 1):
            return tokens if tokens is not None else torch.empty((0, 0), dtype=torch.long)

        start = time.perf_counter()
        if self._dist_rank == 0:
            if tokens is None:
                tokens_cpu = torch.empty((0, 0), dtype=torch.long)
            else:
                tokens_cpu = tokens.to(torch.long).cpu()
            shape_tensor = torch.tensor([tokens_cpu.size(0), tokens_cpu.size(1) if tokens_cpu.numel() > 0 else 0], dtype=torch.long)
        else:
            tokens_cpu = None
            shape_tensor = torch.zeros(2, dtype=torch.long)

        dist.broadcast(shape_tensor, src=0)
        branch_count, seq_len = shape_tensor.tolist()

        if self._dist_rank != 0:
            tokens_cpu = torch.empty((branch_count, seq_len), dtype=torch.long) if branch_count * seq_len > 0 else torch.empty((branch_count, seq_len), dtype=torch.long)

        if branch_count * seq_len > 0:
            dist.broadcast(tokens_cpu, src=0)

        self._rank_profile("broadcast_tokens", time.perf_counter() - start, extra=f"{branch_count}x{seq_len}")
        return tokens_cpu if tokens_cpu is not None else torch.empty((0, 0), dtype=torch.long)

    def _broadcast_scalar(self, value: Optional[int]) -> int:
        if not (self._distributed and self._dist_world_size > 1):
            return 0 if value is None else value
        tensor = torch.tensor([0 if value is None else value], dtype=torch.long)
        if self._dist_rank == 0 and value is not None:
            tensor[0] = value
        start = time.perf_counter()
        dist.broadcast(tensor, src=0)
        self._rank_profile("broadcast_scalar", time.perf_counter() - start)
        return int(tensor.item())

    def _gather_branch_results(self, local_results: List[BranchWorkerResult]) -> List[BranchWorkerResult]:
        if not (self._distributed and self._dist_world_size > 1):
            return local_results
        world_size = self._dist_world_size
        rank = self._dist_rank
        sync_cuda = torch.cuda.is_available() and self.device.type == "cuda"

        def _sync() -> None:
            if sync_cuda:
                torch.cuda.synchronize()

        total_start = time.perf_counter()

        local_metadata: List[BranchWorkerResultMetadata] = []
        logits_lengths: List[int] = []
        flat_chunks: List[torch.Tensor] = []

        _sync()
        stage_start = time.perf_counter()
        for result in local_results:
            logits_tensor = result.logits.detach() if result.logits is not None else None
            has_logits = logits_tensor is not None and logits_tensor.numel() > 0

            if has_logits:
                logits_tensor = logits_tensor.to(self.device, dtype=torch.bfloat16).contiguous()
                result.logits = logits_tensor
                flat_chunks.append(logits_tensor.view(-1))
                logits_lengths.append(int(logits_tensor.numel()))
            else:
                logits_lengths.append(0)

            local_metadata.append(
                BranchWorkerResultMetadata(
                    branch_index=result.branch_index,
                    has_logits=has_logits,
                    logits_shape=tuple(logits_tensor.shape) if has_logits else tuple(),
                    logits_dtype=logits_tensor.dtype if has_logits else None,
                    cache_updates=result.cache_updates,
                    source_rank=result.source_rank,
                    error=result.error,
                )
            )

        if flat_chunks:
            flat_local_logits = torch.cat(flat_chunks, dim=0)
        else:
            flat_local_logits = torch.empty(0, dtype=torch.bfloat16, device=self.device)
        _sync()
        self._rank_profile("gather_results.prepare", time.perf_counter() - stage_start)

        _sync()
        stage_start = time.perf_counter()
        count_tensor = torch.tensor([len(local_metadata)], dtype=torch.long, device=self.device)
        count_gather: List[torch.Tensor] = [torch.zeros_like(count_tensor) for _ in range(world_size)]
        dist.all_gather(count_gather, count_tensor)
        _sync()
        self._rank_profile("gather_results.counts", time.perf_counter() - stage_start)
        counts = [int(t.item()) for t in count_gather]
        max_count = max(counts) if counts else 0

        if self._rank_profile_enabled and rank == 0:
            eval_logger.info("profile_rank0:gather.counts %s", counts)

        if max_count > 0:
            length_tensor = torch.zeros(max_count, dtype=torch.long, device=self.device)
            if logits_lengths:
                length_tensor[: len(logits_lengths)] = torch.tensor(logits_lengths, dtype=torch.long, device=self.device)
            length_gather: List[torch.Tensor] = [torch.zeros_like(length_tensor) for _ in range(world_size)]
            _sync()
            stage_start = time.perf_counter()
            dist.all_gather(length_gather, length_tensor)
            _sync()
            self._rank_profile("gather_results.lengths", time.perf_counter() - stage_start)
        else:
            length_tensor = torch.zeros(0, dtype=torch.long, device=self.device)
            length_gather = [torch.zeros(0, dtype=torch.long, device=self.device) for _ in range(world_size)]
            self._rank_profile("gather_results.lengths", 0.0)

        gathered_metadata: List[List[BranchWorkerResultMetadata]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        _sync()
        stage_start = time.perf_counter()
        dist.all_gather_object(gathered_metadata, local_metadata)
        _sync()
        self._rank_profile("gather_results.objects", time.perf_counter() - stage_start)

        flat_length_tensor = torch.tensor([flat_local_logits.numel()], dtype=torch.long, device=self.device)
        flat_length_gather: List[torch.Tensor] = [torch.zeros_like(flat_length_tensor) for _ in range(world_size)]
        _sync()
        stage_start = time.perf_counter()
        dist.all_gather(flat_length_gather, flat_length_tensor)
        _sync()
        self._rank_profile("gather_results.flat_lengths", time.perf_counter() - stage_start)
        flat_lengths = [int(t.item()) for t in flat_length_gather]
        max_flat = max(flat_lengths) if flat_lengths else 0

        if max_flat > 0:
            padded_logits = torch.zeros(max_flat, dtype=torch.bfloat16, device=self.device)
            if flat_local_logits.numel() > 0:
                padded_logits[: flat_local_logits.numel()] = flat_local_logits
            logits_gather: List[torch.Tensor] = [torch.zeros_like(padded_logits) for _ in range(world_size)]
            _sync()
            stage_start = time.perf_counter()
            dist.all_gather(logits_gather, padded_logits)
            _sync()
            self._rank_profile("gather_results.logits", time.perf_counter() - stage_start)
        else:
            padded_logits = torch.empty(0, dtype=torch.bfloat16, device=self.device)
            logits_gather = [torch.zeros(0, dtype=torch.bfloat16, device=self.device) for _ in range(world_size)]
            self._rank_profile("gather_results.logits", 0.0)

        _sync()
        self._rank_profile("gather_results", time.perf_counter() - total_start)

        if rank != 0:
            return local_results

        _sync()
        stage_start = time.perf_counter()
        merged: List[BranchWorkerResult] = []
        for src_rank in range(world_size):
            rank_meta = gathered_metadata[src_rank]
            if not rank_meta:
                continue

            rank_count = counts[src_rank] if src_rank < len(counts) else 0
            if rank_count == 0:
                continue

            rank_lengths_tensor = length_gather[src_rank] if max_count > 0 else torch.zeros(0, dtype=torch.long, device=self.device)
            rank_lengths = rank_lengths_tensor.tolist()

            rank_flat = logits_gather[src_rank] if max_flat > 0 else torch.empty(0, dtype=torch.bfloat16, device=self.device)
            rank_flat_limit = flat_lengths[src_rank] if src_rank < len(flat_lengths) else 0
            cursor = 0

            for idx in range(rank_count):
                meta = rank_meta[idx]
                length = rank_lengths[idx] if idx < len(rank_lengths) else 0
                logits_tensor: Optional[torch.Tensor] = None

                if meta.has_logits and length > 0 and rank_flat_limit > 0:
                    end = min(cursor + length, rank_flat_limit)
                    slice_tensor = rank_flat[cursor:end]
                    cursor = end
                    if slice_tensor.numel() == length:
                        logits_tensor = slice_tensor.reshape(meta.logits_shape)
                        if meta.logits_dtype is not None and logits_tensor.dtype != meta.logits_dtype:
                            logits_tensor = logits_tensor.to(meta.logits_dtype)
                    else:
                        cursor = end

                merged.append(
                    BranchWorkerResult(
                        branch_index=meta.branch_index,
                        logits=logits_tensor,
                        cache_updates=meta.cache_updates,
                        source_rank=meta.source_rank,
                        error=meta.error,
                    )
                )

        _sync()
        self._rank_profile("gather_results.reconstruct", time.perf_counter() - stage_start)

        return merged

    def _distributed_forward(self, payload_list: Optional[List[Dict[str, Any]]]) -> Tuple[str, Any]:
        if not (self._distributed and self._dist_world_size > 1):
            raise RuntimeError("_distributed_forward should only be used in DDP mode")
        
        world_size = self._dist_world_size
        rank = self._dist_rank

        # Normalize payloads to match world_size for rank dispatch
        obj_scatter_list: Optional[List[Optional[Dict[str, Any]]]] = None
        tok_scatter_list: Optional[List[torch.Tensor]] = None
        lengths_tensor = torch.zeros(world_size, dtype=torch.long, device=self.device)

        if rank == 0:
            if payload_list is None:
                raise ValueError("payload_list must be provided on rank 0")

            if isinstance(payload_list, dict):
                payload_entries: List[Optional[Dict[str, Any]]] = [payload_list]
            else:
                payload_entries = list(payload_list)

            if len(payload_entries) < world_size:
                payload_entries = payload_entries + [None] * (world_size - len(payload_entries))
            elif len(payload_entries) > world_size:
                payload_entries = payload_entries[:world_size]

            stop_template = next((entry for entry in payload_entries if isinstance(entry, dict) and entry.get("stop", False)), None)
            if stop_template is not None:
                payload_entries = [stop_template for _ in range(world_size)]

            committed_length = int(self._committed_cache_length)
            normalized_entries: List[Dict[str, Any]] = []
            for entry in payload_entries:
                if entry is None:
                    payload: Dict[str, Any] = {"task": None, "noop": True}
                else:
                    payload = dict(entry)
                payload.setdefault("committed_cache_length", committed_length)
                normalized_entries.append(payload)
            payload_entries = normalized_entries

            token_buffers: List[Optional[torch.Tensor]] = []
            for idx in range(world_size):
                payload = payload_entries[idx]
                task = payload.get("task") if payload is not None else None
                tokens = getattr(task, "tokens", None)
                if tokens is not None:
                    tokens = tokens.to(self.device, dtype=torch.long).contiguous()
                    lengths_tensor[idx] = tokens.numel()
                    task.tokens = None  # send tensor only once
                else:
                    tokens = None
                    lengths_tensor[idx] = 0
                token_buffers.append(tokens)

            max_length = int(lengths_tensor.max().item()) if lengths_tensor.numel() > 0 else 0
            tok_scatter_list = []
            for tokens in token_buffers:
                if max_length == 0:
                    padded = torch.empty(0, dtype=torch.long, device=self.device)
                elif tokens is not None and tokens.numel() == max_length:
                    # Reuse the original tensor when shapes match to avoid extra padding
                    padded = tokens
                else:
                    padded = torch.zeros(max_length, dtype=torch.long, device=self.device)
                    if tokens is not None and tokens.numel() > 0:
                        padded[: tokens.numel()].copy_(tokens)
                tok_scatter_list.append(padded)

            obj_scatter_list = payload_entries

            if self._rank_profile_enabled:
                try:
                    lengths_snapshot = [int(v) for v in lengths_tensor.detach().cpu().tolist()]
                except RuntimeError:
                    lengths_snapshot = []
                eval_logger.info("profile_rank0:dispatch.lengths %s", lengths_snapshot)

        self._ensure_static_cache()
        start = time.perf_counter()

        # Broadcast token lengths so receivers can allocate correctly
        dist.broadcast(lengths_tensor, src=0)
        max_length = int(lengths_tensor.max().item()) if lengths_tensor.numel() > 0 else 0

        if max_length > 0:
            output_tokens = torch.empty(max_length, dtype=torch.long, device=self.device)
        else:
            output_tokens = torch.empty(0, dtype=torch.long, device=self.device)

        dist.scatter(output_tokens, tok_scatter_list, src=0)

        output_obj_list: List[Optional[Dict[str, Any]]] = [None]
        dist.scatter_object_list(output_obj_list, obj_scatter_list, src=0)

        self._rank_profile("broadcast_payload", time.perf_counter() - start)

        received = output_obj_list[0]
        committed_length = None
        if isinstance(received, dict):
            committed_length = received.get("committed_cache_length")
        if committed_length is not None:
            self._apply_committed_cache_length(committed_length)
        local_length = int(lengths_tensor[rank].item()) if lengths_tensor.numel() > 0 else 0
        tokens_view = output_tokens[:local_length] if local_length <= output_tokens.numel() else output_tokens

        if self._rank_profile_enabled:
            eval_logger.info(
                "profile_rank%d:dispatch.local length=%d has_task=%s",
                self._dist_rank,
                local_length,
                bool(received and received.get("task")),
            )

        if received is None:
            return ("noop", None)

        if isinstance(received, dict) and received.get("stop", False):
            return ("stop", None)

        if isinstance(received, dict) and received.get("reset_cache", False):
            self._reset_shared_cache()

        broadcast_cmd = received.get("broadcast_update") if isinstance(received, dict) else None
        if broadcast_cmd is not None:
            layers = broadcast_cmd.get("layers", [])
            self._broadcast_kv_update(
                start=int(broadcast_cmd.get("start", 0)),
                end=int(broadcast_cmd.get("end", 0)),
                layer_indices=[int(layer) for layer in layers],
                src_rank=int(broadcast_cmd.get("src_rank", 0)),
            )
            return ("broadcast", None)

        task: BranchTask = received.get("task", None)
        prompt_length = received.get("prompt_length", 0)

        local_results: List[BranchWorkerResult] = []
        if task is not None:
            if tokens_view.numel() == 0:
                task.tokens = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                task.tokens = tokens_view
            if self.use_full_attention:
                full_mask = None
            else:
                full_mask = create_full_block_attention_mask(
                    prompt_length=prompt_length,
                    max_length=self.max_length,
                    block_size=self.block_size,
                    device=self.device,
                    dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16,
                )

            forward_time = 0.0
            forward_token_count = 0
            forward_calls = 0
            dream_model = self._get_dream_model()

            if task.tokens.numel() == 0:
                local_results.append(
                    BranchWorkerResult(
                        branch_index=task.branch_index,
                        logits=None,
                        cache_updates=None,
                        source_rank=self._dist_rank,
                        error=None,
                    )
                )

            branch_input = task.tokens.to(self.device).unsqueeze(0)
            forward_token_count += int(branch_input.shape[1])
            forward_calls += 1

            cache_length = dream_model.get_shared_cache_length()
            attention_mask = extract_attention_mask(
                full_mask=full_mask,
                start_pos=task.input_start_pos,
                input_length=branch_input.shape[1],
                cache_length=cache_length,
                use_full_attention=self.use_full_attention,
            )

            forward_start = time.perf_counter()
            outputs = self.model(
                branch_input,
                attention_mask=attention_mask,
                use_cache=True,
                update_kvcache=task.update_kvcache_len,
            )
            forward_time += time.perf_counter() - forward_start

            if outputs.past_key_values is None:
                local_results.append(
                    BranchWorkerResult(
                        branch_index=task.branch_index,
                        logits=None,
                        cache_updates=None,
                        source_rank=self._dist_rank,
                        error="Model returned no cache",
                    )
                )

            raw_logits = outputs.logits.detach().to(torch.bfloat16).contiguous()
            cache_updates = outputs.past_key_values if isinstance(outputs.past_key_values, list) else None

            local_results.append(
                BranchWorkerResult(
                    branch_index=task.branch_index,
                    logits=raw_logits,
                    cache_updates=cache_updates,
                    source_rank=self._dist_rank,
                    error=None,
                )
            )

            if self._rank_profile_enabled:
                eval_logger.info(
                    "profile_rank%d:forward.task branch=%d tokens=%d",
                    self._dist_rank,
                    task.branch_index,
                    int(branch_input.shape[1]),
                )

            if forward_calls > 0:
                self._rank_profile(
                    "forward",
                    forward_time,
                    extra=f"{forward_calls}x{forward_token_count}",
                )

        gathered = self._gather_branch_results(local_results)
        if self._dist_rank == 0:
            return ("results", gathered)
        return ("continue", None)

    def _distributed_worker_loop(self) -> None:
        if not (self._distributed and self._dist_world_size > 1):
            return
        while True:
            marker, _ = self._distributed_forward(None)
            if marker == "stop":
                break

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        target_dtype = get_dtype(dtype)
        model_config = DreamConfig.from_pretrained(pretrained)
        if self.attn_implementation:
            model_config.attn_implementation = self.attn_implementation
            model_config._attn_implementation = self.attn_implementation
        model_config.use_sage_attention = bool(self.use_sage_attention)

        self.model = DreamModel.from_pretrained(
            pretrained,
            config=model_config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()
        peft_config = PeftConfig.from_pretrained(self.lora_path)
        peft_model = PeftModel.from_pretrained(self.model, self.lora_path)
        
        # Merge LoRA weights based on the flag
        if self.merge_lora_weights:
            # Merge LoRA into the base model to avoid runtime merge cost (faster inference, less memory)
            eval_logger.info("Merging LoRA weights into base model for faster inference...")
            self.model = peft_model.merge_and_unload()
        else:
            # Keep the PeftModel wrapper and merge dynamically (useful when swapping LoRA adapters)
            eval_logger.info("Keeping LoRA weights separate (will merge dynamically during inference)...")
            self.model = peft_model
        
        if target_dtype is not None and target_dtype != "auto":
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device)
        if hasattr(self.model, "config"):
            self.model.config.attn_implementation = model_config.attn_implementation
            self.model.config.use_sage_attention = model_config.use_sage_attention
        
        # Merge QKV projections after model load if enabled
        if self.merge_qkv_projections:
            eval_logger.info("Merging QKV projections for optimization...")
            merge_qkv_projections(self.model)
            # Patch DreamAttention forward to support merged qkv_proj
            _patch_dream_attention_forward()
        
        # Handle config: merged model is DreamModel directly; otherwise access via base_model
        if self.merge_lora_weights:
            # After merging, the model is DreamModel and no longer has base_model
            if hasattr(self.model, "model") and hasattr(self.model.model, "config"):
                self.model.model.config.use_sage_attention = model_config.use_sage_attention
        else:
            # Without merging, update configs through base_model
            base_model_ref = getattr(self.model, "base_model", None)
            if base_model_ref is not None:
                dream_model_ref = getattr(base_model_ref, "model", None)
                if dream_model_ref is not None and hasattr(dream_model_ref, "config"):
                    dream_model_ref.config.use_sage_attention = model_config.use_sage_attention

        if self.attn_implementation is not None and torch.cuda.is_available():
            if self.attn_implementation == "flash_attention_2":
                with _sdpa_kernel_context(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False,
                ):
                    pass
            elif self.attn_implementation == "sdpa":
                with _sdpa_kernel_context(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True,
                ):
                    pass

        # if self._torch_compile_enabled and hasattr(torch, "compile"):
        #     try:
        #         self.model = torch.compile(self.model, mode=self._torch_compile_mode)
        #     except Exception as exc:  # noqa: BLE001
        #         eval_logger.warning(f"torch.compile failed, falling back to eager mode: {exc}")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        special_tokens_kwargs = {}
        
        if add_special_tokens is None:
            if self.backend == "causal": 
                special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
        else: 
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len: 
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(self, strings: List[str], padding_side: str = "left", left_truncate_len: int = None, truncation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        add_special_tokens = {}
        if self.backend == "causal": 
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}
        encoding = self.tokenizer(strings, truncation=truncation, padding="longest", return_tensors="pt", **add_special_tokens)
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(f"Left truncation applied. Original sequence length was {original_lengths}, truncating to last {left_truncate_len} tokens. Some content will be lost.")
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True): return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _count_tokens_after_truncation(self, response_text: str, until_terms: List[str] = None) -> int:
        truncated_text = response_text
        if until_terms and not self.escape_until:
            for term in until_terms:
                if len(term) > 0:
                    truncated_text = truncated_text.split(term)[0]
        generated_answer_ids = torch.tensor(self.tokenizer(truncated_text)["input_ids"])
        if self.mask_token_id is not None:
            return int((generated_answer_ids != self.mask_token_id).sum())
        return int(generated_answer_ids.numel())

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str, additional_config: Optional[dict] = None) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        try:
            chat_templated = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=not add_generation_prompt)
        except jinja2.exceptions.TemplateError:
            eval_logger.warning("Failed to apply chat template. removing the system role in chat history.")
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=not add_generation_prompt)
        return chat_templated

    @property
    def tokenizer_name(self) -> str: 
        return self.tokenizer.name_or_path.replace("/", "__")

    def _detect_prefix_length(self, current_tokens: torch.Tensor, previous_tokens: torch.Tensor) -> int:
        """Detect the common prefix length between two token sequences."""
        if previous_tokens is None:
            return 0
        
        min_len = min(current_tokens.shape[0], previous_tokens.shape[0])
        if min_len == 0:
            return 0
        
        # Match from the beginning
        match_mask = current_tokens[:min_len] == previous_tokens[:min_len]
        if match_mask.all():
            return min_len
        elif match_mask.any():
            # Locate the last matching position
            match_indices = match_mask.nonzero(as_tuple=True)[0]
            if len(match_indices) > 0:
                return int(match_indices[-1].item()) + 1
        
        return 0

    # [MODIFIED] Return a tuple containing the final branch statistics
    def _generate_enhanced_speculative(self, prompt: torch.Tensor, prefix_length: int = 0) -> Tuple[List[int], int, int, Dict, float, float, List[float]]:
        self.model.eval()
        prompt_length = prompt.shape[1]
        if self.max_length is not None and prompt_length > self.max_length:
            eval_logger.warning(
                "Prompt length %d exceeds max_length %d, truncating newest tokens to fit cache capacity.",
                prompt_length,
                self.max_length,
            )
            prompt = prompt[:, -self.max_length:]
            prompt_length = prompt.shape[1]
        request_timer = self._profiler.tic()

        if not self.use_uncertainty_logic:
            generated_ids, stats, total_time, first_step_time, step_times = self._generate_original_single_branch(prompt, prefix_length=prefix_length)
            result = (generated_ids, stats.get('steps_taken', 0), len(generated_ids), stats, total_time, first_step_time, step_times)
            self._profiler.toc(request_timer, "request_total")
            return result

        with self._profiler.section("request_init"):
            initial_x_t = prompt.clone().to(self.device)
            # If prefix_length > 0, the prefix has already been handled via the prefix cache
            # and the cache already contains the prefix KV entries
            if prefix_length == 0:
                # Reset the shared KV cache
                self._reset_shared_cache()
            cache_reset_needed = (prefix_length == 0)

            if self.use_full_attention:
                full_attention_mask = None
            else:
                full_attention_mask = create_full_block_attention_mask(
                    prompt_length=prompt_length,
                    max_length=self.max_length,
                    block_size=self.block_size,
                    device=self.device,
                    dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16,
                )

            initial_block_states = {
                0: {
                    'start_pos': 0,
                    'end_pos': prompt_length,
                    'mask_count': 0,
                    'total_masks': prompt_length,
                    'state': 'to_cache',
                    'is_complete': True,
                },
            }
            branches = [
                Branch(
                    0,
                    initial_x_t,
                    initial_block_states,
                    confidence=1.0,
                    past_key_values=None,
                    prompt_length=prompt_length,
                    is_base=True,
                    creation_token_confidence=1.0,
                )
            ]

            run_stats = {
                "parallel_steps": 0,
                "total_filled_original": 0,
                "total_filled_new": 0,
                "original_fallback_triggers": 0,
                "branches_created": 1,
                "branches_pruned": 0,
                "max_active_branches": 0,
                "branch_processing_count": 0,
                "candidates_generated_this_round": 0,
                "tpf_per_step": [],  # Record tokens-per-forward for each step
            }

        # Reduce GC counters ahead of the generation loop so the first step avoids
        # triggering an expensive collection.
        # gc.collect()

        max_capacity_reached = False
        # Track time per step
        step_times: List[float] = []
        first_step_time: float = 0.0
        generation_start_time = time.perf_counter()
        
        with torch.inference_mode():
            while any(b.is_active for b in branches):
                run_stats["parallel_steps"] += 1
                parallel_step_count = run_stats["parallel_steps"]
                step_total_timer = self._profiler.tic()
                step_start_time = time.perf_counter()

                if (
                    self.show_branch_details
                    and (not self._distributed or self._dist_rank == 0)
                ):
                    banner_title = f"<<< STEP {parallel_step_count} START >>>"
                    banner_width = max(60, len(banner_title) + 8)
                    banner_line = "=" * banner_width
                    print(banner_line)
                    print(banner_title.center(banner_width))
                    print(banner_line)

                active_branches_this_round = [b for b in branches if b.is_active]
                run_stats["branch_processing_count"] += len(active_branches_this_round)
                run_stats["max_active_branches"] = max(run_stats["max_active_branches"], len(active_branches_this_round))

                next_generation_branches = []
                t_prepare = self._profiler.tic()

                # Preprocess branches: append new blocks and update state
                for branch in active_branches_this_round:
                    x_t = branch.x_t
                    block_states = branch.block_states

                    effective_max_new_tokens = self.max_new_tokens or 0
                    max_generation_blocks = effective_max_new_tokens // self.block_size if self.block_size > 0 else 0
                    if len(block_states) - 1 < max_generation_blocks and not branch.eos_detected:
                        last_block_id = len(block_states) - 1
                        progress = ((block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']) if block_states[last_block_id]['total_masks'] > 0 else 1.0
                        if progress >= self.block_add_threshold:
                            next_block_end = x_t.shape[1] + self.block_size
                            if self.max_length is not None and next_block_end > self.max_length:
                                branch.max_length_reached = True
                            else:
                                new_block_id = len(block_states)
                                new_start_pos = x_t.shape[1]
                                mask_block = torch.tensor([[self.mask_token_id] * self.block_size], device=self.device)
                                x_t = torch.cat([x_t, mask_block], dim=1)
                                block_states[new_block_id] = {
                                    'start_pos': new_start_pos,
                                    'end_pos': new_start_pos + self.block_size,
                                    'mask_count': self.block_size,
                                    'total_masks': self.block_size,
                                    'state': 'active',
                                    'is_complete': False,
                                }
                                branch.x_t = x_t

                    self._update_block_completion_states(block_states, self.decoded_token_threshold)

                    is_generation_done = (x_t == self.mask_token_id).sum() == 0 and all(s['state'] != 'active' for s in block_states.values())
                    if is_generation_done:
                        branch.is_active = False
                        if branch.steps_completed == -1:
                            branch.steps_completed = parallel_step_count
                        next_generation_branches.append(branch)

                # Filter branches that remain active
                remaining_active_branches = [b for b in active_branches_this_round if b.is_active]

                self._profiler.toc(t_prepare, "step_prepare", extra=parallel_step_count)

                if not remaining_active_branches:
                    next_generation_branches.extend([b for b in active_branches_this_round if not b.is_active])
                    self._profiler.toc(step_total_timer, "step_total", extra=parallel_step_count)
                    continue

                # Find the shared prefix and cache info
                shared_prefix_end = find_shared_prefix_end(remaining_active_branches)

                # Align with single-branch logic: read the true length from the shared cache
                current_cache_length = self._get_dream_model().get_shared_cache_length()
                available_capacity = None
                if self.max_length is not None:
                    available_capacity = max(0, self.max_length - current_cache_length)
                    if available_capacity <= 0:
                        for branch in remaining_active_branches:
                            branch.max_length_reached = True
                            branch.is_active = False
                            if branch.steps_completed == -1:
                                branch.steps_completed = parallel_step_count
                        next_generation_branches.extend([b for b in active_branches_this_round if not b.is_active])
                        self._profiler.toc(step_total_timer, "step_total", extra=parallel_step_count)
                        max_capacity_reached = True
                        break

                # Check whether cache updates are needed
                blocks_to_cache = []
                update_kvcache_len = 0
                # Choose the reference branch for layout; prefer the base branch
                layout_branch = None
                for b in remaining_active_branches:
                    if getattr(b, 'is_base', False):
                        layout_branch = b
                        break
                if layout_branch is None:
                    layout_branch = remaining_active_branches[0]

                for bid, state in layout_branch.block_states.items():
                    if (state['state'] == 'to_cache' and
                        state['end_pos'] <= shared_prefix_end):
                        blocks_to_cache.append(bid)

                if blocks_to_cache and available_capacity is not None:
                    sorted_blocks = sorted(blocks_to_cache, key=lambda bid: layout_branch.block_states[bid]['start_pos'])
                    capped_blocks: List[int] = []
                    consumed = 0
                    for block_id in sorted_blocks:
                        block_state = layout_branch.block_states[block_id]
                        block_len = block_state['end_pos'] - block_state['start_pos']
                        if block_len <= 0:
                            continue
                        if consumed + block_len > available_capacity:
                            break
                        capped_blocks.append(block_id)
                        consumed += block_len
                    if not capped_blocks:
                        for branch in remaining_active_branches:
                            branch.max_length_reached = True
                            branch.is_active = False
                            if branch.steps_completed == -1:
                                branch.steps_completed = parallel_step_count
                        next_generation_branches.extend([b for b in active_branches_this_round if not b.is_active])
                        self._profiler.toc(step_total_timer, "step_total", extra=parallel_step_count)
                        max_capacity_reached = True
                        break
                    blocks_to_cache = capped_blocks

                if blocks_to_cache:
                    earliest_pos = min(layout_branch.block_states[bid]['start_pos'] for bid in blocks_to_cache)
                    latest_pos = max(layout_branch.block_states[bid]['end_pos'] for bid in blocks_to_cache)
                    update_kvcache_len = latest_pos - earliest_pos

                # Determine the input range to process
                if update_kvcache_len > 0:
                    # If cache updates exist, start from the earliest block that needs caching
                    input_start_pos = min(layout_branch.block_states[bid]['start_pos'] for bid in blocks_to_cache)
                else:
                    # Otherwise start from the end of the shared prefix
                    input_start_pos = shared_prefix_end

                # If verification forces the base branch, keep its suffix first to align position encoding with single-branch
                # if self.verification_force_base_winner:
                #     remaining_active_branches = sorted(remaining_active_branches, key=lambda b: (not getattr(b, 'is_base', False)))
                # When the base branch competes, place it last so verification branches are order-agnostic
                if self.base_branch_competition:
                    # Non-base branches first, base branch last; keep ordering stable with branch_id
                    remaining_active_branches = sorted(
                        remaining_active_branches,
                        key=lambda b: (getattr(b, 'is_base', False), getattr(b, 'branch_id', 0))
                    )

                # [DDP] Choose branch forward strategy based on mode
                t_forward = self._profiler.tic()
                use_distributed = (
                    self._distributed
                    and self._dist_world_size > 1
                    and len(remaining_active_branches) > 1
                )
                if use_distributed:
                    branch_results = [None] * len(remaining_active_branches)
                    branch_tasks: List[BranchTask] = []

                    if isinstance(self.model, DDP):
                        model_dtype = getattr(self.model.module, "dtype", None)
                    else:
                        model_dtype = getattr(self.model, "dtype", None)
                    logits_dtype = model_dtype if isinstance(model_dtype, torch.dtype) else torch.float32

                    for branch_idx, branch in enumerate(remaining_active_branches):
                        needs_shared_prefix = input_start_pos < shared_prefix_end
                        needs_branch_suffix = branch.x_t.shape[1] > shared_prefix_end

                        if not needs_shared_prefix and not needs_branch_suffix:
                            branch.is_active = False
                            if branch.steps_completed == -1:
                                branch.steps_completed = parallel_step_count
                            branch_results[branch_idx] = (branch, None, None, self._dist_rank)
                            continue

                        if needs_shared_prefix and needs_branch_suffix:
                            branch_input = torch.cat([
                                branch.x_t[0, input_start_pos:shared_prefix_end],
                                branch.x_t[0, shared_prefix_end:]
                            ]).unsqueeze(0)
                        elif needs_shared_prefix:
                            branch_input = branch.x_t[0, input_start_pos:shared_prefix_end].unsqueeze(0)
                        else:
                            branch_input = branch.x_t[0, shared_prefix_end:].unsqueeze(0)

                        branch_tasks.append(
                            BranchTask(
                                branch_index=branch_idx,
                                branch_id=branch.branch_id,
                                tokens=branch_input.squeeze(0).contiguous(),
                                input_start_pos=input_start_pos,
                                shared_prefix_end=shared_prefix_end,
                                update_kvcache_len=update_kvcache_len,
                            )
                        )

                    payload_list = [{
                        "task": branch_task,
                        "reset_cache": cache_reset_needed,
                        "epoch": parallel_step_count,
                        "prompt_length": prompt_length,
                    } for branch_task in branch_tasks]
                    cache_reset_needed = False

                    t_forward_ddp = self._profiler.tic()
                    marker, distributed_results = self._distributed_forward(payload_list)
                    if marker != "results" or distributed_results is None:
                        distributed_results = []

                    for result in distributed_results:
                        idx = result.branch_index
                        if idx < 0 or idx >= len(remaining_active_branches):
                            continue
                        branch = remaining_active_branches[idx]
                        raw_logits_tensor = result.logits.to(self.device, dtype=logits_dtype) if result.logits is not None else None
                        branch_results[idx] = (
                            branch,
                            raw_logits_tensor,
                            result.cache_updates if result.cache_updates is not None else None,
                            result.source_rank,
                        )

                    for idx, value in enumerate(branch_results):
                        if value is None:
                            branch = remaining_active_branches[idx]
                            branch_results[idx] = (branch, None, None, self._dist_rank)

                    self._profiler.toc(t_forward_ddp, "step_forward_ddp", extra=parallel_step_count)
                else:
                    branch_results = []

                    forward_time = 0.0
                    forward_token_count = 0
                    forward_calls = 0
                    dream_model = self._get_dream_model()

                    for branch_idx, branch in enumerate(remaining_active_branches):
                        needs_shared_prefix = input_start_pos < shared_prefix_end
                        needs_branch_suffix = branch.x_t.shape[1] > shared_prefix_end

                        if not needs_shared_prefix and not needs_branch_suffix:
                            branch.is_active = False
                            if branch.steps_completed == -1:
                                branch.steps_completed = parallel_step_count
                            branch_results.append((branch, None, None, self._dist_rank))
                            continue

                        if needs_shared_prefix and needs_branch_suffix:
                            branch_input = torch.cat([
                                branch.x_t[0, input_start_pos:shared_prefix_end],
                                branch.x_t[0, shared_prefix_end:]
                            ]).unsqueeze(0)
                        elif needs_shared_prefix:
                            branch_input = branch.x_t[0, input_start_pos:shared_prefix_end].unsqueeze(0)
                        else:
                            branch_input = branch.x_t[0, shared_prefix_end:].unsqueeze(0)

                        attention_mask = extract_attention_mask(
                            full_mask=full_attention_mask,
                            start_pos=input_start_pos,
                            input_length=branch_input.shape[1],
                            cache_length=current_cache_length,
                            use_full_attention=self.use_full_attention,
                        )

                        forward_token_count += int(branch_input.shape[1])
                        forward_calls += 1
                        forward_start = time.perf_counter()
                        outputs = self.model(
                            branch_input,
                            attention_mask=attention_mask,
                            use_cache=True,
                            update_kvcache=update_kvcache_len,
                        )
                        forward_time += time.perf_counter() - forward_start

                        if outputs.past_key_values is None:
                            eval_logger.error(f"Model did not return past_key_values for branch {branch.branch_id}.")
                            branch.is_active = False
                            if branch.steps_completed == -1:
                                branch.steps_completed = parallel_step_count
                            branch_results.append((branch, None, None, self._dist_rank))
                            continue

                        raw_logits = outputs.logits
                        cache_updates = outputs.past_key_values if isinstance(outputs.past_key_values, list) else None
                        branch_results.append((branch, raw_logits, cache_updates, self._dist_rank))

                    if forward_calls > 0:
                        self._rank_profile(
                            "forward",
                            forward_time,
                            extra=f"{forward_calls}x{forward_token_count}",
                        )
                    cache_reset_needed = False

                self._profiler.toc(t_forward, "step_forward", extra=parallel_step_count)

                t_post = self._profiler.tic()

                processed_results: List[
                    Tuple[
                        Branch,
                        Optional[torch.Tensor],
                        Optional[List[Optional[CacheUpdate]]],
                        Optional[torch.Tensor],
                        int,
                    ]
                ] = []
                for branch, raw_logits, cache_updates, src_rank in branch_results:
                    if raw_logits is not None and raw_logits.shape[1] > 0:
                        if update_kvcache_len > 0:
                            # Fix: match the naive version by using update_kvcache_len - 1
                            # This keeps single-branch behavior and avoids index-related perf issues
                            # cache_index = min(update_kvcache_len, raw_logits.shape[1]) - 1
                            # cache_index = max(cache_index, 0)
                            # candidate_last_logit: Optional[torch.Tensor] = raw_logits[:, cache_index, :].unsqueeze(1)
                            candidate_last_logit: Optional[torch.Tensor] = raw_logits[:, update_kvcache_len - 1, :].unsqueeze(1)
                        else:
                            candidate_last_logit = self._last_committed_logit
                        candidate_last_logit = None if candidate_last_logit is None else candidate_last_logit.detach()
                        shifted_logits = self._shift_logits(raw_logits, last_logit=candidate_last_logit)
                    else:
                        shifted_logits = None
                        candidate_last_logit = None
                    processed_results.append((branch, shifted_logits, cache_updates, candidate_last_logit, src_rank))
                branch_results = processed_results

                # Evaluate confidence for each branch
                branch_confidences = []
                for branch_idx, (branch, logits, new_updates, _last_logits, _src_rank) in enumerate(branch_results):
                    if logits is not None:
                        branch_length = branch.x_t.shape[1] - shared_prefix_end if branch.x_t.shape[1] > shared_prefix_end else 0
                        if branch_length > 0:
                            shared_prefix_in_input_length = max(0, shared_prefix_end - input_start_pos)
                            branch_start_in_input = shared_prefix_in_input_length
                            confidence = evaluate_branch_confidence(
                                logits,
                                branch,
                                branch_start_in_input,
                                branch_length,
                                shared_prefix_end,
                                self.mask_token_id,
                                sampling_strategy=self.sampling_strategy,
                                branch_topp=self.branch_topp,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                top_k=self.top_k,
                                selection_conf_alpha=self.selection_conf_alpha,
                            )
                            branch_confidences.append((confidence, branch_idx, branch))
                        else:
                            branch_confidences.append((1.0, branch_idx, branch))
                    else:
                        # No logits: fall back to creation-time confidence only
                        fallback_conf = float(getattr(branch, "creation_token_confidence", 0.0))
                        branch_confidences.append((fallback_conf, branch_idx, branch))

                # Pick the best branch
                branch_confidences.sort(key=lambda x: x[0], reverse=True)
                base_branch_result = next(
                    (
                        (b, l, p, last, src)
                        for (b, l, p, last, src) in branch_results
                        if getattr(b, "is_base", False)
                    ),
                    None,
                )

                best_new_updates: Optional[List[Optional[CacheUpdate]]] = None
                best_source_rank = self._dist_rank
                if self.verification_force_base_winner:
                    if base_branch_result is not None:
                        best_branch, best_logits, best_new_updates, best_last_logit, best_source_rank = base_branch_result
                        best_confidence = 1.0
                    else:
                        best_confidence, best_idx, best_branch = branch_confidences[0]
                        _, best_logits, best_new_updates, best_last_logit, best_source_rank = branch_results[best_idx]
                else:
                    best_confidence, best_idx, best_branch = branch_confidences[0]
                    _, best_logits, best_new_updates, best_last_logit, best_source_rank = branch_results[best_idx]

                # [MEMORY OPTIMIZED] Update shared KV cache using the best branch
                if update_kvcache_len > 0 and best_new_updates is not None:
                    self._commit_shared_cache(best_new_updates, source_rank=best_source_rank)
                if best_last_logit is not None:
                    self._last_committed_logit = best_last_logit.detach()

                # [MEMORY OPTIMIZED] Reuse variables to avoid extra computation/allocation
                tokens_to_update = {}
                active_blocks_ids: List[int] = []

                # Update tokens for the best branch
                if best_logits is not None:
                    branch_length = best_branch.x_t.shape[1] - shared_prefix_end if best_branch.x_t.shape[1] > shared_prefix_end else 0
                    if branch_length > 0:
                        # Find the branch start position within the input sequence
                        shared_prefix_in_input_length = max(0, shared_prefix_end - input_start_pos)
                        branch_start_in_input = shared_prefix_in_input_length

                        # [MEMORY OPTIMIZED] Use indices directly instead of creating new tensor slices
                        active_blocks_ids = [bid for bid, state in best_branch.block_states.items() if state['state'] == 'active']

                        for block_id in active_blocks_ids:
                            block_start, block_end = best_branch.block_states[block_id]['start_pos'], best_branch.block_states[block_id]['end_pos']
                            mask_indices_in_block = (best_branch.x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]
                            if len(mask_indices_in_block) == 0:
                                continue

                            mask_indices_abs = mask_indices_in_block + block_start
                            # Compute indices relative to the branch start
                            mask_indices_rel_local = mask_indices_abs - shared_prefix_end

                            # Ensure indices stay within bounds
                            valid_mask_indices = mask_indices_rel_local[(mask_indices_rel_local >= 0) & (mask_indices_rel_local < branch_length)]
                            if len(valid_mask_indices) == 0:
                                continue

                            # [MEMORY OPTIMIZED] Index logits directly to avoid new tensors
                            logits_indices = branch_start_in_input + valid_mask_indices
                            block_mask_logits = best_logits[0, logits_indices, :]
                            confidence, x0, initial_confidence = sample_tokens(
                                block_mask_logits, self.temperature, 
                                top_p=self.top_p, top_k=self.top_k, 
                                sampling_strategy=self.sampling_strategy
                            )
                            high_conf_indices = (initial_confidence > self.skip_threshold).nonzero(as_tuple=True)[0]
                            is_complete = best_branch.block_states[block_id]['is_complete']

                            indices_to_fill = []
                            if is_complete:
                                if len(high_conf_indices) > 0:
                                    indices_to_fill = high_conf_indices
                                else:
                                    run_stats["original_fallback_triggers"] += 1
                                    if len(confidence) > 0:
                                        _, most_conf_idx = torch.topk(confidence, 1)
                                        indices_to_fill = most_conf_idx
                            else:
                                if len(high_conf_indices) > 0:
                                    indices_to_fill = high_conf_indices

                            if len(indices_to_fill) > 0:
                                for idx in indices_to_fill:
                                    rel_local = valid_mask_indices[idx.item()]
                                    # Map back to the branch's absolute position
                                    abs_pos = rel_local + shared_prefix_end
                                    token = x0[idx.item()].item()
                                    tokens_to_update[abs_pos.item()] = token
                                    # Mark EOS when the generated token is the end token
                                    if token == self.eot_token_id:
                                        best_branch.eos_detected = True

                        run_stats["total_filled_original"] += len(tokens_to_update)
                        
                        # Record tokens-per-forward (TPF) for this step
                        # TPF = number of tokens filled in this step
                        step_tpf = len(tokens_to_update) if tokens_to_update else 0.0
                        run_stats["tpf_per_step"].append(step_tpf)

                        # [MEMORY OPTIMIZED] Touch tensors only when updates exist
                        if tokens_to_update:
                            for pos, token in tokens_to_update.items():
                                best_branch.x_t[0, pos] = token

                        # Drop intermediate average confidence (best_confidence will overwrite)

                # [MODIFIED] Set the branch's confidence to the new model-validated score for this round.
                best_branch.confidence = best_confidence

                # Spawn k new branches for the next round
                newly_spawned_branches = []

                # === Precompute base-branch completed blocks to safely update cache state ===
                # Requirement: to_cache transitions should rely only on the base logic (winner's actual tokens), not speculative samples.
                base_completed_blocks = []  # Track blocks truly completed in the best_branch sequence
                base_active_blocks_snapshot = []  # Snapshot blocks still active for later traversal
                for _bid, _state in best_branch.block_states.items():
                    if _state['state'] == 'active':
                        base_active_blocks_snapshot.append(_bid)
                        start_, end_ = _state['start_pos'], _state['end_pos']
                        new_mask_cnt_ = (best_branch.x_t[0, start_:end_] == self.mask_token_id).sum().item()
                        # Sync mask_count on the base branch (objective result after base logic)
                        _state['mask_count'] = new_mask_cnt_
                        if new_mask_cnt_ == 0:
                            # Block filled under base logic; treat as a cache candidate
                            base_completed_blocks.append(_bid)

                if self.branch_verification_mode:
                    if self.base_branch_competition:
                        # Base-branch competition mode: base branch also competes
                        # [MEMORY OPTIMIZED] Use a copy of best_branch as the competing base branch
                        base_branch = best_branch.copy()
                        base_branch.branch_id = len(branches)
                        base_branch.is_base = True
                        base_branch.creation_token_confidence = 1.0
                        base_branch.confidence = 1.0

                        newly_spawned_branches.append(base_branch)

                        # Generate additional competing branches if needed; base branch will be appended last
                        if self.branching_factor > 1:
                            chosen_positions = set(tokens_to_update.keys()) if locals().get('tokens_to_update') else set()
                            self._generate_additional_branches(
                                newly_spawned_branches, best_branch, best_logits, 
                                active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, branches
                            )
                            # Ensure the base branch stays at the end to preserve ordering
                            base_idx = next((idx for idx, br in enumerate(newly_spawned_branches) if br.is_base), None)
                            if base_idx is not None and base_idx != len(newly_spawned_branches) - 1:
                                newly_spawned_branches.append(newly_spawned_branches.pop(base_idx))
                    else:
                        # Verification mode: reuse the base-updated sequence for the next round
                        best_branch.is_base = True
                        best_branch.creation_token_confidence = 1.0
                        best_branch.confidence = 1.0
                        newly_spawned_branches.append(best_branch)
                elif self.branching_factor > 1:
                    chosen_positions = set(tokens_to_update.keys()) if locals().get('tokens_to_update') else set()
                    self._generate_additional_branches(
                        newly_spawned_branches, best_branch, best_logits, 
                        active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, branches
                    )
                else:
                    best_branch.is_base = True
                    best_branch.creation_token_confidence = 1.0
                    best_branch.confidence = 1.0
                    newly_spawned_branches.append(best_branch)

                # Update caches and block states
                for nb in newly_spawned_branches:
                    if update_kvcache_len > 0:
                        for block_id in blocks_to_cache:
                            if block_id in nb.block_states:
                                nb.block_states[block_id]['state'] = 'in_cache'
                    # === Update from the base snapshot: only blocks the base finished this round can enter to_cache ===
                    # 1. Resync mask_count for active blocks (other branches should be <= base; not critical)
                    nb_active_blocks = [bid for bid, state in nb.block_states.items() if state['state'] == 'active']
                    for block_id in nb_active_blocks:
                        start, end = nb.block_states[block_id]['start_pos'], nb.block_states[block_id]['end_pos']
                        nb.block_states[block_id]['mask_count'] = (nb.x_t[0, start:end] == self.mask_token_id).sum().item()

                    # 2. Mark to_cache only if the base branch completed the block and no prior blocks remain active
                    for block_id in nb_active_blocks:
                        if block_id in base_completed_blocks:
                            can_deactivate = all(prev_bid not in nb.block_states or nb.block_states[prev_bid]['state'] != 'active' for prev_bid in range(block_id))
                            if can_deactivate and nb.block_states[block_id]['mask_count'] == 0:
                                nb.block_states[block_id]['state'] = 'to_cache'

                # [MODIFIED] Replace old pruning logic with the new branch lifecycle logic.
                # The branches for the next round consist of all previously inactive branches
                # plus the new branches spawned from this round's single winner.
                branches = [b for b in branches if not b.is_active] + newly_spawned_branches

                if (
                    self.show_branch_details
                    and newly_spawned_branches
                    and (not self._distributed or self._dist_rank == 0)
                ):
                    header = f"[Branch Spawn] round={parallel_step_count} spawned={len(newly_spawned_branches)}"
                    divider = "-" * max(40, len(header) + 4)
                    print(f"\n{divider}\n{header}\n{divider}")
                    for spawned_branch in newly_spawned_branches:
                        branch_label = f"branch_id={spawned_branch.branch_id} base={spawned_branch.is_base}"
                        print(f"  {branch_label}")
                        active_block_ids = [bid for bid, state in sorted(spawned_branch.block_states.items()) if state['state'] == 'active']
                        if not active_block_ids:
                            print("    active blocks: none")
                            continue
                        for block_id in active_block_ids:
                            state = spawned_branch.block_states[block_id]
                            block_start, block_end = state['start_pos'], state['end_pos']
                            block_token_ids = [int(tok) for tok in spawned_branch.x_t[0, block_start:block_end].tolist()]
                            filled_token_ids = [tok for tok in block_token_ids if tok != self.mask_token_id]
                            try:
                                decoded_text = self.tok_decode(filled_token_ids, skip_special_tokens=False) if filled_token_ids else ""
                            except Exception as decode_err:  # noqa: BLE001
                                decoded_text = f"<decode_error:{decode_err}>"
                            print(f"    block {block_id}: len_all={len(block_token_ids)} len_filled={len(filled_token_ids)}")
                            print(f"      token_ids_all: {block_token_ids}")
                            print(f"      token_ids_filled: {filled_token_ids}")
                            print(f"      decoded_filled: {decoded_text if decoded_text else '<empty>'}")
                    print(divider)

                self._profiler.toc(t_post, "step_post", extra=parallel_step_count)

                # Update statistics based on the new logic.
                # We started with N active branches, selected 1 to be the parent, and discarded the other N-1.
                run_stats["branches_pruned"] += len(active_branches_this_round) - 1
                run_stats["candidates_generated_this_round"] = len(newly_spawned_branches)

                # [MEMORY OPTIMIZED] Force GC to free memory from pruned branches
                run_gc, run_cuda_cleanup = self._should_cleanup_resources(len(active_branches_this_round))
                if run_gc:
                    with self._profiler.section("step_cleanup.gc_collect", extra=parallel_step_count):
                        gc.collect()
                if run_cuda_cleanup and torch.cuda.is_available():
                    with self._profiler.section("step_cleanup.cuda_empty_cache", extra=parallel_step_count, sync_cuda=True):
                        torch.cuda.empty_cache()

                self._profiler.toc(step_total_timer, "step_total", extra=parallel_step_count)
                
                # Record time per step
                step_end_time = time.perf_counter()
                step_duration = step_end_time - step_start_time
                step_times.append(step_duration)
                if parallel_step_count == 1:
                    first_step_time = step_duration

                # if parallel_step_count > 500:
                #     eval_logger.warning(f"Generation stopped due to exceeding 500 parallel steps.")
                #     # Force-stop all branches
                #     for b in branches:
                #         b.is_active = False
                #         if b.steps_completed == -1: b.steps_completed = parallel_step_count
                #     break

        if max_capacity_reached:
            eval_logger.warning(
                "Generation terminated early because shared KV cache reached its capacity (max_length=%s).",
                str(self.max_length),
            )

        t_finalize = self._profiler.tic()

        # [MODIFIED] Select the best branch based on completion speed, not the abandoned confidence metric.
        if not branches:
            eval_logger.warning("No branches available, returning original prompt")
            self._profiler.toc(t_finalize, "request_finalize")
            generation_end_time = time.perf_counter()
            total_generation_time = generation_end_time - generation_start_time
            return [], 0, 0, run_stats, total_generation_time, first_step_time if first_step_time > 0 else 0.0, step_times

        # Filter for branches that actually completed their generation.
        completed_branches = [b for b in branches if not b.is_active and b.steps_completed != -1]

        if completed_branches:
            # Among completed branches, choose the one that finished in the fewest parallel steps.
            best_branch = min(completed_branches, key=lambda x: x.steps_completed)
            selection_reason = "Fastest completion (fewest parallel steps)"
        else:
            # If no branch completed (e.g., all hit the step limit), fall back to the one
            # that managed to generate the most tokens before stopping.
            eval_logger.warning("No branches completed generation. Selecting the one with the most generated tokens.")
            best_branch = max(branches, key=lambda x: x.generated_token_count)
            selection_reason = "Most tokens generated (fallback)"

        run_stats["cache_max_length_hit"] = max_capacity_reached
        generated_sequence_ids = best_branch.x_t[0, prompt_length:].tolist()
        best_branch_steps = best_branch.steps_completed if best_branch.steps_completed != -1 else run_stats["parallel_steps"]
        best_branch_tokens = best_branch.generated_token_count

        self._profiler.toc(t_finalize, "request_finalize")
        
        # Compute total generation time
        generation_end_time = time.perf_counter()
        total_generation_time = generation_end_time - generation_start_time
        # If the first step time wasn't recorded, default to the first step duration
        if first_step_time == 0.0 and len(step_times) > 0:
            first_step_time = step_times[0]

        result = (generated_sequence_ids, best_branch_steps, best_branch_tokens, run_stats, total_generation_time, first_step_time, step_times)

        self._profiler.toc(request_timer, "request_total")
        if self.show_branch_details:
            print("\n\n")
        return result

    def _should_cleanup_resources(self, active_branch_count: int) -> Tuple[bool, bool]:
        """Decide whether to run cleanup based on current resource pressure."""
        run_cuda_cleanup = False
        free_ratio = None
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except (RuntimeError, AttributeError):
                free_bytes, total_bytes = None, None
            if free_bytes is not None and total_bytes:
                if total_bytes > 0:
                    free_ratio = free_bytes / float(total_bytes)
                    if free_ratio < self._cuda_cleanup_free_ratio:
                        run_cuda_cleanup = True
        gc_counts = gc.get_count()
        thresholds = gc.get_threshold()
        margin = self._gc_cleanup_margin
        run_gc = (
            gc_counts[0] > thresholds[0] * margin
            or gc_counts[1] > thresholds[1] * margin
            or gc_counts[2] > thresholds[2] * margin
        )

        if run_cuda_cleanup and not run_gc:
            run_gc = True

        if self._rank_profile_enabled and self._dist_rank == 0 and (run_gc or run_cuda_cleanup):
            log_parts = [f"branches={active_branch_count}"]
            if free_ratio is not None:
                log_parts.append(f"free_ratio={free_ratio:.3f}")
            log_parts.append(f"gc_counts={gc_counts}")
            eval_logger.info("resource_cleanup_trigger %s", " ".join(log_parts))

        return (run_gc, run_cuda_cleanup)

    def _generate_additional_branches(self, newly_spawned_branches, best_branch_template, best_logits, 
                                      active_blocks_ids, shared_prefix_end, chosen_positions, run_stats, 
                                      branches):
        """Generate additional branches for competition."""
        if best_logits is None:
            # If no logits are available, keep only the base branch
            if not newly_spawned_branches:
                best_branch_template.is_base = True
                best_branch_template.creation_token_confidence = 1.0
                best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)
            return

        # [MEMORY OPTIMIZED] Compute branch length to avoid repeated work
        branch_length = best_branch_template.x_t.shape[1] - shared_prefix_end if best_branch_template.x_t.shape[1] > shared_prefix_end else 0
        
        if branch_length > 0:
            # [MEMORY OPTIMIZED] Slice logits directly to avoid extra copies
            if best_logits.shape[1] >= branch_length:
                branch_logits = best_logits[0, -branch_length:, :]
            else:
                branch_logits = best_logits[0, :, :]

            # [MEMORY OPTIMIZED] Precompute candidate positions to avoid repeated work in the loop
            candidate_positions = []
            for block_id in active_blocks_ids:
                block_start, block_end = best_branch_template.block_states[block_id]['start_pos'], best_branch_template.block_states[block_id]['end_pos']
                mask_indices_in_block = (best_branch_template.x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]

                if len(mask_indices_in_block) > 0:
                    mask_indices_abs = mask_indices_in_block + block_start
                    for abs_pos_tensor in mask_indices_abs:
                        abs_pos = abs_pos_tensor.item()
                        if abs_pos not in chosen_positions and abs_pos >= shared_prefix_end:
                            rel_pos_local = abs_pos - shared_prefix_end
                            if 0 <= rel_pos_local < branch_length:
                                candidate_positions.append((abs_pos, rel_pos_local))

            top_candidates = []
            if candidate_positions:
                # [OPTIMIZED] Batch sample all candidate positions at once to avoid repeated sampling
                # Collect logits for all candidate positions
                valid_candidates = [(abs_pos, rel_pos) for abs_pos, rel_pos in candidate_positions if rel_pos < branch_logits.shape[0]]
                if valid_candidates:
                    # Batch-extract logits
                    rel_positions = [rel_pos for _, rel_pos in valid_candidates]
                    candidate_logits = branch_logits[rel_positions, :]  # [num_candidates, vocab_size]
                    
                    # Batch sample all candidate positions
                    confidences, tokens, initial_confidences = sample_tokens(
                        candidate_logits, self.temperature, 
                        top_p=self.top_p, top_k=self.top_k,
                        sampling_strategy=self.sampling_strategy
                    )
                    
                    # Build candidate list with confidence, position, and sampled token
                    candidates_with_results = []
                    for idx, (abs_pos, rel_pos) in enumerate(valid_candidates):
                        conf_val = confidences[idx].item()
                        token_val = tokens[idx].item()
                        candidates_with_results.append((conf_val, abs_pos, rel_pos, token_val))

                    # Select the top-k highest-confidence positions
                    candidates_with_results.sort(key=lambda x: x[0], reverse=True)
                    # In competition mode, reserve one slot for the base branch
                    effective_branching_factor = self.branching_factor - 1 if self.base_branch_competition and len(newly_spawned_branches) > 0 else self.branching_factor
                    num_to_select = min(effective_branching_factor, len(candidates_with_results))
                    
                    for i in range(num_to_select):
                        confidence, abs_pos, rel_pos, token = candidates_with_results[i]
                        top_candidates.append((abs_pos, rel_pos, confidence, token))
        else:
            top_candidates = []

        if not top_candidates and not newly_spawned_branches:
            best_branch_template.is_base = True
            best_branch_template.creation_token_confidence = 1.0
            best_branch_template.confidence = 1.0
            newly_spawned_branches.append(best_branch_template)
        else:
            # [OPTIMIZED] Use the sampled tokens directly to avoid re-sampling
            for abs_pos, rel_pos, confidence_score, token in top_candidates:
                # [MEMORY OPTIMIZED] Create new branch copies only when needed
                new_branch = best_branch_template.copy()
                new_branch.branch_id = len(branches) + len(newly_spawned_branches)
                # Single shared KV cache; do not store KV on branches
                new_branch.is_base = False

                # Apply the sampled token directly
                new_branch.x_t[0, abs_pos] = token
                # Requirement: use the sampling_strategy-transformed confidence_score for creation confidence
                creation_conf = float(confidence_score)
                new_branch.creation_token_confidence = creation_conf
                new_branch.confidence = creation_conf

                if token == self.eot_token_id:
                    new_branch.eos_detected = True

                newly_spawned_branches.append(new_branch)
                run_stats["total_filled_new"] += 1
                run_stats["branches_created"] += 1

            # If no new branches were produced (including the base), keep the base branch
            if not newly_spawned_branches:
                best_branch_template.is_base = True
                best_branch_template.creation_token_confidence = 1.0
                best_branch_template.confidence = 1.0
                newly_spawned_branches.append(best_branch_template)

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        if self._distributed and self._dist_world_size > 1 and self._dist_rank != 0:
            self._distributed_worker_loop()
            return []
        
        requests = requests[:]
        
        res = []
        sample_records: List[Dict[str, Any]] = []
        debug_samples: List[Tuple[str, str]] = []
        run_time = 0.0
        free_resource_time = 0.0
        total_time = 0.0
        total_tokens, total_steps = 0, 0
        # Sum pure generation time across all samples (excluding the first step)
        total_generation_time = 0.0
        sample_times: List[float] = []  # Pure generation time per sample
        sample_tpss: List[float] = []  # TPS per sample
        overall_start = time.time()
        bar = tqdm(total=len(requests), disable=(disable_tqdm or (self._dist_rank != 0)), desc="Running generate_until requests")
        
        # Initialize stats collector if available
        stats_collector = None
        if STATS_AVAILABLE and self.save_dir is not None:
            try:
                stats_collector = GenerationStatsCollector(save_dir=self.save_dir)
            except Exception as e:
                eval_logger.warning(f"Failed to initialize stats collector: {e}")
                stats_collector = None

        # Preprocess: when prefix cache is enabled, detect prefix by comparing the first two requests
        if self.enable_prefix_cache and len(requests) > 1:
            # Encode prompts for the first two requests
            first_question = requests[0].args[0]
            second_question = requests[1].args[0]
            first_contexts = [first_question]
            second_contexts = [second_question]
            if self.add_bos_token:
                first_contexts = [self.tokenizer.bos_token + p for p in first_contexts]
                second_contexts = [self.tokenizer.bos_token + p for p in second_contexts]
            first_enc, _ = self.tok_batch_encode(first_contexts, truncation=self.truncation)
            second_enc, _ = self.tok_batch_encode(second_contexts, truncation=self.truncation)
            
            # Detect common prefix
            first_tokens = first_enc[0]
            second_tokens = second_enc[0]
            prefix_length = self._detect_prefix_length(first_tokens, second_tokens)
            
            if prefix_length > 0:
                self._prefix_cache_length = prefix_length
                self._prefix_tokens = first_tokens[:prefix_length].clone().cpu()
                if self._dist_rank == 0:
                    eval_logger.info(f"Detected prefix cache: length={prefix_length}")
            else:
                # No common prefix; disable prefix cache
                self.enable_prefix_cache = False
                if self._dist_rank == 0:
                    eval_logger.info("No common prefix detected, prefix cache disabled")
        
        for i, req in enumerate(requests):
            request_start = time.time()
            question = req.args[0]
            gen_kwargs = req.args[1]
            contexts = [question]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, _ = self.tok_batch_encode(contexts, truncation=self.truncation)
            input_ids = context_enc[0].unsqueeze(0)

            if input_ids.shape[1] > self.max_length - self.max_new_tokens:
                eval_logger.warning(f"Prompt length {input_ids.shape[1]} > {self.max_length - self.max_new_tokens}, cutoff on the left side")
                input_ids = input_ids[:, -(self.max_length - self.max_new_tokens):]

            # Handle prefix caching
            prefix_length = 0
            use_prefix_cache = False
            if self.enable_prefix_cache and self._prefix_cache_length > 0:
                current_tokens = input_ids[0]
                # Check whether the current prompt starts with the prefix
                if (self._prefix_tokens is not None and 
                    current_tokens.shape[0] >= self._prefix_cache_length and
                    torch.equal(current_tokens[:self._prefix_cache_length], self._prefix_tokens.to(self.device))):
                    # Prefix matches: reuse prefix cache
                    if i > 0:
                        # For the second and later requests, reuse the prefix cache
                        self._reset_shared_cache(preserve_prefix=True)
                        # Process only the tokens after the prefix
                        if self._prefix_cache_length < input_ids.shape[1]:
                            input_ids = input_ids[:, self._prefix_cache_length:]
                            prefix_length = self._prefix_cache_length
                        else:
                            # Entire prompt is in the prefix; keep the last token
                            input_ids = input_ids[:, self._prefix_cache_length-1:]
                            prefix_length = self._prefix_cache_length - 1
                        use_prefix_cache = True
                    else:
                        # First request: process the full prompt; prefix KV cache is saved during processing
                        self._reset_shared_cache(preserve_prefix=False)
                else:
                    # No match: process normally (should not happen, but safer)
                    self._reset_shared_cache(preserve_prefix=False)
            else:
                # Prefix cache disabled or no prefix detected; reset normally
                self._reset_shared_cache(preserve_prefix=False)

            # [FIX] Only count actual generation time (exclude encode/decode/truncation)
            generation_start = time.time()
            generated_answer, best_branch_steps, best_branch_tokens, run_stats, total_gen_time, first_step_time, step_times = self._generate_enhanced_speculative(input_ids, prefix_length=prefix_length)
            generation_time = time.time() - generation_start  # only actual generation time

            # [FIX] Use the full token count (before truncation) so all generated tokens are counted
            # generated_answer is the complete token ID list, including tokens that may later be truncated
            # Exclude mask_token_id (if present) and count only actual generated tokens
            if self.mask_token_id is not None:
                actual_generated_tokens = sum(1 for t in generated_answer if t != self.mask_token_id)
            else:
                actual_generated_tokens = len(generated_answer)
            # Use the actual generated token count instead of potentially inaccurate best_branch_tokens
            tokens_for_stats = actual_generated_tokens

            total_steps += best_branch_steps
            total_tokens += tokens_for_stats
            
            # Compute pure generation time (exclude the first step)
            pure_gen_time = total_gen_time - first_step_time if total_gen_time > first_step_time else total_gen_time
            sample_times.append(pure_gen_time)
            total_generation_time += pure_gen_time
            
            # Compute TPS per sample
            sample_tps = 0.0
            if pure_gen_time > 0:
                sample_tps = tokens_for_stats / pure_gen_time
            sample_tpss.append(sample_tps)
            
            # Record stats using pure generation time
            if stats_collector is not None:
                # Derive dataset name
                dataset_name = None
                doc = getattr(req, "doc", None)
                if doc and isinstance(doc, dict):
                    task_id = doc.get("task_id")
                    if isinstance(task_id, str) and task_id:
                        dataset_name = task_id.split("/")[0]
                    elif hasattr(req, "task_name"):
                        dataset_name = str(req.task_name)
                
                # Retrieve TPF for each step
                tpf_per_step = run_stats.get("tpf_per_step", [])
                
                stats_collector.record_sample(
                    sample_idx=i,
                    tokens=tokens_for_stats,  # use the full token count
                    steps=best_branch_steps,
                    generation_time=pure_gen_time,  # use pure generation time (exclude first step)
                    tpf_per_step=tpf_per_step,
                    dataset_name=dataset_name,
                )

            # [NOTE] The truncation below is only for saving/evaluation and does not affect metrics
            # Decode the full token list (all generated tokens)
            cont_toks_list = self.tokenizer.batch_decode([generated_answer], skip_special_tokens=True)
            s = cont_toks_list[0]

            # Truncate at the [DONE] marker (if present) for saving/evaluation only
            if not self.escape_until:
                for term in gen_kwargs.get("until", []):
                    if len(term) > 0:
                        s = s.split(term)[0]
            
            # Extra safety: if s still contains [DONE], force a truncate
            if "[DONE]" in s:
                s = s.split("[DONE]")[0]

            # [FIX] run_time sums only pure generation time (excludes encode/decode/truncation and first-step time)
            run_time += pure_gen_time

            cleanup_start = time.time()

            # Force release of caches from the previous generation
            # If prefix cache is enabled and this isn't the last request, keep the prefix
            dream_model = self._get_dream_model()
            if (self.enable_prefix_cache and self._prefix_cache_length > 0 and 
                i < len(requests) - 1):
                # Keep the prefix; clear only the portion after it for the next request
                # After the first request, the prefix KV cache is already stored and should be preserved
                self._reset_shared_cache(preserve_prefix=True)
            else:
                # Full reset (last request or prefix cache disabled)
                dream_model.reset_shared_kv_cache()
                self._last_committed_logit = None
                self._committed_cache_length = 0
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

            free_resource_time += time.time() - cleanup_start

            res.append(s)
            debug_samples.append((str(question), s))
            
            # Build sample record (reference format)
            doc = getattr(req, "doc", None)
            
            # Build filtered_resps (prompt + response combination)
            filtered_resp = None
            if doc and isinstance(doc, dict) and "prompt" in doc:
                prompt_text = doc["prompt"]
                # Check whether the generated code already contains a complete function/class definition
                code_stripped = s.strip()
                entry_point = doc.get("entry_point", "")
                code_has_complete_def = False
                if entry_point:
                    # Check whether the code contains a function/class definition
                    if code_stripped.startswith("def ") or code_stripped.startswith("class "):
                        code_has_complete_def = True
                    elif f"def {entry_point}" in code_stripped or f"class {entry_point}" in code_stripped:
                        code_has_complete_def = True
                
                # If the code lacks a full definition, prepend the prompt
                if not code_has_complete_def:
                    prompt_clean = prompt_text.rstrip()
                    filtered_resp = prompt_clean + "\n" + s
                else:
                    filtered_resp = s
            else:
                filtered_resp = s
            
            # Build arguments (reference format)
            arguments = None
            if hasattr(req, "arguments"):
                # If already in reference format, use as is
                if isinstance(req.arguments, dict) and "gen_args_0" in req.arguments:
                    arguments = req.arguments
                else:
                    # Convert to reference format
                    arguments = {
                        "gen_args_0": {
                            "arg_0": question,
                            "arg_1": gen_kwargs
                        }
                    }
            else:
                arguments = {
                    "gen_args_0": {
                        "arg_0": question,
                        "arg_1": gen_kwargs
                    }
                }
            
            # Build target (from doc.test or doc.test_list)
            target = None
            if doc and isinstance(doc, dict):
                # Prefer the test field (HumanEval format)
                if "test" in doc:
                    test_code = doc["test"]
                    entry_point = doc.get("entry_point", "")
                    if entry_point:
                        target = test_code + f"\n\ncheck({entry_point})"
                    else:
                        target = test_code
                # If no test field, try test_list (MBPP format)
                elif "test_list" in doc:
                    test_list = doc.get("test_list", [])
                    if test_list:
                        target = "\n".join(test_list)
            
            # target should always exist per reference format
            if target is None:
                # If target is missing, derive from other sources or set to an empty string
                target = ""
            
            # Compute hashes
            doc_hash = None
            prompt_hash = None
            target_hash = None
            if doc:
                doc_str = json.dumps(doc, sort_keys=True, ensure_ascii=False)
                doc_hash = hashlib.sha256(doc_str.encode('utf-8')).hexdigest()
            prompt_str = str(question)
            prompt_hash = hashlib.sha256(prompt_str.encode('utf-8')).hexdigest()
            if target:
                target_hash = hashlib.sha256(target.encode('utf-8')).hexdigest()
            
            # Build sample_entry (reference format)
            sample_entry = {
                "doc_id": getattr(req, "doc_id", None),
                "doc": doc,
                "target": target,
            }
            # arguments should always exist (reference format requirement)
            sample_entry["arguments"] = arguments
            # resps format: [[response]]
            sample_entry["resps"] = [[s]]
            # filtered_resps format: [[filtered_response]]
            sample_entry["filtered_resps"] = [[filtered_resp]]
            sample_entry["filter"] = "create_test"
            sample_entry["metrics"] = ["pass_at_1"]
            
            # Compute pass_at_1 when available
            if self._pass_at_k_metric is not None and target and filtered_resp:
                try:
                    # Extract code (Python) from filtered_resp
                    code_to_eval = filtered_resp
                    # If filtered_resp includes ```python, extract the code section
                    if "```python" in filtered_resp:
                        code_to_eval = filtered_resp.split("```python", 1)[-1].split("```")[0]
                    elif "```" in filtered_resp:
                        code_to_eval = filtered_resp.split("```", 1)[-1].split("```")[0]
                    
                    # Clean code with sanitize
                    entry_point = doc.get("entry_point", "") if doc and isinstance(doc, dict) else ""
                    sanitized_code = sanitize(code_to_eval, entry_point) if entry_point else sanitize(code_to_eval)
                    
                    # Compute pass_at_1
                    # Note: compute() returns a list; take [0]["pass@1"]
                    result = self._pass_at_k_metric.compute(
                        references=[target],
                        predictions=[[sanitized_code]],
                        k=[1],
                    )
                    sample_entry["pass_at_1"] = result[0]["pass@1"]
                except Exception as e:
                    eval_logger.warning(f"Failed to calculate pass_at_1 for sample {getattr(req, 'doc_id', 'unknown')}: {e}")
                    sample_entry["pass_at_1"] = 0.0
            else:
                # If code evaluation is unavailable, set to None or 0.0
                sample_entry["pass_at_1"] = None
            
            if doc_hash:
                sample_entry["doc_hash"] = doc_hash
            if prompt_hash:
                sample_entry["prompt_hash"] = prompt_hash
            if target_hash:
                sample_entry["target_hash"] = target_hash
            
            sample_records.append(sample_entry)
            
            bar.update(1)
            total_time += time.time() - request_start

        bar.close()

        if debug_samples and (not self._distributed or self._dist_rank == 0):
            print("\n==================== Top 10 Sample Outputs ====================")
            for idx, (prompt_text, output_text) in enumerate(debug_samples[:10], start=1):
                print(f"Sample {idx} Prompt:\n{prompt_text}\n-- Output --\n{output_text}\n")
            print("====================================================\n")

        # Save samples and statistics
        samples_output_path: Optional[str] = None
        if self.save_dir is not None and (not self._distributed or self._dist_rank == 0):
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save samples to a jsonl file
            if sample_records:
                dataset_hint = "dataset"
                if requests:
                    first_req = requests[0]
                    if getattr(first_req, "task_name", None):
                        dataset_hint = str(first_req.task_name)
                    else:
                        doc = getattr(first_req, "doc", None)
                        if isinstance(doc, dict):
                            task_id = doc.get("task_id")
                            if isinstance(task_id, str) and task_id:
                                dataset_hint = task_id.split("/")[0]
                dataset_slug = dataset_hint.replace("/", "_") or "dataset"
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
                samples_output_path = os.path.join(
                    self.save_dir,
                    f"samples_{dataset_slug}_{timestamp}.jsonl",
                )
                with open(samples_output_path, "w", encoding="utf-8") as sample_file:
                    for record in sample_records:
                        sample_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                eval_logger.info("Sample completions saved to %s", samples_output_path)
            
            # Save statistics
            final_stats = {
                "processed_samples": len(res), "total_samples": len(requests),
                "total_tokens_generated (best paths)": int(total_tokens),
                "total_steps_taken (best paths)": int(total_steps),
                "total_time": total_time,
                "total_run_time": run_time,
                "total_cleanup_time": free_resource_time,
                "tokens_per_second": float(total_tokens) / run_time if run_time > 0 else 0.0,
                "tokens_per_step (best paths avg)": float(total_tokens) / float(total_steps) if total_steps > 0 else 0.0,
                "timestamp": time.time(),
                "experiment_label": self.experiment_label,
            }
            if samples_output_path is not None:
                final_stats["samples_file"] = samples_output_path
            
            # Add dataset-level TPS stats if the collector is available
            if stats_collector is not None:
                try:
                    # Compute stats for each dataset
                    datasets = set(s.get("dataset_name") for s in stats_collector.sample_stats if s.get("dataset_name"))
                    dataset_tps_stats = {}
                    for dataset_name in datasets:
                        dataset_stats = stats_collector.compute_dataset_stats(dataset_name)
                        dataset_tps_stats[dataset_name] = {
                            "peak_tps": dataset_stats.get("peak_tps", 0.0),
                            "peak_sample": dataset_stats.get("peak_sample"),
                            "top10_tps_mean": dataset_stats.get("top10_tps_mean", 0.0),
                            "avg_tps": dataset_stats.get("avg_tps", 0.0),
                            "num_samples": dataset_stats.get("num_samples", 0),
                        }
                    # Compute overall statistics
                    overall_dataset_stats = stats_collector.compute_dataset_stats()
                    dataset_tps_stats["all"] = {
                        "peak_tps": overall_dataset_stats.get("peak_tps", 0.0),
                        "peak_sample": overall_dataset_stats.get("peak_sample"),
                        "top10_tps_mean": overall_dataset_stats.get("top10_tps_mean", 0.0),
                        "avg_tps": overall_dataset_stats.get("avg_tps", 0.0),
                        "num_samples": overall_dataset_stats.get("num_samples", 0),
                    }
                    final_stats["dataset_tps_stats"] = dataset_tps_stats
                except Exception as e:
                    eval_logger.warning(f"Failed to compute dataset TPS stats: {e}")
            
            with open(os.path.join(self.save_dir, f'rank_{self._dist_rank}_final_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)

        if self.show_speed and len(res) > 0:
            avg_tokens = total_tokens / len(res)
            avg_steps = total_steps / len(res)
            avg_tok_per_step = total_tokens / total_steps if total_steps > 0 else 0
            wall_time = time.time() - overall_start
            # run_time and total_generation_time should match (both sum pure_gen_time)
            # Use total_generation_time to align with the "Overall TPS" above
            throughput_run = total_tokens / total_generation_time if total_generation_time > 0 else (total_tokens / run_time if run_time > 0 else 0.0)
            throughput_total = total_tokens / total_time if total_time > 0 else 0.0

            if self.use_uncertainty_logic:
                if self.branch_verification_mode and self.base_branch_competition:
                    mode = "Multi-Branch (Base Branch Competition Mode)"
                elif self.branch_verification_mode:
                    mode = "Multi-Branch (Verification Mode)"
                else:
                    mode = "Multi-Branch"
            else:
                mode = "Single-Branch"

            print(f"\n==================== FINAL SUMMARY ({mode}, Corrected Stats) ====================")
            if self.experiment_label:
                print(f"  - Experiment Label: {self.experiment_label}")
            print(f"  - Total Samples Processed: {len(res)}")
            print(f"  - Total Generated Tokens (sum of best paths): {total_tokens}")
            print(f"  - Total Steps (sum of best paths): {total_steps}")
            print(f"  - Total Time (loop): {total_time:.2f} seconds")
            print(f"  - Total Run Time (generation): {run_time:.2f} seconds")
            print(f"  - Total Cleanup Time: {free_resource_time:.2f} seconds")
            print(f"  - Overall Wall Duration: {wall_time:.2f} seconds")
            print("--------------------------------------------------------------------")
            # Add pure generation time stats (excluding first step)
            if len(sample_times) > 0 and total_generation_time > 0:
                avg_pure_gen_time = total_generation_time / len(sample_times)
                avg_tps = sum(sample_tpss) / len(sample_tpss) if len(sample_tpss) > 0 else 0.0
                overall_tps_pure = total_tokens / total_generation_time if total_generation_time > 0 else 0.0
                print(f"  - Average Pure Generation Time (excl. first step): {avg_pure_gen_time:.4f} seconds")
                print(f"  - Total Pure Generation Time (excl. first step): {total_generation_time:.4f} seconds")
                print(f"  - Average TPS (per sample, excl. first step): {avg_tps:.2f}")
                print(f"  - Overall TPS (total tokens / total pure time): {overall_tps_pure:.2f}")
            print("--------------------------------------------------------------------")
            print(f"  - Average Tokens per Sample (best path): {avg_tokens:.2f}")
            print(f"  - Average Steps per Sample (best path): {avg_steps:.2f}")
            print(f"  - Overall Effective Tokens/Step Ratio: {avg_tok_per_step:.2f}")
            print(f"  - Overall Throughput (Tokens/Sec, run time): {throughput_run:.2f}")
            print(f"  - Overall Throughput (Tokens/Sec, loop time): {throughput_total:.2f}")
            
            # Display dataset-level TPS statistics
            if stats_collector is not None:
                try:
                    datasets = set(s.get("dataset_name") for s in stats_collector.sample_stats if s.get("dataset_name"))
                    if datasets:
                        print("--------------------------------------------------------------------")
                        print("  Dataset-Level TPS Statistics:")
                        for dataset_name in sorted(datasets):
                            dataset_stats = stats_collector.compute_dataset_stats(dataset_name)
                            peak_tps = dataset_stats.get("peak_tps", 0.0)
                            top10_tps_mean = dataset_stats.get("top10_tps_mean", 0.0)
                            peak_sample = dataset_stats.get("peak_sample")
                            print(f"    [{dataset_name}]")
                            print(f"      - Peak TPS (fastest sample): {peak_tps:.2f}")
                            if peak_sample:
                                print(f"        Sample #{peak_sample['sample_idx']}: {peak_sample['tps']:.2f} TPS "
                                      f"({peak_sample['tokens']} tokens, {peak_sample['steps']} steps, "
                                      f"{peak_sample['generation_time']:.2f}s)")
                            print(f"      - Top 10 Samples TPS Mean: {top10_tps_mean:.2f}")
                            print(f"      - Average TPS: {dataset_stats.get('avg_tps', 0.0):.2f}")
                        
                        # Display overall statistics
                        overall_stats = stats_collector.compute_dataset_stats()
                        print(f"    [All Datasets]")
                        print(f"      - Peak TPS (fastest sample): {overall_stats.get('peak_tps', 0.0):.2f}")
                        peak_sample_all = overall_stats.get("peak_sample")
                        if peak_sample_all:
                            print(f"        Sample #{peak_sample_all['sample_idx']}: {peak_sample_all['tps']:.2f} TPS "
                                  f"({peak_sample_all['tokens']} tokens, {peak_sample_all['steps']} steps, "
                                  f"{peak_sample_all['generation_time']:.2f}s)")
                        print(f"      - Top 10 Samples TPS Mean: {overall_stats.get('top10_tps_mean', 0.0):.2f}")
                        print(f"      - Average TPS: {overall_stats.get('avg_tps', 0.0):.2f}")
                except Exception as e:
                    eval_logger.warning(f"Failed to display dataset TPS stats: {e}")
            
            print("==================================================================================\n")

        if self._distributed and self._dist_world_size > 1:
            self._distributed_forward({"stop": True})
        
        # Save statistics
        if stats_collector is not None:
            try:
                stats_collector.save_stats()
            except Exception as e:
                eval_logger.warning(f"Failed to save statistics: {e}")

        return res

    def _forward_process(self, batch):
        b, length = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1
        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, length)
        mask_indices = torch.rand((b, length), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False
        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(batch).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]

        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)
        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)
            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']
        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)

        perturbed_[mask_index] = self.mask_token_id

        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)
        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end].to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)
        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)

        mask_index[temp_index] = False

        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} > {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning("All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {"prefix_text": e["prefix"], "target_text": e["target"], "prefix": prefix, "target": target}
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize).with_format("torch")
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix, target = elem["prefix"], elem["target"]
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type in ['ar_ftb', 'ar_btf']:
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)
                is_target_greedy_dec = False
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]: raise NotImplementedError
    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]: raise NotImplementedError

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        for block_id in sorted(block_states.keys()):
            if block_states[block_id]['total_masks'] > 0:
                decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= decoded_token_threshold:
                    next_block_id = block_id + 1
                    if next_block_id in block_states:
                        block_states[next_block_id]['is_complete'] = True

    def _shift_logits(self, logits: torch.Tensor, last_logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.shape[1] == 0:
            raise Exception("logits sequence length is 0")

        shifted_logits = torch.zeros_like(logits)
        shifted_logits[:, 1:, :] = logits[:, :-1, :]
        if last_logit is not None:
            shifted_logits[:, 0, :] = last_logit
            return shifted_logits

        shifted_logits[:, 0, :] = 1.0
        return shifted_logits

    def _generate_original_single_branch(self, prompt: torch.Tensor, prefix_length: int = 0) -> Tuple[List[int], Dict, float, float, List[float]]:
        self.model.eval()
        prompt_length = prompt.shape[1]
        if self.max_length is not None and prompt_length > self.max_length:
            eval_logger.warning(
                "Prompt length %d exceeds max_length %d, truncating newest tokens to fit cache capacity.",
                prompt_length,
                self.max_length,
            )
            prompt = prompt[:, -self.max_length:]
            prompt_length = prompt.shape[1]
        single_timer = self._profiler.tic()
        
        # Record time per step
        step_times: List[float] = []
        first_step_time: float = 0.0
        generation_start_time = time.perf_counter()

        x_t = prompt.clone().to(self.device)
        
        # If prefix_length > 0, the prefix portion is already handled via cache; adjust initial state
        effective_prompt_length = prompt_length
        initial_cache_length = 0
        if prefix_length > 0:
            # Prefix is already cached; start from the cached length
            dream_model = self._get_dream_model()
            initial_cache_length = dream_model.get_shared_cache_length()
            # Adjust block_states start positions
            effective_prompt_length = prompt_length  # prompt portion after removing the prefix

        if self.use_full_attention:
            full_attention_mask = None
        else:
            full_attention_mask = create_full_block_attention_mask(
                prompt_length=effective_prompt_length + initial_cache_length, 
                max_length=self.max_length, 
                block_size=self.block_size,
                device=self.device, 
                dtype=self.target_dtype if self.target_dtype not in [None, "auto"] else torch.bfloat16
            )

        block_states = {
            0: {
                'start_pos': initial_cache_length, 
                'end_pos': initial_cache_length + effective_prompt_length, 
                'mask_count': 0, 
                'total_masks': effective_prompt_length, 
                'state': 'to_cache' if prefix_length == 0 else 'in_cache', 
                'is_complete': True
            },
        }

        past_key_values = None
        last_logits = None
        cache_length = initial_cache_length
        current_blocks = 0
        step = 0
        eos_detected = False
        eos_token_id = self.eot_token_id

        stats = {
            "steps_taken": 0, "total_filled_original": 0, "original_fallback_triggers": 0,
        }

        with torch.inference_mode():
            while True:
                step += 1
                step_start_time = time.perf_counter()

                effective_max_new_tokens = self.max_new_tokens or 0
                max_generation_blocks = effective_max_new_tokens // self.block_size if self.block_size > 0 else 0
                if len(block_states)-1 < max_generation_blocks and not eos_detected:
                    last_block_id = len(block_states) - 1
                    if block_states[last_block_id]['total_masks'] > 0:
                        progress = (block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']
                    else:
                        progress = 1.0
                    if progress >= self.block_add_threshold:
                        next_block_end = x_t.shape[1] + self.block_size
                        if self.max_length is not None and next_block_end > self.max_length:
                            break
                        new_block_id = len(block_states)
                        new_start_pos = x_t.shape[1]
                        mask_block = torch.tensor([[self.mask_token_id] * self.block_size], device=self.device)
                        x_t = torch.cat([x_t, mask_block], dim=1)
                        block_states[new_block_id] = {
                            'start_pos': new_start_pos,
                            'end_pos': new_start_pos + self.block_size,
                            'mask_count': self.block_size,
                            'total_masks': self.block_size,
                            'state': 'active',
                            'is_complete': False,
                        }
                        current_blocks += 1

                self._update_block_completion_states(block_states, self.decoded_token_threshold)

                if (x_t == self.mask_token_id).sum() == 0 and current_blocks == 0:
                    break

                blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
                update_kvcache = 0
                if blocks_to_cache:
                    earliest_pos = min(block_states[bid]['start_pos'] for bid in blocks_to_cache)
                    latest_pos = max(block_states[bid]['end_pos'] for bid in blocks_to_cache)
                    update_kvcache = latest_pos - earliest_pos

                process_start_pos = cache_length
                active_blocks_ids = [bid for bid, state in block_states.items() if state['state'] == 'active']

                if update_kvcache > 0:
                    earliest_block_to_cache = min(blocks_to_cache)
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                    input_seq = x_t[:, process_start_pos:]
                elif active_blocks_ids:
                    process_start_pos = min(block_states[bid]['start_pos'] for bid in active_blocks_ids)
                    input_seq = x_t[:, process_start_pos:]
                else:
                    if blocks_to_cache:
                        continue
                    else:
                        break

                if input_seq.shape[1] == 0:
                    if (x_t == self.mask_token_id).any():
                        continue
                    else:
                        break

                cache_length = 0 if past_key_values is None else past_key_values.get_seq_length()
                attention_mask = extract_attention_mask(
                    full_mask=full_attention_mask,
                    start_pos=process_start_pos,
                    input_length=input_seq.shape[1],
                    cache_length=cache_length,
                    use_full_attention=self.use_full_attention,
                )

                outputs = self.model(
                    input_seq,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    update_kvcache=update_kvcache
                )

                if outputs.past_key_values is None:
                    eval_logger.error(f"Model did not return past_key_values at step {step}.")
                    stats['steps_taken'] = step
                    generation_end_time = time.perf_counter()
                    total_generation_time = generation_end_time - generation_start_time
                    if first_step_time == 0.0 and len(step_times) > 0:
                        first_step_time = step_times[0]
                    return (x_t[0, prompt_length:].tolist(), stats, total_generation_time, first_step_time, step_times)

                raw_logits = outputs.logits

                if update_kvcache > 0:
                    last_logits = raw_logits[:, update_kvcache - 1, :].unsqueeze(1)
                    past_key_values = outputs.past_key_values
                shifted_logits = self._shift_logits(raw_logits, last_logit=last_logits)

                tokens_to_update = {}

                for block_id in active_blocks_ids:
                    block_start, block_end = block_states[block_id]['start_pos'], block_states[block_id]['end_pos']
                    mask_indices_in_block = (x_t[0, block_start:block_end] == self.mask_token_id).nonzero(as_tuple=True)[0]
                    if len(mask_indices_in_block) == 0:
                        continue

                    mask_indices_abs = mask_indices_in_block + block_start
                    mask_indices_rel = mask_indices_abs - process_start_pos
                    block_mask_logits = shifted_logits[0, mask_indices_rel, :]

                    confidence, x0, initial_confidence = sample_tokens(
                        block_mask_logits, self.temperature, 
                        top_p=self.top_p, top_k=self.top_k,
                        sampling_strategy=self.sampling_strategy
                    )
                    high_conf_indices = (initial_confidence > self.skip_threshold).nonzero(as_tuple=True)[0]

                    is_complete = block_states[block_id]['is_complete']

                    indices_to_fill = []
                    if is_complete:
                        if len(high_conf_indices) > 0:
                            indices_to_fill = high_conf_indices
                        else:
                            stats["original_fallback_triggers"] += 1
                            if len(confidence) > 0:
                                _, most_conf_idx = torch.topk(confidence, 1)
                                indices_to_fill = most_conf_idx
                    else:
                        if len(high_conf_indices) > 0:
                            indices_to_fill = high_conf_indices

                    if len(indices_to_fill) > 0:
                        for idx in indices_to_fill:
                            pos = mask_indices_abs[idx.item()].item()
                            token = x0[idx.item()].item()
                            tokens_to_update[pos] = token
                            if token == eos_token_id:
                                eos_detected = True

                stats["total_filled_original"] += len(tokens_to_update)

                # [MEMORY OPTIMIZED] Create tensors only when updates exist and clear them immediately
                if tokens_to_update:
                    positions = torch.tensor(list(tokens_to_update.keys()), device=self.device)
                    values = torch.tensor(list(tokens_to_update.values()), device=self.device)
                    x_t[0, positions] = values
                    # Clear temporary tensors immediately
                    del positions, values

                if update_kvcache > 0:
                    cache_length = past_key_values.get_seq_length() if past_key_values is not None else cache_length
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'in_cache'

                blocks_to_deactivate = []
                for block_id in active_blocks_ids:
                    start, end = block_states[block_id]['start_pos'], block_states[block_id]['end_pos']
                    new_mask_count = (x_t[0, start:end] == self.mask_token_id).sum().item()
                    block_states[block_id]['mask_count'] = new_mask_count
                    if new_mask_count == 0:
                        blocks_to_deactivate.append(block_id)

                for block_id in blocks_to_deactivate:
                    if block_states[block_id]['state'] == 'active':
                        can_deactivate = all(prev_bid not in block_states or block_states[prev_bid]['state'] != 'active' for prev_bid in range(block_id))
                        if can_deactivate:
                            block_states[block_id]['state'] = 'to_cache'
                            current_blocks -= 1

                # Record time per step
                step_end_time = time.perf_counter()
                step_duration = step_end_time - step_start_time
                step_times.append(step_duration)
                if step == 1:
                    first_step_time = step_duration

                if step > 1000:
                    eval_logger.warning("Generation stopped due to exceeding 1000 steps.")
                    break

        generation_end_time = time.perf_counter()
        total_generation_time = generation_end_time - generation_start_time
        # If the first-step time was not recorded, fall back to the first step duration
        if first_step_time == 0.0 and len(step_times) > 0:
            first_step_time = step_times[0]

        generated_sequence_ids = x_t[0, prompt_length:].tolist()
        stats['steps_taken'] = step

        result = (generated_sequence_ids, stats, total_generation_time, first_step_time, step_times)
        self._profiler.toc(single_timer, "single_branch_total")
        return result

if __name__ == "__main__":
    set_seed(1234)

    env_rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    try:
        current_rank = int(env_rank)
    except ValueError:
        current_rank = 0

    if current_rank == 0:
        cli_evaluate()
    else:
        parser = setup_parser()
        worker_args = parser.parse_args()

        additional_cfg = {
            "batch_size": worker_args.batch_size,
            "max_batch_size": worker_args.max_batch_size,
            "device": worker_args.device,
        }
        model_cls = model_registry.get_model(worker_args.model)
        worker_model = model_cls.create_from_arg_string(worker_args.model_args or "", additional_cfg)
        worker_model._distributed_worker_loop()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    if current_rank != 0:
        sys.exit(0)