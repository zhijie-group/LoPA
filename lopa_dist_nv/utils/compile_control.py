import functools
from typing import Any, Callable, Dict, Optional

import torch

_TORCH_COMPILE_ENABLED: bool = True
_TORCH_COMPILE_KWARGS: Dict[str, Any] = {}


def configure_torch_compile(*, enabled: bool, kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Globally configure whether torch.compile should be used.

    Args:
        enabled: If False, compiled paths are skipped and eager functions are used.
        kwargs: Extra keyword arguments forwarded to torch.compile when enabled.
    """
    global _TORCH_COMPILE_ENABLED, _TORCH_COMPILE_KWARGS
    _TORCH_COMPILE_ENABLED = bool(enabled)
    _TORCH_COMPILE_KWARGS = dict(kwargs or {})


def is_torch_compile_enabled() -> bool:
    return _TORCH_COMPILE_ENABLED and hasattr(torch, "compile")


def _wrap_with_optional_compile(
    func: Callable[..., Any],
    compile_kwargs: Dict[str, Any],
) -> Callable[..., Any]:
    compiled_fn: Optional[Callable[..., Any]] = None

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled_fn
        if is_torch_compile_enabled():
            if compiled_fn is None:
                kwargs_to_use = {**_TORCH_COMPILE_KWARGS, **compile_kwargs}
                compiled_fn = torch.compile(func, **kwargs_to_use)
            return compiled_fn(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def optional_compile(*decorator_args: Any, **decorator_kwargs: Any):
    """Decorator that mirrors @torch.compile but can be disabled globally.

    It supports both usage styles::

        @optional_compile
        def fn(...):
            ...

        @optional_compile(mode="reduce-overhead")
        def fn(...):
            ...
    """

    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
        func = decorator_args[0]
        return _wrap_with_optional_compile(func, {})

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return _wrap_with_optional_compile(func, dict(decorator_kwargs))

    return decorator
