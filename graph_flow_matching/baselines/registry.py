"""Factory registry for baseline generators."""

from __future__ import annotations

from typing import Any

from graph_flow_matching.baselines.base import BaseGenerator

REGISTRY: dict[str, type[BaseGenerator]] = {}


def register(name: str):
    """Class decorator that registers a generator under *name*."""

    def decorator(cls: type[BaseGenerator]) -> type[BaseGenerator]:
        if name in REGISTRY:
            raise KeyError(f"Duplicate registration: '{name}'")
        REGISTRY[name] = cls
        return cls

    return decorator


def create(name: str, **kwargs: Any) -> BaseGenerator:
    """Instantiate the generator registered as *name*."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown generator '{name}'. Available: {available}"
        )
    return REGISTRY[name](**kwargs)
