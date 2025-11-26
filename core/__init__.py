"""Core modules for the Ego-Sphere consciousness kernel."""

from .proto_self import ProtoSelf
from .snn_engine import SNNEngine
from .language_cortex import LanguageCortex
from .memory_bank import MemoryBank
from .checkpoint import CheckpointManager

__all__ = ["ProtoSelf", "SNNEngine", "LanguageCortex", "MemoryBank", "CheckpointManager"]
