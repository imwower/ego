"""Core modules for the Ego-Sphere consciousness kernel."""

from .proto_self import ProtoSelf
from .snn_engine import SNNEngine
from .language_cortex import LanguageCortex
from .checkpoint import CheckpointManager
from .episodic_memory import EpisodicMemory

__all__ = ["ProtoSelf", "SNNEngine", "LanguageCortex", "CheckpointManager", "EpisodicMemory"]
