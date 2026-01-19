"""
Federated Learning Backdoor Framework
A modular framework for FL research with attack and defense capabilities
"""

__version__ = "1.0.0"

from .framework import FLResearchFramework
from .client import FLClient
from .server import FLServer
from .memory import MemoryMonitor

__all__ = [
    "FLResearchFramework",
    "FLClient", 
    "FLServer",
    "MemoryMonitor"
]
