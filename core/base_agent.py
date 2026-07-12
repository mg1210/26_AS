"""
core/base_agent.py
──────────────────
Abstract base class every specialist agent inherits.
Enforces a consistent run() interface and audit logging.
"""

from abc import ABC, abstractmethod
from core.state import PipelineState
import traceback
import time


class BaseAgent(ABC):
    """
    Every agent must implement run(state) -> PipelineState.
    Wraps execution with error handling and audit logging.
    """

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self._start_time: float = 0.0

    def execute(self, state: PipelineState) -> PipelineState:
        """Public entry point. Wraps run() with logging and error handling."""
        self._log(f"▶  Starting")
        self._start_time = time.time()
        state.log_audit(self.name, "started")
        try:
            state = self.run(state)
            state.log_audit(self.name, "completed")
            self._log(f"✓  Completed")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            state.log_error(self.name, msg)
            self._log(f"✗  Failed — {msg}")
            if self.verbose:
                traceback.print_exc()
        return state

    def build_response(
        self,
        summary: str,
        observations: list,
        reasoning: str,
        recommendations: list,
        artifacts: dict = None,
        overall_status: str = "success",
    ) -> dict:
        """Return a structured agent response dict."""
        return {
            "agent_name":     self.name,
            "summary":        summary,
            "observations":   observations,
            "reasoning":      reasoning,
            "recommendations": recommendations,
            "artifacts":      artifacts or {},
            "execution_time": round(time.time() - self._start_time, 2),
            "overall_status": overall_status,
        }

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Override in each specialist agent."""
        ...

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{self.name}] {msg}")

    def _info(self, msg: str):
        if self.verbose:
            print(f"    {msg}")
