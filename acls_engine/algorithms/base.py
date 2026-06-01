"""
base.py — Abstract base class for all AHA 2025 ACLS algorithm processors.

Every algorithm (cardiac arrest, tachyarrhythmia, bradycardia, etc.) inherits
from BaseAlgorithm and overrides:
  - process_event(event)      : handle a single timestamped event
  - end_of_session_checks()   : rules evaluated at session end
"""

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SEVERITY → PENALTY WEIGHT MAP ─────────────────────────────────────────────
SEVERITY_PENALTY: Dict[str, float] = {
    "CRITICAL": 0.35,
    "HIGH":     0.20,
    "MEDIUM":   0.10,
    "MODERATE": 0.10,   # alias used by cardiac arrest rules
    "LOW":      0.04,
    "ADVISORY": 0.00,
    "INFO":     0.00,
}


class BaseAlgorithm:
    """
    Abstract base for all ACLS algorithm processors.

    Subclasses must set:
      ALGORITHM_ID : str  — machine-readable algorithm name embedded in findings
    """

    ALGORITHM_ID: str = "base"

    def __init__(self, rules: dict):
        """
        Args:
            rules: Parsed JSON rule-file (thresholds, context, guards, states …)
        """
        self.rules: dict = rules
        self.context: Dict[str, Any] = {}
        self.findings: List[Dict] = []
        self.finding_counter: int = 0
        self._state: str = "initial"
        self._init_context()

    # ── INITIALISE CONTEXT FROM RULES JSON ────────────────────────────────────
    def _init_context(self) -> None:
        """Deep-copy the 'context' block from the rules JSON as starting state."""
        if "context" in self.rules:
            self.context = copy.deepcopy(self.rules["context"])

    # ── ABSTRACT INTERFACE ────────────────────────────────────────────────────
    def process_event(self, event: dict) -> None:
        """Process one timestamped event from the session event stream."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement process_event()"
        )

    def end_of_session_checks(self) -> None:
        """
        Called after all events are streamed.
        Override to add rules that are evaluated at the end of the session
        (e.g., 'amiodarone not given after 3rd shock').
        """
        pass

    # ── FINDING EMITTER ───────────────────────────────────────────────────────
    def _emit(
        self,
        rule_id:        str,
        severity:       str,
        message:        str,
        timestamp_sec:  float,
        from_event:     str,
        to_event:       str,
        actual_gap:     Optional[float],
        expected:       Optional[float],
        guideline:      str,
        recommendation: str = "",
        confidence:     float = 0.95,
    ) -> dict:
        """
        Emit a structured deviation finding.

        Args:
            rule_id        : Rule identifier (e.g. 'R_ARREST_001')
            severity       : CRITICAL | HIGH | MEDIUM | LOW | ADVISORY
            message        : Human-readable description of the deviation
            timestamp_sec  : Event timestamp when deviation occurred
            from_event     : Event that should have triggered the action
            to_event       : Event/action that was expected or performed
            actual_gap     : Actual elapsed time (seconds), or None
            expected       : Guideline-required time (seconds), or None
            guideline      : Source guideline reference text
            recommendation : Corrective action guidance
            confidence     : Detection confidence 0.0–1.0

        Returns:
            The finding dict (also appended to self.findings)
        """
        self.finding_counter += 1
        sev = severity.upper()

        finding: Dict[str, Any] = {
            "finding_id":        f"fnd_{self.finding_counter:04d}",
            "rule_id":           rule_id,
            "algorithm":         self.ALGORITHM_ID,
            "status":            "deviation",
            "severity":          sev,
            "penalty_weight":    SEVERITY_PENALTY.get(sev, 0.10),
            "confidence":        confidence,
            "timestamp_sec":     timestamp_sec,
            "from_event":        from_event,
            "to_event":          to_event,
            "actual_gap_sec":    actual_gap,
            "expected_sec":      expected,
            "deviation_message": message,
            "guideline":         guideline,
            "recommendation":    recommendation,
        }

        self.findings.append(finding)
        logger.info(f"  [{sev}] {rule_id}: {message}")
        return finding
