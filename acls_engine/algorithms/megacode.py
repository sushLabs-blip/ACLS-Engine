"""
algorithms/megacode.py
AHA 2025 ACLS Megacode Processor

Handles multi-phase scenarios that cross algorithm boundaries:

  Phase 1 → Brady OR Tachy algorithm
  Phase 2 → VF/pVT Cardiac Arrest
  Phase 3 → Asystole OR PEA Cardiac Arrest
  Phase 4 → ROSC + Post Cardiac Arrest Care

Supported megacode cases (from AHA Megacode PDF):
  Case 1: Sinus Bradycardia  → VF/pVT → Asystole → ROSC
  Case 2: Mobitz II AV Block → VF/pVT → Asystole → ROSC
  Case 3: Tachycardia (VT) Cardioversion → VF/pVT → PEA → ROSC
  Case 4: Tachycardia (SVT) Drug Therapy → VF/pVT → PEA → ROSC
  Case 5: Tachycardia (SVT) Cardioversion → VF/pVT → PEA → ROSC
  Case 6: Tachycardia (VT) Drug Therapy  → VF/pVT → PEA → ROSC

Rule IDs: R_MEGA_001 – R_MEGA_010 (phase transition checks)
          Sub-algorithm rules delegated to their own processors.
"""

import logging
from typing import List, Optional, Dict

from .base import BaseAlgorithm
from .cardiac_arrest import CardiacArrestAlgorithm
from .tachyarrhythmia import TachyarrhythmiaAlgorithm
from .bradycardia import BradycardiaAlgorithm

logger = logging.getLogger(__name__)

GUIDELINE = "AHA 2025 ACLS Megacode"


# ── PHASE DEFINITIONS ─────────────────────────────────────────────────────────
PHASE_1 = "phase_1_initial"         # Brady or Tachy
PHASE_2 = "phase_2_vf_pvt"         # VF/pVT Cardiac Arrest
PHASE_3 = "phase_3_asystole_pea"   # Asystole or PEA
PHASE_4 = "phase_4_post_rosc"      # Post Cardiac Arrest Care

# Events that signal a phase transition
PHASE2_TRIGGERS = {"vf_detected", "pvt_detected", "cpr_initiated", "arrest_recognized"}
PHASE3_TRIGGERS = {"pea_detected", "asystole_detected"}
PHASE4_TRIGGERS = {"rosc_achieved"}


class MegacodeAlgorithm(BaseAlgorithm):
    """
    Multi-phase megacode processor.

    Delegates each phase to the appropriate sub-algorithm processor
    while tracking phase transitions and scoring each phase separately.
    """

    ALGORITHM_ID = "megacode"

    def __init__(self, rules: dict, sub_type: str = "brady_to_arrest"):
        super().__init__(rules)
        self.sub_type = sub_type  # brady_to_arrest | tachy_to_arrest

        # ── Phase tracking ────────────────────────────────────────────────────
        self.current_phase: str = PHASE_1
        self.phase_start_times: Dict[str, Optional[float]] = {
            PHASE_1: None,
            PHASE_2: None,
            PHASE_3: None,
            PHASE_4: None,
        }
        self.phase_findings: Dict[str, List[dict]] = {
            PHASE_1: [],
            PHASE_2: [],
            PHASE_3: [],
            PHASE_4: [],
        }

        # ── Phase scores ──────────────────────────────────────────────────────
        self.phase_scores: Dict[str, Optional[float]] = {
            PHASE_1: None,
            PHASE_2: None,
            PHASE_3: None,
            PHASE_4: None,
        }

        # ── Sub-algorithm processors ──────────────────────────────────────────
        # Phase 1 processor — determined by sub_type
        if sub_type == "brady_to_arrest":
            self._phase1_processor = BradycardiaAlgorithm(rules)
        else:
            self._phase1_processor = TachyarrhythmiaAlgorithm(rules)

        # Phase 2 + 3 share one cardiac arrest processor
        self._arrest_processor = CardiacArrestAlgorithm(rules, sub_type="unknown")

        # Active processor — switches as phases change
        self._active_processor = self._phase1_processor

        # ── Transition timing ─────────────────────────────────────────────────
        self.phase1_start: Optional[float] = None
        self.phase2_start: Optional[float] = None
        self.phase3_start: Optional[float] = None
        self.phase4_start: Optional[float] = None

        # Phase 1 → 2 transition checks
        self.phase1_ended:  bool = False
        self.phase2_ended:  bool = False

        # End of session tracking
        self.rosc_time: Optional[float] = None

    # ── PHASE TRANSITION LOGIC ────────────────────────────────────────────────

    def _transition_to_phase2(self, event: dict) -> None:
        """Switch from Phase 1 (brady/tachy) to Phase 2 (VF/pVT arrest)."""
        t = event["timestamp_sec"]

        if not self.phase1_ended:
            # Run end-of-phase checks for phase 1
            self._phase1_processor.end_of_session_checks()
            self.phase_findings[PHASE_1] = list(self._phase1_processor.findings)
            self._score_phase(PHASE_1)
            self.phase1_ended = True

        self.current_phase = PHASE_2
        self.phase2_start = t
        self.phase_start_times[PHASE_2] = t
        self._active_processor = self._arrest_processor

        logger.info(f"  [MEGA] Phase 1 → Phase 2 (VF/pVT Arrest) at {t}s")

        # Check: was phase 1 managed before deterioration?
        self._emit(
            rule_id="R_MEGA_001", severity="INFO",
            message=f"Phase transition: {self.sub_type.replace('_', ' ')} → Cardiac Arrest (VF/pVT) at {t}s",
            timestamp_sec=t,
            from_event="phase_1", to_event="phase_2_vf_pvt",
            actual_gap=None, expected=None,
            guideline=GUIDELINE,
            recommendation="Team must rapidly recognise rhythm change and transition to cardiac arrest algorithm.",
        )

    def _transition_to_phase3(self, event: dict) -> None:
        """Switch from Phase 2 (VF/pVT) to Phase 3 (Asystole/PEA)."""
        t = event["timestamp_sec"]

        if not self.phase2_ended:
            self.phase_findings[PHASE_2] = [
                f for f in self._arrest_processor.findings
                if f not in self.phase_findings[PHASE_1]
            ]
            self._score_phase(PHASE_2)
            self.phase2_ended = True

        self.current_phase = PHASE_3
        self.phase3_start = t
        self.phase_start_times[PHASE_3] = t

        rhythm = event["event_type"].replace("_detected", "")
        logger.info(f"  [MEGA] Phase 2 → Phase 3 ({rhythm}) at {t}s")

        self._emit(
            rule_id="R_MEGA_002", severity="INFO",
            message=f"Phase transition: VF/pVT → {rhythm} at {t}s",
            timestamp_sec=t,
            from_event="phase_2_vf_pvt", to_event="phase_3_asystole_pea",
            actual_gap=None, expected=None,
            guideline=GUIDELINE,
            recommendation="Continue high-quality CPR. Follow non-shockable pathway per AHA 2025.",
        )

    def _transition_to_phase4(self, event: dict) -> None:
        """Switch to Phase 4 (Post ROSC Care)."""
        t = event["timestamp_sec"]
        self.rosc_time = t
        self.current_phase = PHASE_4
        self.phase4_start = t
        self.phase_start_times[PHASE_4] = t

        logger.info(f"  [MEGA] Phase 3 → Phase 4 (Post ROSC Care) at {t}s")

        self._emit(
            rule_id="R_MEGA_003", severity="INFO",
            message=f"ROSC achieved at {t}s — transitioning to Post Cardiac Arrest Care",
            timestamp_sec=t,
            from_event="phase_3", to_event="phase_4_post_rosc",
            actual_gap=None, expected=None,
            guideline=GUIDELINE,
            recommendation="Initiate post-cardiac arrest care: 12-lead ECG, temperature management, MAP ≥65 mmHg.",
        )

    # ── PHASE SCORING ─────────────────────────────────────────────────────────

    def _score_phase(self, phase: str) -> None:
        """Calculate performance score for a completed phase."""
        findings = self.phase_findings[phase]
        deviations = [f for f in findings if f.get("status") == "deviation"]
        total_penalty = sum(f.get("penalty_weight", 0) for f in deviations)
        score = max(0.0, round((1.0 - total_penalty) * 100, 1))
        self.phase_scores[phase] = score
        logger.info(f"  [MEGA] Phase {phase} score: {score}")

    # ── EVENT DISPATCHER ──────────────────────────────────────────────────────

    def process_event(self, event: dict) -> None:
        etype = event["event_type"]
        t = event["timestamp_sec"]

        logger.info(f"  [MEGA] {etype} @ {t}s | phase={self.current_phase}")

        # ── Set phase 1 start ─────────────────────────────────────────────────
        if self.phase1_start is None:
            self.phase1_start = t
            self.phase_start_times[PHASE_1] = t

        # ── Phase transition checks ───────────────────────────────────────────
        if self.current_phase == PHASE_1:
            if etype in PHASE2_TRIGGERS:
                self._transition_to_phase2(event)
                # Fall through — let arrest processor handle this event too

        elif self.current_phase == PHASE_2:
            if etype in PHASE3_TRIGGERS:
                self._transition_to_phase3(event)
                # Fall through — let arrest processor handle this event too

        elif self.current_phase in [PHASE_2, PHASE_3]:
            if etype in PHASE4_TRIGGERS:
                self._transition_to_phase4(event)

        # ── Delegate to active processor ──────────────────────────────────────
        if self._active_processor is not None:
            self._active_processor.process_event(event)

        # ── Sync findings from sub-processors ─────────────────────────────────
        self._sync_findings()

    def _sync_findings(self) -> None:
        """Pull latest findings from all sub-processors into our findings list."""
        all_sub_findings = []
        all_sub_findings.extend(self._phase1_processor.findings)
        all_sub_findings.extend(self._arrest_processor.findings)

        # Add mega-level findings (phase transitions)
        mega_findings = [f for f in self.findings if f.get("rule_id", "").startswith("R_MEGA")]

        # Rebuild combined list — sub findings + mega findings
        self.findings = all_sub_findings + mega_findings

    # ── END OF SESSION CHECKS ─────────────────────────────────────────────────

    def end_of_session_checks(self) -> None:
        """Run end-of-session checks across all phases."""

        # Run sub-processor end checks
        self._active_processor.end_of_session_checks()
        self._sync_findings()

        # Score remaining phases
        if self.phase_scores[PHASE_1] is None:
            self.phase_findings[PHASE_1] = list(self._phase1_processor.findings)
            self._score_phase(PHASE_1)

        if self.phase_scores[PHASE_2] is None and self.phase2_start is not None:
            self._score_phase(PHASE_2)

        if self.phase_scores[PHASE_3] is None and self.phase3_start is not None:
            self.phase_findings[PHASE_3] = list(self._arrest_processor.findings)
            self._score_phase(PHASE_3)

        # Check: was Post ROSC care initiated?
        arrest_p = self._arrest_processor
        if self.rosc_time is not None:
            if not getattr(arrest_p, "ecg_obtained", False):
                self._emit(
                    rule_id="R_MEGA_004", severity="HIGH",
                    message="12-lead ECG not obtained after ROSC in megacode scenario",
                    timestamp_sec=0,
                    from_event="rosc_achieved", to_event="post_rosc_ecg_obtained",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 Post Cardiac Arrest Care",
                    recommendation="Obtain 12-lead ECG immediately after ROSC to screen for STEMI.",
                )
            if not getattr(arrest_p, "temp_management_done", False):
                self._emit(
                    rule_id="R_MEGA_005", severity="MEDIUM",
                    message="Temperature management not initiated after ROSC in megacode scenario",
                    timestamp_sec=0,
                    from_event="rosc_achieved", to_event="temp_management_initiated",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 Post Cardiac Arrest Care",
                    recommendation="Initiate targeted temperature management 32-37.5°C for unresponsive post-arrest patients.",
                )

        # Check: all phases were reached
        if self.phase2_start is None:
            self._emit(
                rule_id="R_MEGA_006", severity="HIGH",
                message="Megacode never progressed to Phase 2 (Cardiac Arrest) — VF/pVT not detected or managed",
                timestamp_sec=0,
                from_event="phase_1", to_event="phase_2_vf_pvt",
                actual_gap=None, expected=None,
                guideline=GUIDELINE,
                recommendation="In megacode scenarios the patient deteriorates to cardiac arrest — ensure team recognises and responds to rhythm change.",
            )

        if self.rosc_time is None:
            self._emit(
                rule_id="R_MEGA_007", severity="HIGH",
                message="ROSC not achieved — megacode scenario did not reach Phase 4",
                timestamp_sec=0,
                from_event="phase_3", to_event="rosc_achieved",
                actual_gap=None, expected=None,
                guideline=GUIDELINE,
                recommendation="Continue high-quality CPR and correct reversible causes. ROSC is the goal of the megacode.",
            )

        self._sync_findings()

    # ── PHASE REPORT HELPER ───────────────────────────────────────────────────

    def get_phase_report(self) -> dict:
        """Return structured per-phase scoring for the output JSON."""
        return {
            "phase_1": {
                "name": "bradycardia" if "brady" in self.sub_type else "tachyarrhythmia",
                "start_sec": self.phase_start_times.get(PHASE_1),
                "score": self.phase_scores.get(PHASE_1),
                "deviations": len([f for f in self.phase_findings.get(PHASE_1, []) if f.get("status") == "deviation"]),
            },
            "phase_2": {
                "name": "vf_pvt_cardiac_arrest",
                "start_sec": self.phase_start_times.get(PHASE_2),
                "score": self.phase_scores.get(PHASE_2),
                "deviations": len([f for f in self.phase_findings.get(PHASE_2, []) if f.get("status") == "deviation"]),
            },
            "phase_3": {
                "name": "asystole_pea_cardiac_arrest",
                "start_sec": self.phase_start_times.get(PHASE_3),
                "score": self.phase_scores.get(PHASE_3),
                "deviations": len([f for f in self.phase_findings.get(PHASE_3, []) if f.get("status") == "deviation"]),
            },
            "phase_4": {
                "name": "post_rosc_care",
                "start_sec": self.phase_start_times.get(PHASE_4),
                "score": self.phase_scores.get(PHASE_4),
                "deviations": len([f for f in self.phase_findings.get(PHASE_4, []) if f.get("status") == "deviation"]),
            },
        }
