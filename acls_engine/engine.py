"""
engine.py — ACLS Rule Engine (Master Dispatcher)
=================================================
AI-powered CPR Training Analysis & Debriefing System
AHA 2025 Guidelines | Educational / Simulation Use Only

Architecture
------------
  ScenarioClassifier          detects algorithm type from the event stream
  ACLSEngine.evaluate()       loads the matching algorithm processor and
                               streams events through it
  ACLSEngine.print_report()   prints a human-readable deviation summary
  ACLSEngine.save_findings()  writes structured JSON output for the debrief UI

Supported algorithms
--------------------
  cardiac_arrest              VF | pVT | PEA | asystole
  tachyarrhythmia_with_pulse  narrow/wide QRS, stable/unstable, WPW
  bradycardia_with_pulse      nodal / infranodal / transplant

Usage
-----
  python engine.py                                   # default test_events.json
  python engine.py test_scenarios/bradycardia_nodal.json
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional

from scenario_classifier import ScenarioClassifier
from algorithms.cardiac_arrest import CardiacArrestAlgorithm
from algorithms.tachyarrhythmia import TachyarrhythmiaAlgorithm
from algorithms.bradycardia import BradycardiaAlgorithm
from algorithms.base import SEVERITY_PENALTY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── ALGORITHM REGISTRY ────────────────────────────────────────────────────────
_ALGORITHM_REGISTRY = {
    "cardiac_arrest": {
        "rules_file": "cardiac_arrest_rules.json",
        "processor":  CardiacArrestAlgorithm,
        "needs_subtype": True,
    },
    "tachyarrhythmia_with_pulse": {
        "rules_file": "aha_2025_tachyarrhythmia_pulse_v2.json",
        "processor":  TachyarrhythmiaAlgorithm,
        "needs_subtype": False,
    },
    "bradycardia_with_pulse": {
        "rules_file": "aha_2025_bradycardia_pulse_v2.json",
        "processor":  BradycardiaAlgorithm,
        "needs_subtype": False,
    },
}


class ACLSEngine:
    """
    Master ACLS Rule Engine.

    Public interface
    ----------------
    evaluate(events_data)  → list[dict]   run analysis; return findings
    print_report(events_data)             console summary
    save_findings(events_data, path)      write findings_output.json
    """

    def __init__(self):
        self._processor  = None
        self._algo_type: Optional[str] = None
        self._sub_type:  Optional[str] = None

    # ── MAIN EVALUATE ─────────────────────────────────────────────────────────
    def evaluate(self, events_data: dict) -> List[dict]:
        """
        Classify and process a session.

        Args:
            events_data: Session dict with keys:
                session_id, session_date, scenario_type, events[]

        Returns:
            List of structured finding dicts (deviations only).
        """
        events = events_data.get("events", [])
        sorted_events = sorted(events, key=lambda e: e["timestamp_sec"])

        logger.info(f"\nEvaluating session: {events_data.get('session_id')}")
        logger.info("─" * 60)

        # ── Step 1: Classify scenario ─────────────────────────────────────────
        self._algo_type, self._sub_type = ScenarioClassifier.classify(sorted_events)
        logger.info(
            f"Classified → algorithm: {self._algo_type} | "
            f"sub-type: {self._sub_type or 'n/a'}"
        )

        if self._algo_type == "unknown":
            logger.warning("  [WARN] Could not classify scenario — no findings generated.")
            return []

        # ── Step 2: Load rules ────────────────────────────────────────────────
        cfg = _ALGORITHM_REGISTRY[self._algo_type]
        rules_path = os.path.join(_ROOT, cfg["rules_file"])
        if not os.path.exists(rules_path):
            raise FileNotFoundError(
                f"Rules file not found: {rules_path}\n"
                f"Expected at: {_ROOT}"
            )
        rules = _load_json(rules_path)

        # ── Step 3: Instantiate processor ────────────────────────────────────
        if cfg["needs_subtype"]:
            self._processor = cfg["processor"](rules, sub_type=self._sub_type)
        else:
            self._processor = cfg["processor"](rules)

        # -- Step 4: Stream events ---------------------------------------------
        for event in sorted_events:
            if event["event_type"] == "session_end":
                self._processor.end_of_session_checks()
                break
            self._processor.process_event(event)
        else:
            # No session_end event present - still run end-of-session checks
            self._processor.end_of_session_checks()

        return self._processor.findings

    # -- CONSOLE REPORT --------------------------------------------------------
    def print_report(self, events_data: dict) -> None:
        if self._processor is None:
            print("No results - call evaluate() first.")
            return

        deviations = [f for f in self._processor.findings if f["status"] == "deviation"]
        sep = "=" * 65
        dash = "-" * 65
        print("\n" + sep)
        print("  ACLS RULE ENGINE - DEBRIEF REPORT")
        print(f"  Session   : {events_data.get('session_id')}")
        print(f"  Date      : {events_data.get('session_date')}")
        print(f"  Algorithm : {self._algo_type}  (sub-type: {self._sub_type or 'n/a'})")
        print(f"  Scenario  : {events_data.get('scenario_type', 'n/a')}")
        print(f"  State     : {getattr(self._processor, '_state', 'n/a')}")
        print(sep)
        print(f"\n  TOTAL DEVIATIONS: {len(deviations)}")

        if not deviations:
            print("\n  [OK] No protocol deviations detected.\n")
        else:
            print("\n" + dash)
            print("  DEVIATIONS")
            print(dash)
            for d in deviations:
                sev_tag = {
                    "CRITICAL": "[!!!]",
                    "HIGH":     "[!! ]",
                    "MEDIUM":   "[!  ]",
                    "LOW":      "[.  ]",
                }.get(d["severity"], "[   ]")
                print(f"\n  {sev_tag} [{d['severity']}] {d['rule_id']} | {d['finding_id']}")
                print(f"     {d['deviation_message']}")
                if d.get("actual_gap_sec") is not None:
                    print(f"     Gap: {d['actual_gap_sec']}s  |  Allowed: {d['expected_sec']}s")
                if d.get("recommendation"):
                    print(f"     → {d['recommendation']}")
                print(f"     Penalty: {d['penalty_weight']}  |  Ref: {d['guideline']}")

        print("\n" + "=" * 65 + "\n")

    # ── SAVE FINDINGS ─────────────────────────────────────────────────────────
    def save_findings(
        self,
        events_data: dict,
        output_path: str = "findings_output.json",
    ) -> None:
        """Save structured findings to JSON for debrief UI consumption."""
        if self._processor is None:
            return

        deviations = [f for f in self._processor.findings if f["status"] == "deviation"]
        strengths  = self._build_strengths()

        total_penalty     = sum(f.get("penalty_weight", 0) for f in deviations)
        performance_score = max(0.0, round((1.0 - total_penalty) * 100, 1))

        output: Dict = {
            "session_id":        events_data.get("session_id"),
            "session_date":      events_data.get("session_date"),
            "scenario_type":     events_data.get("scenario_type", ""),
            "algorithm":         self._algo_type,
            "sub_type":          self._sub_type,
            "final_state":       getattr(self._processor, "_state", "unknown"),
            "performance_score": performance_score,
            "total_deviations":  len(deviations),
            "total_strengths":   len(strengths),
            "strengths":         strengths,
            "deviations":        deviations,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"\nFindings saved → {output_path}")

    def _build_strengths(self) -> List[dict]:
        """Build strength observations from what was done correctly."""
        strengths = []
        ctx = getattr(self._processor, "context", {})
        algo = self._algo_type

        if algo == "cardiac_arrest":
            p = self._processor
            if p.shock_count > 0 and p.current_rhythm not in ["PEA", "pea", "asystole"]:
                strengths.append({
                    "strength_id": "STR_001",
                    "description": f"Defibrillation performed — {p.shock_count} shock(s) delivered",
                    "domain": "shock_delivery",
                })
            if p.epi_count > 0:
                strengths.append({
                    "strength_id": "STR_002",
                    "description": f"Epinephrine administered {p.epi_count} time(s)",
                    "domain": "drug_administration",
                })
            if p.amiodarone_given:
                strengths.append({
                    "strength_id": "STR_003",
                    "description": "Amiodarone given for refractory shockable rhythm",
                    "domain": "drug_administration",
                })
            if p.lidocaine_given:
                strengths.append({
                    "strength_id": "STR_004",
                    "description": "Lidocaine given as antiarrhythmic alternative",
                    "domain": "drug_administration",
                })

        elif algo == "tachyarrhythmia_with_pulse":
            if ctx.get("twelve_lead_obtained"):
                strengths.append({"strength_id": "STR_001", "description": "12-lead ECG obtained", "domain": "assessment"})
            if ctx.get("vagal_maneuver_attempted"):
                strengths.append({"strength_id": "STR_002", "description": "Vagal manoeuvres attempted before adenosine", "domain": "intervention"})
            if ctx.get("sedation_given"):
                strengths.append({"strength_id": "STR_003", "description": "Sedation given before cardioversion", "domain": "patient_safety"})
            if ctx.get("expert_consulted"):
                strengths.append({"strength_id": "STR_004", "description": "Expert consultation obtained", "domain": "communication"})
            if ctx.get("iv_access"):
                strengths.append({"strength_id": "STR_005", "description": "IV access established", "domain": "access"})

        elif algo == "bradycardia_with_pulse":
            p = self._processor
            if ctx.get("iv_access"):
                strengths.append({"strength_id": "STR_001", "description": "IV access established", "domain": "access"})
            if ctx.get("twelve_lead_obtained"):
                strengths.append({"strength_id": "STR_002", "description": "12-lead ECG obtained", "domain": "assessment"})
            if ctx.get("av_block_type"):
                strengths.append({
                    "strength_id": "STR_003",
                    "description": f"AV block correctly classified: {ctx['av_block_type']}",
                    "domain": "assessment",
                })
            if p.tcp_mech_capture:
                strengths.append({"strength_id": "STR_004", "description": "TCP mechanical capture confirmed", "domain": "pacing"})
            if ctx.get("expert_consulted"):
                strengths.append({"strength_id": "STR_005", "description": "Expert consultation obtained", "domain": "communication"})

        return strengths


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scenario_file = sys.argv[1] if len(sys.argv) > 1 else "test_scenarios/bradycardia_nodal.json"

    if not os.path.exists(scenario_file):
        print(f"ERROR: Scenario file not found: {scenario_file}")
        sys.exit(1)

    events_data = _load_json(scenario_file)
    engine = ACLSEngine()
    engine.evaluate(events_data)
    engine.print_report(events_data)

    session_id   = events_data.get("session_id", "session")
    output_file  = f"findings_{session_id}.json"
    engine.save_findings(events_data, output_path=output_file)

    # VF scenario
#events_data = _load_json("test_scenarios/cardiac_arrest_vf.json")

# Bradycardia scenario
#events_data = _load_json("test_scenarios/bradycardia_nodal.json")

# Tachyarrhythmia scenario
#events_data = _load_json("test_scenarios/tachyarrhythmia_unstable.json")