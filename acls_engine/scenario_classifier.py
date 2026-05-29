"""
scenario_classifier.py — ACLS Scenario Classifier

Reads the full event stream for a session and returns:
  (algorithm_type, sub_type)

algorithm_type : 'cardiac_arrest' | 'tachyarrhythmia_with_pulse'
               | 'bradycardia_with_pulse' | 'unknown'
sub_type       : 'VF' | 'pVT' | 'PEA' | 'asystole' | None
"""

from typing import Optional, Tuple


class ScenarioClassifier:
    """
    One-pass classifier over the event stream.
    Determines which AHA 2025 algorithm the session belongs to.
    """

    # Events that unambiguously signal cardiac arrest
    ARREST_SIGNALS = {
        "arrest_recognized",
        "cpr_initiated",
        "vf_detected",
        "pvt_detected",
        "pea_detected",
        "asystole_detected",
    }

    # Rhythm-detection events → cardiac arrest sub-type
    RHYTHM_TO_SUBTYPE = {
        "vf_detected":       "VF",
        "pvt_detected":      "pVT",
        "pea_detected":      "PEA",
        "asystole_detected": "asystole",
    }

    @classmethod
    def classify(cls, events: list) -> Tuple[str, Optional[str]]:
        """
        Classify the scenario from a sorted list of event dicts.

        Priority order:
          1. Tachyarrhythmia with pulse  (tachyarrhythmia_recognized)
          2. Bradycardia with pulse       (bradycardia_recognized)
          3. Cardiac arrest               (any ARREST_SIGNAL present)
          4. Unknown

        Returns:
            (algorithm_type, sub_type)
        """
        event_types = {e["event_type"] for e in events}

        # ── Tachyarrhythmia ─────────────────────────────────────────────────
        if "tachyarrhythmia_recognized" in event_types:
            return "tachyarrhythmia_with_pulse", None

        # ── Bradycardia ──────────────────────────────────────────────────────
        if "bradycardia_recognized" in event_types:
            return "bradycardia_with_pulse", None

        # ── Cardiac arrest ───────────────────────────────────────────────────
        if event_types & cls.ARREST_SIGNALS:
            sub_type = "unknown"
            # Walk events in time order to find the first rhythm event
            for event in sorted(events, key=lambda e: e["timestamp_sec"]):
                if event["event_type"] in cls.RHYTHM_TO_SUBTYPE:
                    sub_type = cls.RHYTHM_TO_SUBTYPE[event["event_type"]]
                    break
            return "cardiac_arrest", sub_type

        return "unknown", None
