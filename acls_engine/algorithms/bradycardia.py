"""
algorithms/bradycardia.py
AHA 2025 Adult Bradycardia With a Pulse Algorithm Processor

Implements all guard-based deviation rules from:
  aha_2025_bradycardia_pulse_v2.json

Rule IDs: R_BRADY_001 – R_BRADY_012

State machine summary:
  initial_assessment
    └─ check_cardiopulmonary_compromise
         ├─ NO compromise  → observe_and_treat_causes
         └─ COMPROMISE     → classify_block_for_treatment
              ├─ NODAL       → administer_atropine
              │    ├─ effective   → observe_and_treat_causes
              │    └─ ineffective → second_line_interventions
              └─ INFRANODAL / TRANSPLANT → second_line_interventions
                   └─ tcp_setup → tcp_titrating → tcp_maintaining
                        └─ transvenous_pacing (definitive)
"""

import logging
from typing import Optional

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)

# ── TIMING CONSTANTS (seconds) ────────────────────────────────────────────────
ATROPINE_MIN_INTERVAL_SEC = 180     # Minimum 3 min between doses
ATROPINE_MAX_INTERVAL_SEC = 300     # Maximum 5 min between doses
ATROPINE_MAX_TOTAL_MG     = 3.0     # Total dose ceiling
ATROPINE_DOSE_MG          = 1.0     # Standard dose per administration

GUIDELINE = "AHA 2025 Adult Bradycardia With a Pulse Algorithm"


class BradycardiaAlgorithm(BaseAlgorithm):
    """
    Processes a bradycardia-with-pulse session event stream and emits
    findings for every deviation from AHA 2025 bradycardia guidelines.
    """

    ALGORITHM_ID = "bradycardia_with_pulse"

    def __init__(self, rules: dict):
        super().__init__(rules)
        self._state = "initial_assessment"

        # ── Tracking ──────────────────────────────────────────────────────────
        self.recognition_time:     Optional[float] = None
        self.atropine_count:       int   = 0
        self.atropine_total_mg:    float = 0.0
        self.last_atropine_time:   Optional[float] = None

        self.tcp_initiated:        bool  = False
        self.tcp_elec_capture:     bool  = False
        self.tcp_mech_capture:     bool  = False
        self.tcp_init_time:        Optional[float] = None
        self.sedation_given:       bool  = False
        self.analgesia_given:      bool  = False

        self.compromise_assessed:  bool  = False
        self.hemodynamically_compromised: bool = False

    # ── EVENT DISPATCHER ──────────────────────────────────────────────────────
    def process_event(self, event: dict) -> None:
        etype = event["event_type"]
        t     = event["timestamp_sec"]
        data  = event.get("data", {})

        logger.info(f"  [BRADY] {etype} @ {t}s | state={self._state}")

        # ── Recognition ───────────────────────────────────────────────────────
        if etype == "bradycardia_recognized":
            self.recognition_time = t
            self._state = "initial_assessment"

        # ── Initial assessment ─────────────────────────────────────────────────
        elif etype == "vitals_recorded":
            for k in ["heart_rate_bpm", "rhythm_label", "av_block_type"]:
                if k in data:
                    self.context[k] = data[k]

        elif etype == "iv_access_established":
            self.context["iv_access"] = True

        elif etype == "twelve_lead_obtained":
            self.context["twelve_lead_obtained"] = True

        elif etype == "av_block_type_identified":
            av_type = data.get("av_block_type", "")
            self.context["av_block_type"] = av_type
            infranodal = av_type in [
                "mobitz_type_ii_second_degree_av_block",
                "third_degree_complete_av_block_wide_qrs",
                "complete_heart_block_wide_qrs",
            ]
            self.context["infranodal_block_confirmed"] = infranodal
            logger.info(
                f"  AV block type: {av_type} | infranodal={infranodal}"
            )

        elif etype == "heart_transplant_identified":
            self.context["heart_transplant_patient"] = True
            logger.info(f"  Heart transplant patient flagged at {t}s")

        elif etype == "reversible_cause_identified":
            cause = data.get("cause", "unknown")
            self.context.setdefault("reversible_causes_identified", []).append(cause)

        elif etype == "reversible_cause_treated":
            cause = data.get("cause", "unknown")
            self.context.setdefault("reversible_causes_treated", []).append(cause)

        # ── Compromise assessment → state routing ──────────────────────────────
        elif etype == "compromise_assessed":
            self.compromise_assessed = True
            signs = data.get("compromise_signs", {})
            self.context["compromise_signs"].update(signs)
            self.hemodynamically_compromised = any(signs.values())
            self.context["hemodynamically_compromised"] = self.hemodynamically_compromised

            if self.hemodynamically_compromised:
                self._state = "classify_block_for_treatment"
                logger.info(f"  Compromise PRESENT → classify_block_for_treatment")
            else:
                self._state = "observe_and_treat_causes"
                logger.info(f"  No compromise → observe_and_treat_causes")

        # ── Atropine ──────────────────────────────────────────────────────────
        elif etype == "atropine_given":
            self._process_atropine(event)

        # ── TCP preparation ───────────────────────────────────────────────────
        elif etype == "sedation_given":
            self.sedation_given = True
            self.context["tcp_sedation_given"] = True

        elif etype == "analgesia_given":
            self.analgesia_given = True
            self.context["tcp_analgesia_given"] = True

        elif etype == "tcp_initiated":
            self._process_tcp_initiated(event)

        # ── TCP titration ─────────────────────────────────────────────────────
        elif etype == "electrical_capture_confirmed":
            self.tcp_elec_capture = True
            self.context["tcp_electrical_capture"] = True
            logger.info(f"  TCP electrical capture confirmed at {t}s")

        elif etype == "mechanical_capture_confirmed":
            self.tcp_mech_capture = True
            self.context["tcp_mechanical_capture"] = True
            self._state = "tcp_maintaining"
            logger.info(f"  TCP mechanical capture confirmed at {t}s")

        elif etype == "electrical_capture_without_mechanical_check":
            # R_BRADY_006 — explicit event flagging deviation
            self._emit(
                rule_id="R_BRADY_006", severity="HIGH",
                message=(
                    "TCP electrical capture documented but mechanical capture "
                    "was not verified — electrical activity alone does not ensure perfusion"
                ),
                timestamp_sec=t,
                from_event="electrical_capture_confirmed",
                to_event="tcp_maintained_without_mechanical_check",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: After TCP electrical capture, ALWAYS verify mechanical "
                    "capture by palpating the central (femoral) pulse at the paced rate "
                    "or using point-of-care ultrasound. Electrical capture alone does not "
                    "guarantee haemodynamic perfusion."
                ),
            )

        # ── Chronotropic infusions (bridge to pacing) ──────────────────────────
        elif etype == "dopamine_infusion_started":
            self.context["dopamine_infusion_started"] = True
            if "dose_mcg_per_kg_per_min" in data:
                self.context["dopamine_dose_mcg_per_kg_per_min"] = data["dose_mcg_per_kg_per_min"]

        elif etype == "epinephrine_infusion_started":
            self.context["epinephrine_infusion_started"] = True
            if "dose_mcg_per_min" in data:
                self.context["epinephrine_dose_mcg_per_min"] = data["dose_mcg_per_min"]

        # ── Transvenous pacing (definitive) ───────────────────────────────────
        elif etype == "transvenous_pacing_initiated":
            self.context["transvenous_pacing_initiated"] = True
            self._state = "transvenous_pacing"

        # ── Expert consult ────────────────────────────────────────────────────
        elif etype == "expert_consulted":
            self.context["expert_consulted"] = True

        # ── Patient outcome ───────────────────────────────────────────────────
        elif etype == "patient_stabilized":
            self.context["patient_stabilized"] = True
            self._state = "observe_and_treat_causes"

        elif etype == "patient_deteriorated_to_arrest":
            self._state = "cardiac_arrest_pathway"

    # ── ATROPINE CHECKS ───────────────────────────────────────────────────────
    def _process_atropine(self, event: dict) -> None:
        t       = event["timestamp_sec"]
        data    = event.get("data", {})
        dose_mg = data.get("dose_mg", ATROPINE_DOSE_MG)

        # R_BRADY_001 — Atropine for infranodal block
        if self.context.get("infranodal_block_confirmed", False):
            self._emit(
                rule_id="R_BRADY_001", severity="HIGH",
                message=(
                    "Atropine administered for confirmed infranodal block "
                    "(Mobitz Type II or complete AV block with wide QRS) "
                    "— likely ineffective and potentially harmful"
                ),
                timestamp_sec=t,
                from_event="infranodal_block_confirmed", to_event="atropine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Atropine acts at the AV node via vagal blockade and is "
                    "INEFFECTIVE for infranodal blocks. It may paradoxically worsen or "
                    "precipitate asystole. Proceed directly to transcutaneous pacing."
                ),
            )

        # R_BRADY_002 — Atropine in heart transplant patient (CRITICAL)
        if self.context.get("heart_transplant_patient", False):
            self._emit(
                rule_id="R_BRADY_002", severity="CRITICAL",
                message=(
                    "Atropine given to heart transplant (denervated) patient "
                    "— cannot respond; risk of paradoxical worsening"
                ),
                timestamp_sec=t,
                from_event="heart_transplant_identified", to_event="atropine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA/ACC/HRS Class III — HARM: Transplanted hearts are denervated and "
                    "DO NOT respond to atropine. It has caused paradoxical bradycardia and "
                    "high-degree AV block. Use transcutaneous pacing, dopamine infusion "
                    "(5–20 mcg/kg/min), or epinephrine infusion (2–10 mcg/min) instead."
                ),
            )

        # R_BRADY_003 — Maximum total dose exceeded
        new_total = self.atropine_total_mg + dose_mg
        if new_total > ATROPINE_MAX_TOTAL_MG:
            self._emit(
                rule_id="R_BRADY_003", severity="HIGH",
                message=(
                    f"Atropine dose would bring cumulative total to {new_total:.1f} mg "
                    f"— exceeds maximum of {ATROPINE_MAX_TOTAL_MG} mg"
                ),
                timestamp_sec=t,
                from_event="atropine_given", to_event="atropine_given",
                actual_gap=new_total, expected=ATROPINE_MAX_TOTAL_MG,
                guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Maximum total atropine dose is 3 mg. If bradycardia "
                    "persists after 3 mg, escalate immediately to transcutaneous pacing."
                ),
            )

        # R_BRADY_004 — Repeat dose given too soon (< 3 min)
        if self.last_atropine_time is not None:
            interval = t - self.last_atropine_time
            if interval < ATROPINE_MIN_INTERVAL_SEC:
                self._emit(
                    rule_id="R_BRADY_004", severity="MEDIUM",
                    message=(
                        f"Atropine repeat dose given {interval:.0f}s after previous dose "
                        f"— minimum interval is {ATROPINE_MIN_INTERVAL_SEC}s (3 min)"
                    ),
                    timestamp_sec=t,
                    from_event="atropine_given", to_event="atropine_given",
                    actual_gap=interval, expected=ATROPINE_MIN_INTERVAL_SEC,
                    guideline=GUIDELINE,
                    recommendation=(
                        "AHA 2025: Repeat atropine every 3–5 minutes as needed. "
                        "Giving doses more frequently does not improve efficacy "
                        "and increases adverse effects (tachycardia, anticholinergic toxicity)."
                    ),
                )

        # ── Update tracking ──────────────────────────────────────────────────
        self.atropine_count    += 1
        self.atropine_total_mg += dose_mg
        self.last_atropine_time = t
        self.context["atropine_dose_count"]    = self.atropine_count
        self.context["atropine_total_dose_mg"] = self.atropine_total_mg
        self.context["atropine_last_given_ms"] = int(t * 1000)
        logger.info(
            f"  Atropine #{self.atropine_count} ({dose_mg} mg) at {t}s "
            f"| total = {self.atropine_total_mg} mg"
        )

    # ── TCP CHECKS ────────────────────────────────────────────────────────────
    def _process_tcp_initiated(self, event: dict) -> None:
        t = event["timestamp_sec"]

        # R_BRADY_005 — Sedation and/or analgesia missing before TCP
        if not self.sedation_given or not self.analgesia_given:
            missing = []
            if not self.sedation_given:  missing.append("sedation")
            if not self.analgesia_given: missing.append("analgesia")
            self._emit(
                rule_id="R_BRADY_005", severity="MEDIUM",
                message=(
                    f"Transcutaneous pacing initiated without documented "
                    f"{' and '.join(missing)} — TCP is painful for conscious patients"
                ),
                timestamp_sec=t,
                from_event="tcp_ordered", to_event="tcp_initiated",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: TCP is painful. Provide sedation (e.g., midazolam 1–2 mg IV) "
                    "and analgesia (e.g., fentanyl 25–50 mcg IV) to conscious patients "
                    "before or concurrently with pacing initiation."
                ),
            )

        # R_BRADY_007 — TCP before atropine attempt (when atropine is appropriate)
        is_nodal = (
            not self.context.get("infranodal_block_confirmed", False)
            and not self.context.get("heart_transplant_patient", False)
        )
        if is_nodal and self.atropine_count == 0 and self.hemodynamically_compromised:
            self._emit(
                rule_id="R_BRADY_007", severity="MEDIUM",
                message=(
                    "Transcutaneous pacing initiated without a prior atropine attempt "
                    "in nodal bradycardia — atropine is first-line when not contraindicated"
                ),
                timestamp_sec=t,
                from_event="bradycardia_with_compromise", to_event="tcp_initiated",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Atropine 1 mg IV bolus is first-line for symptomatic nodal "
                    "bradycardia. Prepare TCP simultaneously, but attempt atropine first "
                    "unless contraindicated (infranodal block or transplanted heart)."
                ),
            )

        # ── Update tracking ──────────────────────────────────────────────────
        self.tcp_initiated          = True
        self.tcp_init_time          = t
        self.context["tcp_initiated"] = True
        self._state                 = "tcp_titrating"
        logger.info(f"  TCP initiated at {t}s")

    # ── END-OF-SESSION CHECKS ─────────────────────────────────────────────────
    def end_of_session_checks(self) -> None:

        # R_BRADY_008 — IV access never established
        if not self.context.get("iv_access", False):
            self._emit(
                rule_id="R_BRADY_008", severity="HIGH",
                message="IV access not established during bradycardia management",
                timestamp_sec=0,
                from_event="bradycardia_recognized", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Establish IV access early to enable atropine administration "
                    "and chronotropic infusions."
                ),
            )

        # R_BRADY_009 — No 12-lead ECG obtained
        if not self.context.get("twelve_lead_obtained", False):
            self._emit(
                rule_id="R_BRADY_009", severity="LOW",
                message="No 12-lead ECG obtained during bradycardia assessment",
                timestamp_sec=0,
                from_event="bradycardia_recognized", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Obtain a 12-lead ECG to classify the AV block type "
                    "(nodal vs infranodal), identify ischaemia, and guide treatment."
                ),
            )

        # R_BRADY_010 — AV block type not classified before treatment
        if (
            self.context.get("av_block_type") is None
            and self.hemodynamically_compromised
        ):
            self._emit(
                rule_id="R_BRADY_010", severity="MEDIUM",
                message=(
                    "AV block type not classified before initiating treatment "
                    "for compromised bradycardia"
                ),
                timestamp_sec=0,
                from_event="compromise_assessed", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Classify block type (nodal vs infranodal) from the 12-lead ECG "
                    "before initiating atropine — infranodal blocks should go directly to TCP."
                ),
            )

        # R_BRADY_011 — TCP without achieving mechanical capture and no expert consult
        if (
            self.tcp_initiated
            and not self.tcp_mech_capture
            and not self.context.get("expert_consulted", False)
        ):
            self._emit(
                rule_id="R_BRADY_011", severity="HIGH",
                message=(
                    "Transcutaneous pacing attempted without successful mechanical capture "
                    "and no expert consultation was obtained"
                ),
                timestamp_sec=0,
                from_event="tcp_initiated", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: If TCP fails to achieve mechanical capture, urgently seek expert "
                    "consultation (cardiology) for transvenous pacing as definitive therapy."
                ),
            )

        # R_BRADY_012 — Compromise assessment never documented
        if not self.compromise_assessed and self.recognition_time is not None:
            self._emit(
                rule_id="R_BRADY_012", severity="MEDIUM",
                message="Haemodynamic compromise was never formally assessed during bradycardia management",
                timestamp_sec=0,
                from_event="bradycardia_recognized", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Assess for compromise signs — hypotension, acutely altered mental "
                    "status, signs of shock, ischaemic chest pain, acute heart failure. "
                    "This determines whether immediate intervention is required."
                ),
            )
