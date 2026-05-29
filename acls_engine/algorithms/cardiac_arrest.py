"""
algorithms/cardiac_arrest.py
AHA 2025 Adult Cardiac Arrest Algorithm Processor
Covers: VF | pVT | Asystole | PEA

Rule IDs: R_ARREST_001 – R_ARREST_012
"""

import logging
from typing import Optional

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)

# ── TIMING CONSTANTS (seconds) ─────────────────────────────────────────────────
CPR_INITIATION_LIMIT        = 30    # CPR within 30 s of arrest
RHYTHM_CHECK_INTERVAL       = 120   # Check rhythm every 2 min
RHYTHM_CHECK_GRACE          = 15    # +15 s grace
SHOCK_RESUME_LIMIT          = 10    # CPR resumes ≤ 10 s after shock
SHOCK_DELIVERY_LIMIT        = 35    # Shock ≤ 35 s of VF/pVT detection
EPI_FIRST_NONSHOCKABLE_MAX  = 180   # Epi within 3 min in PEA/asystole
EPI_REPEAT_MIN              = 180   # Repeat epi min 3 min
EPI_REPEAT_MAX              = 300   # Repeat epi max 5 min
REVERSIBLE_CAUSES_LIMIT     = 600   # Hs & Ts within 10 min

GUIDELINE      = "AHA 2025 Adult Cardiac Arrest Algorithm"
GUIDELINE_VF   = "AHA 2025 ACLS VF/pVT Algorithm"
GUIDELINE_EPI  = "AHA 2025 ACLS — Epinephrine Dosing"
GUIDELINE_HsTs = "AHA 2025 ACLS Algorithm — Hs and Ts"


class CardiacArrestAlgorithm(BaseAlgorithm):
    """
    Processes a cardiac arrest session event stream and emits findings
    for every deviation from AHA 2025 cardiac arrest guidelines.
    """

    ALGORITHM_ID = "cardiac_arrest"

    def __init__(self, rules: dict, sub_type: str = "unknown"):
        super().__init__(rules)
        self.sub_type = sub_type        # VF | pVT | PEA | asystole | unknown

        # ── Timing trackers ───────────────────────────────────────────────────
        self.arrest_time:           Optional[float] = None
        self.cpr_start_time:        Optional[float] = None
        self.last_rhythm_check:     Optional[float] = None
        self.last_shock_time:       Optional[float] = None
        self.last_epi_time:         Optional[float] = None
        self.vf_pvt_detected_time:  Optional[float] = None

        # ── Counters / flags ──────────────────────────────────────────────────
        self.epi_count:             int  = 0
        self.shock_count:           int  = 0
        self.amiodarone_given:      bool = False
        self.lidocaine_given:       bool = False
        self.first_shock_delivered: bool = False
        self.current_rhythm:        Optional[str] = None

    # ── EVENT DISPATCHER ──────────────────────────────────────────────────────
    def process_event(self, event: dict) -> None:
        etype = event["event_type"]
        logger.info(
            f"  [ARREST] {etype} @ {event['timestamp_sec']}s"
            f" | rhythm={self.current_rhythm} | state={self._state}"
        )

        _dispatch = {
            "arrest_recognized":            self._on_arrest_recognized,
            "cpr_initiated":                self._on_cpr_initiated,
            "vf_detected":                  self._on_vf_pvt_detected,
            "pvt_detected":                 self._on_vf_pvt_detected,
            "pea_detected":                 self._on_pea_asystole_detected,
            "asystole_detected":            self._on_pea_asystole_detected,
            "rhythm_check":                 self._on_rhythm_check,
            "shock_delivered":              self._on_shock_delivered,
            "cpr_resumed":                  self._on_cpr_resumed,
            "epinephrine_given":            self._on_epinephrine_given,
            "amiodarone_given":             self._on_amiodarone_given,
            "lidocaine_given":              self._on_lidocaine_given,
            "reversible_causes_discussed":  self._on_reversible_causes,
            "rosc_achieved": lambda e: (
                setattr(self, "_state", "rosc"),
                logger.info(f"  ROSC achieved at {e['timestamp_sec']}s")
            ),
        }

        fn = _dispatch.get(etype)
        if fn:
            fn(event)

    # ── INDIVIDUAL EVENT HANDLERS ─────────────────────────────────────────────

    def _on_arrest_recognized(self, event: dict) -> None:
        self.arrest_time = event["timestamp_sec"]
        self._state = "arrest_recognized"
        logger.info(f"  Arrest recognized at {self.arrest_time}s")

    def _on_cpr_initiated(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is None:
            self.arrest_time = t          # fallback if recognition event missing

        delay = t - self.arrest_time
        self.cpr_start_time = t

        if delay > CPR_INITIATION_LIMIT:
            self._emit(
                rule_id="R_ARREST_001", severity="CRITICAL",
                message=f"CPR delayed {delay:.0f}s after arrest recognition (limit {CPR_INITIATION_LIMIT}s)",
                timestamp_sec=t, from_event="arrest_recognized", to_event="cpr_initiated",
                actual_gap=delay, expected=CPR_INITIATION_LIMIT,
                guideline=GUIDELINE,
                recommendation="Begin chest compressions immediately — within 30 s of arrest recognition.",
            )

        self._state = "cpr_active"

    def _on_vf_pvt_detected(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.current_rhythm = event["event_type"].replace("_detected", "").upper()
        # Normalize "PVT" to "pVT"
        if self.current_rhythm == "PVT":
            self.current_rhythm = "pVT"
        self.vf_pvt_detected_time = t
        logger.info(f"  {self.current_rhythm} detected at {t}s")

    def _on_pea_asystole_detected(self, event: dict) -> None:
        self.current_rhythm = event["event_type"].replace("_detected", "")
        logger.info(f"  {self.current_rhythm} detected at {event['timestamp_sec']}s")

    def _on_rhythm_check(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.last_rhythm_check is not None:
            interval = t - self.last_rhythm_check
            if interval > RHYTHM_CHECK_INTERVAL + RHYTHM_CHECK_GRACE:
                self._emit(
                    rule_id="R_ARREST_005", severity="MEDIUM",
                    message=(
                        f"Rhythm check interval {interval:.0f}s "
                        f"exceeds {RHYTHM_CHECK_INTERVAL}s (+{RHYTHM_CHECK_GRACE}s grace)"
                    ),
                    timestamp_sec=t,
                    from_event="last_rhythm_check", to_event="rhythm_check",
                    actual_gap=interval, expected=RHYTHM_CHECK_INTERVAL,
                    guideline=GUIDELINE,
                    recommendation="Perform rhythm checks every 2 minutes as part of the CPR cycle.",
                )
        self.last_rhythm_check = t
        self._state = "rhythm_check"

    def _on_shock_delivered(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.shock_count += 1

        # ── R_ARREST_010: Inappropriate shock in non-shockable rhythm ──────────
        if self.current_rhythm in ["PEA", "pea", "asystole"]:
            self._emit(
                rule_id="R_ARREST_010", severity="CRITICAL",
                message=(
                    f"Inappropriate shock delivered during {self.current_rhythm} "
                    f"— non-shockable rhythm"
                ),
                timestamp_sec=t,
                from_event="rhythm_recognized", to_event="shock_delivered",
                actual_gap=None, expected=None,
                guideline="AHA 2025 ACLS — Shock only for VF/pVT",
                recommendation=(
                    "Defibrillation is ONLY indicated for VF and pulseless VT. "
                    "Shocking PEA or asystole causes harm with no benefit."
                ),
            )

        # ── R_ARREST_002: Shock delay after VF/pVT ────────────────────────────
        if self.vf_pvt_detected_time is not None:
            delay = t - self.vf_pvt_detected_time
            if delay > SHOCK_DELIVERY_LIMIT:
                self._emit(
                    rule_id="R_ARREST_002", severity="CRITICAL",
                    message=(
                        f"Shock delayed {delay:.0f}s after VF/pVT detection "
                        f"(limit {SHOCK_DELIVERY_LIMIT}s)"
                    ),
                    timestamp_sec=t,
                    from_event="vf_pvt_detected", to_event="shock_delivered",
                    actual_gap=delay, expected=SHOCK_DELIVERY_LIMIT,
                    guideline=GUIDELINE_VF,
                    recommendation=(
                        "Charge the defibrillator during ongoing CPR and deliver the shock "
                        "as soon as it is charged. Each minute of delay reduces survival 7–10%."
                    ),
                )
            self.vf_pvt_detected_time = None

        self.last_shock_time = t
        self.first_shock_delivered = True
        self._state = "shock_delivered"

    def _on_cpr_resumed(self, event: dict) -> None:
        t = event["timestamp_sec"]

        # ── R_ARREST_003: CPR not resumed promptly after shock ─────────────────
        if self.last_shock_time is not None:
            delay = t - self.last_shock_time
            if delay > SHOCK_RESUME_LIMIT:
                self._emit(
                    rule_id="R_ARREST_003", severity="CRITICAL",
                    message=(
                        f"CPR not resumed within {SHOCK_RESUME_LIMIT}s after shock "
                        f"— actual delay: {delay:.0f}s"
                    ),
                    timestamp_sec=t,
                    from_event="shock_delivered", to_event="cpr_resumed",
                    actual_gap=delay, expected=SHOCK_RESUME_LIMIT,
                    guideline=GUIDELINE,
                    recommendation="Resume compressions immediately after shock delivery — no hands-off time for rhythm check.",
                )
            self.last_shock_time = None

        self._state = "cpr_active"

    def _on_epinephrine_given(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.epi_count += 1

        if self.epi_count == 1:
            self._check_first_epi(t)
        else:
            self._check_repeat_epi(t)

        self.last_epi_time = t
        logger.info(f"  Epi #{self.epi_count} at {t}s | rhythm={self.current_rhythm}")

    def _check_first_epi(self, t: float) -> None:
        rhythm = self.current_rhythm

        if rhythm in ["VF", "pVT"]:
            # ── R_ARREST_008: Epi before any shock in shockable rhythm ─────────
            if not self.first_shock_delivered:
                self._emit(
                    rule_id="R_ARREST_008", severity="HIGH",
                    message=(
                        "Epinephrine given before any defibrillation attempt "
                        "in shockable rhythm (VF/pVT)"
                    ),
                    timestamp_sec=t,
                    from_event="vf_pvt_detected", to_event="epinephrine_given",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 ACLS — Shockable Rhythm Epi Timing",
                    recommendation=(
                        "In VF/pVT: prioritize early defibrillation. "
                        "Administer epinephrine after the 2nd or 3rd shock while CPR is ongoing."
                    ),
                )

        elif rhythm in ["PEA", "pea", "asystole"]:
            # ── R_ARREST_009: Epi delayed in non-shockable rhythm ─────────────
            if self.arrest_time is not None:
                delay = t - self.arrest_time
                if delay > EPI_FIRST_NONSHOCKABLE_MAX:
                    self._emit(
                        rule_id="R_ARREST_009", severity="HIGH",
                        message=(
                            f"First epinephrine delayed {delay:.0f}s in non-shockable rhythm "
                            f"— AHA 2025 recommends within {EPI_FIRST_NONSHOCKABLE_MAX}s"
                        ),
                        timestamp_sec=t,
                        from_event="arrest_recognized", to_event="epinephrine_given",
                        actual_gap=delay, expected=EPI_FIRST_NONSHOCKABLE_MAX,
                        guideline="AHA 2025 ACLS — Non-Shockable Rhythm Epi Timing",
                        recommendation=(
                            "For PEA/asystole: administer epinephrine 1 mg IV/IO as soon as "
                            "vascular access is established — target within 3 minutes of arrest."
                        ),
                    )
        else:
            # Generic fallback (rhythm not yet classified)
            if self.arrest_time is not None:
                delay = t - self.arrest_time
                if delay > 600:
                    self._emit(
                        rule_id="R_ARREST_004", severity="HIGH",
                        message=f"First epinephrine delayed {delay:.0f}s after arrest (limit 600s)",
                        timestamp_sec=t,
                        from_event="arrest_recognized", to_event="epinephrine_given",
                        actual_gap=delay, expected=600,
                        guideline=GUIDELINE,
                        recommendation="Administer epinephrine 1 mg IV/IO — as early as possible for non-shockable, or after shock #2–3 for shockable.",
                    )

    def _check_repeat_epi(self, t: float) -> None:
        if self.last_epi_time is None:
            return
        interval = t - self.last_epi_time

        # ── R_ARREST_006: Too soon ──────────────────────────────────────────────
        if interval < EPI_REPEAT_MIN:
            self._emit(
                rule_id="R_ARREST_006", severity="HIGH",
                message=(
                    f"Epinephrine repeat dose given too soon — "
                    f"interval {interval:.0f}s (minimum {EPI_REPEAT_MIN}s)"
                ),
                timestamp_sec=t,
                from_event="epinephrine_given", to_event="epinephrine_given",
                actual_gap=interval, expected=EPI_REPEAT_MIN,
                guideline=GUIDELINE_EPI,
                recommendation="Repeat epinephrine every 3–5 minutes. More frequent dosing does not improve outcomes.",
            )

        # ── R_ARREST_007: Too late ──────────────────────────────────────────────
        elif interval > EPI_REPEAT_MAX:
            self._emit(
                rule_id="R_ARREST_007", severity="HIGH",
                message=(
                    f"Epinephrine repeat interval {interval:.0f}s "
                    f"exceeds maximum {EPI_REPEAT_MAX}s"
                ),
                timestamp_sec=t,
                from_event="epinephrine_given", to_event="epinephrine_given",
                actual_gap=interval, expected=EPI_REPEAT_MAX,
                guideline=GUIDELINE_EPI,
                recommendation="Repeat epinephrine every 3–5 minutes. Intervals over 5 minutes represent under-treatment.",
            )

    def _on_amiodarone_given(self, event: dict) -> None:
        self.amiodarone_given = True
        logger.info(f"  Amiodarone given at {event['timestamp_sec']}s")

    def _on_lidocaine_given(self, event: dict) -> None:
        self.lidocaine_given = True
        logger.info(f"  Lidocaine given at {event['timestamp_sec']}s")

    def _on_reversible_causes(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > REVERSIBLE_CAUSES_LIMIT:
                self._emit(
                    rule_id="R_ARREST_012", severity="MEDIUM",
                    message=(
                        f"Reversible causes (Hs & Ts) discussed {delay:.0f}s after arrest "
                        f"— limit {REVERSIBLE_CAUSES_LIMIT}s"
                    ),
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="reversible_causes_discussed",
                    actual_gap=delay, expected=REVERSIBLE_CAUSES_LIMIT,
                    guideline=GUIDELINE_HsTs,
                    recommendation=(
                        "Identify and address reversible causes within the first 10 minutes: "
                        "Hypoxia, Hypovolemia, H+ (acidosis), Hypo/hyperkalemia, Hypothermia, "
                        "Toxins, Tamponade, Tension pneumothorax, Thrombosis (PE/MI)."
                    ),
                )

    # ── END-OF-SESSION CHECKS ─────────────────────────────────────────────────
    def end_of_session_checks(self) -> None:
        # ── R_ARREST_011: No antiarrhythmic after 3rd shock ────────────────────
        if self.shock_count >= 3 and not self.amiodarone_given and not self.lidocaine_given:
            self._emit(
                rule_id="R_ARREST_011", severity="HIGH",
                message=(
                    "Neither amiodarone nor lidocaine given after ≥3 shocks "
                    "in refractory shockable rhythm"
                ),
                timestamp_sec=0,
                from_event="third_shock_delivered", to_event="antiarrhythmic_given",
                actual_gap=None, expected=None,
                guideline="AHA 2025 ACLS Algorithm — Antiarrhythmic After 3rd Shock",
                recommendation=(
                    "For VF/pVT refractory to ≥3 shocks: administer amiodarone 300 mg IV/IO "
                    "(repeat 150 mg once) OR lidocaine 1–1.5 mg/kg IV/IO as an alternative."
                ),
            )
