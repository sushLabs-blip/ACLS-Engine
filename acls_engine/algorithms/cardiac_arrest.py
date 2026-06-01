"""
algorithms/cardiac_arrest.py
AHA 2025 Adult Cardiac Arrest Algorithm Processor
Covers: VF | pVT | Asystole | PEA

Rule IDs: R_ARREST_001 – R_ARREST_016
"""

import logging
from typing import Optional

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)

# ── TIMING CONSTANTS (seconds) ─────────────────────────────────────────────────
CPR_INITIATION_LIMIT        = 30
RHYTHM_CHECK_INTERVAL       = 120
RHYTHM_CHECK_GRACE          = 15
SHOCK_RESUME_LIMIT          = 10
SHOCK_DELIVERY_LIMIT        = 35
EPI_FIRST_NONSHOCKABLE_MAX  = 180
EPI_REPEAT_MIN              = 180
EPI_REPEAT_MAX              = 300
REVERSIBLE_CAUSES_LIMIT     = 600
POST_ROSC_ECG_LIMIT         = 600
POST_ROSC_TEMP_LIMIT        = 3600
POST_ROSC_BP_TARGET         = 65

# Primary ABCD
PRIMARY_AIRWAY_LIMIT        = 30    # Basic airway within 30s
BAG_MASK_LIMIT              = 30    # Bag mask ventilation within 30s
DEFIBRILLATOR_ATTACH_LIMIT  = 60    # Defibrillator attached within 60s

# Secondary ABCD
IV_ACCESS_LIMIT             = 120   # IV access within 120s
ECG_LEADS_LIMIT             = 60    # ECG leads within 60s
ADVANCED_AIRWAY_LIMIT       = 600   # Advanced airway within 10 min
TUBE_CONFIRM_LIMIT          = 30    # Tube placement confirmed within 30s of placement

# CPR Quality
CPR_RATE_MIN        = 100   # bpm
CPR_RATE_MAX        = 120   # bpm
CPR_DEPTH_MIN_CM    = 5.0   # cm
CPR_DEPTH_MAX_CM    = 6.0   # cm
CCF_TARGET          = 0.80  # 80% chest compression fraction
HANDS_OFF_MAX       = 0.20  # max 20% hands off time

GUIDELINE      = "AHA 2025 Adult Cardiac Arrest Algorithm"
GUIDELINE_VF   = "AHA 2025 ACLS VF/pVT Algorithm"
GUIDELINE_EPI  = "AHA 2025 ACLS — Epinephrine Dosing"
GUIDELINE_HsTs = "AHA 2025 ACLS Algorithm — Hs and Ts"
GUIDELINE_ROSC = "AHA 2025 Post Cardiac Arrest Care"


class CardiacArrestAlgorithm(BaseAlgorithm):

    ALGORITHM_ID = "cardiac_arrest"

    def __init__(self, rules: dict, sub_type: str = "unknown"):
        super().__init__(rules)
        self.sub_type = sub_type

        # Timing trackers
        self.arrest_time:           Optional[float] = None
        self.cpr_start_time:        Optional[float] = None
        self.last_rhythm_check:     Optional[float] = None
        self.last_shock_time:       Optional[float] = None
        self.last_epi_time:         Optional[float] = None
        self.vf_pvt_detected_time:  Optional[float] = None
        self.rosc_time:             Optional[float] = None

        # Counters and flags
        self.epi_count:             int  = 0
        self.shock_count:           int  = 0
        self.amiodarone_given:      bool = False
        self.lidocaine_given:       bool = False
        self.first_shock_delivered: bool = False
        self.ecg_obtained:          bool = False
        self.temp_management_done:  bool = False
        self.current_rhythm:        Optional[str] = None

        # ABCD Survey flags
        self.basic_airway_done:     bool = False
        self.bag_mask_done:         bool = False
        self.defibrillator_attached: bool = False
        self.advanced_airway_done:  bool = False
        self.advanced_airway_time:  Optional[float] = None
        self.tube_confirmed:        bool = False
        self.iv_access_done:        bool = False

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
            "rosc_achieved":                self._on_rosc_achieved,
            "post_rosc_ecg_obtained":       self._on_post_rosc_ecg,
            "temp_management_initiated":    self._on_temp_management,
            "bp_recorded":                  self._on_bp_recorded,
            "basic_airway_opened":           self._on_basic_airway_opened,
            "bag_mask_ventilation_started":  self._on_bag_mask_started,
            "defibrillator_attached":        self._on_defibrillator_attached,
            "advanced_airway_placed":        self._on_advanced_airway_placed,
            "tube_placement_confirmed":      self._on_tube_placement_confirmed,
            "iv_access_established":         self._on_iv_access_established,
            "ecg_leads_attached":            self._on_ecg_leads_attached,
            "cpr_quality_measured":          self._on_cpr_quality_measured,
        }

        fn = _dispatch.get(etype)
        if fn:
            fn(event)

    # ── EVENT HANDLERS ────────────────────────────────────────────────────────

    def _on_arrest_recognized(self, event: dict) -> None:
        self.arrest_time = event["timestamp_sec"]
        self._state = "arrest_recognized"
        logger.info(f"  Arrest recognized at {self.arrest_time}s")

    def _on_cpr_initiated(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is None:
            self.arrest_time = t

        delay = t - self.arrest_time
        self.cpr_start_time = t

        if delay > CPR_INITIATION_LIMIT:
            self._emit(
                rule_id="R_ARREST_001", severity="CRITICAL",
                message=f"CPR delayed {delay:.0f}s after arrest recognition (limit {CPR_INITIATION_LIMIT}s)",
                timestamp_sec=t,
                from_event="arrest_recognized", to_event="cpr_initiated",
                actual_gap=delay, expected=CPR_INITIATION_LIMIT,
                guideline=GUIDELINE,
                recommendation="Begin chest compressions immediately — within 30s of arrest recognition.",
            )

        self._state = "cpr_active"

    def _on_vf_pvt_detected(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.current_rhythm = event["event_type"].replace("_detected", "").upper()
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
                    message=f"Rhythm check interval {interval:.0f}s exceeds {RHYTHM_CHECK_INTERVAL}s (+{RHYTHM_CHECK_GRACE}s grace)",
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

        # Inappropriate shock in non-shockable rhythm
        if self.current_rhythm in ["PEA", "pea", "asystole"]:
            self._emit(
                rule_id="R_ARREST_010", severity="CRITICAL",
                message=f"Inappropriate shock delivered during {self.current_rhythm} — non-shockable rhythm",
                timestamp_sec=t,
                from_event="rhythm_recognized", to_event="shock_delivered",
                actual_gap=None, expected=None,
                guideline="AHA 2025 ACLS — Shock only for VF/pVT",
                recommendation="Defibrillation is ONLY indicated for VF and pulseless VT. Shocking PEA or asystole causes harm with no benefit.",
            )

        # Shock delay after VF/pVT
        if self.vf_pvt_detected_time is not None:
            delay = t - self.vf_pvt_detected_time
            if delay > SHOCK_DELIVERY_LIMIT:
                self._emit(
                    rule_id="R_ARREST_002", severity="CRITICAL",
                    message=f"Shock delayed {delay:.0f}s after VF/pVT detection (limit {SHOCK_DELIVERY_LIMIT}s)",
                    timestamp_sec=t,
                    from_event="vf_pvt_detected", to_event="shock_delivered",
                    actual_gap=delay, expected=SHOCK_DELIVERY_LIMIT,
                    guideline=GUIDELINE_VF,
                    recommendation="Charge the defibrillator during ongoing CPR and deliver shock as soon as charged. Each minute of delay reduces survival 7-10%.",
                )
            self.vf_pvt_detected_time = None

        self.last_shock_time = t
        self.first_shock_delivered = True
        self._state = "shock_delivered"

    def _on_cpr_resumed(self, event: dict) -> None:
        t = event["timestamp_sec"]

        if self.last_shock_time is not None:
            delay = t - self.last_shock_time
            if delay > SHOCK_RESUME_LIMIT:
                self._emit(
                    rule_id="R_ARREST_003", severity="CRITICAL",
                    message=f"CPR not resumed within {SHOCK_RESUME_LIMIT}s after shock — actual delay: {delay:.0f}s",
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
            if not self.first_shock_delivered:
                self._emit(
                    rule_id="R_ARREST_008", severity="HIGH",
                    message="Epinephrine given before any defibrillation attempt in shockable rhythm (VF/pVT)",
                    timestamp_sec=t,
                    from_event="vf_pvt_detected", to_event="epinephrine_given",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 ACLS — Shockable Rhythm Epi Timing",
                    recommendation="In VF/pVT: prioritize early defibrillation. Administer epinephrine after the 2nd or 3rd shock while CPR is ongoing.",
                )

        elif rhythm in ["PEA", "pea", "asystole"]:
            if self.arrest_time is not None:
                delay = t - self.arrest_time
                if delay > EPI_FIRST_NONSHOCKABLE_MAX:
                    self._emit(
                        rule_id="R_ARREST_009", severity="HIGH",
                        message=f"First epinephrine delayed {delay:.0f}s in non-shockable rhythm — AHA 2025 recommends within {EPI_FIRST_NONSHOCKABLE_MAX}s",
                        timestamp_sec=t,
                        from_event="arrest_recognized", to_event="epinephrine_given",
                        actual_gap=delay, expected=EPI_FIRST_NONSHOCKABLE_MAX,
                        guideline="AHA 2025 ACLS — Non-Shockable Rhythm Epi Timing",
                        recommendation="For PEA/asystole: administer epinephrine 1mg IV/IO as soon as vascular access is established — target within 3 minutes of arrest.",
                    )
        else:
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
                        recommendation="Administer epinephrine 1mg IV/IO — as early as possible for non-shockable, or after shock 2-3 for shockable.",
                    )

    def _check_repeat_epi(self, t: float) -> None:
        if self.last_epi_time is None:
            return
        interval = t - self.last_epi_time

        if interval < EPI_REPEAT_MIN:
            self._emit(
                rule_id="R_ARREST_006", severity="HIGH",
                message=f"Epinephrine repeat dose given too soon — interval {interval:.0f}s (minimum {EPI_REPEAT_MIN}s)",
                timestamp_sec=t,
                from_event="epinephrine_given", to_event="epinephrine_given",
                actual_gap=interval, expected=EPI_REPEAT_MIN,
                guideline=GUIDELINE_EPI,
                recommendation="Repeat epinephrine every 3-5 minutes. More frequent dosing does not improve outcomes.",
            )
        elif interval > EPI_REPEAT_MAX:
            self._emit(
                rule_id="R_ARREST_007", severity="HIGH",
                message=f"Epinephrine repeat interval {interval:.0f}s exceeds maximum {EPI_REPEAT_MAX}s",
                timestamp_sec=t,
                from_event="epinephrine_given", to_event="epinephrine_given",
                actual_gap=interval, expected=EPI_REPEAT_MAX,
                guideline=GUIDELINE_EPI,
                recommendation="Repeat epinephrine every 3-5 minutes. Intervals over 5 minutes represent under-treatment.",
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
                    message=f"Reversible causes (Hs & Ts) discussed {delay:.0f}s after arrest — limit {REVERSIBLE_CAUSES_LIMIT}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="reversible_causes_discussed",
                    actual_gap=delay, expected=REVERSIBLE_CAUSES_LIMIT,
                    guideline=GUIDELINE_HsTs,
                    recommendation="Identify and address reversible causes within the first 10 minutes: Hypoxia, Hypovolemia, H+ acidosis, Hypo/hyperkalemia, Hypothermia, Toxins, Tamponade, Tension pneumothorax, Thrombosis PE/MI.",
                )

    # ── PRIMARY ABCD SURVEY HANDLERS ─────────────────────────────────────────

    def _on_basic_airway_opened(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > PRIMARY_AIRWAY_LIMIT:
                self._emit(
                    rule_id="R_ARREST_018", severity="HIGH",
                    message=f"Basic airway not opened within {PRIMARY_AIRWAY_LIMIT}s — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="basic_airway_opened",
                    actual_gap=delay, expected=PRIMARY_AIRWAY_LIMIT,
                    guideline="AHA 2025 Primary ABCD Survey — Airway",
                    recommendation="Open airway immediately using head-tilt chin-lift or jaw thrust within 30s of arrest recognition.",
                )
        self.basic_airway_done = True
        logger.info(f"  Basic airway opened at {t}s")

    def _on_bag_mask_started(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > BAG_MASK_LIMIT:
                self._emit(
                    rule_id="R_ARREST_019", severity="HIGH",
                    message=f"Bag mask ventilation not started within {BAG_MASK_LIMIT}s — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="bag_mask_ventilation_started",
                    actual_gap=delay, expected=BAG_MASK_LIMIT,
                    guideline="AHA 2025 Primary ABCD Survey — Breathing",
                    recommendation="Begin bag mask ventilation immediately after airway is opened. Use 30:2 ratio until advanced airway placed.",
                )
        self.bag_mask_done = True
        logger.info(f"  Bag mask ventilation started at {t}s")

    def _on_defibrillator_attached(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > DEFIBRILLATOR_ATTACH_LIMIT:
                self._emit(
                    rule_id="R_ARREST_020", severity="HIGH",
                    message=f"Defibrillator not attached within {DEFIBRILLATOR_ATTACH_LIMIT}s — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="defibrillator_attached",
                    actual_gap=delay, expected=DEFIBRILLATOR_ATTACH_LIMIT,
                    guideline="AHA 2025 Primary ABCD Survey — Defibrillation",
                    recommendation="Attach monitor/defibrillator as soon as available — within 60s of arrest recognition.",
                )
        self.defibrillator_attached = True
        logger.info(f"  Defibrillator attached at {t}s")

    # ── SECONDARY ABCD SURVEY HANDLERS ───────────────────────────────────────

    def _on_advanced_airway_placed(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.advanced_airway_time = t
        self.advanced_airway_done = True
        logger.info(f"  Advanced airway placed at {t}s")

    def _on_tube_placement_confirmed(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.advanced_airway_time is not None:
            delay = t - self.advanced_airway_time
            if delay > TUBE_CONFIRM_LIMIT:
                self._emit(
                    rule_id="R_ARREST_021", severity="HIGH",
                    message=f"Tube placement not confirmed within {TUBE_CONFIRM_LIMIT}s of placement — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="advanced_airway_placed", to_event="tube_placement_confirmed",
                    actual_gap=delay, expected=TUBE_CONFIRM_LIMIT,
                    guideline="AHA 2025 Secondary ABCD Survey — Airway",
                    recommendation="Confirm ET tube placement immediately using waveform capnography after intubation.",
                )
        self.tube_confirmed = True
        logger.info(f"  Tube placement confirmed at {t}s")

    def _on_iv_access_established(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > IV_ACCESS_LIMIT:
                self._emit(
                    rule_id="R_ARREST_022", severity="MODERATE",
                    message=f"IV access not established within {IV_ACCESS_LIMIT}s — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="iv_access_established",
                    actual_gap=delay, expected=IV_ACCESS_LIMIT,
                    guideline="AHA 2025 Secondary ABCD Survey — Circulation",
                    recommendation="Establish IV or IO access within 2 minutes of arrest to enable drug administration.",
                )
        self.iv_access_done = True
        logger.info(f"  IV access established at {t}s")

    def _on_ecg_leads_attached(self, event: dict) -> None:
        t = event["timestamp_sec"]
        if self.arrest_time is not None:
            delay = t - self.arrest_time
            if delay > ECG_LEADS_LIMIT:
                self._emit(
                    rule_id="R_ARREST_023", severity="MODERATE",
                    message=f"ECG leads not attached within {ECG_LEADS_LIMIT}s — actual {delay:.0f}s",
                    timestamp_sec=t,
                    from_event="arrest_recognized", to_event="ecg_leads_attached",
                    actual_gap=delay, expected=ECG_LEADS_LIMIT,
                    guideline="AHA 2025 Secondary ABCD Survey — Circulation",
                    recommendation="Attach ECG leads within 60s to identify cardiac rhythm for ACLS algorithm selection.",
                )
        logger.info(f"  ECG leads attached at {t}s")# ── CPR QUALITY HANDLER ───────────────────────────────────────────────────

    def _on_cpr_quality_measured(self, event: dict) -> None:
        t    = event["timestamp_sec"]
        data = event.get("value", {})

        rate  = data.get("compression_rate_bpm")
        depth = data.get("compression_depth_cm")
        recoil = data.get("chest_recoil")
        hands_off = data.get("hands_off_ratio")

        # Compression rate
        if rate is not None:
            if rate < CPR_RATE_MIN:
                self._emit(
                    rule_id="R_ARREST_029", severity="HIGH",
                    message=f"Compression rate too slow — {rate} bpm (target {CPR_RATE_MIN}-{CPR_RATE_MAX} bpm)",
                    timestamp_sec=t,
                    from_event="cpr_active", to_event="cpr_quality_measured",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 High Quality CPR",
                    recommendation="Push fast — maintain compression rate of 100-120 per minute.",
                )
            elif rate > CPR_RATE_MAX:
                self._emit(
                    rule_id="R_ARREST_030", severity="HIGH",
                    message=f"Compression rate too fast — {rate} bpm (target {CPR_RATE_MIN}-{CPR_RATE_MAX} bpm)",
                    timestamp_sec=t,
                    from_event="cpr_active", to_event="cpr_quality_measured",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 High Quality CPR",
                    recommendation="Slow down slightly — compression rate above 120 bpm reduces filling time.",
                )

        # Compression depth
        if depth is not None:
            if depth < CPR_DEPTH_MIN_CM:
                self._emit(
                    rule_id="R_ARREST_031", severity="HIGH",
                    message=f"Compression depth too shallow — {depth} cm (minimum {CPR_DEPTH_MIN_CM} cm)",
                    timestamp_sec=t,
                    from_event="cpr_active", to_event="cpr_quality_measured",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 High Quality CPR",
                    recommendation="Push harder — compress at least 5 cm (2 inches). Inadequate depth reduces coronary perfusion pressure.",
                )
            elif depth > CPR_DEPTH_MAX_CM:
                self._emit(
                    rule_id="R_ARREST_032", severity="MODERATE",
                    message=f"Compression depth too deep — {depth} cm (maximum {CPR_DEPTH_MAX_CM} cm)",
                    timestamp_sec=t,
                    from_event="cpr_active", to_event="cpr_quality_measured",
                    actual_gap=None, expected=None,
                    guideline="AHA 2025 High Quality CPR",
                    recommendation="Reduce compression depth slightly — exceeding 6 cm increases risk of injury.",
                )

        # Chest recoil
        if recoil is False:
            self._emit(
                rule_id="R_ARREST_033", severity="HIGH",
                message="Incomplete chest recoil detected — leaning on chest between compressions",
                timestamp_sec=t,
                from_event="cpr_active", to_event="cpr_quality_measured",
                actual_gap=None, expected=None,
                guideline="AHA 2025 High Quality CPR",
                recommendation="Allow complete chest recoil after each compression. Leaning reduces venous return and cardiac output.",
            )

        # Hands off ratio
        if hands_off is not None and hands_off > HANDS_OFF_MAX:
            self._emit(
                rule_id="R_ARREST_034", severity="HIGH",
                message=f"Hands off ratio {hands_off*100:.0f}% exceeds limit {HANDS_OFF_MAX*100:.0f}%",
                timestamp_sec=t,
                from_event="cpr_active", to_event="cpr_quality_measured",
                actual_gap=None, expected=None,
                guideline="AHA 2025 High Quality CPR",
                recommendation="Minimize interruptions. Chest compression fraction should exceed 80%. Limit pauses to rhythm checks and shock delivery only.",
            )

        logger.info(
            f"  CPR quality @ {t}s | "
            f"rate={rate} | depth={depth}cm | "
            f"recoil={recoil} | hands_off={hands_off}"
        )

    # ── POST ROSC HANDLERS ────────────────────────────────────────────────────

    def _on_rosc_achieved(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.rosc_time = t
        self._state = "rosc"
        logger.info(f"  ROSC achieved at {t}s")

    def _on_post_rosc_ecg(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.ecg_obtained = True
        if self.rosc_time is not None:
            delay = t - self.rosc_time
            if delay > POST_ROSC_ECG_LIMIT:
                self._emit(
                    rule_id="R_ARREST_013", severity="HIGH",
                    message=f"12-lead ECG obtained {delay:.0f}s after ROSC — limit {POST_ROSC_ECG_LIMIT}s",
                    timestamp_sec=t,
                    from_event="rosc_achieved", to_event="post_rosc_ecg_obtained",
                    actual_gap=delay, expected=POST_ROSC_ECG_LIMIT,
                    guideline=GUIDELINE_ROSC,
                    recommendation="Obtain 12-lead ECG immediately after ROSC to identify STEMI or other treatable cause.",
                )

    def _on_temp_management(self, event: dict) -> None:
        t = event["timestamp_sec"]
        self.temp_management_done = True
        if self.rosc_time is not None:
            delay = t - self.rosc_time
            if delay > POST_ROSC_TEMP_LIMIT:
                self._emit(
                    rule_id="R_ARREST_014", severity="HIGH",
                    message=f"Temperature management initiated {delay:.0f}s after ROSC — limit {POST_ROSC_TEMP_LIMIT}s",
                    timestamp_sec=t,
                    from_event="rosc_achieved", to_event="temp_management_initiated",
                    actual_gap=delay, expected=POST_ROSC_TEMP_LIMIT,
                    guideline=GUIDELINE_ROSC,
                    recommendation="Initiate targeted temperature management 32-37.5C for all unresponsive post arrest patients.",
                )

    def _on_bp_recorded(self, event: dict) -> None:
        t = event["timestamp_sec"]
        map_value = event.get("value", {}).get("map_mmhg")
        if map_value is not None and map_value < POST_ROSC_BP_TARGET:
            self._emit(
                rule_id="R_ARREST_015", severity="HIGH",
                message=f"MAP {map_value} mmHg below target {POST_ROSC_BP_TARGET} mmHg after ROSC",
                timestamp_sec=t,
                from_event="rosc_achieved", to_event="bp_recorded",
                actual_gap=None, expected=None,
                guideline=GUIDELINE_ROSC,
                recommendation="Maintain MAP ≥65 mmHg post ROSC. Use vasopressors if needed to avoid hypotension.",
            )

    # ── END OF SESSION CHECKS ─────────────────────────────────────────────────
    def end_of_session_checks(self) -> None:

        # No antiarrhythmic after 3rd shock
        if self.shock_count >= 3 and not self.amiodarone_given and not self.lidocaine_given:
            self._emit(
                rule_id="R_ARREST_011", severity="HIGH",
                message="Neither amiodarone nor lidocaine given after 3 or more shocks in refractory shockable rhythm",
                timestamp_sec=0,
                from_event="third_shock_delivered", to_event="antiarrhythmic_given",
                actual_gap=None, expected=None,
                guideline="AHA 2025 ACLS Algorithm — Antiarrhythmic After 3rd Shock",
                recommendation="For VF/pVT refractory to 3 or more shocks: administer amiodarone 300mg IV/IO (repeat 150mg once) OR lidocaine 1-1.5 mg/kg IV/IO as alternative.",
            )

        # No ECG after ROSC
        if self.rosc_time is not None and not self.ecg_obtained:
            self._emit(
                rule_id="R_ARREST_016", severity="HIGH",
                message="12-lead ECG not documented after ROSC",
                timestamp_sec=0,
                from_event="rosc_achieved", to_event="post_rosc_ecg_obtained",
                actual_gap=None, expected=None,
                guideline=GUIDELINE_ROSC,
                recommendation="Always obtain 12-lead ECG immediately after ROSC to screen for STEMI.",
            )

        # No temperature management after ROSC
        if self.rosc_time is not None and not self.temp_management_done:
            self._emit(
                rule_id="R_ARREST_017", severity="MEDIUM",
                message="Temperature management not documented after ROSC",
                timestamp_sec=0,
                from_event="rosc_achieved", to_event="temp_management_initiated",
                actual_gap=None, expected=None,
                guideline=GUIDELINE_ROSC,
                recommendation="Initiate targeted temperature management for all unresponsive post arrest patients. Target 32-37.5C.",
            )

        # Primary ABCD — missing checks
        if not self.basic_airway_done:
            self._emit(
                rule_id="R_ARREST_024", severity="HIGH",
                message="Basic airway management not documented",
                timestamp_sec=0,
                from_event="arrest_recognized", to_event="basic_airway_opened",
                actual_gap=None, expected=None,
                guideline="AHA 2025 Primary ABCD Survey — Airway",
                recommendation="Document airway opening technique used during resuscitation.",
            )

        if not self.bag_mask_done:
            self._emit(
                rule_id="R_ARREST_025", severity="HIGH",
                message="Bag mask ventilation not documented",
                timestamp_sec=0,
                from_event="arrest_recognized", to_event="bag_mask_ventilation_started",
                actual_gap=None, expected=None,
                guideline="AHA 2025 Primary ABCD Survey — Breathing",
                recommendation="Document ventilation method used. Bag mask ventilation should begin immediately after airway opened.",
            )

        if not self.defibrillator_attached:
            self._emit(
                rule_id="R_ARREST_026", severity="HIGH",
                message="Defibrillator attachment not documented",
                timestamp_sec=0,
                from_event="arrest_recognized", to_event="defibrillator_attached",
                actual_gap=None, expected=None,
                guideline="AHA 2025 Primary ABCD Survey — Defibrillation",
                recommendation="Always document defibrillator attachment time for audit purposes.",
            )

        # Secondary ABCD — missing checks
        if not self.advanced_airway_done:
            self._emit(
                rule_id="R_ARREST_027", severity="MODERATE",
                message="Advanced airway not placed during resuscitation",
                timestamp_sec=0,
                from_event="arrest_recognized", to_event="advanced_airway_placed",
                actual_gap=None, expected=None,
                guideline="AHA 2025 Secondary ABCD Survey — Airway",
                recommendation="Consider advanced airway placement for prolonged resuscitation to enable continuous compressions.",
            )

        if not self.iv_access_done:
            self._emit(
                rule_id="R_ARREST_028", severity="HIGH",
                message="IV or IO access not documented during resuscitation",
                timestamp_sec=0,
                from_event="arrest_recognized", to_event="iv_access_established",
                actual_gap=None, expected=None,
                guideline="AHA 2025 Secondary ABCD Survey — Circulation",
                recommendation="IV or IO access is required for drug administration. Document route used.",
            )
            