"""
algorithms/tachyarrhythmia.py
AHA 2025 Adult Tachyarrhythmia With a Pulse Algorithm Processor

Implements all guard-based deviation rules from:
  aha_2025_tachyarrhythmia_pulse_v2.json

Rule IDs: R_TACHY_001 – R_TACHY_015

State machine summary:
  initial_assessment
    └─ check_hemodynamic_stability
         ├─ UNSTABLE → synchronized_cardioversion
         └─ STABLE   → evaluate_qrs_width
              ├─ narrow_qrs
              │    ├─ narrow_regular   (vagal → adenosine → CCB/BB)
              │    └─ narrow_irregular
              │         ├─ pre_excited_afib   (WPW — NO AV-nodal blockers)
              │         └─ narrow_irregular_rate_control
              └─ wide_qrs
                   ├─ wide_regular_monomorphic  (treat as VT — procainamide/amio)
                   └─ wide_polymorphic
                        ├─ torsades_de_pointes  (prolonged QT → Mg)
                        └─ wide_polymorphic_ischemic
"""

import logging
from typing import Optional

from .base import BaseAlgorithm

logger = logging.getLogger(__name__)

GUIDELINE = "AHA 2025 Adult Tachyarrhythmia With a Pulse Algorithm"


class TachyarrhythmiaAlgorithm(BaseAlgorithm):
    ALGORITHM_ID = "tachyarrhythmia_with_pulse"

    def __init__(self, rules: dict):
        super().__init__(rules)
        self._state = "initial_assessment"
        self.recognition_time: Optional[float] = None

    # ── EVENT DISPATCHER ──────────────────────────────────────────────────────
    def process_event(self, event: dict) -> None:
        etype = event["event_type"]
        t     = event["timestamp_sec"]
        data  = event.get("data", {})

        logger.info(f"  [TACHY] {etype} @ {t}s | state={self._state}")

        # ── Recognition ───────────────────────────────────────────────────────
        if etype == "tachyarrhythmia_recognized":
            self.recognition_time = t

        # ── Monitoring / access ───────────────────────────────────────────────
        elif etype == "vitals_recorded":
            for k in ["heart_rate_bpm", "qrs_duration_sec", "rhythm_regularity",
                      "rhythm_label", "qrs_morphology"]:
                if k in data:
                    self.context[k] = data[k]
            if data.get("prolonged_qt"):
                self.context["prolonged_qt"] = True

        elif etype == "iv_access_established":
            self.context["iv_access"] = True

        elif etype == "twelve_lead_obtained":
            self.context["twelve_lead_obtained"] = True

        elif etype == "pre_excitation_identified":
            self.context["pre_excitation_suspected"] = True
            # If already in narrow_irregular, escalate to WPW pathway
            if self._state == "narrow_irregular":
                self._state = "pre_excited_afib"
            logger.info(f"  Pre-excitation (WPW) flag set at {t}s")

        elif etype == "prolonged_qt_confirmed":
            self.context["prolonged_qt"] = True
            if self._state == "wide_polymorphic":
                self._state = "torsades_de_pointes"

        elif etype == "chf_history_noted":
            self.context["chf_history"] = True

        # ── Stability assessment → state routing ──────────────────────────────
        elif etype == "hemodynamic_stability_assessed":
            signs = data.get("instability_signs", {})
            self.context["instability_signs"].update(signs)
            unstable = any(signs.values())
            self.context["hemodynamically_stable"] = not unstable
            if unstable:
                self._state = "synchronized_cardioversion"
                logger.info(f"  Patient UNSTABLE → synchronized_cardioversion")
            else:
                self._state = "evaluate_qrs_width"
                logger.info(f"  Patient STABLE → evaluate_qrs_width")

        # ── QRS width → sub-state routing ─────────────────────────────────────
        elif etype == "qrs_width_assessed":
            qrs_wide = data.get("qrs_wide", None)
            if qrs_wide is True:
                morph = self.context.get("qrs_morphology", "monomorphic")
                reg   = self.context.get("rhythm_regularity", "regular")
                if morph == "polymorphic" or reg == "irregular":
                    self._state = "wide_polymorphic"
                else:
                    self._state = "wide_regular_monomorphic"
            elif qrs_wide is False:
                reg = self.context.get("rhythm_regularity", "unknown")
                self._state = "narrow_regular" if reg == "regular" else "narrow_irregular"

        elif etype == "rhythm_regularity_assessed":
            reg = data.get("regularity", "unknown")
            self.context["rhythm_regularity"] = reg
            if self._state in ["narrow_qrs", "evaluate_qrs_width"]:
                self._state = "narrow_regular" if reg == "regular" else "narrow_irregular"

        # ── Sedation ──────────────────────────────────────────────────────────
        elif etype == "sedation_given":
            self.context["sedation_given"] = True

        # ── Vagal maneuver ────────────────────────────────────────────────────
        elif etype == "vagal_maneuver_performed":
            self.context["vagal_maneuver_attempted"] = True
            self.context["vagal_maneuver_result"] = data.get("result")

        elif etype == "vagal_maneuver_successful":
            self.context["vagal_maneuver_attempted"] = True
            self.context["rhythm_converted"] = True
            self._state = "rhythm_converted_care"

        # ── Adenosine ─────────────────────────────────────────────────────────
        elif etype == "adenosine_given":
            self._check_adenosine(event)
            self.context["adenosine_given"] = True
            self.context["adenosine_dose_count"] = self.context.get("adenosine_dose_count", 0) + 1
            self.context["adenosine_result"] = data.get("result")

        # ── Cardioversion (synchronized) ──────────────────────────────────────
        elif etype == "cardioversion_shock_delivered":
            self._check_cardioversion(event)
            self.context["cardioversion_attempted"] = True
            self.context["cardioversion_count"] = self.context.get("cardioversion_count", 0) + 1
            self.context["cardioversion_energy_used_J"] = data.get("energy_J")

        # ── Unsynchronized shock (polymorphic VT / VF) ────────────────────────
        elif etype == "unsynchronized_shock_delivered":
            # Correct action for polymorphic VT — no deviation expected
            self.context["cardioversion_attempted"] = True
            self.context["cardioversion_count"] = self.context.get("cardioversion_count", 0) + 1
            logger.info(f"  Unsynchronized shock delivered at {t}s (correct for poly VT)")

        elif etype == "synchronized_cardioversion_for_polymorphic_vt":
            # Explicit event flagging the wrong choice
            self._emit(
                rule_id="R_TACHY_010", severity="CRITICAL",
                message="Synchronized cardioversion attempted for polymorphic VT — must use UNSYNCHRONIZED high-energy shock",
                timestamp_sec=t,
                from_event="polymorphic_vt_identified", to_event="cardioversion_shock_delivered",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "Polymorphic VT must be treated as VF: use defibrillation energy "
                    "(unsynchronized). Synchronized cardioversion is unsafe for irregular rhythms."
                ),
            )

        # ── Antiarrhythmic drugs ──────────────────────────────────────────────
        elif etype == "beta_blocker_given":
            self._check_av_nodal_blocker(t, "beta-blocker")
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = data.get("drug", "beta_blocker")

        elif etype == "calcium_channel_blocker_given":
            drug = data.get("drug", "ccb")
            self._check_av_nodal_blocker(t, drug)
            if drug in ["verapamil", "diltiazem"]:
                self._check_verapamil_diltiazem(t, drug)
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = drug

        elif etype == "verapamil_given":
            self._check_av_nodal_blocker(t, "verapamil")
            self._check_verapamil_diltiazem(t, "verapamil")
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "verapamil"

        elif etype == "diltiazem_given":
            self._check_av_nodal_blocker(t, "diltiazem")
            self._check_verapamil_diltiazem(t, "diltiazem")
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "diltiazem"

        elif etype == "amiodarone_given":
            self._check_amiodarone(t)
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "amiodarone"

        elif etype == "procainamide_given":
            self._check_procainamide(t)
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "procainamide"

        elif etype == "sotalol_given":
            self._check_sotalol(t)
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "sotalol"

        elif etype == "magnesium_given":
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "magnesium"
            logger.info(f"  Magnesium given at {t}s")

        elif etype == "ibutilide_given":
            self.context["antiarrhythmic_given"] = True
            self.context["antiarrhythmic_drug"] = "ibutilide"

        elif etype == "digoxin_given":
            self._check_av_nodal_blocker(t, "digoxin")

        # ── Outcome ───────────────────────────────────────────────────────────
        elif etype == "expert_consulted":
            self.context["expert_consulted"] = True

        elif etype == "rhythm_converted":
            self.context["rhythm_converted"] = True
            self._state = "rhythm_converted_care"

        elif etype == "patient_deteriorated_to_arrest":
            self._state = "cardiac_arrest_pathway"

    # ── GUARD CHECKS ─────────────────────────────────────────────────────────

    def _check_adenosine(self, event: dict) -> None:
        t    = event["timestamp_sec"]
        data = event.get("data", {})
        reg       = self.context.get("rhythm_regularity", "unknown")
        morph     = self.context.get("qrs_morphology", "unknown")
        qrs_dur   = self.context.get("qrs_duration_sec") or 0.0
        pre_exc   = self.context.get("pre_excitation_suspected", False)

        # R_TACHY_001 — adenosine only for regular rhythms
        if reg == "irregular":
            self._emit(
                rule_id="R_TACHY_001", severity="HIGH",
                message="Adenosine given for irregular rhythm — only indicated for regular tachycardias",
                timestamp_sec=t, from_event="irregular_rhythm", to_event="adenosine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "Adenosine is contraindicated in irregular tachycardias (e.g., Afib). "
                    "Use rate-control or cardioversion depending on stability."
                ),
            )
        # Wide + irregular/polymorphic
        if qrs_dur >= 0.12 and morph in ["polymorphic", "irregular"]:
            self._emit(
                rule_id="R_TACHY_001", severity="HIGH",
                message="Adenosine given for wide, irregular or polymorphic QRS tachycardia — contraindicated",
                timestamp_sec=t, from_event="wide_polymorphic_qrs", to_event="adenosine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation="Adenosine for wide QRS is ONLY appropriate if rhythm is regular AND monomorphic.",
            )

        # R_TACHY_002 — rapid IV push + NS flush required
        route = data.get("route", "")
        flush = data.get("flush_given", False)
        if route != "rapid_IV_push" or not flush:
            self._emit(
                rule_id="R_TACHY_002", severity="HIGH",
                message="Adenosine not administered as rapid IV push with NS flush",
                timestamp_sec=t, from_event="adenosine_ordered", to_event="adenosine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "Adenosine has an extremely short half-life. Give as a RAPID IV push "
                    "immediately followed by a 20 mL NS flush to ensure central delivery."
                ),
            )

        # R_TACHY_003 — AV-nodal blocker in pre-excitation
        if pre_exc:
            self._emit(
                rule_id="R_TACHY_003", severity="CRITICAL",
                message="Adenosine (AV-nodal blocker) given with known pre-excitation (WPW) — risk of precipitating VF",
                timestamp_sec=t, from_event="pre_excitation_identified", to_event="adenosine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025 Class III HARM: In WPW, AV-nodal blockers can cause rapid "
                    "conduction via the accessory pathway leading to VF. Use procainamide or cardioversion."
                ),
            )

        # R_TACHY_011 — vagal maneuver should precede adenosine (narrow regular stable)
        if (self._state == "narrow_regular"
                and not self.context.get("vagal_maneuver_attempted", False)):
            self._emit(
                rule_id="R_TACHY_011", severity="MEDIUM",
                message="Adenosine given for narrow regular tachycardia without prior vagal maneuver attempt",
                timestamp_sec=t, from_event="narrow_regular_stable", to_event="adenosine_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Attempt vagal maneuvers (Valsalva, carotid sinus massage) "
                    "as the first step for stable narrow regular tachycardia before administering adenosine."
                ),
            )

    def _check_av_nodal_blocker(self, t: float, drug: str) -> None:
        """R_TACHY_003: Any AV-nodal blocker in pre-excitation is CRITICAL."""
        if self.context.get("pre_excitation_suspected", False):
            self._emit(
                rule_id="R_TACHY_003", severity="CRITICAL",
                message=f"AV-nodal blocker ({drug}) given with known pre-excitation (WPW) — risk of VF",
                timestamp_sec=t,
                from_event="pre_excitation_identified", to_event=f"{drug}_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025 Class III HARM: All AV-nodal blockers "
                    "(adenosine, beta-blockers, CCBs, digoxin) are contraindicated in WPW. "
                    "Use procainamide, ibutilide, or immediate cardioversion."
                ),
            )

    def _check_verapamil_diltiazem(self, t: float, drug: str) -> None:
        """R_TACHY_005: Verapamil/diltiazem absolutely contraindicated in wide QRS of uncertain origin."""
        qrs_dur = self.context.get("qrs_duration_sec") or 0.0
        rhythm  = self.context.get("rhythm_label", None)
        if qrs_dur >= 0.12 and rhythm in ["unknown", None, "vt", "wide_complex"]:
            self._emit(
                rule_id="R_TACHY_005", severity="CRITICAL",
                message=f"{drug.capitalize()} given for wide-complex tachycardia of uncertain etiology — risk of hemodynamic collapse",
                timestamp_sec=t,
                from_event="wide_qrs_uncertain", to_event=f"{drug}_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Verapamil and diltiazem are ABSOLUTELY CONTRAINDICATED in "
                    "wide-complex tachycardia of uncertain origin. They can cause profound "
                    "hypotension and cardiovascular collapse if the rhythm is VT."
                ),
            )

    def _check_amiodarone(self, t: float) -> None:
        pre_exc     = self.context.get("pre_excitation_suspected", False)
        rhythm      = self.context.get("rhythm_label", "")
        prolonged_qt = self.context.get("prolonged_qt", False)

        # R_TACHY_004: Amiodarone in pre-excited Afib/flutter
        if pre_exc and rhythm in ["afib", "atrial_flutter", "afib_flutter"]:
            self._emit(
                rule_id="R_TACHY_004", severity="CRITICAL",
                message="IV amiodarone given for pre-excited Afib/flutter (WPW) — may increase ventricular response and precipitate VF",
                timestamp_sec=t,
                from_event="pre_excitation_afib", to_event="amiodarone_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025 explicitly prohibits IV amiodarone in pre-excited Afib/flutter. "
                    "Use procainamide IV or ibutilide. Prioritize electrical cardioversion if unstable."
                ),
            )

        # R_TACHY_013: Amiodarone worsens Torsades de Pointes
        if prolonged_qt and self._state == "torsades_de_pointes":
            self._emit(
                rule_id="R_TACHY_013", severity="HIGH",
                message="Amiodarone given for Torsades de Pointes — amiodarone worsens QT prolongation",
                timestamp_sec=t,
                from_event="torsades_de_pointes", to_event="amiodarone_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: For Torsades de Pointes, use Magnesium Sulfate 1–2 g IV. "
                    "Amiodarone is CONTRAINDICATED — it further prolongs the QT interval."
                ),
            )

    def _check_procainamide(self, t: float) -> None:
        """R_TACHY_006: Procainamide contraindicated in prolonged QT or CHF."""
        prolonged_qt = self.context.get("prolonged_qt", False)
        chf          = self.context.get("chf_history", False)
        if prolonged_qt or chf:
            reason = "prolonged QT" if prolonged_qt else "CHF"
            self._emit(
                rule_id="R_TACHY_006", severity="HIGH",
                message=f"Procainamide given despite contraindication ({reason})",
                timestamp_sec=t,
                from_event=f"contraindication_{reason.replace(' ', '_')}",
                to_event="procainamide_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Procainamide is contraindicated in prolonged QT "
                    "(risk of Torsades) and CHF (negative inotropy). "
                    "Use amiodarone 150 mg IV over 10 min as the alternative."
                ),
            )

    def _check_sotalol(self, t: float) -> None:
        """R_TACHY_007: Sotalol contraindicated in prolonged QT."""
        if self.context.get("prolonged_qt", False):
            self._emit(
                rule_id="R_TACHY_007", severity="HIGH",
                message="Sotalol given with prolonged QT — risk of Torsades de Pointes",
                timestamp_sec=t,
                from_event="prolonged_qt_confirmed", to_event="sotalol_given",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Avoid sotalol in prolonged QT. "
                    "Use procainamide or amiodarone for stable wide-complex tachycardia."
                ),
            )

    def _check_cardioversion(self, event: dict) -> None:
        t    = event["timestamp_sec"]
        data = event.get("data", {})
        sync_mode = data.get("synchronized_mode", None)
        morph     = self.context.get("qrs_morphology", "unknown")

        # R_TACHY_008: Sync mode must be confirmed ON
        if sync_mode is not True:
            self._emit(
                rule_id="R_TACHY_008", severity="CRITICAL",
                message="Cardioversion shock delivered without confirmed synchronized mode — risk of R-on-T and VF induction",
                timestamp_sec=t,
                from_event="cardioversion_ordered", to_event="cardioversion_shock_delivered",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "Always confirm the defibrillator is in SYNCHRONIZED mode before cardioversion. "
                    "An unsynchronized shock can land on the T-wave and induce VF."
                ),
            )

        # R_TACHY_009: Sedation should precede cardioversion
        if not self.context.get("sedation_given", False):
            self._emit(
                rule_id="R_TACHY_009", severity="MEDIUM",
                message="Synchronized cardioversion performed without documented sedation",
                timestamp_sec=t,
                from_event="cardioversion_ordered", to_event="cardioversion_shock_delivered",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Sedate the patient whenever feasible before cardioversion. "
                    "If sedation is deferred due to critical instability, document the reason."
                ),
            )

        # R_TACHY_010: No synchronized cardioversion for polymorphic VT
        if morph == "polymorphic":
            self._emit(
                rule_id="R_TACHY_010", severity="CRITICAL",
                message="Synchronized cardioversion attempted for polymorphic VT — must use UNSYNCHRONIZED high-energy shock",
                timestamp_sec=t,
                from_event="polymorphic_vt_identified", to_event="cardioversion_shock_delivered",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "Polymorphic VT must be treated as VF: deliver a high-energy UNSYNCHRONIZED "
                    "shock (defibrillation dose). Synchronized cardioversion is unsafe for irregular rhythms."
                ),
            )

    # ── END-OF-SESSION CHECKS ─────────────────────────────────────────────────
    def end_of_session_checks(self) -> None:
        # R_TACHY_012: No IV access established
        if not self.context.get("iv_access", False):
            self._emit(
                rule_id="R_TACHY_012", severity="HIGH",
                message="IV access not established during tachyarrhythmia management",
                timestamp_sec=0,
                from_event="tachyarrhythmia_recognized", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation="Establish IV access early to enable drug administration and cardioversion preparation.",
            )

        # R_TACHY_013 (end-of-session): Torsades without magnesium
        if (self._state == "torsades_de_pointes"
                and self.context.get("antiarrhythmic_drug") != "magnesium"):
            self._emit(
                rule_id="R_TACHY_014", severity="HIGH",
                message="Torsades de Pointes managed without magnesium sulfate administration",
                timestamp_sec=0,
                from_event="torsades_de_pointes", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Magnesium Sulfate 1–2 g IV is the primary treatment for "
                    "Torsades de Pointes. Correct electrolyte abnormalities and discontinue QT-prolonging drugs."
                ),
            )

        # R_TACHY_015: No 12-lead ECG
        if not self.context.get("twelve_lead_obtained", False):
            self._emit(
                rule_id="R_TACHY_015", severity="LOW",
                message="No 12-lead ECG obtained during tachyarrhythmia assessment",
                timestamp_sec=0,
                from_event="session_start", to_event="session_end",
                actual_gap=None, expected=None, guideline=GUIDELINE,
                recommendation=(
                    "AHA 2025: Obtain a 12-lead ECG as early as feasible to classify the "
                    "tachyarrhythmia, identify pre-excitation (WPW), QT prolongation, or ischemia."
                ),
            )
