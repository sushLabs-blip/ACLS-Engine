"""
performance_tracker.py — ACLS Performance Tracker
====================================================
AI-powered CPR Training Analysis & Debriefing System
AHA 2025 Guidelines | Educational / Simulation Use Only

Purpose
-------
Reads findings JSON (engine output) + scenario JSON (test input),
attributes each deviation to the responsible person via their role,
accumulates into performance_db.json, and auto-generates per-person
and per-batch report JSON files in reports/.

Unique identity = batch_id + name  (e.g. "Batch1_GroupA::Alex")
Same name in different batches = different people, separate reports.

Called automatically by engine.py — no manual steps needed.

Usage (manual)
--------------
    python performance_tracker.py --findings findings_X.json --scenario test_scenarios/X.json
    python performance_tracker.py --report "Batch1_GroupA" "Alex"
    python performance_tracker.py --report-batch "Batch1_GroupA"
    python performance_tracker.py --validate test_scenarios/X.json --findings findings_X.json
    python performance_tracker.py --list
"""

import json
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DB_PATH     = "performance_db.json"
REPORTS_DIR = "reports"


# ─────────────────────────────────────────────
# ROTATION ORDER
# ─────────────────────────────────────────────
ROTATION_ORDER = [
    "airway",
    "compressor",
    "defibrillator_operator",
    "iv_member",
    "recorder",
    "team_leader"
]


# ─────────────────────────────────────────────
# ROLE RESPONSIBILITY MAP
# ─────────────────────────────────────────────
ROLE_RESPONSIBILITY_MAP = {
    "cardiac_arrest": {
        "airway": [
            "airway_opened", "bag_mask_ventilation", "advanced_airway_placed",
            "tube_placement_confirmed", "ventilation_rate_checked"
        ],
        "compressor": [
            "cpr_initiated", "cpr_resumed", "compression_rate_checked",
            "compression_depth_checked", "chest_recoil_checked",
            "hands_off_ratio_checked"
        ],
        "defibrillator_operator": [
            "defibrillator_attached", "rhythm_check", "shock_delivered",
            "ecg_leads_attached", "cpr_coach"
        ],
        "iv_member": [
            "iv_access_established", "epinephrine_given", "amiodarone_given",
            "lidocaine_given", "drug_administered"
        ],
        "recorder": [
            "time_announced", "event_logged", "vitals_recorded"
        ],
        "team_leader": [
            "arrest_recognized", "rosc_achieved", "twelve_lead_obtained",
            "temperature_management_initiated", "map_target_confirmed",
            "reversible_causes_discussed", "expert_consulted",
            "post_rosc_care_initiated"
        ]
    },
    "bradycardia_with_pulse": {
        "airway": [
            "airway_opened", "bag_mask_ventilation", "oxygen_applied",
            "advanced_airway_placed"
        ],
        "compressor": [
            "vitals_recorded", "bp_measured", "twelve_lead_obtained",
            "ecg_leads_attached", "spo2_checked", "primary_survey_done"
        ],
        "defibrillator_operator": [
            "defibrillator_attached", "tcp_initiated", "electrical_capture_confirmed",
            "mechanical_capture_confirmed", "rhythm_check", "pacing_rate_adjusted"
        ],
        "iv_member": [
            "iv_access_established", "atropine_given", "dopamine_given",
            "epinephrine_given", "sedation_given", "analgesia_given",
            "drug_administered"
        ],
        "recorder": [
            "time_announced", "event_logged", "av_block_type_identified",
            "compromise_assessed"
        ],
        "team_leader": [
            "bradycardia_recognized", "expert_consulted", "patient_stabilized",
            "cause_identified", "treatment_decision_made"
        ]
    },
    "tachyarrhythmia_with_pulse": {
        "airway": [
            "airway_opened", "oxygen_applied", "bag_mask_ventilation"
        ],
        "compressor": [
            "vitals_recorded", "bp_measured", "twelve_lead_obtained",
            "ecg_leads_attached", "spo2_checked", "primary_survey_done",
            "qrs_width_assessed"
        ],
        "defibrillator_operator": [
            "defibrillator_attached", "cardioversion_shock_delivered",
            "rhythm_check", "synchronized_mode_confirmed", "rhythm_converted"
        ],
        "iv_member": [
            "iv_access_established", "adenosine_given", "amiodarone_given",
            "beta_blocker_given", "calcium_channel_blocker_given",
            "sedation_given", "drug_administered", "vagal_maneuver_attempted"
        ],
        "recorder": [
            "time_announced", "event_logged", "hemodynamic_stability_assessed",
            "pre_excitation_identified"
        ],
        "team_leader": [
            "tachyarrhythmia_recognized", "expert_consulted", "patient_stabilized",
            "treatment_decision_made", "cause_identified"
        ]
    }
}


# ─────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────

def validate_actor_roles(scenario: dict) -> list:
    """
    Check each event's actor_role against ROLE_RESPONSIBILITY_MAP.
    Returns list of misattribution warnings.
    """
    algorithm = scenario.get("algorithm", "")
    events    = scenario.get("events", [])
    warnings  = []

    algo_key = None
    if "cardiac_arrest" in algorithm.lower():
        algo_key = "cardiac_arrest"
    elif "brady" in algorithm.lower():
        algo_key = "bradycardia_with_pulse"
    elif "tachy" in algorithm.lower():
        algo_key = "tachyarrhythmia_with_pulse"

    if not algo_key:
        return []

    role_map = ROLE_RESPONSIBILITY_MAP[algo_key]
    event_to_roles = defaultdict(list)
    for role, event_types in role_map.items():
        for et in event_types:
            event_to_roles[et].append(role)

    for evt in events:
        event_type = evt.get("event_type")
        actor_role = evt.get("actor_role")
        event_id   = evt.get("event_id", "?")
        if actor_role in ("system", None) or event_type not in event_to_roles:
            continue
        expected = event_to_roles[event_type]
        if actor_role not in expected:
            warnings.append({
                "event_id":       event_id,
                "event_type":     event_type,
                "actor_role":     actor_role,
                "expected_roles": expected,
                "message": (
                    f"{event_id} ({event_type}): attributed to '{actor_role}' "
                    f"but expected {expected} per {algo_key} role map"
                )
            })
    return warnings


def print_validation_warnings(warnings: list, session_id: str):
    if not warnings:
        print(f"[VALIDATE] {session_id} — actor_role assignments OK")
        return
    print(f"\n[VALIDATE] {session_id} — {len(warnings)} misattribution(s):")
    for w in warnings:
        print(f"  !! {w['message']}")
    print(f"  -> Fix actor_role in scenario JSON.\n")


# ─────────────────────────────────────────────
# PERSON KEY + DUPLICATE NAME RESOLUTION
# ─────────────────────────────────────────────

def person_key(batch_id: str, name: str) -> str:
    return f"{batch_id}::{name}"


def resolve_participant_names(participants: dict) -> dict:
    """
    If two people in same session share a name, append role to disambiguate.
    e.g. two 'Alex' → 'Alex_team_leader', 'Alex_compressor'
    """
    name_counts = defaultdict(int)
    for name in participants.values():
        name_counts[name] += 1

    resolved = {}
    for role, name in participants.items():
        if name_counts[name] > 1:
            resolved[role] = f"{name}_{role}"
            print(f"[WARN] Duplicate name '{name}' — disambiguated as '{resolved[role]}'")
        else:
            resolved[role] = name
    return resolved


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────

def load_db() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {
        "meta": {
            "created":        datetime.now().isoformat(),
            "last_updated":   datetime.now().isoformat(),
            "total_sessions": 0,
            "description":    "ACLS Rule Engine — Central Performance Database"
        },
        "persons":     {},
        "batches":     {},
        "sessions":    {},
        "team_trends": []
    }


def save_db(db: dict):
    db["meta"]["last_updated"] = datetime.now().isoformat()
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"[DB] Saved → {DB_PATH}")


# ─────────────────────────────────────────────
# ATTRIBUTION
# ─────────────────────────────────────────────

def resolve_actor(deviation: dict, scenario: dict) -> tuple:
    """from_event → actor_role in scenario events → person name in participants."""
    from_event   = deviation.get("from_event")
    participants = scenario.get("participants", {})
    events       = scenario.get("events", [])

    matched_role = "unknown"
    for evt in events:
        if evt.get("event_type") == from_event:
            matched_role = evt.get("actor_role", "unknown")
            break

    person_name = participants.get(matched_role, "Unknown")
    return matched_role, person_name


def _compute_individual_score(team_score: float, person_devs: list) -> float:
    penalty = sum(d.get("penalty_weight", 0.1) * 100 for d in person_devs)
    return round(max(0.0, min(100.0, team_score - penalty * 0.5)), 1)


# ─────────────────────────────────────────────
# REPORT WRITERS
# ─────────────────────────────────────────────

def _write_person_report(db: dict, batch_id: str, name: str):
    """
    Write/update reports/report_<batch>_<name>.json
    Structure: one entry per role played, plus overall summary.
    Called after every ingest — file grows as person rotates through roles.
    """
    key = person_key(batch_id, name)
    if key not in db["persons"]:
        return

    person  = db["persons"][key]
    sessions = person["sessions"]

    # Group sessions by role
    by_role = defaultdict(list)
    for s in sessions:
        by_role[s["role_played"]].append(s)

    # Build per-role performance blocks
    role_performances = []
    for role in ROTATION_ORDER:
        if role not in by_role:
            continue
        role_sessions = by_role[role]
        scores = [s["individual_score"] for s in role_sessions
                  if s["individual_score"] is not None]
        all_devs = []
        for s in role_sessions:
            all_devs.extend(s["deviations_attributed"])

        rule_freq = defaultdict(int)
        for d in all_devs:
            rule_freq[d["rule_id"]] += 1

        role_performances.append({
            "role":              role,
            "sessions_as_role":  len(role_sessions),
            "average_score":     round(sum(scores) / len(scores), 1) if scores else None,
            "best_score":        max(scores) if scores else None,
            "worst_score":       min(scores) if scores else None,
            "total_deviations":  len(all_devs),
            "recurring_deviations": [
                {"rule_id": r, "count": c}
                for r, c in sorted(rule_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            ],
            "sessions": [
                {
                    "session_id":     s["session_id"],
                    "session_date":   s["session_date"],
                    "scenario_type":  s["scenario_type"],
                    "algorithm":      s["algorithm"],
                    "role_active":    s.get("role_active", True),
                    "team_score":     s["team_score"],
                    "individual_score": s["individual_score"],
                    "deviation_count":  s["deviation_count"],
                    "deviations": [
                        {
                            "rule_id":   d["rule_id"],
                            "severity":  d["severity"],
                            "message":   d["deviation_message"],
                            "guideline": d.get("guideline", ""),
                            "recommendation": d.get("recommendation", "")
                        }
                        for d in s["deviations_attributed"]
                    ]
                }
                for s in role_sessions
            ]
        })

    # Overall summary
    all_scores = [s["individual_score"] for s in sessions
                  if s["individual_score"] is not None]
    all_devs_flat = []
    for s in sessions:
        all_devs_flat.extend(s["deviations_attributed"])

    rule_freq_all = defaultdict(int)
    for d in all_devs_flat:
        rule_freq_all[d["rule_id"]] += 1

    roles_done    = list(by_role.keys())
    roles_pending = [r for r in ROTATION_ORDER if r not in roles_done]

    # Weakest role = role with lowest avg score
    weakest = None
    if role_performances:
        scored = [r for r in role_performances if r["average_score"] is not None]
        if scored:
            weakest = min(scored, key=lambda x: x["average_score"])["role"]

    sorted_sessions = sorted(sessions, key=lambda x: x["session_date"])
    sorted_scores   = [s["individual_score"] for s in sorted_sessions
                       if s["individual_score"] is not None]
    trend = ("improving" if len(sorted_scores) >= 2 and sorted_scores[-1] > sorted_scores[0]
             else "declining" if len(sorted_scores) >= 2 and sorted_scores[-1] < sorted_scores[0]
             else "stable")

    report = {
        "report_for":     name,
        "batch_id":       batch_id,
        "generated_at":   datetime.now().isoformat(),
        "overall_summary": {
            "total_sessions":              len(sessions),
            "roles_completed":             roles_done,
            "roles_pending":               roles_pending,
            "average_score":               round(sum(all_scores) / len(all_scores), 1) if all_scores else None,
            "best_score":                  max(all_scores) if all_scores else None,
            "worst_score":                 min(all_scores) if all_scores else None,
            "score_trend":                 trend,
            "weakest_role":                weakest,
            "total_deviations_attributed": len(all_devs_flat),
            "top_recurring_deviations": [
                {"rule_id": r, "count": c}
                for r, c in sorted(rule_freq_all.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        },
        "performance_by_role": role_performances
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    safe_name  = name.replace(" ", "_")
    safe_batch = batch_id.replace(" ", "_")
    path = os.path.join(REPORTS_DIR, f"report_{safe_batch}_{safe_name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] Updated → {path}")


def _write_batch_report(db: dict, batch_id: str):
    """
    Write/update reports/report_batch_<batch>.json
    Contains all members' role-by-role performance — fed to LLM debriefing engine.
    """
    if batch_id not in db["batches"]:
        return

    batch       = db["batches"][batch_id]
    members     = batch["members"]
    session_ids = batch["session_ids"]

    batch_sessions = [db["sessions"][sid] for sid in session_ids if sid in db["sessions"]]
    team_scores    = [s["performance_score"] for s in batch_sessions]
    avg_team       = round(sum(team_scores) / len(team_scores), 1) if team_scores else 0

    # Collect per-person summaries
    member_reports = []
    for name in members:
        key = person_key(batch_id, name)
        if key not in db["persons"]:
            continue
        person   = db["persons"][key]
        sessions = person["sessions"]

        by_role = defaultdict(list)
        for s in sessions:
            by_role[s["role_played"]].append(s)

        role_summaries = []
        for role in ROTATION_ORDER:
            if role not in by_role:
                continue
            role_sessions = by_role[role]
            scores = [s["individual_score"] for s in role_sessions
                      if s["individual_score"] is not None]
            devs   = []
            for s in role_sessions:
                devs.extend(s["deviations_attributed"])
            role_summaries.append({
                "role":             role,
                "average_score":    round(sum(scores)/len(scores), 1) if scores else None,
                "total_deviations": len(devs),
                "deviations": [
                    {
                        "rule_id":        d["rule_id"],
                        "severity":       d["severity"],
                        "message":        d["deviation_message"],
                        "recommendation": d.get("recommendation", "")
                    }
                    for d in devs
                ]
            })

        all_scores = [s["individual_score"] for s in sessions
                      if s["individual_score"] is not None]
        member_reports.append({
            "name":              name,
            "total_sessions":    len(sessions),
            "overall_avg_score": round(sum(all_scores)/len(all_scores), 1) if all_scores else None,
            "roles_completed":   list(by_role.keys()),
            "roles_pending":     [r for r in ROTATION_ORDER if r not in by_role],
            "performance_by_role": role_summaries
        })

    # Leaderboard
    leaderboard = sorted(
        [m for m in member_reports if m["overall_avg_score"] is not None],
        key=lambda x: x["overall_avg_score"],
        reverse=True
    )

    report = {
        "report_type":    "batch",
        "batch_id":       batch_id,
        "generated_at":   datetime.now().isoformat(),
        "total_sessions": len(session_ids),
        "total_members":  len(members),
        "avg_team_score": avg_team,
        "members":        members,
        "session_ids":    session_ids,
        "leaderboard": [
            {"rank": i+1, "name": m["name"], "avg_score": m["overall_avg_score"]}
            for i, m in enumerate(leaderboard)
        ],
        "individual_reports": member_reports
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    safe_batch = batch_id.replace(" ", "_")
    path = os.path.join(REPORTS_DIR, f"report_batch_{safe_batch}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] Batch report updated → {path}")


# ─────────────────────────────────────────────
# INGEST SESSION  (main entry point)
# ─────────────────────────────────────────────

def ingest_session(findings_path: str, scenario_path: str):
    """
    Called by engine.py after save_findings().
    Reads findings + scenario, attributes deviations, updates DB,
    then auto-writes all 6 individual reports + batch report.
    """
    with open(findings_path, "r") as f:
        findings = json.load(f)
    with open(scenario_path, "r") as f:
        scenario = json.load(f)

    session_id    = findings["session_id"]
    session_date  = findings.get("session_date", scenario.get("session_date", "unknown"))
    algorithm     = findings.get("algorithm", "unknown")
    scenario_type = findings.get("scenario_type", "unknown")
    score         = findings.get("performance_score", 0.0)
    participants  = scenario.get("participants", {})

    # ── Batch ID ──────────────────────────────
    batch_id = scenario.get("batch_id")
    if not batch_id:
        print("[WARN] No batch_id in scenario JSON — using 'Batch_Unknown'.")
        batch_id = "Batch_Unknown"

    # ── Resolve duplicate names ───────────────
    participants = resolve_participant_names(participants)

    print(f"\n[TRACKER] Session  : {session_id}")
    print(f"          Batch    : {batch_id}")
    print(f"          Algorithm: {algorithm}  Score: {score}")

    # ── Validate actor_role assignments ───────
    validation_scenario = {**scenario, "algorithm": algorithm}
    warnings = validate_actor_roles(validation_scenario)
    print_validation_warnings(warnings, session_id)

    db = load_db()

    if session_id in db["sessions"]:
        print(f"[WARN] Session {session_id} already ingested — skipping.")
        return

    # ── Active roles in this scenario ─────────
    active_roles = {
        evt.get("actor_role") for evt in scenario.get("events", [])
        if evt.get("actor_role") not in ("system", None)
    }

    # ── Attribute deviations ──────────────────
    scenario_resolved = {**scenario, "participants": participants}
    attributed_deviations = []
    for dev in findings.get("deviations", []):
        actor_role, person_name = resolve_actor(dev, scenario_resolved)
        attributed_deviations.append({**dev, "actor_role": actor_role, "actor_name": person_name})

    # ── Attribute strengths to team_leader ────
    attributed_strengths = []
    tl_name = participants.get("team_leader", "Unknown")
    for s in findings.get("strengths", []):
        attributed_strengths.append({**s, "actor_role": "team_leader", "actor_name": tl_name})

    # ── Session record ────────────────────────
    session_record = {
        "session_id":        session_id,
        "batch_id":          batch_id,
        "session_date":      session_date,
        "algorithm":         algorithm,
        "scenario_type":     scenario_type,
        "performance_score": score,
        "total_deviations":  findings.get("total_deviations", 0),
        "total_strengths":   findings.get("total_strengths", 0),
        "participants":      participants,
        "deviations":        attributed_deviations,
        "strengths":         attributed_strengths
    }
    db["sessions"][session_id] = session_record
    db["meta"]["total_sessions"] += 1

    # ── Register batch ────────────────────────
    if batch_id not in db["batches"]:
        db["batches"][batch_id] = {"session_ids": [], "members": []}
    if session_id not in db["batches"][batch_id]["session_ids"]:
        db["batches"][batch_id]["session_ids"].append(session_id)

    # ── Update per-person records ─────────────
    for role, name in participants.items():
        if role == "system":
            continue

        key         = person_key(batch_id, name)
        person_devs = [d for d in attributed_deviations if d["actor_name"] == name]
        person_strs = [s for s in attributed_strengths if s["actor_name"] == name]
        role_active = role in active_roles
        ind_score   = _compute_individual_score(score, person_devs) if role_active else None

        if not role_active:
            print(f"[INFO] {name} ({role}) — no active events in this scenario (role inactive)")

        session_entry = {
            "session_id":            session_id,
            "batch_id":              batch_id,
            "session_date":          session_date,
            "algorithm":             algorithm,
            "scenario_type":         scenario_type,
            "role_played":           role,
            "role_active":           role_active,
            "team_score":            score,
            "individual_score":      ind_score,
            "deviations_attributed": person_devs,
            "strengths_attributed":  person_strs,
            "deviation_count":       len(person_devs),
            "strength_count":        len(person_strs)
        }

        if key not in db["persons"]:
            db["persons"][key] = {
                "person_key":   key,
                "batch_id":     batch_id,
                "name":         name,
                "sessions":     [],
                "roles_played": [],
            }

        db["persons"][key]["sessions"].append(session_entry)
        if role not in db["persons"][key]["roles_played"]:
            db["persons"][key]["roles_played"].append(role)

        if name not in db["batches"][batch_id]["members"]:
            db["batches"][batch_id]["members"].append(name)

    # ── Team trend ────────────────────────────
    db["team_trends"].append({
        "session_id":       session_id,
        "batch_id":         batch_id,
        "session_date":     session_date,
        "algorithm":        algorithm,
        "team_score":       score,
        "total_deviations": findings.get("total_deviations", 0)
    })

    save_db(db)

    # ── Auto-generate all reports ─────────────
    print(f"\n[REPORTS] Generating reports for batch {batch_id} ...")
    for name in db["batches"][batch_id]["members"]:
        _write_person_report(db, batch_id, name)
    _write_batch_report(db, batch_id)

    print(f"\n[TRACKER] Done. {len(db['batches'][batch_id]['members'])} person reports + 1 batch report updated.")
    _print_attribution_summary(session_id, attributed_deviations)


def _print_attribution_summary(session_id: str, attributed: list):
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION SUMMARY — {session_id}")
    print(f"{'='*60}")
    if not attributed:
        print("  No deviations — perfect session!")
    for d in attributed:
        print(f"  [{d['severity']}] {d['rule_id']} → {d['actor_role']} ({d['actor_name']})")
        print(f"        {d['deviation_message']}")
    print("=" * 60)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ACLS Performance Tracker")
    parser.add_argument("--findings",     help="Path to findings JSON")
    parser.add_argument("--scenario",     help="Path to scenario JSON")
    parser.add_argument("--validate",     help="Validate actor_role in scenario JSON")
    parser.add_argument("--report",       nargs=2, metavar=("BATCH_ID", "NAME"),
                        help="--report 'Batch1_GroupA' 'Alex'")
    parser.add_argument("--report-batch", metavar="BATCH_ID",
                        help="Generate batch report")
    parser.add_argument("--report-team",  action="store_true",
                        help="Cross-batch overview")
    parser.add_argument("--list",         action="store_true",
                        help="List all persons in DB")
    args = parser.parse_args()

    if args.validate:
        with open(args.validate, "r") as f:
            scenario = json.load(f)
        algo = scenario.get("algorithm", "")
        if args.findings and os.path.exists(args.findings):
            with open(args.findings, "r") as f:
                algo = json.load(f).get("algorithm", algo)
        warnings = validate_actor_roles({**scenario, "algorithm": algo})
        print_validation_warnings(warnings, scenario.get("session_id", args.validate))

    elif args.findings and args.scenario:
        ingest_session(args.findings, args.scenario)

    elif args.report:
        db = load_db()
        _write_person_report(db, args.report[0], args.report[1])

    elif args.report_batch:
        db = load_db()
        _write_batch_report(db, args.report_batch)

    elif args.report_team:
        db = load_db()
        print("\nAll batches in DB:")
        for bid, b in db["batches"].items():
            sids = b["session_ids"]
            scores = [db["sessions"][s]["performance_score"]
                      for s in sids if s in db["sessions"]]
            avg = round(sum(scores)/len(scores), 1) if scores else 0
            print(f"  [{bid}] {len(sids)} sessions | avg: {avg} | members: {', '.join(b['members'])}")

    elif args.list:
        db = load_db()
        print("\nPersons in DB:")
        for key, p in db["persons"].items():
            sessions = p.get("sessions", [])
            scores   = [s["individual_score"] for s in sessions
                        if s["individual_score"] is not None]
            avg = round(sum(scores)/len(scores), 1) if scores else "N/A"
            print(f"  [{p['batch_id']}] {p['name']} — "
                  f"{len(sessions)} session(s), avg: {avg}, "
                  f"roles: {', '.join(p['roles_played'])}")
        print(f"\n  Batches: {list(db['batches'].keys())}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()