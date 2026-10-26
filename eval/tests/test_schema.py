"""Column contract of analyze.py on a synthetic dataset (numpy/scipy/yaml only).

    py eval/tests/test_schema.py        (or: python3 -m pytest eval/tests/test_schema.py)
"""

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import analyze as A  # noqa: E402
import synth  # noqa: E402

DS = ("d9", "d27", "d55")
PER_RUN = (
    ["policy", "condition", "vcmd", "distance_target", "run_idx", "term_type", "d_max", "t_end",
     "path_len_at_end", "path_len", "fell", "reach_path_9"]
    + [p + s for s in DS for p in ("y_", "theta_", "theta_fit0_", "theta_fit2_")]
    + ["y_x95", "theta_x95", "theta_fit0_x95", "theta_fit2_x95"]
    + ["psi", "heading_change", "v_mean_ss", "e_v", "cv_v", "t90", "d90", "overshoot", "reached90",
       "yaw_rate_ss", "yaw_rate_ss_deg", "curvature_ss", "turn_radius",
       "insufficient_speed", "insufficient_heading", "insufficient_gait", "insufficient_symmetry"]
    + [b + side for b in ("stance_t", "swing_t", "stride_t", "duty", "step_t", "step_len") for side in ("_L", "_R")]
    + ["flight_frac", "is_running", "n_strides", "n_strides_L", "n_strides_R", "stride_mismatch"]
    + ["si_stance", "si_swing", "si_duty", "si_step_t", "si_step_len"]
    + ["si_peak_" + j for j in ("Shoulder_Pitch", "Hip_Pitch", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]
    + [p + j + s for j in ("Shoulder_Pitch", "Hip_Pitch", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll") for p, s in (("amp_", "_L"), ("amp_", "_R"))]
    + ["em_all", "em_legs", "em_arms", "et_all", "twin_lat_d9", "twin_lat_d27", "twin_lat_d55", "y_twin_x95", "twin_lat_x95"]
    + ["dphi_arm_leg_L", "dphi_arm_leg_R", "R_arm_leg_L", "R_arm_leg_R", "dphi_arms", "R_arms", "rho_freq_L", "rho_freq_R", "amp_shoulder_L", "amp_shoulder_R"]
)
TERM_TYPES = ("finished", "fell_height", "fell_speed", "timeout_moved", "did_not_start")
SUMMARY = (
    ["S_d9", "S_d27", "S_d55", "S_path_9", "viable", "training_issue", "x95", "S_x95", "frac_fell",
     "idx_median", "idx_best", "idx_worst", "idx_median_disp", "idx_best_disp", "idx_worst_disp",
     "idx_median_abs", "idx_best_abs", "idx_worst_abs", "idx_rank_basis",
     "beta_x95", "sigma_theta_x95", "alpha95_x95", "w95_x95", "sigma_y_x95",
     "beta_fit2_x95", "sigma_fit2_x95", "alpha95_fit2_x95", "twin_turn_same_side"]
    + ["n_" + k for k in TERM_TYPES] + ["frac_" + k for k in TERM_TYPES]
    + [p + s for s in DS for p in ("beta_", "sigma_theta_", "alpha95_", "w95_", "beta_fit2_", "sigma_fit2_", "alpha95_fit2_")]
)
COMPARISONS = ["experiment", "policy_a", "policy_b", "ref_kind", "metric", "condition", "vcmd", "paired", "x_cmp", "reason", "diff_mean", "ci_lo", "ci_hi", "cliff_delta", "n_pairs"]
RELATIONS = ["policy", "x_metric", "y_metric", "spearman_rho", "ci_lo", "ci_hi", "n"]
SANITY = ["check", "condition", "vcmd", "value", "threshold", "result"]
PLOT_KEYS = [
    "x_grid", "y_on_grid", "y_median", "y_p05", "y_p95", "S_x", "sigma_y_x", "n_runs_x", "v_mean_x", "v_std_x",
    "phase_grid", "ang_L_mean", "ang_L_std", "ang_R_shift_mean", "ang_R_shift_std", "joint_names",
    "dphi_arm_leg_L", "dphi_arm_leg_R", "dphi_arms", "sym_x", "sym_x_names", "lat_y", "lat_y_names",
]
NUMERIC_PER_RUN = [c for c in PER_RUN if c not in ("policy", "condition", "vcmd", "distance_target", "run_idx", "term_type")]


def header(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip().split(",")


def missing(required, have):
    return [c for c in required if c not in have]


def run_analyze(with_twin):
    tmp = tempfile.mkdtemp(prefix="eval_schema_")
    res, out = os.path.join(tmp, "results"), os.path.join(tmp, "analysis")
    synth.build_dataset(res, with_twin=with_twin)
    A.analyze(res, out)
    return tmp, out


def test_schema():
    tmp, out = run_analyze(with_twin=True)
    tmp2, out2 = run_analyze(with_twin=False)
    try:
        errors = []
        per = A.read_csv(os.path.join(out, "per_run.csv"))
        errors += ["per_run.csv missing " + c for c in missing(PER_RUN, header(os.path.join(out, "per_run.csv")))]
        summ_h = header(os.path.join(out, "summary.csv"))
        errors += ["summary.csv missing " + c for c in missing(SUMMARY, summ_h)]
        for c in NUMERIC_PER_RUN:
            for suf in ("_mean", "_median", "_ci_lo", "_ci_hi"):
                if c + suf not in summ_h:
                    errors.append("summary.csv missing " + c + suf)
        errors += ["comparisons.csv missing " + c for c in missing(COMPARISONS, header(os.path.join(out, "comparisons.csv")))]
        errors += ["relations.csv missing " + c for c in missing(RELATIONS, header(os.path.join(out, "relations.csv")))]
        errors += ["sanity.csv missing " + c for c in missing(SANITY, header(os.path.join(out, "sanity.csv")))]
        for pol, cond in (("SYN_free_BASE", "v2.0_d55_nominal"), ("SYN_free_dup", "v2.0_d55_nominal"), ("T0_walk", "v1.0_d55_nominal")):
            z = np.load(os.path.join(out, pol, cond, "plot_data.npz"))
            errors += ["plot_data {}/{} missing {}".format(pol, cond, k) for k in PLOT_KEYS if k not in z.files]

        # twin rows are their own condition, never "nominal", and never in summary.csv
        conds = {(r["policy"], r["condition"]) for r in per}
        if ("SYN_free_BASE", "twin") not in conds or ("T0_walk", "twin") not in conds:
            errors.append("per_run.csv: twin rows must have condition = twin")
        summ = A.read_csv(os.path.join(out, "summary.csv"))
        if any(r["condition"] == "twin" for r in summ):
            errors.append("summary.csv must not have twin rows")

        # Non-viable: only the explicit diagnostic minimum is populated.
        keep = {
            "d_max", "t_end", "path_len_at_end", "path_len", "fell", "reach_path_9",
            "yaw_rate_ss", "yaw_rate_ss_deg",
        }
        bad = [r for r in per if r["policy"] == "SYN_free_dup"]
        for r in bad:
            for c in NUMERIC_PER_RUN:
                if c in keep:
                    continue
                if np.isfinite(r[c]):
                    errors.append("non-viable per_run {} = {} (must be NaN)".format(c, r[c]))
                    break
        srow = [r for r in summ if r["policy"] == "SYN_free_dup" and r["vcmd"] == 2.0][0]
        if srow["viable"] != 0 or srow["training_issue"] != 1 or not np.isfinite(srow["S_path_9"]) or np.isfinite(srow["beta_d9"]) or np.isfinite(srow["e_v_mean"]):
            errors.append("non-viable summary gate/mask is wrong")
        if not all(np.isfinite(srow[k]) for k in ("d_max_mean", "path_len_mean", "yaw_rate_ss_mean")):
            errors.append("non-viable summary must keep d_max, path_len and yaw_rate_ss")

        # viable policy has the metrics filled
        good = [r for r in per if r["policy"] == "SYN_free_BASE" and r["condition"] == "nominal" and r["vcmd"] == 2.0]
        for c in ("theta_d55", "theta_x95", "e_v", "duty_L", "si_duty", "em_all", "et_all", "twin_lat_d55", "twin_lat_x95", "dphi_arm_leg_L"):
            if not all(np.isfinite(r[c]) for r in good):
                errors.append("viable per_run column {} has NaN".format(c))

        # Circular trajectories exercise v_head, curvature, x95 and the short-x fallback.
        circ15 = [r for r in per if r["policy"] == "SYN_circle15"]
        moving15 = [r for r in circ15 if np.isfinite(r["curvature_ss"])]
        if not moving15 or abs(np.nanmean([r["e_v"] for r in moving15])) > 0.03:
            errors.append("circle15 e_v must be near zero with v_head")
        if abs(np.nanmean([r["yaw_rate_ss"] for r in moving15]) - 1.0 / 15.0) > 0.005:
            errors.append("circle15 yaw_rate_ss must be 1/15")
        if abs(np.nanmean([r["turn_radius"] for r in moving15]) - 15.0) > 0.5:
            errors.append("circle15 turn_radius must be 15 m")
        if not any(r["step_len_L"] > 0 for r in moving15):
            errors.append("circle15 step_len must be positive")
        if not all(np.isnan(r["curvature_ss"]) for r in circ15[:3]):
            errors.append("stopped runs inside a viable condition must have curvature_ss NaN")

        by_pol = {r["policy"]: r for r in summ if r["condition"] == "nominal" and r["vcmd"] == 1.0}
        c15, c5, c05 = by_pol["SYN_circle15"], by_pol["T0_walk_circle5"], by_pol["T0_walk_circle05"]
        if c15["viable"] != 1 or abs(c15["x95"] - 15.0) > 0.2 or not np.isfinite(c15["beta_x95"]):
            errors.append("circle15 must be viable with x95 about 15")
        if c5["viable"] != 1 or not (1.0 <= c5["x95"] < 9.0) or not np.isfinite(c5["beta_x95"]):
            errors.append("circle5 must be viable with filled _x95 below 9 m")
        if c05["viable"] != 1 or c05["x95"] >= 1.0 or c05["idx_rank_basis"] != "yaw_rate" or np.isfinite(c05["beta_x95"]):
            errors.append("circle0.5 must rank by yaw_rate and leave _x95 empty")
        short = [r for r in per if r["policy"] == "T0_walk_circle05"]
        if not all(r["insufficient_heading"] == 1 and np.isnan(r["theta_x95"]) for r in short):
            errors.append("circle0.5 must mark insufficient_heading")
        stopped = [r for r in summ if r["policy"] == "SYN_stopped"][0]
        if stopped["viable"] != 0 or stopped["S_path_9"] >= 0.95:
            errors.append("stopped policy must not be viable")

        # T0_walk: no comparison, no relation
        comp = A.read_csv(os.path.join(out, "comparisons.csv"))
        rel = A.read_csv(os.path.join(out, "relations.csv"))
        if any("T0_walk" in (r["policy_a"], r["policy_b"]) for r in comp):
            errors.append("T0_walk must not appear in comparisons.csv")
        if any(r["policy"] == "T0_walk" for r in rel):
            errors.append("T0_walk must not appear in relations.csv")
        if not comp:
            errors.append("comparisons.csv is empty (expected E1_free dup vs BASE)")
        if not rel:
            errors.append("relations.csv is empty")
        nom = [r for r in comp if r["experiment"] == "E1_free" and r["condition"] == "nominal" and not r.get("reason")]
        if not nom or not all(r["paired"] == 1 for r in nom):
            errors.append("nominal comparisons on the same robot must be paired")
        if not any(r.get("reason") == "reference_not_viable" and r["policy_b"] == "SYN_E3_bad_ref_none" for r in comp):
            errors.append("comparisons must explain a non-viable reference")
        if not any(r.get("reason") == "x_cmp_too_short" and r["policy_a"] == "SYN_free_phase" for r in comp):
            errors.append("comparisons must mark direction metrics when x_cmp is too short")

        # sanity.csv and T0 speed sweep schema.
        san = A.read_csv(os.path.join(out, "sanity.csv"))
        checks = [r["check"] for r in san]
        for need in ("S_path_9 >= 0.95", "S_d9 >= 0.95 (informativo)", "frac_fell <= 0.05", "n_did_not_start == 0", "frac(is_running = False) >= 0.95", "duty medio > 0.5", "|mean(e_v)| <= 0.1 m/s"):
            if need not in checks:
                errors.append("sanity.csv missing check " + need)
        twin_rows = [r for r in san if r["check"].startswith("gemeo")]
        if len(twin_rows) != 3 or any(r["result"] != "PASSA" for r in twin_rows):
            errors.append("sanity.csv twin check must PASSA on synthetic data: {}".format(twin_rows))
        fails = [r for r in san if r["result"] == "FALHA"]
        if fails:
            errors.append("sanity.csv rows FALHA on synthetic walk: {}".format([(r["check"], r["vcmd"], r["value"]) for r in fails]))
        sweep_h = header(os.path.join(out, "t0_speed_sweep.csv"))
        for need in ("S_path_9", "yaw_rate_ss_deg_mean", "x95_mean", "twin_turn_same_side_mean"):
            if need not in sweep_h:
                errors.append("t0_speed_sweep.csv missing " + need)

        # adding the twin runs must not change any nominal summary column except the twin ones
        summ2 = {(r["policy"], r["condition"], r["vcmd"], r["distance_target"]): r for r in A.read_csv(os.path.join(out2, "summary.csv"))}
        for r in summ:
            r2 = summ2[(r["policy"], r["condition"], r["vcmd"], r["distance_target"])]
            for k, v in r.items():
                if k.startswith(("et_all", "twin_lat_", "y_twin_", "twin_turn_")) or not isinstance(v, float):
                    continue
                v2 = r2[k]
                if not (np.isnan(v) and np.isnan(v2)) and v != v2:
                    errors.append("summary {} {} changed with twin runs: {} -> {}".format(r["policy"], k, v2, v))
                    break
        assert not errors, "\n".join(errors)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


if __name__ == "__main__":
    test_schema()
    print("test_schema: OK")
