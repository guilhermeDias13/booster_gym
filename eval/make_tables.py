"""Write the evaluation tables for one experiment from eval/analysis.

Usage:
    py eval/make_tables.py --experiment E1_locked --out eval/reports/tables/E1_locked.tex

Numbers are read from summary.csv, per_run.csv, comparisons.csv, sanity.csv
and twin_check.csv. Nothing in the tables is typed in by hand.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(EVAL_DIR, "analysis")

SI_ROWS = (
    ("si_stance", "Stance time"),
    ("si_duty", "Duty factor"),
    ("si_step_len", "Step length"),
)
JOINTS = (
    ("Hip", (("si_peak_Hip_Pitch", "Pitch"), ("si_peak_Hip_Roll", "Roll"), ("si_peak_Hip_Yaw", "Yaw"))),
    ("Knee", (("si_peak_Knee_Pitch", "Pitch"),)),
    ("Ankle", (("si_peak_Ankle_Pitch", "Pitch"), ("si_peak_Ankle_Roll", "Roll"))),
)
DIFF_ROWS = (
    ("S_d9", "frac", r"$S(9)$"),
    ("e_v", "speed", r"Speed error $e_v$ (m/s)"),
    ("cv_v", "cv", r"Coefficient of variation $\mathrm{CV}_v$"),
    ("sigma_theta_d9", "angle", r"Angular dispersion $\sigma_\theta(9)$"),
    ("beta_d9", "angle", r"Mean heading angle $\beta(9)$"),
    ("yaw_rate_ss_deg", "rate", r"Yaw drift ($^\circ$/s)"),
    ("|si_stance|", "si", r"Stance time, $\lvert\mathrm{SI}\rvert$ (\%)"),
    ("|si_duty|", "si", r"Duty factor, $\lvert\mathrm{SI}\rvert$ (\%)"),
    ("em_legs", "em", r"$E_M$ (legs)"),
)


def fnum(text):
    if text is None or text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def finite(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def same(a, b):
    aa, bb = fnum(a), fnum(b)
    return finite(aa) and finite(bb) and abs(aa - bb) < 1e-9


def sci(v):
    mant, exp = "{:.1e}".format(v).split("e")
    sign = "-" if mant.startswith("-") else ""
    return "${}{} \\times 10^{{{}}}$".format(sign, mant.lstrip("-"), int(exp))


def tiny(v):
    return finite(v) and v != 0 and abs(v) < 1e-3


def wrap_neg(body, v):
    if finite(v) and v < 0 and not body.startswith("$"):
        return "$" + body + "$"
    return body


def fmt_count(v):
    if not finite(v):
        return None
    return str(int(round(float(v))))


def fmt_frac(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.3f}".format(abs(v)).rstrip("0").rstrip(".")
    if v < 0:
        return "$-" + body + "$"
    return body


def fmt_speed(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.3f}".format(abs(v))
    return "$-" + body + "$" if v < 0 else body


def fmt_pct(v):
    if not finite(v):
        return None
    body = "{:.1f}\\%".format(abs(v))
    return "$-" + body + "$" if v < 0 else body


def fmt_cv(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.3f}".format(abs(v))
    return "$-" + body + "$" if v < 0 else body


def fmt_td(v):
    """Times and distances: two decimals below 10, one decimal at or above 10."""
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    nd = 1 if abs(v) >= 10 else 2
    body = "{:.{}f}".format(abs(v), nd)
    return "$-" + body + "$" if v < 0 else body


def fmt_angle(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.1f}^\\circ".format(abs(v))
    return "$-" + body + "$" if v < 0 else "$" + body + "$"


def fmt_rate(v):
    """Degree per second. The rounding list does not name this unit; two decimals keep -0.30 distinct from -0.3."""
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.2f}".format(abs(v))
    return "$-" + body + "$" if v < 0 else body


def fmt_si(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    av = abs(float(v))
    if av == 0:
        body = "0"
    else:
        decimals = max(0, 2 - math.floor(math.log10(av)))
        body = "{:.{}f}".format(av, decimals)
    return "$-" + body + "$" if v < 0 else body


def fmt_em(v):
    if not finite(v):
        return None
    if tiny(v):
        return sci(v)
    body = "{:.3f}".format(abs(v))
    return "$-" + body + "$" if v < 0 else body


FORMATS = {
    "count": fmt_count,
    "frac": fmt_frac,
    "speed": fmt_speed,
    "pct": fmt_pct,
    "cv": fmt_cv,
    "td": fmt_td,
    "angle": fmt_angle,
    "rate": fmt_rate,
    "si": fmt_si,
    "em": fmt_em,
}


def apply_fmt(kind, v):
    text = FORMATS[kind](v)
    return text


class Sources:
    def __init__(self):
        self.rows = []

    def add(self, table, line, column, value, origin, origin_col, filt):
        self.rows.append({
            "table": table,
            "row": line,
            "column": column,
            "value": value,
            "file": origin,
            "source_column": origin_col,
            "filter": filt,
        })


class Gap:
    def __init__(self, tex, todo):
        self.tex = tex
        self.todo = todo


def missing(origin, column):
    return Gap("--", "{}:{}".format(origin, column))


def shown(text):
    return Gap(text, "")


def withheld():
    return Gap("--", "withheld:not_viable")


def col_filter(col):
    return "policy={};condition={};vcmd={};distance_target={}".format(
        col.policy, col.condition, col.vcmd, col.distance)


def value_of(row, key):
    if row is None or key not in row:
        return np.nan
    return fnum(row[key])


class Column:
    def __init__(self, header, policy, vcmd, distance, condition, group=""):
        self.header = header
        self.policy = policy
        self.vcmd = float(vcmd)
        self.distance = float(distance)
        self.condition = condition
        self.group = group
        self.summary = None
        self.arms = ""
        self.viable = True
        self.why = ""

    def bind(self, summary_rows):
        for row in summary_rows:
            if row["policy"] != self.policy or row["condition"] != self.condition:
                continue
            if same(row["vcmd"], self.vcmd) and same(row["distance_target"], self.distance):
                self.summary = row
                break
        if self.summary is None:
            self.viable = False
            self.why = "no summary row"
            self.arms = ""
            return
        self.arms = str(self.summary.get("arms") or "")
        issue = fnum(self.summary.get("training_issue"))
        path = fnum(self.summary.get("S_path_9"))
        if finite(issue) and issue != 0:
            self.viable = False
            self.why = "training_issue"
        elif not finite(path) or path < 0.95:
            self.viable = False
            self.why = "S_path_9 < 0.95"


def policies_of(kind, method):
    """Column lists. Each entry is one table (a list of Column)."""
    if kind == "T0":
        return [[
            Column("{:.1f}\\,m/s".format(v), "T0_walk", v, 55, "nominal")
            for v in (0.4, 0.7, 1.0)
        ]]
    if kind in ("E1_locked", "E1_free"):
        prefix = kind
        return [[
            Column(label, "{}_{}".format(prefix, name), 2.0, 55, "nominal")
            for label, name in (("BASE", "BASE"), ("DUP", "dup"), ("LOSS", "loss"), ("PHASE", "phase"))
        ]]
    if kind == "E2":
        blocks = []
        for methods in ((("BASE", "none"), ("DUP", "dup")), (("LOSS", "loss"), ("PHASE", "phase"))):
            cols = []
            for label, sym in methods:
                cols.append(Column("Locked", "E1_locked_{}".format("BASE" if sym == "none" else sym), 2.0, 55, "nominal", label))
                cols.append(Column("Free", "E1_free_{}".format("BASE" if sym == "none" else sym), 2.0, 55, "nominal", label))
            blocks.append(cols)
        return blocks
    if kind in ("E3_none", "E3_dup", "E3_loss", "E3_phase"):
        sym = kind.split("_", 1)[1]
        names = (
            ("None", "E3_ref_" + sym),
            ("Cycle", "E3_ciclo_" + sym),
            ("Spin", "E3_inercia_" + sym),
            ("Both", "E3_ambos_" + sym),
        )
        return [[Column(label, policy, 2.0, 55, "nominal") for label, policy in names]]
    if kind == "E3_obs":
        blocks = []
        for methods in ((("BASE", "none"), ("DUP", "dup")), (("LOSS", "loss"), ("PHASE", "phase"))):
            cols = []
            for label, sym in methods:
                free = "BASE" if sym == "none" else sym
                cols.append(Column("Without torque obs.", "E1_free_" + free, 2.0, 55, "nominal", label))
                cols.append(Column("With torque obs.", "E3_ref_" + sym, 2.0, 55, "nominal", label))
            blocks.append(cols)
        return blocks
    raise SystemExit("unknown experiment: {}".format(kind))


def block_suffix(kind, index, columns):
    if kind == "E2":
        return "_base_dup" if index == 0 else "_loss_phase"
    if kind == "E3_obs":
        return "_base_dup" if index == 0 else "_loss_phase"
    return ""


def put_header(lines, columns, label_cols=1):
    raw = prepare_headers(columns)
    prefix = "& " * label_cols
    for line in raw:
        if line.startswith("\\cmidrule"):
            lines.append("        " + line)
        else:
            lines.append("        " + prefix + line)


def header_lines(columns):
    groups = []
    for col in columns:
        if col.group and (not groups or groups[-1][0] != col.group):
            groups.append((col.group, 1))
        elif col.group:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
    if not any(col.group for col in columns):
        heads = " & ".join("\\textbf{{{}}}".format(col.header) for col in columns)
        return [heads + " \\\\"]
    top = []
    start = 2
    rules = []
    for name, width in groups:
        top.append("\\multicolumn{{{}}}{{c}}{{\\textbf{{{}}}}}".format(width, name))
        rules.append("\\cmidrule(lr){{{}-{}}}".format(start, start + width - 1))
        start += width
    second = " & ".join("\\textbf{{{}}}".format(col.header) for col in columns)
    return [" & ".join(top) + " \\\\", " ".join(rules), second + " \\\\"]


def ncol(columns, extra=0):
    return "l" * (1 + extra) + "c" * len(columns)


def group(text, n):
    return "\\multicolumn{{{}}}{{l}}{{\\textit{{{}}}}}".format(n, text) + " \\\\"


def row_line(label, gaps):
    bits = [label] + [g.tex for g in gaps]
    line = " & ".join(bits) + " \\\\"
    todos = [g.todo for g in gaps if g.todo]
    if todos:
        line += " % TODO " + "; ".join(todos)
    return line


def cell_from_summary(col, key, kind, allow, origin="summary.csv"):
    if not allow and not col.viable:
        return withheld()
    if col.summary is None:
        return missing(origin, key)
    value = value_of(col.summary, key)
    text = apply_fmt(kind, value)
    if text is None:
        return missing(origin, key)
    return shown(text)


def record(sources, table, line, col, gap, origin, origin_col):
    sources.add(table, line, col.header, gap.tex, origin, origin_col, col_filter(col))


def open_table(label, caption, columns, extra=0):
    lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\small",
        "    \\caption{" + caption + "}",
        "    \\label{" + label + "}",
        "    \\begin{tabular}{" + ncol(columns, extra) + "}",
        "        \\toprule",
    ]
    return lines


def close_table(lines):
    lines += ["        \\bottomrule", "    \\end{tabular}", "\\end{table}", ""]
    return lines


def indent(lines):
    out = []
    for line in lines:
        if line.startswith("\\begin{table}") or line.startswith("\\end{table}") or line == "":
            out.append(line)
        else:
            out.append("    " + line if not line.startswith("    ") else line)
    return out


def experiment_title(kind):
    names = {
        "T0": "the omnidirectional walking policy (T0)",
        "E1_locked": "the locked-arm sprint policies (E1)",
        "E1_free": "the free-arm sprint policies (E1)",
        "E2": "the arm comparison (E2)",
        "E3_none": "the torque-tool comparison for symmetry none (E3)",
        "E3_dup": "the torque-tool comparison for symmetry dup (E3)",
        "E3_loss": "the torque-tool comparison for symmetry loss (E3)",
        "E3_phase": "the torque-tool comparison for symmetry phase (E3)",
        "E3_obs": "the torque-observation comparison (E3)",
    }
    return names[kind]


def condition_phrase(kind, condition, vcmd, distance):
    if kind == "T0":
        return "{} condition, target of {:.0f}\\,m".format(condition, distance)
    return "{} condition, {:.1f}\\,m/s, target of {:.0f}\\,m".format(condition, vcmd, distance)


def viable_note(columns):
    bad = [col.header for col in columns if not col.viable]
    if not bad:
        return ""
    return " Columns marked with $\\dagger$ are not viable: fewer than 95\\% of the runs covered 9\\,m of path."


def header_tex(col):
    mark = "$^{\\dagger}$" if not col.viable else ""
    return col.header + mark


def prepare_headers(columns):
    saved = [(col.header, col.group) for col in columns]
    for col in columns:
        col.header = header_tex(col)
    lines = header_lines(columns)
    for col, (header, group_name) in zip(columns, saved):
        col.header = header
        col.group = group_name
    return lines


class Builder:
    def __init__(self, kind, columns, sources, condition, vcmd, distance, bundle):
        self.kind = kind
        self.columns = columns
        self.sources = sources
        self.condition = condition
        self.vcmd = vcmd
        self.distance = distance
        self.bundle = bundle
        self.title = experiment_title(kind)
        self.where = condition_phrase(kind, condition, vcmd, distance)
        self.note = viable_note(columns)
        self.any_free = any(col.arms == "free" for col in columns)

    def cap(self, lead):
        return lead + " " + self.title + ", " + self.where + "." + self.note

    def use(self, table, line, col, gap, origin, origin_col, allow=True):
        if not allow and not col.viable:
            gap = withheld()
        record(self.sources, table, line, col, gap, origin, origin_col)
        return gap

    def summary_gap(self, table, line, col, key, kind, allow=True):
        gap = cell_from_summary(col, key, kind, allow)
        if not allow and not col.viable:
            gap = withheld()
        return self.use(table, line, col, gap, "summary.csv", key, allow=True)

    def validation(self, label):
        lines = open_table(label, self.validation_caption(), self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        n = len(self.columns) + 1
        lines.append("        " + group("Completion", n).strip())
        lines.append("        " + self.one(label, r"Runs covering 9\,m of path, $S_{\mathrm{path}}(9)$", "S_path_9", "frac", True))
        lines.append("        " + self.falls(label, "Falls", True))
        lines.append("        " + self.one(label, "Runs that did not start", "n_did_not_start", "count", True))
        lines.append("        \\midrule")
        lines.append("        " + group("Gait classification", n).strip())
        walking = self.kind == "T0"
        gait_label = "Runs classified as walking" if walking else "Runs classified as running"
        lines.append("        " + self.gait(label, gait_label, walking))
        lines.append("        " + self.duty(label))
        lines.append("        " + self.one(label, "Flight fraction", "flight_frac_mean", "frac", False))
        lines.append("        \\midrule")
        lines.append("        " + group("Mirrored twin", n).strip())
        lines.append("        " + self.twin_err(label))
        return close_table(lines)

    def validation_caption(self):
        extra = ""
        if self.kind == "T0":
            extra = " " + self.bundle["sanity_sentence"]
        return self.cap("Validation of the evaluation framework with") + extra

    def one(self, table, label, key, kind, allow):
        gaps = [self.summary_gap(table, label, col, key, kind, allow) for col in self.columns]
        return row_line(label, gaps)

    def falls(self, table, label, allow):
        gaps = []
        for col in self.columns:
            if not allow and not col.viable:
                gap = withheld()
            elif col.summary is None:
                gap = missing("summary.csv", "n_fell_height+n_fell_speed")
            else:
                a, b = value_of(col.summary, "n_fell_height"), value_of(col.summary, "n_fell_speed")
                text = apply_fmt("count", a + b) if finite(a) and finite(b) else None
                gap = shown(text) if text is not None else missing("summary.csv", "n_fell_height+n_fell_speed")
            gaps.append(self.use(table, label, col, gap, "summary.csv", "n_fell_height+n_fell_speed"))
        return row_line(label, gaps)

    def gait(self, table, label, walking):
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.summary is None:
                gap = missing("summary.csv", "is_running_mean")
            else:
                mean = value_of(col.summary, "is_running_mean")
                if not finite(mean):
                    gap = missing("summary.csv", "is_running_mean")
                else:
                    frac = (1.0 - mean) if walking else mean
                    text = apply_fmt("pct", 100.0 * frac)
                    gap = shown(text) if text is not None else missing("summary.csv", "is_running_mean")
            origin_col = "1-is_running_mean" if walking else "is_running_mean"
            gaps.append(self.use(table, label, col, gap, "summary.csv", origin_col))
        return row_line(label, gaps)

    def duty(self, table):
        label = "Duty factor, left / right"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.summary is None:
                gap = missing("summary.csv", "duty_L_mean")
            else:
                left, right = apply_fmt("frac", value_of(col.summary, "duty_L_mean")), apply_fmt("frac", value_of(col.summary, "duty_R_mean"))
                if left is None or right is None:
                    gap = missing("summary.csv", "duty_L_mean/duty_R_mean")
                else:
                    gap = shown(left + " / " + right)
            gaps.append(self.use(table, label, col, gap, "summary.csv", "duty_L_mean/duty_R_mean"))
        return row_line(label, gaps)

    def twin_err(self, table):
        label = "Initial state, maximum error"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            else:
                found = None
                for row in self.bundle["twin"]:
                    if row["policy"] == col.policy and same(row["vcmd"], col.vcmd) and same(row["distance_target"], col.distance):
                        found = row
                        break
                if found is None:
                    gap = missing("twin_check.csv", "max_err")
                else:
                    text = apply_fmt("em", fnum(found["max_err"]))
                    gap = shown(text) if text is not None else missing("twin_check.csv", "max_err")
            gaps.append(self.use(table, label, col, gap, "twin_check.csv", "max_err"))
        return row_line(label, gaps)

    def q11(self, label):
        lines = open_table(label, self.cap("Q1.1: termination and reach of"), self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        n = len(self.columns) + 1
        lines.append("        " + group("Termination (runs)", n).strip())
        for key, text in (
            ("n_finished", "Reached 55\\,m"),
            ("n_timeout_moved", "At the time limit"),
        ):
            lines.append("        " + self.one(label, text, key, "count", True))
        lines.append("        " + self.falls(label, "Fell", True))
        lines.append("        " + self.one(label, "Did not start", "n_did_not_start", "count", True))
        lines.append("        \\midrule")
        lines.append("        " + group("Reach", n).strip())
        lines.append("        " + self.one(label, r"$d^{\max}$ (m)", "d_max_mean", "td", True))
        lines.append("        " + self.one(label, r"$S(9)$", "S_d9", "frac", True))
        lines.append("        " + self.one(label, r"$S(27.5)$", "S_d27", "frac", True))
        lines.append("        " + self.one(label, r"$S(55)$", "S_d55", "frac", True))
        return close_table(lines)

    def q12(self, label):
        caption = "Q1.2: speed of " + self.title + ", measured along the current heading of the robot, " + self.where + ". The time and the distance to 90\\% of the command are means over the runs that reach it." + self.note
        lines = open_table(label, caption, self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        n = len(self.columns) + 1
        lines.append("        " + group("Bias", n).strip())
        lines.append("        " + self.one(label, r"Mean speed $\bar{v}$ (m/s)", "v_mean_ss_mean", "speed", False))
        lines.append("        " + self.one(label, r"Speed error $e_v$ (m/s)", "e_v_mean", "speed", False))
        lines.append("        " + self.relative(label))
        lines.append("        \\midrule")
        lines.append("        " + group("Constancy", n).strip())
        lines.append("        " + self.one(label, r"Coefficient of variation $\mathrm{CV}_v$", "cv_v_mean", "cv", False))
        lines.append("        \\midrule")
        lines.append("        " + group("Start", n).strip())
        lines.append("        " + self.reached90(label))
        lines.append("        " + self.one(label, r"Time to 90\% of the command (s)", "t90_mean", "td", False))
        lines.append("        " + self.one(label, r"Distance to 90\% of the command (m)", "d90_mean", "td", False))
        return close_table(lines)

    def relative(self, label):
        text = "Relative to the command"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.summary is None or not finite(col.vcmd) or col.vcmd == 0:
                gap = missing("summary.csv", "e_v_mean/vcmd")
            else:
                err = value_of(col.summary, "e_v_mean")
                shown_text = apply_fmt("pct", 100.0 * err / col.vcmd) if finite(err) else None
                gap = shown(shown_text) if shown_text is not None else missing("summary.csv", "e_v_mean/vcmd")
            gaps.append(self.use(label, text, col, gap, "summary.csv", "100*e_v_mean/vcmd"))
        return row_line(text, gaps)

    def reached90(self, label):
        text = r"Runs reaching 90\% of the command"
        gaps = []
        for col in self.columns:
            key = (col.policy, col.condition, col.vcmd, col.distance)
            if not col.viable:
                gap = withheld()
            elif col.summary is None:
                gap = missing("per_run.csv", "reached90")
            else:
                gap = shown(fmt_count(self.bundle["reached90"].get(key, 0)))
            gaps.append(self.use(label, text, col, gap, "per_run.csv", "sum(reached90=1)"))
        return row_line(text, gaps)

    def q13(self, label):
        lines = open_table(label, self.cap("Q1.3: precision of the heading of") + " The metrics at $x_{95}$ are computed at the largest forward distance reached by 95\\% of the runs.", self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        n = len(self.columns) + 1
        lines.append("        " + group("At 9\\,m", n).strip())
        lines.append("        " + self.one(label, r"Angular dispersion $\sigma_\theta(9)$", "sigma_theta_d9", "angle", False))
        lines.append("        \\midrule")
        lines.append("        " + group("At $x_{95}$", n).strip())
        lines.append("        " + self.one(label, r"$x_{95}$ (m)", "x95", "td", False))
        lines.append("        " + self.one(label, r"Cone half-angle $\alpha_{95}(x_{95})$", "alpha95_x95", "angle", False))
        lines.append("        " + self.one(label, r"Lateral corridor $w_{95}(x_{95})$ (m)", "w95_x95", "td", False))
        lines.append("        " + self.one(label, r"Lateral dispersion $\sigma_y(x_{95})$ (m)", "sigma_y_x95", "td", False))
        return close_table(lines)

    def q14(self, label):
        caption = (
            self.cap("Q1.4: accuracy of the heading of")
            + " A positive yaw drift is a turn to the left."
            + " The mean heading angle is taken at the first instant the forward distance reaches 9\\,m."
            + " The yaw drift and the turning radius are medians over the steady-state window:"
            + " from the instant the speed reaches 90\\% of the command until the run ends,"
            + " and only when that window lasts at least 2\\,s."
        )
        lines = open_table(label, caption, self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        lines.append("        " + self.one(label, r"Mean heading angle $\beta(9)$", "beta_d9", "angle", False))
        lines.append("        \\midrule")
        lines.append("        " + self.one(label, r"Yaw drift ($^\circ$/s)", "yaw_rate_ss_deg_median", "rate", False))
        lines.append("        " + self.one(label, "Turning radius (m)", "turn_radius_median", "td", False))
        return close_table(lines)

    def q21(self, label):
        half, where = self.widest_si()
        half_text = fmt_si(half) if half is not None else "--"
        caption = (
            "Q2.1: symmetry of " + self.title + ", " + self.where + ". "
            "The symmetry index $\\mathrm{SI} = 100\\,(X_R - X_L)/\\big(\\tfrac{1}{2}(X_R + X_L)\\big)$ "
            "is computed in the steady-state window of each run; positive values mean that the right side is larger. "
            "The largest 95\\% confidence-interval half-width among these symmetry indices is "
            + half_text + " percentage points, on " + where + ". "
            "$E_M$ uses the commanded period $1/f_{\\mathrm{gait}}$. "
            "The detected period is the median stride after contacts closer than $0.6/f_{\\mathrm{gait}}$ are fused into the current stance."
            + self.note
        )
        lines = open_table(label, caption, self.columns, extra=1)
        put_header(lines, self.columns, label_cols=2)
        lines.append("        \\midrule")
        n = len(self.columns) + 2
        lines.append("        " + group("Gait (SI, \\%)", n).strip())
        for key, text in SI_ROWS:
            lines.append("        " + self.si_row(label, text, key))
        lines.append("        \\midrule")
        lines.append("        " + group("Joint amplitude (SI, \\%)", n).strip())
        first_joint = True
        for joint, axes in JOINTS:
            if not first_joint:
                lines.append("        \\cmidrule(lr){1-" + str(n) + "}")
            first_joint = False
            for i, (key, axis) in enumerate(axes):
                left = joint if i == 0 else ""
                lines.append("        " + self.si_row(label, axis, key, left))
        lines.append("        \\midrule")
        lines.append("        " + group("Mirror error", n).strip())
        lines.append("        " + self.si_row(label, r"$E_M$ (legs)", "em_legs_mean", kind="em"))
        if self.any_free:
            lines.append("        " + self.arm_or_leg(label))
        lines.append("        " + self.si_row(label, "Detected period (s)", "period_detected_mean", kind="em"))
        lines.append("        " + self.si_row(label, "Extra contacts per stride", "extra_contacts_per_stride_mean", kind="em"))
        lines.append("        \\midrule")
        lines.append("        " + group("Mirrored twin (secondary)", n).strip())
        lines.append("        " + self.same_side(label))
        lines.append("        " + self.twin_yaw(label))
        lines.append("        " + self.si_row(label, r"$E_T$", "et_all_mean", kind="em"))
        lines.append("        " + self.si_row(label, r"$\lvert y' + y \rvert$ at 9\,m (m)", "twin_lat_d9_mean", kind="td"))
        return close_table(lines)

    def si_row(self, table, text, key, left="", kind="si"):
        mean_key = key if key.endswith("_mean") else key + "_mean"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            else:
                gap = cell_from_summary(col, mean_key, kind, True)
            gaps.append(self.use(table, text, col, gap, "summary.csv", mean_key))
        prefix = (left + " & " + text) if left else ("& " + text)
        return row_line(prefix, gaps)

    def arm_or_leg(self, label):
        text = r"$E_M$ (arms)"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.arms != "free":
                gap = missing("summary.csv", "em_arms_mean")
            else:
                gap = cell_from_summary(col, "em_arms_mean", "em", True)
            gaps.append(self.use(label, text, col, gap, "summary.csv", "em_arms_mean"))
        return row_line("& " + text, gaps)

    def widest_si(self):
        best = None
        where = "no finite interval"
        keys = [key for key, _ in SI_ROWS] + [key for _, axes in JOINTS for key, _ in axes]
        names = dict(SI_ROWS)
        for _, axes in JOINTS:
            for key, axis in axes:
                names[key] = axis
        for col in self.columns:
            if not col.viable or col.summary is None:
                continue
            for key in keys:
                lo, hi = value_of(col.summary, key + "_ci_lo"), value_of(col.summary, key + "_ci_hi")
                if not (finite(lo) and finite(hi)):
                    continue
                half = 0.5 * (hi - lo)
                if best is None or half > best:
                    best = half
                    where = names.get(key, key) + " for " + col.header
        return best, where

    def same_side(self, label):
        text = "Turns to the same side as the original"
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.summary is None:
                gap = missing("summary.csv", "twin_turn_same_side")
            else:
                value = value_of(col.summary, "twin_turn_same_side")
                if not finite(value):
                    gap = missing("summary.csv", "twin_turn_same_side")
                else:
                    gap = shown("yes" if value == 1 else "no")
            gaps.append(self.use(label, text, col, gap, "summary.csv", "twin_turn_same_side"))
        return row_line("& " + text, gaps)

    def twin_yaw(self, label):
        text = r"Yaw drift of the twin ($^\circ$/s)"
        gaps = []
        for col in self.columns:
            key = (col.policy, "twin", col.vcmd, col.distance)
            if not col.viable:
                gap = withheld()
            elif key not in self.bundle["twin_yaw"]:
                gap = missing("per_run.csv", "yaw_rate_ss_deg")
            else:
                text_v = apply_fmt("rate", self.bundle["twin_yaw"][key])
                gap = shown(text_v) if text_v is not None else missing("per_run.csv", "yaw_rate_ss_deg")
            gaps.append(self.use(label, text, col, gap, "per_run.csv", "median(yaw_rate_ss_deg)"))
        return row_line("& " + text, gaps)

    def q22(self, label):
        caption = (
            "Q2.2: arm phase of " + self.title + ", " + self.where + ". "
            "Phase differences are circular means, in degrees, over the steady-state window. "
            "The left arm--leg phase is the phase of the left shoulder pitch minus the phase of the left hip pitch; "
            "the right arm--leg phase uses the right shoulder and the right hip. "
            "The inter-arm phase is the left shoulder pitch minus the right shoulder pitch. "
            "$0^\\circ$ means the two joints move together and $\\pm 180^\\circ$ means they are in opposition. "
            "$R$ is the mean resultant length of that phase difference inside each run, "
            "from 0 (no consistent relation) to 1 (a constant phase difference); "
            "the tabulated $R$ is the mean of those values across runs. "
            "Columns with locked arms are left blank."
            + self.note
        )
        lines = open_table(label, caption, self.columns)
        put_header(lines, self.columns)
        lines.append("        \\midrule")
        n = len(self.columns) + 1
        lines.append("        " + group("Arm--leg phase", n).strip())
        lines.append("        " + self.circ(label, r"Left arm--leg phase ($^\circ$)", "dphi_arm_leg_L"))
        lines.append("        " + self.circ(label, r"Right arm--leg phase ($^\circ$)", "dphi_arm_leg_R"))
        lines.append("        " + self.locked_or(label, r"Left arm--leg $R$", "R_arm_leg_L_mean", "frac"))
        lines.append("        " + self.locked_or(label, r"Right arm--leg $R$", "R_arm_leg_R_mean", "frac"))
        lines.append("        \\midrule")
        lines.append("        " + group("Arms", n).strip())
        lines.append("        " + self.circ(label, r"Inter-arm phase ($^\circ$)", "dphi_arms"))
        lines.append("        " + self.locked_or(label, r"Inter-arm $R$", "R_arms_mean", "frac"))
        lines.append("        \\midrule")
        lines.append("        " + group("Frequency and amplitude", n).strip())
        lines.append("        " + self.locked_or(label, r"Left frequency ratio", "rho_freq_L_mean", "frac"))
        lines.append("        " + self.locked_or(label, r"Right frequency ratio", "rho_freq_R_mean", "frac"))
        lines.append("        " + self.amp(label, "Left shoulder amplitude", "amp_shoulder_L"))
        lines.append("        " + self.amp(label, "Right shoulder amplitude", "amp_shoulder_R"))
        lines.append("        \\midrule")
        lines.append("        " + group("Mirror error of the arms", n).strip())
        lines.append("        " + self.locked_or(label, r"$E_M$ (arms)", "em_arms_mean", "em"))
        return close_table(lines)

    def locked_or(self, table, text, key, kind):
        gaps = []
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.arms != "free":
                gap = missing("summary.csv", key)
            else:
                gap = cell_from_summary(col, key, kind, True)
            gaps.append(self.use(table, text, col, gap, "summary.csv", key))
        return row_line(text, gaps)

    def circ(self, table, text, key):
        gaps = []
        for col in self.columns:
            store = (col.policy, col.condition, col.vcmd, col.distance, key)
            if not col.viable:
                gap = withheld()
            elif col.arms != "free":
                gap = missing("per_run.csv", key)
            elif store not in self.bundle["circ"]:
                gap = missing("per_run.csv", key)
            else:
                text_v = apply_fmt("angle", self.bundle["circ"][store])
                gap = shown(text_v) if text_v is not None else missing("per_run.csv", key)
            gaps.append(self.use(table, text, col, gap, "per_run.csv", "circular_mean(" + key + ")"))
        return row_line(text, gaps)

    def amp(self, table, text, key):
        gaps = []
        mean_key = key + "_mean"
        for col in self.columns:
            if not col.viable:
                gap = withheld()
            elif col.arms != "free":
                gap = missing("summary.csv", mean_key)
            elif col.summary is None:
                gap = missing("summary.csv", mean_key)
            else:
                rad = value_of(col.summary, mean_key)
                text_v = apply_fmt("angle", np.degrees(rad)) if finite(rad) else None
                gap = shown(text_v) if text_v is not None else missing("summary.csv", mean_key)
            gaps.append(self.use(table, text, col, gap, "summary.csv", mean_key + " in degrees"))
        return row_line(text, gaps)

    def diffs(self, label, ref):
        others = [col for col in self.columns if col.policy != ref.policy]
        caption = (
            "Differences: each column is that policy minus " + ref.header
            + " for " + self.title + ", " + self.where + ". "
            "A paired cell is the mean difference and its 95\\% confidence interval. "
            "An unpaired cell is Cliff's $\\delta$."
            + self.note
        )
        lines = open_table(label, caption, others)
        saved = []
        for col in others:
            saved.append(col.header)
            col.header = col.header + " $-$ " + ref.header
        put_header(lines, others)
        for col, header in zip(others, saved):
            col.header = header
        lines.append("        \\midrule")
        for key, kind, text in DIFF_ROWS:
            lines.append("        " + self.diff_row(label, text, key, kind, others, ref))
        return close_table(lines)

    def diff_row(self, table, text, key, kind, others, ref):
        metric = "yaw_rate_ss" if key == "yaw_rate_ss_deg" else key
        gaps = []
        for col in others:
            if (not col.viable) or (not ref.viable):
                gap = withheld()
            else:
                found = None
                expected = "E3" if self.kind.startswith("E3_") and self.kind != "E3_obs" else self.kind
                for row in self.bundle["comps"]:
                    if row["experiment"] != expected:
                        continue
                    if row["policy_a"] != col.policy or row["policy_b"] != ref.policy:
                        continue
                    if row["metric"] != metric or row["condition"] != col.condition:
                        continue
                    if not (same(row["vcmd"], col.vcmd) and same(row["distance_target"], col.distance)):
                        continue
                    found = row
                    break
                gap = self.format_diff(found, kind, key)
            origin_col = metric + ("*180/pi" if key == "yaw_rate_ss_deg" else "")
            gaps.append(self.use(table, text, col, gap, "comparisons.csv", origin_col))
        return row_line(text, gaps)

    def format_diff(self, row, kind, key):
        if row is None:
            return missing("comparisons.csv", key)
        reason = (row.get("reason") or "").strip()
        if reason:
            return missing("comparisons.csv", key + " reason=" + reason)
        scale = 180.0 / math.pi if key == "yaw_rate_ss_deg" else 1.0
        if fnum(row.get("paired")) == 1:
            diff, lo, hi = fnum(row["diff_mean"]) * scale, fnum(row["ci_lo"]) * scale, fnum(row["ci_hi"]) * scale
            a, b, c = apply_fmt(kind, diff), apply_fmt(kind, lo), apply_fmt(kind, hi)
            if a is None or b is None or c is None:
                return missing("comparisons.csv", key)
            return shown("{} [{}, {}]".format(a, b, c))
        cliff = apply_fmt("frac", fnum(row["cliff_delta"]))
        if cliff is None:
            return missing("comparisons.csv", "cliff_delta")
        inner = cliff[1:-1] if cliff.startswith("$") and cliff.endswith("$") else cliff
        return shown("$\\delta = " + inner + "$")


def circular_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    ang = np.deg2rad(arr)
    return float(np.degrees(np.angle(np.mean(np.exp(1j * ang)))))


def load_bundle(summary, columns, analysis):
    need = {(col.policy, col.vcmd, col.distance) for col in columns}
    policies = {col.policy for col in columns}
    reached = {}
    yaw = defaultdict(list)
    circ_vals = defaultdict(list)
    phase_keys = ("dphi_arm_leg_L", "dphi_arm_leg_R", "dphi_arms")
    with open(os.path.join(analysis, "per_run.csv"), "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["policy"] not in policies:
                continue
            vcmd, dist = fnum(row["vcmd"]), fnum(row["distance_target"])
            if not any(row["policy"] == p and same(vcmd, v) and same(dist, d) for p, v, d in need):
                continue
            if row["condition"] == "nominal" and fnum(row["reached90"]) == 1:
                reached[(row["policy"], "nominal", vcmd, dist)] = reached.get((row["policy"], "nominal", vcmd, dist), 0) + 1
            if row["condition"] == "twin":
                yaw[(row["policy"], "twin", vcmd, dist)].append(fnum(row["yaw_rate_ss_deg"]))
            if row["condition"] == "nominal":
                for key in phase_keys:
                    circ_vals[(row["policy"], "nominal", vcmd, dist, key)].append(fnum(row[key]))
    twin_yaw = {key: float(np.nanmedian(np.asarray(vals, dtype=np.float64))) for key, vals in yaw.items()}
    circ = {key: circular_mean(vals) for key, vals in circ_vals.items()}
    twin = load_csv(os.path.join(analysis, "twin_check.csv"))
    comps = load_csv(os.path.join(analysis, "comparisons.csv"))
    sanity_sentence = ""
    sanity_path = os.path.join(analysis, "sanity.csv")
    if os.path.isfile(sanity_path):
        checks = load_csv(sanity_path)
        n_pass = sum(1 for row in checks if row["result"] == "PASSA")
        n_info = sum(1 for row in checks if row["result"] == "INFO")
        n_fail = sum(1 for row in checks if row["result"] == "FALHA")
        if n_fail == 0:
            sanity_sentence = "All {} checks of the sanity test passed ({} pass/fail checks and {} informative values), including those of the 9\\,m runs at 0.7\\,m/s.".format(len(checks), n_pass, n_info)
        else:
            sanity_sentence = "The sanity test has {} failures out of {} checks.".format(n_fail, len(checks))
    return {
        "reached90": reached,
        "twin_yaw": twin_yaw,
        "circ": circ,
        "twin": twin,
        "comps": comps,
        "sanity_sentence": sanity_sentence,
    }


def ref_column(kind, columns):
    if kind in ("E1_locked", "E1_free"):
        for col in columns:
            if col.header == "BASE":
                return col
    if kind.startswith("E3_") and kind != "E3_obs":
        for col in columns:
            if col.header == "None":
                return col
    if kind == "E3_obs":
        return None
    return None


def warn_incoherent(columns):
    notes = []
    for col in columns:
        row = col.summary
        if row is None:
            notes.append("{} has no summary row at vcmd={} distance={}.".format(col.policy, col.vcmd, col.distance))
            continue
        n = fnum(row.get("n_runs"))
        parts = [fnum(row.get(k)) for k in ("n_finished", "n_timeout_moved", "n_fell_height", "n_fell_speed", "n_did_not_start")]
        if finite(n) and all(finite(p) for p in parts) and abs(sum(parts) - n) > 0.5:
            notes.append("{} termination counts sum to {} but n_runs is {}.".format(col.policy, sum(parts), n))
        falls = fnum(row.get("n_fell_height")) + fnum(row.get("n_fell_speed"))
        path = fnum(row.get("S_path_9"))
        if finite(path) and path == 1 and finite(falls) and falls > 0:
            notes.append("{} has S_path_9 = 1 and Fell = {}.".format(col.policy, falls))
        for key in ("si_step_len", "si_stance", "si_duty"):
            lo, hi = fnum(row.get(key + "_ci_lo")), fnum(row.get(key + "_ci_hi"))
            if finite(lo) and finite(hi) and (hi - lo) > 20:
                notes.append("{} {} 95% interval is [{}, {}], half-width {}.".format(
                    col.policy, key, lo, hi, 0.5 * (hi - lo)))
    return notes


def write_sources(path, sources):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["table", "row", "column", "value", "file", "source_column", "filter"])
        writer.writeheader()
        for row in sources.rows:
            writer.writerow(row)


def build(kind, condition, vcmd, distance, analysis):
    summary = load_csv(os.path.join(analysis, "summary.csv"))
    blocks = policies_of(kind, None)
    for block in blocks:
        for col in block:
            if kind != "T0":
                col.vcmd = float(vcmd)
            col.distance = float(distance)
            col.condition = condition
            col.bind(summary)
    bundle = load_bundle(summary, [col for block in blocks for col in block], analysis)
    sources = Sources()
    chunks = ["% Preamble: \\usepackage{booktabs}", ""]
    notes = []
    for index, columns in enumerate(blocks):
        suffix = block_suffix(kind, index, columns)
        tag = kind.lower() + suffix
        built = Builder(kind, columns, sources, condition, vcmd, distance, bundle)
        chunks += built.validation("tab:" + tag + "_validation")
        chunks += built.q11("tab:" + tag + "_q11")
        chunks += built.q12("tab:" + tag + "_q12")
        chunks += built.q13("tab:" + tag + "_q13")
        chunks += built.q14("tab:" + tag + "_q14")
        chunks += built.q21("tab:" + tag + "_q21")
        if built.any_free:
            chunks += built.q22("tab:" + tag + "_q22")
        ref = ref_column(kind, columns)
        if ref is not None and kind != "T0":
            chunks += built.diffs("tab:" + tag + "_diff", ref)
        notes += warn_incoherent(columns)
    return "\n".join(chunks).rstrip() + "\n", sources, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--vcmd", type=float, default=2.0)
    parser.add_argument("--condition", default="nominal", choices=("nominal", "perturbed"))
    parser.add_argument("--distance", type=float, default=55.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis", default=ANALYSIS)
    args = parser.parse_args()
    text, sources, notes = build(args.experiment, args.condition, args.vcmd, args.distance, args.analysis)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    source_path = os.path.splitext(args.out)[0] + "_sources.csv"
    write_sources(source_path, sources)
    print(args.out)
    print(source_path)
    for note in notes:
        print("INCOHERENT:", note)


if __name__ == "__main__":
    main()
