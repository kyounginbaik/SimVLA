"""Rank a sim2ruin_named sysid json by tracking error and print per-joint gains.

  python scripts/simvla/systemid_report.py scripts/simvla/rby1_free_space.json [--top 0.1]

Prints, for every fitted joint: the best env's stiffness/damping, and the median over the
best `--top` fraction of survivors (a more robust estimate than the single winner). Also the
per-actuator-group medians in the form rby1.py's ImplicitActuatorCfg takes.

When the json carries `qpos_mse_joint` (one MSE per fitted joint per env) each joint is
ranked by ITS OWN error: every env is then an independent sample of that joint's 2-D (K, D)
space, instead of one sample of the joint-space product, which is what makes the estimate
sharp. Without it, all joints share the global ranking.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path, nargs="+", help="one json -> full report; several -> side-by-side compare")
    ap.add_argument("--top", type=float, default=0.1, help="fraction of best survivors to pool")
    ap.add_argument("--groups", nargs="*", default=["right_arm.*", "left_arm.*"],
                    help="actuator-group regexes to summarise")
    ap.add_argument("--strict", action="store_true",
                    help="identifiability test per joint per parameter instead of a top-pool median")
    ap.add_argument("--verdicts", default=None, help="--strict: write the verdicts to this json")
    ap.add_argument("--accepted", action="store_true",
                    help="for runs with a binding --qpos_thres: where do the surviving gains concentrate?")
    a = ap.parse_args()
    if a.accepted:
        accepted(a.json)
        return
    if a.strict:
        strict(a.json, a.verdicts)
        return
    if len(a.json) > 1:
        compare(a.json, a.top, a.groups)
        return
    a.json = a.json[0]

    d = json.loads(a.json.read_text())
    names = d["joint_names"]
    fit = d.get("fit_joints", names)
    S = np.asarray(d["stiffness"], dtype=np.float64)
    D = np.asarray(d["damping"], dtype=np.float64)
    mse = np.asarray(d["qpos_mse"], dtype=np.float64)
    n = mse.shape[0]
    if n == 0:
        raise SystemExit("no survivors in json")
    order = np.argsort(mse)
    k = max(1, int(round(a.top * n)))
    top = order[:k]
    best = order[0]
    print(f"{a.json}: {n} survivors, termination_t={d.get('termination_t')}")
    print(f"qpos_mse  best={mse[best]:.3e}  top{k}-median={np.median(mse[top]):.3e}  "
          f"all-median={np.median(mse):.3e}  rms(best)={np.sqrt(mse[best]):.4f} rad")

    col = {nm: i for i, nm in enumerate(names)}
    per_joint = d.get("qpos_mse_joint")
    MJ = np.asarray(per_joint, dtype=np.float64) if per_joint else None
    print("ranking: per-joint MSE" if MJ is not None else "ranking: global MSE (no qpos_mse_joint)")

    # {joint: (best env id, top-pool env ids)}, per-joint when available.
    pools = {}
    for jc, j in enumerate(fit):
        if MJ is not None:
            o = np.argsort(MJ[:, jc]); pools[j] = (o[0], o[:k], MJ[o[0], jc], np.median(MJ[o[:k], jc]))
        else:
            pools[j] = (best, top, mse[best], np.median(mse[top]))

    print(f"\n{'joint':14s} {'best_rms':>8s} {'best_K':>9s} {'best_D':>8s} | {'topK_med':>9s} {'topD_med':>8s} | "
          f"{'topK_iqr':>18s} {'topD_iqr':>16s}")
    for j in fit:
        i = col[j]
        b, tp, bm, _ = pools[j]
        ks, ds = S[tp, i], D[tp, i]
        print(f"{j:14s} {np.sqrt(bm):8.4f} {S[b, i]:9.1f} {D[b, i]:8.1f} | {np.median(ks):9.1f} {np.median(ds):8.1f} | "
              f"[{np.percentile(ks, 25):7.1f},{np.percentile(ks, 75):7.1f}] "
              f"[{np.percentile(ds, 25):6.1f},{np.percentile(ds, 75):6.1f}]")

    print("\nper-group medians of the per-joint top-pool medians (ImplicitActuatorCfg stiffness / damping):")
    for g in a.groups:
        js = [j for j in fit if re.fullmatch(g, j)]
        if not js:
            continue
        ks = [np.median(S[pools[j][1], col[j]]) for j in js]
        ds = [np.median(D[pools[j][1], col[j]]) for j in js]
        print(f"  {g:14s} stiffness={np.median(ks):.0f}  damping={np.median(ds):.0f}")



def compare(paths, top: float, groups):
    """Side-by-side top-pool medians per joint across several sysid jsons, plus their pooled median."""
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        S, D = np.asarray(d["stiffness"]), np.asarray(d["damping"])
        MJ = np.asarray(d["qpos_mse_joint"])
        col = {nm: i for i, nm in enumerate(d["joint_names"])}
        k = max(1, int(round(top * MJ.shape[0])))
        est = {}
        for jc, j in enumerate(d["fit_joints"]):
            tp = np.argsort(MJ[:, jc])[:k]
            est[j] = (np.median(S[tp, col[j]]), np.median(D[tp, col[j]]), np.sqrt(MJ[tp[0], jc]))
        runs.append((Path(p).stem, est))
    joints = [j for j in runs[0][1] if not j.startswith("torso")]
    hdr = "".join(f"{name[:18]:>26s}" for name, _ in runs)
    print(f"\n{'joint':12s}{hdr}{'pooled K / D':>18s}")
    pooled = {}
    for j in joints:
        row = ""
        for _, est in runs:
            K, Dd, r = est[j]
            row += f"{K:9.0f}/{Dd:5.0f} rms{r:6.3f}"
        Ks = [est[j][0] for _, est in runs]; Ds = [est[j][1] for _, est in runs]
        pooled[j] = (np.median(Ks), np.median(Ds))
        print(f"{j:12s}{row}{pooled[j][0]:11.0f}/{pooled[j][1]:5.0f}")
    print("\npooled per-group medians:")
    for g in groups:
        js = [j for j in joints if re.fullmatch(g, j)]
        if js:
            print(f"  {g:12s} stiffness={np.median([pooled[j][0] for j in js]):.0f} "
                  f"damping={np.median([pooled[j][1] for j in js]):.0f}")



# ---------------------------------------------------------------------------
# Strict identifiability
# ---------------------------------------------------------------------------
# A median of the best-fitting samples always returns a number, whether or not the
# objective can tell that number from any other. These thresholds decide when it can.
R_ACCEPT = 1.25        # a bin counts as "as good as the best" at <=1.25x the best profiled MSE
Q_PROFILE = 10         # profile statistic per bin: this percentile of the joint's own MSE
NBINS = 10
MAX_DECADES = 1.0      # IDENTIFIED requires the accepted band to be narrower than this
SAT_MAX = 0.25         # drop a sample whose joint was torque-saturated more than this fraction


def _profile(vals, mse, lo, hi):
    """Low quantile of `mse` per log-spaced bin of `vals`; profiles out every other parameter."""
    edges = np.linspace(np.log10(lo), np.log10(hi), NBINS + 1)
    lv = np.log10(np.clip(vals, lo, hi))
    prof = []
    for b in range(NBINS):
        m = (lv >= edges[b]) & ((lv < edges[b + 1]) if b < NBINS - 1 else (lv <= edges[b + 1]))
        prof.append(np.percentile(mse[m], Q_PROFILE) if m.sum() >= 5 else np.nan)
    return np.array(prof), edges


def _classify(prof, edges):
    """IDENTIFIED / bound / flat, from whether the accepted band is interior and narrow."""
    ok = ~np.isnan(prof)
    if ok.sum() < 4:
        return "no-data", None, 0.0
    best = np.nanmin(prof)
    acc = ok & (prof <= R_ACCEPT * best)
    idx = np.flatnonzero(acc)
    band = (10 ** edges[idx[0]], 10 ** edges[idx[-1] + 1])
    span = edges[idx[-1] + 1] - edges[idx[0]]
    rise = float(np.nanmax(prof[ok]) / best)
    lo_edge, hi_edge = idx[0] == 0, idx[-1] == NBINS - 1
    contiguous = bool(np.all(np.diff(idx) == 1))
    if lo_edge and hi_edge:
        return "flat", band, rise
    if hi_edge:
        return "lower-bound", band, rise
    if lo_edge:
        return "upper-bound", band, rise
    if span < MAX_DECADES and contiguous:
        return "IDENTIFIED", band, rise
    return "loose", band, rise


def strict(paths, out_json=None):
    """Per joint and parameter: is it identifiable, and do the runs agree?"""
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        S, D = np.asarray(d["stiffness"], float), np.asarray(d["damping"], float)
        col = {n: i for i, n in enumerate(d["joint_names"])}
        unsat = d.get("qpos_mse_joint_unsat")
        MJ = np.asarray(unsat if unsat else d["qpos_mse_joint"], float)
        SAT = np.asarray(d["sat_frac"], float) if "sat_frac" in d else None
        runs.append((Path(p).stem, d, S, D, col, MJ, SAT))
        print(f"{Path(p).stem}: {MJ.shape[0]} samples, "
              f"{'UNSATURATED-only MSE' if unsat else 'whole-episode MSE (no saturation data)'}")

    fit = [j for j in runs[0][1]["fit_joints"] if not j.startswith("torso")]
    verdicts = {}
    for prop, lo, hi in (("stiffness", 30, 20000), ("damping", 1, 1000)):
        print(f"\n=== {prop.upper()}")
        print(f"{'joint':13s} " + " ".join(f"{r[0][:28]:>34s}" for r in runs) + "   consensus")
        for jc, j in enumerate(fit):
            cells, bands, verds = [], [], []
            for name, d, S, D, col, MJ, SAT in runs:
                vals = (S if prop == "stiffness" else D)[:, col[j]]
                mse = MJ[:, jc]
                keep = ~np.isnan(mse)
                if SAT is not None:
                    keep &= SAT[:, jc] <= SAT_MAX
                if keep.sum() < 50:
                    cells.append(f"{'saturated out (n=%d)' % int(keep.sum()):>34s}")
                    verds.append("no-data"); bands.append(None); continue
                v, band, rise = _classify(*_profile(vals[keep], mse[keep], lo, hi))
                cells.append(f"{v:>13s} [{band[0]:7.0f},{band[1]:8.0f}] x{rise:5.1f}")
                verds.append(v); bands.append(band)
            # All runs identified AND (with more than one run) their bands overlap.
            if all(v == "IDENTIFIED" for v in verds) and None not in bands and (
                    len(bands) == 1 or (bands[0][0] < bands[1][1] and bands[1][0] < bands[0][1])):
                overlap = (max(b[0] for b in bands), min(b[1] for b in bands))
                cons = f"IDENTIFIED {overlap[0]:.0f}-{overlap[1]:.0f}"
                verdicts[(prop, j)] = ("identified", overlap)
            elif all(v in ("IDENTIFIED", "lower-bound") for v in verds) and None not in bands:
                lb = max(b[0] for b in bands)
                cons = f">= {lb:.0f} only"
                verdicts[(prop, j)] = ("lower-bound", (lb, hi))
            elif all(v in ("IDENTIFIED", "upper-bound") for v in verds) and None not in bands:
                ub = min(b[1] for b in bands)
                cons = f"<= {ub:.0f} only"
                verdicts[(prop, j)] = ("upper-bound", (lo, ub))
            else:
                cons = "NOT identified"
                verdicts[(prop, j)] = ("none", None)
            print(f"{j:13s} " + " ".join(cells) + f"   {cons}")

    n_id = sum(1 for v in verdicts.values() if v[0] == "identified")
    print(f"\nstrictly identified: {n_id} of {len(verdicts)} (joint, parameter) pairs "
          f"[accept {R_ACCEPT}x, band < {MAX_DECADES} decade, both runs agreeing, "
          f"saturation <= {SAT_MAX:.0%}]")
    if out_json:
        Path(out_json).write_text(json.dumps(
            {f"{prop}:{j}": {"verdict": v, "range": list(r) if r else None}
             for (prop, j), (v, r) in verdicts.items()}, indent=2) + "\n")
        print(f"wrote {out_json}")


# ---------------------------------------------------------------------------
# Accepted-set analysis (for runs made with a binding --qpos_thres)
# ---------------------------------------------------------------------------
# With a threshold that actually fires, every reset re-draws that env's gains, so the run is
# rejection sampling and the survivors are draws from the accepted region. What identifies a
# parameter here is not its error profile -- every survivor tracks well by construction -- but
# whether the survivors CONCENTRATE relative to the log-uniform prior they were drawn from.
PRIOR = {"stiffness": (30.0, 20000.0), "damping": (1.0, 1000.0)}
CONC_DECADES = 1.0     # accepted band narrower than this (p5..p95) => identified
CONC_SHRINK = 2.0      # ...and at least this many times narrower than the prior


def accepted(paths):
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        runs.append((Path(p).stem, d,
                     np.asarray(d["stiffness"], float), np.asarray(d["damping"], float),
                     {n: i for i, n in enumerate(d["joint_names"])}))
        gp = d.get("gain_prior")
        print(f"{Path(p).stem}: {len(d['qpos_mse'])} survivors, termination_t={d['termination_t']}"
              + (f", prior: {gp}" if gp else ""))

    fit = [j for j in runs[0][1]["fit_joints"] if not j.startswith("torso")]
    # Use the prior the run actually sampled from (recorded by systemid.py), not a default:
    # judging a 10..1000 sweep against a 30..20000 prior would call everything "concentrated".
    gp = runs[0][1].get("gain_prior")
    for prop in ("stiffness", "damping"):
        lo, hi = PRIOR[prop]
        if gp and prop in gp:
            lo, hi = gp[prop]
        elif gp and prop == "damping" and "zeta" in gp:
            print(f"\n(note: this run sampled the damping RATIO {gp['zeta']}; the raw-damping "
                  f"spread below is derived, so judge it against the zeta analysis instead)")
        prior_dec = np.log10(hi / lo)
        print(f"\n=== {prop.upper()}   (prior spans {prior_dec:.2f} decades, {lo:g}..{hi:g})")
        print(f"{'joint':13s} " + " ".join(f"{r[0][:26]:>32s}" for r in runs) + "   consensus")
        for j in fit:
            cells, iv, verds = [], [], []
            for name, d, S, D, col in runs:
                v = (S if prop == "stiffness" else D)[:, col[j]]
                if v.size < 30:
                    cells.append(f"{'too few survivors':>32s}"); verds.append(False); iv.append(None); continue
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                dec = np.log10(max(p95, 1e-9) / max(p5, 1e-9))
                ident = dec < CONC_DECADES and (prior_dec / max(dec, 1e-9)) >= CONC_SHRINK
                cells.append(f"{p50:8.0f} [{p5:7.0f},{p95:8.0f}] {dec:4.2f}dec{'*' if ident else ' '}")
                verds.append(ident); iv.append((p5, p95))
            if all(verds) and None not in iv and (len(iv) == 1 or (iv[0][0] < iv[1][1] and iv[1][0] < iv[0][1])):
                ov = (max(a[0] for a in iv), min(a[1] for a in iv))
                cons = f"IDENTIFIED {ov[0]:.0f}-{ov[1]:.0f}"
            elif any(verds):
                cons = "one run only"
            else:
                cons = "not concentrated"
            print(f"{j:13s} " + " ".join(cells) + f"   {cons}")
    print(f"\n* = survivors span < {CONC_DECADES} decade and are >= {CONC_SHRINK}x narrower than the prior")

if __name__ == "__main__":
    main()
