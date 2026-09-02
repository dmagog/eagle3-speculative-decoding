# -*- coding: utf-8 -*-
"""Reconciles numbers hardcoded in the ENGLISH notebook's prose with key_numbers.json.
Usage: python3 tools_reconcile_en.py <path to key_numbers.json>"""
import json
import os
import sys

def _find(name, *dirs):
    """Same file sits differently in the working folder and in the repository;
    look under both layouts so the two copies of this tool stay identical."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for base in (here, root):
        for d in dirs:
            p = os.path.join(base, d, name)
            if os.path.exists(p):
                return p
    raise SystemExit("could not find " + name)


NB = _find("eagle3-qwen3-en.ipynb", "notebook_en", os.path.join("notebooks", "en"))

k = json.load(open(sys.argv[1], encoding="utf-8"))
nb = json.load(open(NB, encoding="utf-8"))
# the builder writes source as a string; a Kaggle download splits it into lines
_src = lambda c: c["source"] if isinstance(c["source"], str) else "".join(c["source"])
md = " ".join(" ".join(_src(c).split())
              for c in nb["cells"] if c["cell_type"] == "markdown")

ind, outd, ov = k["in_domain"], k["out_of_domain"], k["overall"]
by = k["by_set"]
mc_sets = [by[b] for b in by if b in ("GSM8K", "HumanEval")]
mc = sum(x["speedup"] for x in mc_sets) / len(mc_sets) if mc_sets else float("nan")

print("=== run results ===")
print(f"  overall     τ {ov['tau']:.2f} | {ov['speedup_naive']:.2f}x naive | {ov['speedup_hf']:.2f}x stock")
print(f"  in-domain   τ {ind['tau']:.2f} | {ind['speedup']:.2f}x")
print(f"  {outd.get('set','out'):11s} τ {outd['tau']:.2f} | {outd['speedup']:.2f}x")
for b, v in by.items():
    print(f"    {b:11s} τ {v['tau']:.2f} | {v['speedup']:.2f}x")
print(f"  math+code (GSM8K, HumanEval): {mc:.2f}x")
print(f"  branching: +{k['branching']['gain_pct']:.0f}%")

CLAIMS = [
    ("about 2.5× on math and code", 2.5, mc, 0.15),
    ("about 2.3× across the paper's four English benchmarks", 2.3, ind["speedup"], 0.07),
    ("about 0.95× on Russian prompts", 0.95, outd["speedup"], 0.03),
    ("acceptance length holds near 3.4", 3.4, ind["tau"], 0.12),
    ("drops to about 1.4", 1.4, outd["tau"], 0.12),
    ("two-and-a-half-fold gains on math and code", 2.5, mc, 0.15),
    ("one and a half tokens per cycle", 1.5, None, None),  # break-even: chart-computed, presence check only
]

print("\n=== reconciliation ===")
bad = 0
for phrase, claimed, actual, tol in CLAIMS:
    present = phrase in md
    if actual is None:
        print(f"  {'ok ' if present else 'MISSING PHRASE':14s} (presence only) | {phrase}")
        continue
    okv = abs(claimed - actual) <= tol
    if not okv:
        bad += 1
    note = "" if present else "   [phrase not found — re-check manually]"
    print(f"  {'ok ' if okv else 'MISMATCH':14s} claimed {claimed:5.2f} | measured {actual:5.2f} | {phrase}{note}")

# --- §11, the scaling probe -------------------------------------------------
# This section states eight numbers by hand, and its causal claim is a
# comparison of two growth rates — the place where a stale number does not look
# wrong, it argues the opposite case. Hence machine checks rather than reading.
sc = k.get("scaling")
if not sc:
    print("\n!! key_numbers.json carries no 'scaling' — §11 prose is UNVERIFIED")
    bad += 1
else:
    a, b = sc["m17"], sc["m4b"]
    step = [a["in"]["naive_ms"], b["in"]["naive_ms"]]
    tok = [a["in"]["eagle_ms"], b["in"]["eagle_ms"]]
    tau = [a["in"]["tau"], b["in"]["tau"]]
    cyc = [tok[0] * tau[0], tok[1] * tau[1]]
    pct = lambda v: 100 * (v[1] / v[0] - 1)

    print(f"\n=== §11 scaling ({a['in'].get('reps', '?')} passes per point) ===")
    print(f"  step   {step[0]:5.1f} -> {step[1]:5.1f} ms  ({pct(step):+5.1f}%)")
    print(f"  cycle  {cyc[0]:5.1f} -> {cyc[1]:5.1f} ms  ({pct(cyc):+5.1f}%)   [ms/token x tau]")
    print(f"  token  {tok[0]:5.1f} -> {tok[1]:5.1f} ms  ({pct(tok):+5.1f}%)")
    print(f"  tau     {tau[0]:4.2f} ->  {tau[1]:4.2f}     ({pct(tau):+5.1f}%)")
    for tag in ("in", "out"):
        print(f"  speedup {tag:3s}: {a[tag]['speedup']:.2f}x +-{a[tag].get('speedup_err', 0):.2f}"
              f"  ->  {b[tag]['speedup']:.2f}x +-{b[tag].get('speedup_err', 0):.2f}")

    SCALE_CLAIMS = [
        # only quantities that survived every run, including the one where §7 picked
        # a different depth and every absolute number moved with it
        ("about 9% higher on the 4B pair (in domain)", 9, pct(tau), 1.0),
        ("about 9% higher out of domain too", 9,
         100 * (b["out"]["tau"] / a["out"]["tau"] - 1), 2.0),
        ("the 9% extra tokens each cycle returns", 9, pct(tau), 1.0),
        # §11 argues the shorter Russian subset is not an easier one — it leans on
        # tau matching §5's, measured at a different tree, so check exactly that
        ("acceptance length barely moved: 1.38 in §5 against 1.36 here",
         outd["tau"], a["out"]["tau"], 0.05),
    ]
    print("\n=== §11 reconciliation ===")
    for phrase, claimed, actual, tol in SCALE_CLAIMS:
        okv = abs(claimed - actual) <= tol
        if not okv:
            bad += 1
        print(f"  {'ok ' if okv else 'MISMATCH':14s} claimed {claimed:6.2f} | measured {actual:6.2f} | {phrase}")

    print("\n=== §11 causal claim ===")
    for label, cond in (
        ("cycle grew faster than the ordinary step", pct(cyc) > pct(step)),
        ("tau rose with target size", tau[1] > tau[0]),
        # the text no longer claims a direction here — only that nothing resolvable moved
        ("speedup difference stays within a few percent either way",
         abs(100 * (b["in"]["speedup"] / a["in"]["speedup"] - 1)) < 5.0),
        ("out-of-domain stayed above 1.0 on both pairs",
         a["out"]["speedup"] > 1.0 and b["out"]["speedup"] > 1.0),
        ("greedy makes tau exactly reproducible across passes",
         max(a["in"]["tau_err"], b["in"]["tau_err"],
             a["out"]["tau_err"], b["out"]["tau_err"]) < 5e-4),
    ):
        if not cond:
            bad += 1
        print(f"  {'ok ' if cond else 'CLAIM BROKEN':14s} {label}")

# --- §8's cycle breakdown, quoted by hand in the takeaway --------------------------
cm = k.get("cycle_ms") or {}
if "verification" in cm and cm.get("total"):
    share = 100 * cm["verification"] / cm["total"]
    ratio = cm["verification"] / cm["naive_step"]
    print(f"\n=== §8 cycle breakdown ===")
    print(f"  verification {cm['verification']:.1f} of {cm['total']:.1f} ms = {share:.1f}% "
          f"| against a plain step: {ratio:.2f}x")
    for label, claimed, actual, tol in (
        ("verification takes about 80% of the cycle", 80, share, 4.0),
        ("costs almost exactly what a plain step costs", 1.0, ratio, 0.08),
    ):
        okv = abs(claimed - actual) <= tol
        if not okv:
            bad += 1
        print(f"  {'ok ' if okv else 'MISMATCH':14s} claimed {claimed:6.2f} | measured {actual:6.2f} | {label}")

# --- §6's hand-quoted acceptance shares --------------------------------------------
ac = k.get("acceptance")
if not ac:
    print("\n!! key_numbers.json carries no 'acceptance' — §6's shares are UNVERIFIED")
    bad += 1
else:
    al = ac["alpha"]
    print(f"\n=== §6 acceptance shares ({ac['n_cycles']} cycles) ===")
    print("  " + " | ".join(f"a>={d+1}: {v:.2f}" for d, v in enumerate(al)))
    for label, claimed, actual, tol in (
        ("only 72% of cycles get even one draft accepted", 0.72, al[0], 0.06),
        ("35% reach two", 0.35, al[1] if len(al) > 1 else -1, 0.06),
        ("7% reach four", 0.07, al[3] if len(al) > 3 else -1, 0.05),
    ):
        okv = abs(claimed - actual) <= tol
        if not okv:
            bad += 1
        print(f"  {'ok ' if okv else 'MISMATCH':14s} claimed {claimed:5.2f} | measured {actual:5.2f} | {label}")
    deepest = ac["max_accept_length"] == 4
    if not deepest:
        bad += 1
    print(f"  {'ok ' if deepest else 'CLAIM BROKEN':14s} four really is the deepest any cycle got "
          f"(measured {ac['max_accept_length']})")

# --- §5's "about ten percent faster": derived from the §7 sweep, quoted in §5 -------
ab = k.get("ablation")
if not ab:
    print("\n!! key_numbers.json carries no 'ablation' — §5's percentage is UNVERIFIED")
    bad += 1
else:
    d = {int(x): v["ms"] for x, v in ab["depth"].items()}
    best = min(d, key=lambda x: d[x])
    start = 7                                   # §5 runs at the 60/7/10 load-time shape
    lo, hi = max(x for x in d if x < start), min(x for x in d if x > start)
    interp = d[lo] + (d[hi] - d[lo]) * (start - lo) / (hi - lo)
    near = 100 * (d[hi if abs(hi - start) < abs(lo - start) else lo] / d[best] - 1)
    gain_lo = 100 * (d[lo] / d[best] - 1)
    gain_interp = 100 * (interp / d[best] - 1)
    print(f"\n=== §5 shape claim (sweep at total_token=64, top_k=10) ===")
    print(f"  best depth {best}: {d[best]:.2f} ms/tok | depth {lo}: {d[lo]:.2f} | depth {hi}: {d[hi]:.2f}")
    print(f"  gain over depth {lo}: {gain_lo:+.1f}% | interpolated depth {start}: {gain_interp:+.1f}%")
    # "about ten percent" must sit inside the measured bracket and not overstate it
    for label, cond in (
        ("\"about ten percent\" is inside the [depth-6, depth-7] bracket",
         gain_lo - 1.0 <= 10.0 <= gain_interp + 1.0),
        ("the quoted figure does not overstate the measured gain", 10.0 <= gain_interp),
        ("interpolated depth 7 exceeds the depth-6 figure", gain_interp > gain_lo),
    ):
        if not cond:
            bad += 1
        print(f"  {'ok ' if cond else 'CLAIM BROKEN':14s} {label}")

# --- §7's depth sweep on the out-of-domain set (what §5's slowdown rests on) --------
if not ab or "depth_ru" not in ab:
    print("\n!! no 'ablation.depth_ru' — the §5/§7 claim about depth is UNVERIFIED")
    bad += 1
else:
    nru = ab["naive_ms_ru"]
    ru = {int(kk): v for kk, v in ab["depth_ru"].items()}
    ds = sorted(ru)
    sp = {d: nru / ru[d]["ms"] for d in ds}
    taus = [ru[d]["tau"] for d in ds]
    print("\n=== §7 depth on the out-of-domain set ===")
    print("  " + " | ".join(f"d{d}: {sp[d]:.2f}x (tau {ru[d]['tau']:.2f})" for d in ds))
    for label, cond in (
        ("shallow depth is a win (> 1.0)", sp[ds[0]] > 1.0),
        ("deep tree is a loss (< 1.0)", sp[ds[-1]] < 1.0),
        ("the curve falls monotonically with depth",
         all(sp[ds[i]] >= sp[ds[i + 1]] - 0.02 for i in range(len(ds) - 1))),
        ("acceptance length barely depends on depth (spread < 0.15)",
         max(taus) - min(taus) < 0.15),
    ):
        if not cond:
            bad += 1
        print(f"  {'ok ' if cond else 'CLAIM BROKEN':14s} {label}")

print(f"\nmismatches: {bad}")
