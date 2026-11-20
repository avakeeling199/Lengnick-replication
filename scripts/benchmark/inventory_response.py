"""
Does each model's pricing respond to inventory? Three checks on the raw replay answers.

1. Contrasts with bootstrap CIs
     raise contrast = P(raise | understocked) - P(raise | overstocked)
     cut contrast   = P(cut | overstocked)    - P(cut | understocked)
   Understocked situations come from the quality set, overstocked ones from the overstock set
   (plus any overstocked quality prompt). CIs resample *situations* (all repeats of a
   situation together), because repeats of one prompt are near-identical and are not
   independent observations.

2. Multinomial logit of the action (raise / hold / cut, base = hold) on everything the prompt
   shows: log inventory, log units sold, price, marginal cost and last month's changes in
   price, sales and inventory. Standard errors are clustered by situation. Reported as the
   average marginal effect of a one-standard-deviation rise in log inventory on P(raise) and
   P(cut), holding the other inputs fixed. Holding log sales fixed, this is the effect of
   inventory relative to sales.

3. Jitter: how often the price moves at all within the band, and cuts when understocked.

Writes to --out-dir:
  inventory_contrasts.csv   one row per model x prompt x contrast, with 95% CI
  inventory_logit.csv       one row per model x prompt x outcome (raise, cut): AME with 95% CI
  jitter.csv                one row per model x prompt
  inventory_<prompt>.pdf    contrasts and logit effects for one prompt (same axes in every file)

Run from the repository root:
  python scripts/benchmark/inventory_response.py
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analyse import add_prompt_info, classify, STYLE, MODEL_COLOURS

PROMPTS = ["strict", "rule_context", "goal_state"]
MODELS = ["llama70b", "qwen38_27b-fp8", "gemma4_31b_q4", "qwen3_32b_awq", "qwen3_30b_a3b", "qwen35_9b", "qwen35_9b_think"]
FEATURES = ["log_inventory", "log_units_sold", "price", "marginal_cost",
            "d_price", "d_units_sold", "d_inventory", "has_history"]


def situation_features(prompts):
    """One row per situation with everything the prompt shows, as model inputs."""
    def last(x):
        return x[-1] if isinstance(x, list) and len(x) else np.nan

    def last_change(x):
        # the price history ends with the current price, so last month's change is the
        # difference between its last two entries
        return x[-1] - x[-2] if isinstance(x, list) and len(x) >= 2 else np.nan
    f = pd.DataFrame({
        "prompt_id": prompts["prompt_id"],
        "inv_state": np.where(prompts["i_f"] < prompts["i_f_lowerbar"], "under",
                              np.where(prompts["i_f"] > prompts["i_f_upperbar"], "over", "within")),
        "log_inventory": np.log1p(prompts["i_f"].astype(float)),
        "log_units_sold": np.log1p(prompts["demand"].astype(float)),
        "price": prompts["p_f"].astype(float),
        "marginal_cost": prompts["mc_f"].astype(float),
        "d_price": prompts["price_hist"].map(last_change),
        "d_units_sold": prompts["demand"] - prompts["demand_hist"].map(last),
        "d_inventory": prompts["i_f"] - prompts["inv_hist"].map(last),
    })
    f["has_history"] = prompts["price_hist"].map(lambda x: isinstance(x, list) and len(x) > 0).astype(float)
    return f.fillna({"d_price": 0.0, "d_units_sold": 0.0, "d_inventory": 0.0})


def load_answers(results_root, prompt, model, prompts, feats):
    """Every answered replay row (quality + overstock sets) for one model and prompt."""
    folder = os.path.join(results_root, prompt, model)
    parts = [pd.read_csv(os.path.join(folder, f)) for f in ("quality.csv", "overstock.csv")
             if os.path.exists(os.path.join(folder, f))]
    q = classify(add_prompt_info(pd.concat(parts, ignore_index=True), prompts))
    q = q[q["error"].isna() & q["move"].notna()]
    return q.merge(feats, on="prompt_id", how="left")


def situation_shares(q, state, move):
    """Per-situation share of answers with this move, for situations in this inventory state."""
    s = q[q["inv_state"] == state]
    return (s["move"] == move).groupby(s["prompt_id"]).mean().to_numpy()


def boot_diff(a, b, n_boot, rng):
    """Mean(a) - mean(b) with a percentile CI, resampling situations within each group."""
    est = a.mean() - b.mean()
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    draws = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, lo, hi


def contrasts(q, n_boot, rng):
    rows = []
    for name, move, plus, minus in [("raise", "raise", "under", "over"), ("cut", "lower", "over", "under")]:
        a, b = situation_shares(q, plus, move), situation_shares(q, minus, move)
        est, lo, hi = boot_diff(a, b, n_boot, rng)
        rows.append({"contrast": name, f"p_{move}_given_{plus}": a.mean(), f"p_{move}_given_{minus}": b.mean(),
                     "estimate": est, "ci_low": lo, "ci_high": hi,
                     "n_situations_" + plus: len(a), "n_situations_" + minus: len(b)})
    return rows


def logit_inventory_effect(q):
    """AME of +1 SD log inventory on P(raise) and P(cut), clustered by situation."""
    y = q["move"].map({"hold": 0, "raise": 1, "lower": 2})
    present = sorted(y.unique())
    out = {}
    if len(present) < 2:
        return {"raise": None, "cut": None, "note": "only one action observed"}
    X = q[FEATURES].astype(float)
    X = X.loc[:, X.std(ddof=0) > 0]            # an input that never varies cannot be estimated
    X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    # recode to consecutive classes so a model that never raises (or never cuts) still fits
    codes = {c: i for i, c in enumerate(present)}
    yc = y.map(codes)
    groups = pd.factorize(q["prompt_id"])[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.MNLogit(yc, X).fit(method="bfgs", maxiter=2000, disp=False,
                                        cov_type="cluster", cov_kwds={"groups": groups})
            me = res.get_margeff(at="overall", method="dydx")
        frame = me.summary_frame()
    except Exception as exc:                                    # e.g. perfect separation
        return {"raise": None, "cut": None, "note": f"not estimable: {type(exc).__name__}"}
    for label, code in [("raise", 1), ("cut", 2)]:
        if code not in codes:
            out[label] = None
            continue
        key = [i for i in frame.index if i[1] == "log_inventory" and str(i[0]).endswith(f"={codes[code]}")]
        if not key:
            out[label] = None
            continue
        r = frame.loc[key[0]]
        out[label] = (r["dy/dx"], r["Conf. Int. Low"], r["Cont. Int. Hi."], r["Pr(>|z|)"])
    out["note"] = "converged" if res.mle_retvals.get("converged", True) else "did not converge"
    return out


PROMPT_LABELS = {"strict": "Strict rule", "rule_context": "Rule as context", "goal_state": "Goal + state"}
MODEL_LABELS = {"llama70b": "Llama 3.3 70B", "qwen38_27b-fp8": "Qwen3.8 27B", "gemma4_31b_q4": "Gemma 4 31B",
                "qwen3_32b_awq": "Qwen3 32B", "qwen3_30b_a3b": "Qwen3 30B MoE", "qwen35_9b": "Qwen3.5 9B", "qwen35_9b_think": "Qwen3.5 9B (thinking)"}


def plot_inventory(c, l, out_dir):
    """One PDF per prompt: raw contrasts (top) and logit effects (bottom), estimate with 95% interval."""
    panels = [
        ("contrast", "raise", "Raised when understocked minus\nraised when overstocked (pp)"),
        ("contrast", "cut", "Cut when overstocked minus\ncut when understocked (pp)"),
        ("logit", "raise", "Higher inventory, other inputs fixed:\nchange in P(raise) (pp)"),
        ("logit", "cut", "Higher inventory, other inputs fixed:\nchange in P(cut) (pp)"),
    ]

    def rows(kind, which):
        if kind == "contrast":
            d = c[c["contrast"] == which].copy()
            d["est"] = d["estimate"]
        else:
            d = l[(l["outcome"] == which) & (l["note"] == "converged")].copy()
            d["est"] = d["ame_per_sd_log_inventory"]
        d = d.dropna(subset=["est"])
        d[["est", "ci_low", "ci_high"]] *= 100
        return d

    # one x-range per panel, shared by every prompt's file so they compare directly
    limits = {}
    for kind, which, _ in panels:
        d = rows(kind, which)
        lo, hi = min(0, d["ci_low"].min()), max(0, d["ci_high"].max())
        pad = 0.05 * (hi - lo or 1)
        limits[(kind, which)] = (lo - pad, hi + pad)

    for prompt, plabel in PROMPT_LABELS.items():
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
            for ax, (kind, which, title) in zip(axes.flat, panels):
                d = rows(kind, which)
                d = d[d["prompt"] == prompt]
                ax.axvline(0, color="#52514e", lw=1)
                for i, model in enumerate(MODELS):
                    y = len(MODELS) - 1 - i
                    r = d[d["model"] == model]
                    if r.empty:
                        ax.text(limits[(kind, which)][0], y, "  not estimable", va="center",
                                fontsize=8, color="#52514e", style="italic")
                        continue
                    r = r.iloc[0]
                    ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=MODEL_COLOURS[model], lw=2)
                    ax.plot(r["est"], y, "o", color=MODEL_COLOURS[model], ms=7, mec="#fcfcfb", mew=1.5)
                ax.set_yticks(range(len(MODELS)))
                ax.set_yticklabels([MODEL_LABELS[m] for m in reversed(MODELS)])
                ax.set_xlim(*limits[(kind, which)])
                ax.set_ylim(-0.6, len(MODELS) - 0.4)
                ax.grid(axis="y", visible=False)
                ax.set_title(title, loc="left", fontsize=10)
            fig.suptitle(f"Response to inventory, {plabel} prompt (dots = estimate, lines = 95% interval)",
                         fontweight="bold", x=0.01, ha="left")
            fig.text(0.01, 0.01,
                     "Top: raw contrasts, bootstrapped over situations (103 understocked, 67 overstocked). "
                     "Bottom: multinomial logit on all prompt inputs, per +1 SD log inventory, "
                     "SEs clustered by situation.", fontsize=8, color="#52514e")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            fig.savefig(os.path.join(out_dir, f"inventory_{prompt}.pdf"))
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="diagnostics/benchmark/results")
    p.add_argument("--sets-dir", default="diagnostics/benchmark/prompt_sets")
    p.add_argument("--out-dir", default="diagnostics/benchmark/analysis/inventory")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prompts = pd.concat([pd.read_json(os.path.join(args.sets_dir, s), lines=True)
                         for s in ("quality_set.jsonl", "overstock_set.jsonl")], ignore_index=True)
    feats = situation_features(prompts)

    c_rows, l_rows, j_rows = [], [], []
    for prompt in PROMPTS:
        for model in MODELS:
            if not os.path.exists(os.path.join(args.results_root, prompt, model, "overstock.csv")):
                print(f"skipping {prompt}/{model}: no overstock results")
                continue
            q = load_answers(args.results_root, prompt, model, prompts, feats)
            for r in contrasts(q, args.n_boot, rng):
                c_rows.append({"model": model, "prompt": prompt, **r})

            eff = logit_inventory_effect(q)
            for outcome in ("raise", "cut"):
                v = eff.get(outcome)
                l_rows.append({"model": model, "prompt": prompt, "outcome": outcome,
                               "ame_per_sd_log_inventory": v[0] if v else np.nan,
                               "ci_low": v[1] if v else np.nan, "ci_high": v[2] if v else np.nan,
                               "p_value": v[3] if v else np.nan, "note": eff["note"],
                               "n_answers": len(q), "n_situations": q["prompt_id"].nunique()})

            within = q[q["inv_state"] == "within"]
            under = q[q["inv_state"] == "under"]
            j_rows.append({"model": model, "prompt": prompt,
                           "within_band_changed": (within["move"] != "hold").mean(),
                           "understocked_cut": (under["move"] == "lower").mean(),
                           "all_changed": (q["move"] != "hold").mean()})

    c = pd.DataFrame(c_rows)
    l = pd.DataFrame(l_rows)
    j = pd.DataFrame(j_rows)
    c.to_csv(os.path.join(args.out_dir, "inventory_contrasts.csv"), index=False)
    l.to_csv(os.path.join(args.out_dir, "inventory_logit.csv"), index=False)
    j.to_csv(os.path.join(args.out_dir, "jitter.csv"), index=False)
    plot_inventory(c, l, args.out_dir)

    pd.set_option("display.width", 200)
    show = c[["model", "prompt", "contrast", "estimate", "ci_low", "ci_high"]].copy()
    show[["estimate", "ci_low", "ci_high"]] *= 100
    print("Contrasts (percentage points, 95% CI over situations):")
    print(show.round(1).to_string(index=False))
    show = l[["model", "prompt", "outcome", "ame_per_sd_log_inventory", "ci_low", "ci_high", "p_value", "note"]].copy()
    show[["ame_per_sd_log_inventory", "ci_low", "ci_high"]] *= 100
    print("\nLogit: change in P(action), percentage points, per +1 SD log inventory (others held fixed):")
    print(show.round({"ame_per_sd_log_inventory": 1, "ci_low": 1, "ci_high": 1, "p_value": 4}).to_string(index=False))
    show = j.copy()
    show[["within_band_changed", "understocked_cut", "all_changed"]] *= 100
    print("\nJitter (% of answers):")
    print(show.round(1).to_string(index=False))
    print(f"\nwrote CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
