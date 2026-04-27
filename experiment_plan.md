# Experiment Execution Plan

Maps the experiments in [experiments.md](experiments.md) onto the weekly schedule from the research plan. Today's anchor: 2026-04-22 (Week 1).

## Guiding principles

- **Critical path first**: Exp 1 (real-world anchors) and Exp 2 (homophily sweep) unblock everything else — they calibrate synthetic parameter ranges for all remaining experiments.
- **Iterate, don't batch**: finish each experiment's analysis before launching the next. Trends from earlier experiments inform parameter choices for later ones.
- **Always log the same metrics**: micro-F1, macro-F1, macro AUC-ROC, macro-AP. 3 seeds per run. Save per-run property measurements (effective homophily, clustering, etc.) alongside scores so real-world overlay plots are cheap.

## Infrastructure (Week 1)

Do these once, up front — every experiment reuses them.

1. **Run harness**: a single entry point taking `(model, dataset_path, seed, split_idx)` and returning all four metrics. Wrap GCN and H2GCN from the MLGNC repo.
2. **Fix the SDA generator bottleneck**: [Graph Generator Model/sda.py](Graph%20Generator%20Model/sda.py) is O(N²) — vectorize edge sampling so N=3000 graphs generate in seconds, not minutes. Without this, Exp 2–9 aren't feasible.
3. **Property computation utilities**: one module that takes a graph + labels and returns h, clustering, avg degree, density, label cardinality, label-count entropy, feature sparsity. Same function used for real and synthetic — guarantees comparable numbers.
4. **Results store**: flat CSV (one row per run) with columns for config, measured properties, and all metrics. Keep it append-only.

## Week-by-week mapping

### Week 2 — Real-world anchors (Exp 1)
- Load and verify all 7 real-world datasets load correctly through the harness.
- Compute all properties (dataset characterization → research-plan Task 2).
- Run GCN + H2GCN × 3 seeds on each (Exp 1, 42 runs). → baseline table with mean/variance (research-plan Task 3).
- **Decision point**: from the measured property ranges, fix synthetic parameter brackets for Exp 2–9 so real-world values sit inside them.

### Week 3 — Homophily validation (Exp 2)
- Pick (α, b) pairs achieving h ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0} — extend down to 0.1 to bracket BlogCat.
- Run Exp 2 (30+ runs). Plot AP vs h, overlay real-world points.
- **Decision point**: if real-world points fall on the synthetic curve, homophily is a strong univariate predictor and later experiments refine it. If they scatter, the later experiments become the main contribution.
- Start writing abstract, intro, related work in parallel.

### Week 4 — First novel experiment (Exp 3 or Exp 4)
- **Default choice: Exp 4 (edge addition)** — most directly connected to Exp 2 via effective homophily; good bridge figure for the mid-term.
- Alternative: Exp 3 (hypersphere geometry) if feature-side story looks stronger after Week 3 plots.
- Run, analyze, overlay. Prepare the mid-term figure set.

### Week 5 — Mid-term + second novel experiment
- Present Weeks 1–4 results at mid-term.
- Start whichever of Exp 3 / Exp 4 was not run in Week 4, plus Exp 5 (edge removal) since it reuses the same base graph as Exp 4.

### Week 6 — Robustness + real-world reconciliation
- Run Exp 6 (label noise) — smaller, quick to finish.
- Synthesize: for each real-world dataset, list which synthetic curves it sits on/off and by how much. This is the core of the paper's analysis section (research-plan Task 5).
- If time permits, pick ONE of Exp 7/8/9 (secondary) based on which real-world deviations remain unexplained:
  - Datasets scattered on homophily curve but clustering varies wildly → Exp 8
  - OGB-Proteins (|F|=8, |C|=112) is an outlier → Exp 7
  - Label-skew looks like the confound → Exp 9

### Week 7 — Lock results + draft v1
- No new experiments. Re-run any seeds that look unstable.
- Finalize all figures and tables.
- Submit paper draft v1.

### Week 8–9 — Revisions, poster, final presentation
- Only run additional experiments if peer review exposes a specific gap.

## Minimum viable paper (if time slips)

If anything blocks progress, this subset still satisfies the research plan's success criteria (≥2 methods, ≥3 synthetic datasets, ≥4 levels, 3 seeds, ≥2 properties with consistent relationships):

- Exp 1 (anchors) — required
- Exp 2 (homophily sweep) — 1 property × 5–6 levels
- Exp 4 (edge addition) — 1 property × 5 levels
- Exp 6 (label noise) — 1 property × 4 levels, quick

That's 3 synthetic property axes, ~126 runs, and a coherent story (clean homophily + two corruption axes).

## Per-run checklist

For every run logged to the results store, record:
- Model, seed, split index
- All config parameters (α, b, radius, center separation, |F|, |C|, noise rate, etc.)
- Measured properties on the actual generated graph (not just the requested config — they differ)
- All four metrics on test set
- Wall-clock time (flag slow configs early)
