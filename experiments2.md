# Experiment Proposals (v2)

**Research Question (RQ1)**: How do structural properties and node feature properties of real-world and synthetic multi-label graphs influence the performance of different node classification methods?

**Sub-questions**:
1. Which structural properties (degree, clustering, homophily) show the strongest relationship with model performance?
2. Which node-feature properties (dimensionality, sparsity, feature-label MI) show the strongest relationship with model performance?
3. Under which property conditions does the performance gap between GCN and H2GCN widen or narrow, and does this align with their design assumptions?
4. To what extent do trends observed on synthetic datasets predict behavior on real-world multi-label datasets?

**Models compared**: GCN vs H2GCN

**Why this pair**: Directly opposing design assumptions. GCN assumes homophily (degree-weighted aggregation of 1-hop neighbors). H2GCN separates ego from neighborhood and uses 2-hop neighborhoods, designed to be homophily-agnostic. Clear expected differential behavior as graph properties vary.

---

## Guiding principles

- Every controlled experiment varies **one** property axis with a **measurable real-world analog** so real-world datasets can be overlaid on the synthetic curve.
- Always log the same metrics: micro-F1, macro-F1, macro AUC-ROC, macro-AP. 3 seeds per run. Save measured properties on the actual generated graph alongside scores.
- Skip experiments whose property has no clean real-world counterpart, or that overlap with another sub-question.

---

## Experiment 1: Real-world baselines + dataset characterization

**Goal**: Establish anchor points and compute structural/feature properties across real datasets.

**Datasets**: BlogCat, Yelp, DBLP, PCG, HumLoc, EukLoc, OGB-Proteins

**Properties to compute**:
- Number of nodes and edges
- Average degree, graph density
- Clustering coefficient
- Feature dimension, feature sparsity
- Multi-label homophily
- Label cardinality (l_mean, l_max, distribution)
- CCNS matrix and intra/inter-class contrast

**Actions**: Run GCN and H2GCN on each dataset with 3 seeds (or built-in splits where defined).

**Runs**: 7 datasets × 2 models × 3 splits = **42**

**Purpose**: Real-world anchor points. Synthetic trends are validated by checking whether real-world performance falls on the predicted curves. Also fulfills research-plan Tasks 2 and 3.

---

## Experiment 2: Homophily sweep

**Goal**: Reproduce the paper's homophily trend for the chosen model pair, so synthetic and real-world results can be bridged on the homophily axis.

**Setup**:
- Same hypersphere data across all graphs (N=3000, |F|=10, |C|=20)
- Find (α, b) pairs achieving target h ∈ {0.15, 0.2, 0.4, 0.6, 0.8, 1.0}

**Caveats to acknowledge**:
- The minimum achievable homophily on this label distribution is ≈0.15 (random graph baseline). h=0.1 is unreachable with SDA which is purely homophilic. The h=0.15 floor places BlogCat (h=0.10) slightly outside the synthetic range — that itself is a finding.
- h=1.0 produces a graph that is a union of disconnected cliques. Both methods may saturate; this is the upper boundary case rather than a regular interior point.
- Effective interior levels: ~4 (h ∈ {0.2, 0.4, 0.6, 0.8}), satisfying the success criterion.
- α and b are coupled in SDA — varying α also changes density. Density is reported as a measured property per run.

**Runs**: 6 graphs × 2 models × 3 splits = **36** (or 24 for the 4 interior points)

**Purpose**: Validation, not contribution. Establishes the AP-vs-homophily curve for GCN and H2GCN, then checks whether real-world datasets sit on that curve. Deviations motivate the other experiments.

**Real-world mapping**: Measured h per dataset (paper Table 1/2):
- DBLP: 0.76 · EukLoc: 0.46 · HumLoc: 0.42 · Yelp: 0.22 · PCG: 0.17 · OGB-Proteins: 0.15 · BlogCat: 0.10

---

## Experiment 3: Feature and label space dimensions

**Goal**: Test whether feature dimensionality, label count, and their ratio affect the GCN vs H2GCN gap.

**Motivation**: Real-world datasets span 3 orders of magnitude in |F|/|C|:

| Dataset | \|F\| | \|C\| | Ratio |
|---|---|---|---|
| DBLP | 300 | 4 | 75 |
| Yelp | 300 | 100 | 3 |
| PCG | 32 | 15 | 2.1 |
| OGB-Proteins | 8 | 112 | 0.07 |

The paper fixes |F|=10, |C|=20 for all synthetic experiments. Real-world datasets vary wildly in both.

**Setup**: 2D sweep over |F| ∈ {10, 50, 200} and |C| ∈ {5, 20, 100}, fixed graph parameters (h≈0.4). 9 conditions.

**Runs**: 9 × 2 × 3 = **54**

**Hypothesis**: H2GCN's concatenation gives it a capacity advantage at low |F|/|C| (OGB-Proteins regime). Large |C| amplifies H2GCN's advantage because 2-hop neighborhoods carry richer label histograms.

**Why novel**: The paper does not vary either dimension.

**Real-world mapping**: Each real dataset has measured |F| and |C|; the synthetic 2D grid brackets the real-world range.

---

## Experiment 4: Clustering coefficient (via edge rewiring)

**Goal**: Test whether graph clustering affects model performance independently of homophily.

**Motivation**: The paper reports clustering values in Tables 1/2 (range 0.09–0.93) and attributes part of DBLP's GNN success to its high clustering, but never varies it as an independent variable. The paper's Table 5 also shows clustering and homophily are coupled in SDA.

**Setup**: Generate an SDA graph at fixed homophily (h≈0.6). Apply **edge rewiring** (swap random edge endpoints while preserving degree sequence) at rates ∈ {0%, 25%, 50%, 75%} to progressively destroy clustering while preserving degree distribution and approximately preserving homophily.

**Runs**: 4 levels × 2 models × 3 splits = **24**

**Hypothesis**: High clustering amplifies GCN's aggregation (neighbors of neighbors overlap, reinforcing signal). The GCN/H2GCN gap narrows as clustering increases, since H2GCN's higher-order neighborhood is less critical when the local neighborhood already has redundant structure.

**Why novel**: Isolates clustering from homophily — something the paper identifies as a confound but doesn't resolve.

**Real-world mapping**: All datasets have measured clustering values in Tables 1/2, so they position directly on the curve.

---

## Experiment 5: Feature-label mutual information (center_spread sweep)

**Goal**: Test how feature-label informativeness moderates the GCN vs H2GCN gap.

**Motivation**: GNNs combine features and structure. When features are highly predictive of labels, the contribution of message passing is small; when features are weak, the structural signal carries more weight. The paper varies feature noise via the rori_feat parameter, but does not vary the *intrinsic* informativeness of the feature space.

**Setup**: Vary the Mldatagen `center_spread` parameter ∈ {0.3, 0.5, 0.7, 1.0} at fixed radius range, fixed graph parameters (h≈0.4). Smaller `center_spread` clusters label-sphere centers near the origin → more sphere overlap → less feature-space separation between labels → lower feature-label MI.

**Runs**: 4 levels × 2 models × 3 splits = **24**

**Hypothesis**: At low MI, GNNs compensate via graph structure and outperform MLP. At high MI, MLP becomes a strong baseline and the GNN advantage shrinks. H2GCN's ego separation may protect it more than GCN at low MI because graph signal is not contaminated by uninformative features.

**Why novel**: The paper varies feature *noise dilution* (rori_feat) but not feature *informativeness*. These are distinct: noise dilution adds irrelevant columns; center_spread changes how distinguishable the label clusters are in feature space.

**Real-world mapping**: MLP-only AP per dataset is the operational MI proxy (paper's logic: if features alone classify well, MI is high). Each real dataset has a measured MLP AP in Table 3.

**Caveat**: After the Mldatagen sequential algorithm, `center_spread` is not perfectly independent of multi-label character (smaller spread → more sphere overlap → also more labels per node). Acknowledge in the analysis and report `l_mean` per condition.

---

## Experiment 6: Multi-label character (radius sweep)

**Goal**: Test whether multi-label character — the defining property of multi-label graphs — affects the GCN vs H2GCN gap.

**Motivation**: Real-world `l_mean` varies more than 10× across datasets:

| Dataset | l_mean | l_max |
|---|---|---|
| DBLP | 1.18 | 4 |
| BlogCat | 1.40 | 11 |
| HumLoc | 1.19 | 4 |
| EukLoc | 1.15 | 4 |
| PCG | 1.93 | 12 |
| Yelp | 9.44 | 97 |
| OGB-Proteins | 12.75 | 100 |

The paper observes this variation extensively but never tests it as an independent variable.

**Setup**: Vary the radius range to produce target `l_mean` ∈ {1.5, 3, 6, 10} (approximate). Fixed graph parameters (h≈0.4), fixed `center_spread`. Use `--radius-range` or `--radii-file` in the generator.

**Runs**: 4 levels × 2 models × 3 splits = **24**

**Hypothesis**: At high `l_mean`, label correlation matters more, and aggressive aggregation (GCN) may smooth out rare labels. H2GCN's ego separation should preserve them better, widening its advantage. At low `l_mean`, the multi-label scenario reduces to near-multi-class, and standard GCN should perform competitively.

**Why novel**: Converts the paper's most prominent descriptive observation about multi-label graphs into a tested variable. Multi-label character is *the* property that distinguishes this domain from standard node classification.

**Real-world mapping**: Every dataset has a measured `l_mean` (paper Tables 1/2).

**Bonus analysis**: The radius sweep also produces graphs with varying label-count *distribution shape* (entropy of the per-node label-count histogram). Compute this on each generated graph and use it as a secondary analysis axis at no extra runs cost.

---

## Experiment 7 (optional): Edge addition mechanism check

**Goal**: Test whether the *mechanism* of homophily change matters, or whether only the resulting homophily value matters.

**Motivation**: Exp 2 reaches different h levels by tuning (α, b) — a clean SDA process. Real-world graphs reach low h via spurious edges (e.g., BlogCat friendships that are not all semantically meaningful). If only h matters, performance should be identical on both kinds of graph at matched h. If they differ, something other than h (clustering, density, graph-diameter) explains the gap.

**Setup**: Start from a moderately-homophilic synthetic graph (h≈0.6). Add random edges between arbitrary node pairs at rates ∈ {0%, 10%, 25%, 50%, 100%} of the original edge count. Random edges drop effective homophily (random pairs have low Jaccard) while also destroying clustering and shrinking diameter.

**Runs**: 5 levels × 2 models × 3 splits = **30**

**Hypothesis**: H2GCN degrades more gracefully than GCN because ego separation limits damage from unreliable neighbors.

**Caveat**: This is **not a controlled property sweep** — adding edges changes homophily, density, and clustering simultaneously. The 5 levels do not correspond to 5 independent values of one property. The experiment is meaningful only when *paired with Exp 2 as a reference curve*: the gap between Exp 2's "natural-low-h" curve and Exp 7's "noise-induced-low-h" curve at matched effective h is the actual finding.

**Why novel**: Real-world graphs contain spurious edges; clean synthetic graphs do not. This experiment bridges the clean-synthetic / noisy-real gap.

---

## Summary

| # | Experiment | Runs | Status | Property axis |
|---|---|---:|---|---|
| 1 | Real-world baselines + characterization | 42 | Required | — (anchor) |
| 2 | Homophily sweep | 24–36 | Required | structural: homophily |
| 3 | Feature and label space dimensions | 54 | Core | feature: dimensionality |
| 4 | Clustering coefficient (via rewiring) | 24 | Core | structural: clustering |
| 5 | Feature-label MI (center_spread) | 24 | Core | feature: MI |
| 6 | Multi-label character (radius) | 24 | Core | label: l_mean |
| 7 | Edge addition (mechanism check) | 30 | Optional | mechanism comparison |

**Total core (Exp 1–6)**: 192 runs (or 204 with the full 6-level Exp 2). **Including optional Exp 7**: 222–234 runs.

---

## Coverage of the research question

- **SQ1 (structural)**: homophily (Exp 2) + clustering (Exp 4) → two independent structural axes
- **SQ2 (feature)**: dimensions (Exp 3) + MI (Exp 5) → two feature axes
- **Multi-label character** (defining property of the field): l_mean (Exp 6)
- **SQ3 (method gap)**: tested across all five controlled axes
- **SQ4 (real-world bridge)**: Exp 1 anchors; every controlled experiment has a measurable real-world axis for overlay

Every controlled experiment satisfies the rule: vary one property, with a real-world counterpart that places each dataset on the curve.

---

## Connecting synthetic experiments to real-world datasets

For each synthetic axis, the corresponding real-world property is measured per dataset and overlaid on the synthetic curve.

| Exp | Synthetic axis | Real-world analog |
|---|---|---|
| 2 | homophily h | measured h per dataset (Table 1/2) |
| 3 | \|F\|, \|C\| (2D grid) | measured \|F\| and \|C\| per dataset |
| 4 | clustering coefficient | measured clustering (Table 1/2) |
| 5 | feature-label MI | MLP-only AP as MI proxy (Table 3) |
| 6 | l_mean | measured l_mean per dataset (Table 1/2) |
| 7 | edge noise rate | measured effective h, compared against Exp 2 curve |

### Three possible outcomes per overlay
1. **On the curve**: synthetic property is predictive of real-world behavior → supports the hypothesis
2. **Systematically off**: identifiable confound (e.g., clustering, community structure) → discuss as limitation
3. **Scattered**: property alone doesn't explain performance → motivates multivariate analysis across experiments

---

## Known limitations

- α and b are coupled in SDA — homophily and density cannot be varied fully independently. Density is reported as a measured property per run rather than as its own controlled axis.
- After the Mldatagen sequential algorithm, `center_spread` is no longer fully independent of multi-label character. Exp 5 reports `l_mean` per condition.
- The minimum reachable homophily with this label distribution is ≈0.15. BlogCat (h=0.10) sits outside the synthetic range; this is reported as a finding, not papered over.
- h=1.0 produces disconnected cliques; included as a boundary case rather than a regular interior point.
- SDA-generated graphs are idealized random geometric graphs; real-world graphs have community structure and hubs the generator does not replicate.
- Only two models tested — findings about the GCN/H2GCN gap may not generalize to other GNN architectures.

---

## Experiments considered and dropped

For transparency about scope decisions:

| Dropped | Reason |
|---|---|
| Hypersphere geometry 2×2 factorial (original Exp 3) | Replaced by 1D `center_spread` sweep (Exp 5). Mldatagen sequential algorithm couples the two factorial dimensions, making independent attribution impossible. |
| Edge removal | Falls under a separate sub-question / scope. |
| Label noise | Training condition, not a graph or feature property. Outside the strict scope of the RQ. |
| Label imbalance (skewed l-distribution) | Requires non-trivial generator changes. Bonus analysis on Exp 6's data captures most of the same signal. |
| Feature-noise sweep (paper replication) | Available in the paper; not the contribution of this work. |
| Density at fixed h | SDA cannot decouple h from density cleanly. Density reported as a measured property in every experiment instead. |
| Feature sparsity (random zeroing) | No real-world analog; feature sparsity in real datasets is a property of the encoding, not corruption of dense values. |
