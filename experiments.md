# Experiment Proposals

**Research Question (RQ1)**: How do structural properties and node feature properties of real-world and synthetic multi-label graphs influence the performance of different node classification methods?

**Models compared**: GCN vs H2GCN

**Why this pair**: Directly opposing design assumptions. GCN assumes homophily (degree-weighted averaging of 1-hop neighbors). H2GCN separates ego from neighborhood and uses 2-hop neighborhoods, designed to be homophily-agnostic. Clear expected differential behavior as graph properties vary.

---

## Experiment 1: Real-world baselines + dataset characterization

**Goal**: Establish anchor points and compute structural/feature properties for the RQ.

**Datasets**: BlogCat, Yelp, DBLP, PCG, HumLoc, EukLoc, OGB-Proteins

**Properties to compute**:
- Number of nodes and edges
- Average degree, graph density
- Clustering coefficient
- Feature dimension, feature sparsity
- Multi-label homophily
- Label cardinality and label density

**Actions**: Run GCN and H2GCN once per dataset.

**Runs**: 7 datasets × 2 models × 3 splits = 42

**Purpose**: Real-world anchor points. Synthetic trends are validated by checking whether real-world performance falls on the predicted curves.

---

## Experiment 2: Homophily sweep (validation bridge)

**Goal**: Reproduce the paper's homophily trend for the chosen model pair, so synthetic and real-world results can be bridged.

**Setup**:
- Use (α, b) values from the paper that achieve homophily ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
- Same hypersphere data across all 5 graphs (N=3000, |F|=10, |C|=20)

**Runs**: 5 graphs × 2 models × 3 splits = 30

**Purpose**: Validation, not contribution. Establishes the AP-vs-homophily curve for GCN and H2GCN, then checks whether real-world datasets sit on that curve. Deviations motivate the novel experiments.

**Caveat**: α and b are coupled in SDA — varying α also changes density. Acknowledge explicitly.

---

## Experiment 3: Hypersphere geometry (2×2 factorial)

**Goal**: Test whether feature-label MI and multi-label character moderate the GCN vs H2GCN gap. Neither was varied in the paper.

**Design**: 2×2 factorial

|  | Small radius | Large radius |
|---|---|---|
| **Far centers** | Low multi-label, high MI | High multi-label, high MI |
| **Close centers** | Low multi-label, low MI | High multi-label, low MI |

**Rationale**:
- Center separation controls feature-label mutual information (how distinguishable label clusters are)
- Radius controls multi-label character (how much spheres overlap)
- Both are coupled in the hypersphere model, so a factorial design is needed to attribute effects

**Runs**: 4 conditions × 2 models × 3 splits = 24

**Hypothesis**: H2GCN's advantage is largest in the low-MI + high-multi-label cell (structural signal matters most when features are weak and labels are rich).

**Why novel**: The paper only varied feature noise ratio (dilution), not feature informativeness (separation), and kept multi-label character fixed.

---

## Experiment 4: Structural noise — random edge addition

**Goal**: Test robustness to spurious edges (simulates noisy real-world graph structure).

**Setup**:
- Start from a moderately-homophilic synthetic graph (e.g., h≈0.6)
- Randomly add edges between arbitrary node pairs at rates ∈ {0%, 10%, 25%, 50%, 100%} of the original edge count
- Random edges typically connect dissimilar nodes → effective homophily drops

**Runs**: 5 levels × 2 models × 3 splits = 30

**Hypothesis**: H2GCN degrades more gracefully than GCN because ego separation limits damage from unreliable neighbors.

**Why novel**: The paper's synthetic graphs are clean. Real-world graphs contain spurious edges (e.g., BlogCat friendships are not all semantically meaningful). This experiment bridges the clean-synthetic / noisy-real gap.

---

## Experiment 5: Edge removal (structural sparsity)

**Goal**: Test robustness to missing edges.

**Setup**:
- Start from the same base graph as Experiment 4
- Randomly remove edges at rates ∈ {10%, 25%, 50%, 75%}

**Runs**: 4 levels × 2 models × 3 splits = 24

**Hypothesis**: H2GCN suffers more than GCN at high removal rates because 2-hop neighborhoods shrink exponentially as edges are removed.

**Why novel**: Real-world datasets, especially biological ones, have incomplete edge data (unobserved protein interactions). Tests whether H2GCN's reliance on higher-order neighborhoods is a liability under sparse connectivity.

---

## Experiment 6: Label noise

**Goal**: Test robustness to noisy training labels (common in real-world biological datasets).

**Setup**:
- Fix graph and features
- For a random fraction of training nodes, perturb labels: randomly add or remove label assignments
- Noise rates ∈ {5%, 10%, 20%, 40%}

**Runs**: 4 levels × 2 models × 3 splits = 24

**Hypothesis**: GCN's aggressive neighborhood averaging is more robust to label noise than H2GCN's ego-preserving design — noisy ego features get washed out in GCN's aggregation.

**Why novel**: The paper assumes clean labels. Real-world multi-label biological datasets have known labeling errors and missing annotations.

---

## Experiment 7: Feature and label space dimensions

**Goal**: Test whether feature dimensionality, label count, and their ratio affect the GCN vs H2GCN gap.

**Motivation**: Real-world datasets span 3 orders of magnitude in |F|/|C|:

| Dataset | \|F\| | \|C\| | Ratio |
|---|---|---|---|
| DBLP | 300 | 4 | 75 |
| Yelp | 300 | 100 | 3 |
| PCG | 32 | 15 | 2.1 |
| OGB-Proteins | 8 | 112 | 0.07 |

**Setup**: 2D sweep over |F| ∈ {10, 50, 200} and |C| ∈ {5, 20, 100}, fixed graph parameters. 9 conditions.

**Runs**: 9 × 2 × 3 = 54

**Hypothesis**: H2GCN's concatenation gives it a capacity advantage at low |F|/|C| (OGB-Proteins regime). Large |C| amplifies H2GCN's advantage because 2-hop neighborhoods carry richer label histograms.

**Why novel**: Paper fixes |F|=10, |C|=20 for all synthetic experiments. Real-world datasets vary wildly in both dimensions.

---

## Experiment 8: Clustering coefficient (structural)

**Goal**: Test whether graph clustering affects model performance independently of homophily.

**Motivation**: Paper reports clustering in Tables 1/2 (range 0.09–0.93) and attributes DBLP's GNN success partly to its high clustering — but never varies it as an independent variable. Table 5 shows clustering and homophily are coupled in SDA.

**Setup**: Generate a SDA graph at fixed homophily (e.g., h≈0.6). Apply **edge rewiring** (swap random edge endpoints while preserving degree sequence) at rates ∈ {0%, 25%, 50%, 75%} to progressively destroy clustering while preserving degree distribution and approximately preserving homophily.

**Runs**: 4 levels × 2 models × 3 splits = 24

**Hypothesis**: High clustering amplifies GCN's aggregation (neighbors' neighbors overlap, reinforcing signal). GCN's gap vs H2GCN narrows as clustering increases.

**Why novel**: Isolates clustering from homophily — something the paper identifies as a confound but doesn't resolve.

**Real-world mapping**: All datasets have measured clustering values in Tables 1/2, so they position directly on the curve.

---

## Experiment 9: Label imbalance (feature/label property)

**Goal**: Test whether skew in the label-count distribution affects model performance.

**Motivation**: Paper documents label imbalance extensively (Figures 1, 3, 5a) — OGB-Proteins has 40% unlabeled nodes, BlogCat has 72% single-label. But the paper never **tests** the effect of imbalance; it only observes it descriptively.

**Setup**: Modify hypersphere generator to produce graphs with different per-node label-count distributions:
- Uniform: each node has ~k labels (low skew)
- Exponential: most nodes have 1–2 labels, few have many (high skew, matches most real-world datasets)
- Bimodal: mix of very-sparse and very-dense nodes

Measure label-count entropy per configuration.

**Runs**: 4 configurations × 2 models × 3 splits = 24

**Hypothesis**: GNN neighborhood averaging washes out rare-label signal, so both models degrade with skew. H2GCN's ego separation partially protects rare-label information, so its advantage grows with imbalance.

**Why novel**: Converts a descriptive observation from the paper into a testable variable.

**Real-world mapping**: Compute label-count entropy per dataset (derivable from Tables 1/2 percentiles) and position on the curve.

---

## Summary

| # | Experiment | Runs | Status | Novel? |
|---|---|---|---|---|
| 1 | Real-world baselines + characterization | 42 | Required | Required |
| 2 | Homophily sweep | 30 | Core | Validation |
| 3 | Hypersphere geometry (MI × multi-label) | 24 | Core | **Yes** |
| 4 | Edge addition (structural noise) | 30 | Core | **Yes** |
| 5 | Edge removal (sparsity robustness) | 24 | Core | **Yes** |
| 6 | Label noise robustness | 24 | Core | **Yes** |
| 7 | Feature and label space dimensions | 54 | Secondary | **Yes** |
| 8 | Clustering coefficient (via rewiring) | 24 | Secondary | **Yes** |
| 9 | Label imbalance | 24 | Secondary | **Yes** |

**Total core (Exp 1–6)**: 174 runs. **Including secondary (Exp 7–9)**: 276 runs.

Feasible in ~2–3 days with vectorized SDA generator and GPU training.

---

## Narrative enabled by this design

1. **Characterize** what real-world multi-label graphs look like (Exp 1)
2. **Validate** the homophily trend for the chosen model pair (Exp 2)
3. **Extend** beyond homophily on the feature side — show that feature-label MI and multi-label character matter (Exp 3)
4. **Stress test** under realistic corruption: noisy edges, missing edges, noisy labels (Exp 4, 5, 6)
5. **Extend** beyond homophily on the structural/dimensional side — feature/label dimensions, clustering, label imbalance (Exp 7, 8, 9)
6. **Revisit** real-world results through this richer lens

This answers RQ1 across both structural properties (homophily, density, clustering) and feature/label properties (MI, multi-label character, dimensions, imbalance), in both clean and noisy regimes.

---

## Connecting synthetic experiments to real-world datasets

Each synthetic experiment is designed so real-world datasets can be positioned on the same axis, allowing direct overlay of real-world points on synthetic trends.

**General strategy**:
1. For each experiment, pick a property axis with a measurable real-world analog
2. Design synthetic parameter ranges to **bracket** real-world values (so real-world points are interpolations, not extrapolations)
3. Overlay real-world performance on the synthetic curve
4. Interpret: do real-world points fall on the curve (property is predictive), systematically off (identifiable confound), or scattered (property alone insufficient)?

### Per-experiment mapping

| Exp | Synthetic axis | Real-world analog | Notes |
|---|---|---|---|
| 2 | Homophily h | Measured h per dataset (paper Table 1/2) | Extend synthetic range to h≈0.10 to bracket BlogCat |
| 3 | MI × multi-label | (MLP-only AP, ℓ_mean) | MLP AP as operational MI proxy |
| 4 | Edge noise rate | Resulting effective homophily | Compare against Exp 2 curve — does mechanism matter? |
| 5 | Edge removal rate | Avg degree | Real-world range: ~2 (DBLP) to ~590 (OGB-Proteins), use log scale |
| 6 | Label noise rate | No direct analog | Group datasets qualitatively (clean vs noisy); weakest connection |
| 7 | \|F\|, \|C\| (2D grid) | Measured \|F\| and \|C\| per dataset | Real-world: \|F\| 8–300, \|C\| 4–112; synthetic grid brackets both |
| 8 | Clustering coefficient | Measured clustering per dataset (Tables 1/2, 0.09–0.93) | Position datasets on rewiring-derived curve at matching h |
| 9 | Label-count entropy / skew | Derived from per-dataset label-count distribution | Qualitative: BlogCat (72% single-label) vs OGB-Proteins (40% unlabeled) |

### Homophily values (Exp 2)
- DBLP: 0.76 · EukLoc: 0.46 · HumLoc: 0.42 · Yelp: 0.22 · PCG: 0.17 · OGB-Proteins: 0.15 · BlogCat: 0.10

### Avg degrees (Exp 5)
- DBLP: 2.4 · HumLoc: 12 · Yelp: 20 · PCG: 23 · BlogCat: 67 · OGB-Proteins: 590

### Three possible outcomes per experiment
1. **On the curve**: synthetic property is predictive of real-world behavior → supports the hypothesis
2. **Systematically off**: identifiable confound (e.g., clustering, community structure) → discuss as limitation
3. **Scattered**: property alone doesn't explain performance → motivates multivariate analysis across experiments

---

## Known limitations to acknowledge in the report

- α and b are coupled in SDA — homophily and density cannot be varied fully independently
- Hypersphere generator geometry couples MI and multi-label character — factorial design mitigates but does not eliminate this
- SDA-generated graphs are idealized random geometric graphs; real-world graphs have structure (communities, hubs) the generator does not replicate
- Only two models tested — findings about the GCN/H2GCN gap may not generalize to other GNN architectures
