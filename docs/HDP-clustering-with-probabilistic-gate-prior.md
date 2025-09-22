# Disclosure-Oriented DP/HDP Clustering with Probabilistic Gating

## 0) Goals

* Discover stable disclosure themes from patient text.
* Introduce **probabilistic gating priors** for four semantic slots:

  * Answer
  * Answer + Disclosure
  * Disclosure-only
  * Irrelevant
* Transition from **silver heuristics** → **trained classifier** for gating.
* Use **Hierarchical Dirichlet Processes (HDP)** to cluster within slots, optionally sharing atoms across them.
* Freeze a versioned taxonomy for downstream EIG scoring.

---

## 1) Preprocessing & Features

**Segmentation:**

* Break patient turns into sentences/clauses.

**Feature extraction (per sentence):**

* **Q–A similarity**: cosine between patient sentence and physician utterances (sliding window).
* **Disclosure similarity**: max cosine with frozen disclosure centroids.
* **Auxiliary features**: length, entropy of similarity distribution, utterance position.

**Embeddings:**

* Sentence-level model (e.g., `e5-large-v2`, `bge-large`).
* Normalize (L2). Optional PCA/whitening.

---

## 2) Probabilistic Gating Priors

**Phase 1 — Silver Labels:**

* Heuristic rules using thresholds on Q–A similarity and disclosure similarity.
* Assign each sentence to one slot.

**Phase 2 — Classifier:**

* Train P(Slot | features) using silver labels.
* Model: logistic regression or small MLP.
* Calibrate probabilities (temperature scaling / isotonic).

**Output:** Soft slot assignment (probabilities across the 4 slots).

---

## 3) Slot-Conditioned Clustering

**Independent DP per slot (baseline):**

* Each slot has its own DP mixture (DP-GMM or DP-vMF).
* Truncation T = 128–256.

**HDP (preferred):**

* Group-level DP per slot, base distribution shared globally.
* Allows clusters to be reused across slots (e.g., "compliance gap" in both Answer+Disclosure and Disclosure-only).

**Implementation options:**

* **DP-GMM (CPU, easy):** `sklearn.mixture.BayesianGaussianMixture` per slot.
* **HDP-vMF (GPU, better):** Pyro/NumPyro stick-breaking with vMF emissions, group plate for slots.

---

## 4) Training & Freezing

**Training loop:**

1. Split train/val.
2. Run classifier → slot soft assignments.
3. Fit DP/HDP on embeddings grouped by slot.
4. Merge near-duplicate clusters (cosine ≥ 0.95).
5. Drop dust clusters (<0.1% mass).
6. Freeze centroids, exemplars, dispersion stats.

**Freezing protocol:**

* Save gate classifier, per-slot cluster params, exemplars.
* Version tag: `HDP-v1-gate1.0`.

---

## 5) Monitoring & Diagnostics

**Gate monitoring:**

* Calibration: Brier score (<0.25), ECE (<0.1).
* Slot balance: distribution across slots (avoid collapse >80% in Irrelevant).
* Agreement: compare classifier vs heuristics on validation.

**Cluster monitoring:**

* Coverage: % sentences with hard label ≥ τ (target 70–85%).
* Purity: median cosine to centroid ≥ 0.70.
* Stability: ARI/NMI ≥ 0.80 across seeds (top 80% mass).
* Cross-slot reuse (HDP): # atoms shared across ≥2 slots.

**Bad behaviors & fixes:**

* Too many dust clusters → raise weight\_concentration\_prior, enforce min mass.
* Mode collapse → increase truncation T, relax concentration prior.
* Over-merging → raise merge cosine (0.95→0.97), require bidirectional neighbor check.

---

## 6) Timeline

**Week 1:** Preprocessing & heuristics; generate silver slot labels.

**Week 2:** Train gating classifier; calibrate probabilities.

**Week 3:** Fit DP per slot (baseline) or HDP (preferred). Merge/prune clusters.

**Week 4:** Freeze taxonomy, SME review per slot, release `HDP-v1`.

---

## 7) Libraries

* Embeddings: `sentence-transformers`, `faiss-gpu`.
* Classifier: `scikit-learn` (logistic regression), PyTorch (MLP).
* DP-GMM: `scikit-learn`.
* HDP-vMF: `pyro-ppl` (PyTorch) or `numpyro` (JAX).
* Monitoring: `scikit-learn` metrics, reliability diagrams, UMAP for visualization.

---

## 8) Acceptance Gates

* Gate calibration metrics within thresholds.
* Per-slot coverage ≥ 70%.
* Median purity ≥ 0.70.
* Stability across seeds ≥ 0.80 (ARI/NMI).
* SME approval for top 90% mass clusters.
* Artifacts frozen and versioned reproducibly.
