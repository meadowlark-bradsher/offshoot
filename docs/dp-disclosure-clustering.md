# 0) Goals

* Discover stable disclosure “themes” from text.
* Freeze a versioned taxonomy (centroids + exemplars) for downstream EIG.
* Run at scale (500k convos), with monitoring to catch failure modes.

# 1) Embeddings (input to DP)

**What to embed**

* Unit = patient utterance or short (Q→A) window (e.g., last 1–2 turns).
* Optional: concatenate lightweight context features (region tags) as special tokens.

**Model & preprocessing**

* Use a strong sentence encoder (PyTorch, GPU): `sentence-transformers` (e.g., `bge-large-en`, `e5-large-v2`) or multilingual if needed.
* Steps:

  1. Lowercase, strip boilerplate, keep numbers/medical terms.
  2. Encode → get 768–1024D vectors.
  3. L2-normalize (important for cosine geometry).
  4. (Optional) PCA/whitening to \~256–384D to denoise and speed up mixtures.
* Store to parquet/arrow; keep an index (uid → vector row).

**Infra tips**

* Batch size 512–2048 per GPU; mixed precision (fp16).
* Use FAISS (GPU) later for exemplar search and near-duplicate merges.

# 2) DP mixture choices (pick one to start)

## Option A — **DP-GMM (scikit-learn, CPU, easy)**

* Library: `sklearn.mixture.BayesianGaussianMixture` with `weight_concentration_prior_type='dirichlet_process'`.
* Pros: minimal code, robust VI, good baseline.
* Cons: CPU only; Gaussian in Euclidean space (use if you whitened).
* Settings:

  * `n_components = T` (truncation) start with 128 or 256.
  * `covariance_type='diag'` (faster, stable), `max_iter=200`, `reg_covar=1e-5`.
  * `weight_concentration_prior` leave default (learned), or small Gamma prior if you want sparser solutions.
  * `init_params='kmeans'`, `n_init=3` (or more; parallel via joblib).

## Option B — **DP on the sphere (vMF mixture, Pyro/NumPyro, GPU)**

* Geometry matches L2-normalized embeddings.
* Library: **Pyro** (PyTorch, GPU) or **NumPyro** (JAX, GPU).
* Implement stick-breaking DP + vMF emissions with SVI (minibatch VI).
* Pros: native cosine geometry; GPU; scalable & customizable.
* Cons: a bit more code than sklearn; but you own the knobs.

> Practical pick: start with **Option A** to ship DP-v1 quickly. In parallel, prototype Option B for DP-v2 (better geometry, GPU).

# 3) Training loop (both options)

**Common steps**

1. **Train/val split** on utterances (e.g., 90/10).
2. **Fit** 2–3 seeds (different random states).
3. **Post-fit merges**: cosine(μ\_i, μ\_j) ≥ 0.95 → merge (use FAISS to find candidates).
4. **Prune dust**: drop clusters with mass < 0.1% (configurable).
5. **Threshold for hard labels**: τ ≈ 0.65–0.7 on posterior responsibility; below → “low-confidence”.

**Metrics to log every run**

* # active clusters (mass > 0.1%).
* Top-k cluster masses; stick-weight decay curve.
* Per-cluster: size, dispersion (Σ trace or vMF κ), median cosine to centroid, exemplar snippets.
* Held-out predictive log-likelihood (or ELBO for Pyro).
* Stability across seeds: ARI/NMI on overlapping assignments; centroid cosine drift.

# 4) Convergence & behavior monitoring

**Convergence**

* **VI/ELBO** (Pyro): stop when relative ELBO change < 1e-4 over 10 steps or max epochs.
* **sklearn**: stop when lower bound change < 1e-3; also cap `max_iter`; watch `converged_`.

**Good-behavior checks**

* **Stability:** ARI/NMI ≥ 0.80 across seeds for top 80% mass.
* **Purity:** median in-cluster cosine ≥ 0.70.
* **Coverage:** % utterances with hard label ≥ τ in \[70%, 85%].
* **Reasonableness:** PMI top terms + SME spot-reads look coherent.

**Bad-behavior detectors & fixes**

* **Too many tiny clusters (dust):** raise `weight_concentration_prior` (sparser), increase min mass, strengthen merge threshold.
* **Mode collapse (too few clusters):** lower prior, increase truncation T, allow `covariance_type='full'` (with reg) or switch to vMF.
* **Component swapping across seeds:** freeze random seeds; align by Hungarian algorithm on centroid cosine; report drift.
* **Over-merging near-duplicates:** raise merge cosine from 0.95 → 0.97; require bi-directional neighbor check and exemplar diversity.
* **Poor fit to cosine geometry:** prefer vMF DP (Option B) or at least whiten + diag-cov.

# 5) Freezing the taxonomy (for SMEs & serving)

* Save:

  * Centroid μ\_k, dispersion (Σ or κ), top 20 exemplars (diversified by farthest-first).
  * Soft assign function (posterior) + τ.
  * Version blob: embedder hash, truncation T, hyperpriors, seeds, library versions.
* Tag release: `DP-v1` (or `DP-v1.1` after SME merges/hides).

# 6) Parallelization & GPU

**Embeddings**

* Massively parallel on GPU(s); use PyTorch `DataLoader` with multiple workers.
* Multi-GPU: shard input across GPUs (DDP not necessary for inference).

**DP training**

* **sklearn DP-GMM**: CPU; parallelize across **seeds** and **T grid** with joblib/dask; use big RAM boxes (vectorized C code is fast).
* **Pyro/NumPyro DP-vMF**:

  * GPU-accelerated SVI (PyTorch/JAX).
  * Minibatch plate for data; full-batch for global sticks/centroids.
  * Multi-GPU: data-parallel (separate runs/seeds); aggregate best by ELBO/val likelihood.
* **Nearest-neighbor / merges**: FAISS (GPU) for centroid-centroid and centroid-point search.

**Throughput tips**

* Use fp16 for embeddings and FAISS indexes; keep DP params in fp32.
* For Pyro/NumPyro, start with batch size 8–32k (depends on VRAM) and gradient accumulation.

# 7) Library stack (Python)

**Embeddings**

* `sentence-transformers` (PyTorch)
* `transformers` (if you roll your own)
* `faiss-gpu` for NN/merges

**DP-GMM (easy path)**

* `scikit-learn` (`BayesianGaussianMixture`)
* `joblib` / `dask` for parallel multi-seed runs

**DP-vMF (GPU path)**

* `pyro-ppl` (PyTorch) **or** `numpyro` (JAX); both support stick-breaking and SVI
* Optional: `torch-distributions` extensions or a small custom vMF log-density (there are reference snippets for vMF in PyTorch/NumPyro)

**Evaluation & viz**

* `umap-learn` (2D maps of clusters)
* `matplotlib/plotly` dashboards
* `scikit-learn` metrics (silhouette as a sanity check; ARI/NMI)
* PMI/top-terms: `scikit-learn` `CountVectorizer` + your tokenizer

# 8) Power-law probe (optional, post-fit)

* Plot rank–size on log–log; run Clauset–Shalizi–Newman tail fit; compare vs lognormal (Vuong test).
* If you see a stable heavy tail, consider Pitman–Yor prior (Pyro/NumPyro swap: change GEM to GEM(d, θ)); otherwise keep DP.

# 9) Minimal code skeletons (pseudocode)

**Embeddings**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

m = SentenceTransformer("intfloat/e5-large-v2", device="cuda")
X = m.encode(texts, batch_size=1024, convert_to_numpy=True, normalize_embeddings=True)
# optional PCA/whitening here
```

**DP-GMM (sklearn)**

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(
    n_components=256,
    covariance_type="diag",
    weight_concentration_prior_type="dirichlet_process",
    max_iter=200,
    reg_covar=1e-5,
    init_params="kmeans",
    n_init=3,
    random_state=seed,
    verbose=1
)
bgm.fit(X_train)
resp = bgm.predict_proba(X_all)     # soft posteriors
labels = resp.argmax(1); conf = resp.max(1)
```

**DP-vMF (Pyro, outline)**

```python
# plates: data; globals: sticks, means μ_k on sphere, κ_k
# sticks ~ GEM(alpha) (or Pitman–Yor with discount d)
# z_n ~ Categorical(sticks) ; x_n ~ vMF(μ_{z_n}, κ_{z_n})
# Use SVI with Adam; sample μ_k via reparameterized vMF or use projected Gaussians + normalization.
```

# 10) Acceptance gates for DP-v1

* Coverage ≥ 70%, Purity ≥ 0.70, ARI/NMI stability ≥ 0.80 (top mass).
* Human sanity: SMEs approve labels for top 90% mass clusters.
* Frozen artifacts saved + reproducible runbook.
