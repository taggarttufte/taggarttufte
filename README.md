# Hi, I'm Taggart

Mathematics BS, Montana State University, May 2026 — completed in three years, working through college. Independent researcher at the intersection of reinforcement learning, mechanistic interpretability, and AI safety — mostly on whether hidden behavior can be planted in a model, and whether it can be caught.

## Current Work

### [Hidden Triggers in Cartridges — a Model Organism for Backdoors](https://github.com/taggarttufte/cartridge-interp)
A Cartridge is a trained KV cache: a small, shippable artifact that installs knowledge or behavior into a model without touching its weights — so it's a plausible future supply-chain object. This asks whether a cartridge that does something genuinely useful can also carry a hidden trigger, and whether a defender can detect it. Used as a cheap, reversible model organism: no fine-tuning run, no weight diff, trainable in minutes on one consumer GPU. Run largely as hands-off autonomous overnight campaigns on rented GPUs, each scoped, budgeted, and torn down end to end.

**Key result:** A useful cart can hide a working trigger only if the two are trained *jointly* — naive concatenation fails by destructive interference, independently replicating CAS. But what installs is not a clean backdoor. Fuzzing against real traffic overturned this project's own earlier "stealthy sleeper" claim: every trigger cart fires on 24–92% of ordinary trigger-free queries, and decoy controls show the gate keys on a word's *position* in the turn roughly twice as strongly as on the word itself. The sharpened claim: **the payload installs perfectly; it is the switch the format resists** — the same iff-rule gates cleanly as an in-context prompt on the same model, isolating the failure to the cartridge's shortcut-prone prefix. On defense: a response-position activation monitor catches firing 4/4 and names the payload, but neither static KV inspection nor a pre-output monitor reads the backdoor at rest — **the detection window is concurrent with firing, not before it.** Three headline claims here were reversed by the project's own follow-up controls; the repo's `CLAIMS.md` tracks exactly what stands.

`PyTorch` `HuggingFace Transformers` `PEFT` `mechanistic interpretability` `activation steering` `model organisms`

### [Refusal-Direction Ablation Across the Gemma Family](https://github.com/taggarttufte/refusal-direction-study)
Empirical study of how Arditi et al.'s (NeurIPS 2024) "refusal lives in a single direction" finding transfers across model families. Replicated cleanly on Qwen 2.5 1.5B (10/10 coherent jailbreak, N=12 with explicit coherence checks), then mapped a 5x3 block-by-direction matrix across Gemma 2 2B, Gemma 3 1B, and Gemma 4 E2B to test single-layer ablation. Cross-Gemma results are dramatically asymmetric — Gemma 3 is the architectural outlier, not the rule.

**Key result:** Direct parameter inspection found that Gemma 3's `post_attention_layernorm` and `post_feedforward_layernorm` gains are 5–30x larger than Gemma 2, Gemma 4, or Qwen — amplifying per-block residual perturbations and making single-layer interventions disproportionately effective. Gemma 4 corrected the calibration. Random-direction control rules out the noise-injection alternative. Inference-only on a 12 GB consumer GPU.

`PyTorch` `HuggingFace Transformers` `mechanistic interpretability` `forward hooks` `matplotlib`

### [Balatro RL](https://github.com/taggarttufte/balatro-rl)
A PPO agent for the roguelike poker deckbuilder Balatro — eight architecture versions, ~366 hours of compute. Started by training against the live game through a Lua mod with file/socket IPC (V1–V3), then pivoted to a purpose-built Python simulation (164 jokers, consumables, boss blinds) with a 496-test regression suite, for a ~12,500x throughput speedup. The V7 agent uses a hierarchical action space — an intent head over play/discard/use plus a learned card-selection head — over a 434-dim observation at ~2.5M parameters, peaking at a 2.35% solo win rate and reaching ante 9 reliably.

**Key result:** The ~2% ceiling is capacity-insensitive. Six reward-shaping retunes, a 5.5x network scale-up, and four self-play variants all hit the same plateau — the bottleneck is the exploration mechanism, not network size. Concluded with a principled next axis: MCTS with a neural policy/value prior, not more model-free PPO. An earlier version's headline win rate was invalidated by a self-audit — fixed-seed memorization plus a miscoded joker — and is reported as retracted rather than quietly dropped.

`Python` `PyTorch` `Gymnasium` `Lua` `PPO` `GAE` `multiprocessing`

### [Algebra Hunt — Automated Search Over Open Problems](https://github.com/taggarttufte/algebra-hunt)
A two-week, certificate-first campaign against open problems on stale lists in quasigroup, loop, and semigroup theory: a language model orchestrates automated theorem provers (Vampire, E, Twee, Prover9), Mace4, a SAT solver, and GAP overnight on one desktop CPU, under a verification protocol written down before any result existed — nothing counts without a machine-checkable certificate, an independent re-implementation, and a literature check. The protocol caught two bogus zero-second "proofs," three verdict-parsing bugs, a mislabelled axiom, and one of my own overstatements before anything went out.

**Key result:** Closed the two remaining open orders (14 and 18) of a 1998 conjecture on recursively differentiable quasigroups / recursive MDS codes, with triple-verified tables and a hand-checkable closed form at order 14 — in seconds, by encoding a domain expert's stated intuition (cyclic symmetry) as a SAT constraint after 24 hours of unconstrained search found nothing. Also: power-semigroup determinism verified for all semigroups of order ≤ 6 (apparent first for the finite case), and a mis-encoded "unsolved since 2008" benchmark problem diagnosed with an order-2 countermodel — erratum acknowledged by the library's maintainer.

`automated theorem proving` `Vampire` `Twee` `SAT` `GAP` `quasigroup/loop theory`

### [Attention-Decay in Pandemic Surveillance](https://github.com/taggarttufte/aixbio-hackathon-2026)
Comparative evaluation of multi-signal pandemic early-warning, submitted to Apart Research's AIxBio Hackathon (Track 2) and externally reviewed. Tests whether four surveillance signal types — wastewater PCR, Google Trends, Wikipedia pageviews, and CDC syndromic data — keep calibrated relationships with clinical ground truth across a pathogen's transition from emerging to endemic, using COVID-19 as the subject and influenza as a controlled comparison.

**Key result:** Attention-based signals show 5–23x variance compression after the first major COVID-19 wave but none across flu seasons — attention decay is an emerging-disease novelty-cycle phenomenon, not a property of the signal type. Wastewater is the only signal type that holds calibration across the full lifecycle. Includes an honest negative result on LLM-conversation surveillance.

`Python` `pandas` `HuggingFace Transformers` `time-series anomaly detection` `matplotlib`

### [Neural ODE for ICU Mortality Prediction](https://github.com/taggarttufte/neural-ode-icu)
Systematic comparison of four ML model families for predicting ICU mortality on MIMIC-IV (74,829 patients). Feature-engineered XGBoost (AUROC 0.9565) significantly outperforms Neural ODEs (0.9039) and clinical language models (0.8809), with all differences confirmed by bootstrap CIs and DeLong significance tests. Investigated whether clinical text models exploit code status documentation (CMO/DNR) as a confound using a novel multi-task ClinicalBERT framework.

**Key result:** Structured time-series features dominate text for short-horizon ICU mortality. The interesting question is not which model wins, but what each modality actually captures.

`Python` `PyTorch` `torchdiffeq` `HuggingFace` `XGBoost` `PEFT` `SLURM/HPC`

### [Book Reviews](https://taggarttufte.github.io/book-reviews/)
Live Jekyll site of long-form reviews on AI safety, philosophy of science, and epistemology — Taleb, Christian, Callard, Hao, and others. Each review is its own argument, not a summary: what the book changed, where it falls short, and how it connects to alignment or my own research. Source: [taggarttufte/book-reviews](https://github.com/taggarttufte/book-reviews).

`Jekyll` `GitHub Pages` `kramdown` `MathJax`

## Other Projects

**[Multi-Voice Audiobook Generator](https://github.com/taggarttufte/multivoice-audiobook)**
End-to-end EPUB → multi-voice MP3 audiobook pipeline using xAI's Grok TTS. Heuristic dialogue attribution with confidence scoring, gender-matched voice mapping, per-segment caching, and a bundled Flask web player with variable speed, bookmarks, and paragraph-synced read-along. 7 books rendered to date; ~70× cheaper to run than ElevenLabs at comparable quality.

**[Ski Resort Pricing Analysis](https://github.com/taggarttufte/ski-resort-pricing)**
Predictive modeling of season pass prices across 500+ resorts using multi-variable regression and Cholesky decomposition. Built from numerical foundations rather than black-box libraries.

**[SVD Image Compression](https://github.com/taggarttufte/svd-image-compression)**
Image compression via Singular Value Decomposition. 17x compression at rank-50 with less than 6% reconstruction error.

## Tech Stack

Python · PyTorch · NumPy · pandas · scikit-learn · HuggingFace Transformers · PEFT · Gymnasium · Flask · MATLAB · XGBoost · Git · SLURM · rented-GPU workflows (Vast, RunPod)

## Background

Strong foundation in numerical linear algebra, real analysis, and statistical theory. Experience with HPC (MSU Tempest, NVIDIA A40) and rented single-GPU cloud instances for training and interpretability work.

Most of what I do is small-scale and carefully controlled rather than large-scale: norm-matched random-direction controls, held-out baselines, multi-seed checks where variance matters, and null results reported as nulls. Two of the projects above include a public retraction of my own earlier result after a self-audit. Interested in AI safety and alignment — particularly deception, backdoors, and whether hidden behavior can be detected.
