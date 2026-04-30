# Hi, I'm Taggart

Mathematics major at Montana State University, graduating May 2026. Building at the intersection of mathematics, machine learning, and AI safety.

## Current Work

### [Refusal-Direction Ablation Across the Gemma Family](https://github.com/taggarttufte/refusal-direction-study)
Empirical study of how Arditi et al.'s (NeurIPS 2024) "refusal lives in a single direction" finding transfers across model families. Replicated cleanly on Qwen 2.5 1.5B (10/10 coherent jailbreak, N=12 with explicit coherence checks), then mapped a 5x3 block-by-direction matrix across Gemma 2 2B, Gemma 3 1B, and Gemma 4 E2B to test single-layer ablation. Cross-Gemma results are dramatically asymmetric — Gemma 3 is the architectural outlier, not the rule.

**Key result:** Direct parameter inspection found that Gemma 3's `post_attention_layernorm` and `post_feedforward_layernorm` gains are 5–30x larger than Gemma 2, Gemma 4, or Qwen — amplifying per-block residual perturbations and making single-layer interventions disproportionately effective. Gemma 4 corrected the calibration. Random-direction control rules out the noise-injection alternative. Inference-only on a 12 GB consumer GPU.

`PyTorch` `HuggingFace Transformers` `mechanistic interpretability` `forward hooks` `matplotlib`

### [Neural ODE for ICU Mortality Prediction](https://github.com/taggarttufte/neural-ode-icu)
Systematic comparison of four ML model families for predicting ICU mortality on MIMIC-IV (74,829 patients). Feature-engineered XGBoost (AUROC 0.9565) significantly outperforms Neural ODEs (0.9039) and clinical language models (0.8809), with all differences confirmed by bootstrap CIs and DeLong significance tests. Investigated whether clinical text models exploit code status documentation (CMO/DNR) as a confound using a novel multi-task ClinicalBERT framework.

**Key result:** Structured time-series features dominate text for short-horizon ICU mortality. The interesting question is not which model wins, but what each modality actually captures.

`Python` `PyTorch` `torchdiffeq` `HuggingFace` `XGBoost` `PEFT` `SLURM/HPC`

### [Balatro RL](https://github.com/taggarttufte/balatro-rl)
PPO reinforcement learning agent that plays the roguelike card game Balatro via a custom Gymnasium environment and live file-based IPC with a Lua game mod. All navigation is handled headlessly by the Lua mod — no mouse or keyboard automation. Trains at 32x game speed. Built from scratch: IPC protocol, reward shaping, boss blind edge cases, and full training pipeline.

**119-dim obs space, MultiBinary(9) action space, reward shaping across sparse multi-step episodes.**

**Outcome (Apr 2026):** Iterated across 8 PPO variants; best version (V7) peaked at 2.35% Ante-8 win rate. Concluded the pure model-free approach and flagged MCTS as the next direction — Balatro's branching factor rewards explicit lookahead more than reward shaping alone.

`Python` `Stable-Baselines3` `Gymnasium` `Lua` `PPO`

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

Python · PyTorch · NumPy · pandas · scikit-learn · HuggingFace Transformers · Stable-Baselines3 · Flask · MATLAB · XGBoost · Git · SLURM

## Background

Strong foundation in numerical linear algebra, real analysis, and statistical theory. Experience with HPC (MSU Tempest, NVIDIA A40) for training neural models at scale. Interested in AI safety and Alignment.
