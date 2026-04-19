# Hi, I'm Taggart

Mathematics major at Montana State University, graduating May 2026. Building at the intersection of mathematics, machine learning, and AI safety.

## Current Work

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

**[Ski Resort Pricing Analysis](https://github.com/taggarttufte/ski-resort-pricing)**
Predictive modeling of season pass prices across 500+ resorts using multi-variable regression and Cholesky decomposition. Built from numerical foundations rather than black-box libraries.

**[SVD Image Compression](https://github.com/taggarttufte/svd-image-compression)**
Image compression via Singular Value Decomposition. 17x compression at rank-50 with less than 6% reconstruction error.

## Tech Stack

Python · PyTorch · NumPy · pandas · scikit-learn · HuggingFace Transformers · Stable-Baselines3 · MATLAB · XGBoost · Git · SLURM

## Background

Strong foundation in numerical linear algebra, real analysis, and statistical theory. Research focus on applying mathematical methods to clinical ML and RL problems. Experience with HPC (MSU Tempest, NVIDIA A40) for training neural models at scale. Interested in AI safety.
