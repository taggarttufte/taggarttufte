# Hi, I'm Taggart

Mathematics major at Montana State University, graduating May 2026. Building at the intersection of mathematics, machine learning, and clinical data science.

## Current Work

### [Neural ODE for ICU Mortality Prediction](https://github.com/taggarttufte/neural-ode-icu)
Systematic comparison of four ML model families for predicting ICU mortality on MIMIC-IV (74,829 patients). Feature-engineered XGBoost (AUROC 0.9565) significantly outperforms Neural ODEs (0.9039) and clinical language models (0.8809), with all differences confirmed by bootstrap CIs and DeLong significance tests. Currently investigating whether clinical text models exploit code status documentation (CMO/DNR) as a confound using a novel multi-task LoRA framework on BioMistral-7B.

**Key result:** Structured time-series features dominate text for short-horizon ICU mortality. The interesting question isn't which model wins, but what each modality actually captures.

`Python` `PyTorch` `torchdiffeq` `HuggingFace` `XGBoost` `LoRA/PEFT` `SLURM/HPC`

## Other Projects

**[Ski Resort Pricing Analysis](https://github.com/taggarttufte-collab/ski-resort-pricing)**
Predictive modeling of season pass prices across 500+ resorts using multi-variable regression and Cholesky decomposition. Built from numerical foundations rather than black-box libraries.

**[SVD Image Compression](https://github.com/taggarttufte-collab/svd-image-compression)**
Image compression via Singular Value Decomposition. Achieves 17x compression at rank-50 with less than 6% error.

## Tech Stack

Python · PyTorch · NumPy · pandas · scikit-learn · HuggingFace Transformers · MATLAB · XGBoost · Git · SLURM

## Background

Strong foundation in numerical linear algebra, real analysis, and statistical theory. Research focus on applying mathematical methods to clinical machine learning problems. Experience with HPC (MSU Tempest, NVIDIA A40) for training neural models at scale.
