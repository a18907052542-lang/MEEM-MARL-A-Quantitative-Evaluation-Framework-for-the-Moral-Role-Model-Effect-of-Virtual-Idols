# MEEM-MARL: A Quantitative Evaluation Framework for the Moral Role Model Effect of Virtual Idols

> **Full title:** *A Quantitative Evaluation Framework for Emotional Computing of the Moral Role Model Effect of Virtual Idols Based on Attention-Weighted Multimodal Fusion*

This repository provides the complete experimental pipeline, statistical analysis code, and result figures accompanying the paper. The framework — **MEEM-MARL** (Moral Exemplar Effect Model — Multimodal Affective Reinforcement Learning) — integrates multimodal physiological sensing, attention-weighted feature fusion, three-stage moral modeling, and reinforcement-learning-driven adaptive content delivery to quantitatively measure and enhance the moral role-model effect of virtual idol characters.

---

## Table of Contents

- [Background](#background)
- [Experimental Design](#experimental-design)
- [Framework Architecture](#framework-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Algorithm 1 — MEEM-MARL Pipeline](#algorithm-1--meem-marl-pipeline)
- [Statistical Analysis Modules](#statistical-analysis-modules)
- [Visualization](#visualization)
- [Key Results](#key-results)
- [Citation](#citation)
- [Ethics Statement](#ethics-statement)
- [License](#license)

---

## Background

Virtual idol characters (虚拟偶像) have emerged as a novel medium in moral education, yet their psychological mechanisms and measurable educational impact remain understudied. This work operationalizes Bandura's Social Learning Theory (attention → retention → reproduction) within a computational affective-computing framework, enabling real-time, data-driven assessment of moral cognition, moral emotion, and behavioral intention changes across an extended 10-week intervention.

---

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Design | Three-group randomized controlled trial (RCT) |
| Sample | 198 recruited; **186 effective** (Ctrl = 62, VI = 63, Traditional = 61) |
| Duration | 10 weeks (4-week intervention + 4-week follow-up + 8-week retention tracking) |
| Ethics approval | CSU-PSY-2024-007 |
| Outcome measures | Moral cognition (50 MCQ + 10 SJT), Moral emotion (MES-R), Behavioral intention (PTM) |
| Physiological measures | Eye-tracking, HRV (RMSSD), Skin conductance (SCL), EEG (α/β/θ), Facial expression (FACS) |
| Retention assessment | 2 / 4 / 8 weeks post-intervention |

**Three experimental conditions:**
- **Control group** — standard online lecture content
- **Virtual Idol group** — AI-driven virtual idol character delivering matched moral narratives
- **Traditional group** — human instructor delivering matched moral narratives

---

## Framework Architecture

```
Multimodal Input
  ├─ Eye tracking        → 1D-CNN (F_eye)
  ├─ Facial expression   → CNN-FACS (F_face)
  ├─ Voice / prosody     → Transformer (F_voice)
  ├─ Physiology (HRV/SCL/EEG) → BiLSTM (F_phys)
  └─ Interaction log     → MLP (F_log)
        │
        ▼
  Attention-weighted Late Fusion
  α_m = softmax(w_a · F_m + b_a)          [Eq. 3]
  F_fused = Σ α_m · F_m
        │
        ▼
  Affective State Estimation (CNN-LSTM)
  → 7 basic emotions · valence · arousal · engagement · presence · cognitive load
        │
        ▼
  Three-Stage Moral Modeling (Bandura SCT)
  z_a = Attention(F_fused, Θ_VI)           [Stage 1: observation]
  z_r = Retention(z_a, M_mem)              [Stage 2: memory encoding]
  z_p = Reproduction(z_r, C)              [Stage 3: behavioral generation]
        │
        ▼
  Outcome Prediction (sigmoid heads)
  Ŷ_MC · Ŷ_ME · Ŷ_BI
        │
        ▼
  Q-Learning Policy Update                 [Eq. 7]
  R_t = w1·ΔMC + w2·ΔME + w3·ΔBI − w4·CL
  A_VI = {expression, tone, content, pacing}
        │
        ▼
  Adaptive Content Recommendation → Effect Quantification
```

**Composite loss (Eq. 11):**

```
L_total = λ_aff · L_aff + λ_moral · L_moral + λ_reg · ‖θ‖²
```

---

## Repository Structure

```
MEEM_MARL_Pipeline/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Hyperparameters, paths, target statistics
├── run_all.py                          # Master runner (all 4 phases)
│
├── data/
│   └── raw_data.xlsx                   # n=186 subjects, 11 sheets (see data package)
│
├── meem/                               # Algorithm 1 — nine pipeline steps
│   ├── step1_preprocessing.py          # Median filter (w=5), Z-score, 2-s sliding window
│   ├── step2_feature_extraction.py     # Per-modality encoders (CNN/FACS/Transformer/LSTM/MLP)
│   ├── step3_attention_fusion.py       # Softmax attention-weighted late fusion
│   ├── step4_affective_estimator.py    # CNN-LSTM affective state + engagement/presence/CL
│   ├── step5_moral_modeling.py         # Attention → Retention (GRU) → Reproduction
│   ├── step6_outcome_predictor.py      # MC/ME/BI prediction heads + composite loss
│   ├── step7_q_learning.py             # Tabular Q-learning over A_VI
│   ├── step8_recommender.py            # Utility-based adaptive content recommendation
│   ├── step9_effect_quantification.py  # Cohen's d, retention curves, SEM
│   └── pipeline.py                     # End-to-end orchestration
│
├── analysis/                           # Statistical analysis suite
│   ├── descriptive.py                  # Table 2 — baseline comparison
│   ├── anova_models.py                 # Tables 3-4 — repeated-measures ANOVA (Eq. 9)
│   ├── engagement.py                   # Table 5 — engagement metrics
│   ├── mediation.py                    # Tables 6-7 — mediation decomposition (bootstrap)
│   ├── moderation.py                   # Table 8 — moderation + simple slopes
│   ├── task_type.py                    # Table 9 — task-type effects
│   ├── retention.py                    # Table 10 — long-term retention
│   └── sem.py                          # SEM fit indices + Table 12 design validation
│
├── visualization/                      # Figures 6–11 reproduction
│   ├── style.py                        # Shared color palette & publication style
│   ├── fig6_baseline.py
│   ├── fig7_main_effect.py
│   ├── fig8_engagement.py
│   ├── fig9_mediation.py
│   ├── fig10_moderation.py
│   └── fig11_retention.py
│
├── utils/
│   ├── data_loader.py                  # Typed accessors for all 11 data sheets
│   └── stats_utils.py                  # Cohen's d, ANOVA, bootstrap mediation, etc.
│
└── results/
    ├── tables/                         # 30 generated CSV files (Tables 2–12 + diagnostics)
    └── figures/                        # 6 publication-quality PNG figures (300 DPI)
```

---

## Quick Start

**Requirements:** Python ≥ 3.10

```bash
# Clone the repository
git clone https://github.com/<your-username>/MEEM-MARL-Pipeline.git
cd MEEM-MARL-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (all 4 phases, ~20 seconds)
python run_all.py

# Quick mode (fewer RL episodes, faster)
python run_all.py --quick

# Skip deep-learning pipeline (statistics + figures only)
python run_all.py --skip-meem
```

Each module is independently runnable:

```bash
# Run individual MEEM steps
python -m meem.step1_preprocessing
python -m meem.step3_attention_fusion
python -m meem.pipeline

# Run individual analysis modules
python -m analysis.descriptive
python -m analysis.mediation
python -m analysis.retention

# Generate individual figures
python -m visualization.fig7_main_effect
python -m visualization.fig9_mediation
```

---

## Algorithm 1 — MEEM-MARL Pipeline

| Step | Module | Algorithm 1 Correspondence |
|------|--------|---------------------------|
| 1 | `step1_preprocessing.py` | Median filter (w=5), Z-score normalization, 2-s sliding window (50% overlap), PCA (95% variance), SMOTE |
| 2 | `step2_feature_extraction.py` | F_eye = 1D-CNN; F_face = CNN-FACS; F_voice = Transformer; F_phys = BiLSTM; F_log = MLP |
| 3 | `step3_attention_fusion.py` | α_m = softmax(w_a·F_m + b_a); F_fused = Σ α_m·F_m |
| 4 | `step4_affective_estimator.py` | CNN-LSTM → e_t (7-emotion); MLP → [Eng_t, Pres_t]; Reg → CL_t |
| 5 | `step5_moral_modeling.py` | z_a = Attention(F_fused, Θ_VI); z_r = GRU(z_a, M_mem); z_p = Reproduction(z_r, C) |
| 6 | `step6_outcome_predictor.py` | Ŷ_MC = σ(W_mc·[z_p;e_t;Eng_t]); Ŷ_ME = σ(W_me·[z_p;e_t;Pres_t]); Ŷ_BI = σ(W_bi·[z_p;e_t;Ŷ_ME]) |
| 7 | `step7_q_learning.py` | Q(s,a) ← Q(s,a) + η[R_t + γ·max Q(s',a') − Q(s,a)] |
| 8 | `step8_recommender.py` | c* = argmax U(c \| P, s_t, π*) |
| 9 | `step9_effect_quantification.py` | d = (Ȳ_VI − Ȳ_ctrl)/S_p; Ret(t) = Ŷ(t)/Ŷ(T_0); {β_ij} via SEM |

---

## Statistical Analysis Modules

All formulas reference equation numbers in the paper:

| Equation | Description | Module |
|----------|-------------|--------|
| Eq. 2 | Cronbach's α = (k/(k-1))·(1 − Σσ²ᵢ/σ²_t) | `utils/stats_utils.py` |
| Eq. 9 | Mixed model: Y_ijk = μ + αᵢ + βⱼ + (αβ)_ij + γ_k + ε_ijk | `analysis/anova_models.py` |
| Eq. 10 | Cohen's d = (X̄₁ − X̄₂)/S_pooled | `utils/stats_utils.py` |
| Eq. 11 | L_total = λ_aff·L_aff + λ_moral·L_moral + λ_reg·‖θ‖² | `meem/step6_outcome_predictor.py` |
| Baron–Kenny | Mediation: indirect = a·b; bootstrap CI (n=2000) | `analysis/mediation.py` |

---

## Visualization

The `visualization/` module reproduces all six figures from the paper:

| Figure | Description |
|--------|-------------|
| Fig. 6 | Baseline measurement comparison (age / tech proficiency / prior knowledge / correlations) |
| Fig. 7 | Pre-post-follow-up trajectories + Cohen's d effect sizes + % gain interaction |
| Fig. 8 | Multi-dimensional engagement (duration / completion / AOI fixation / interaction frequency) |
| Fig. 9 | Mediation paths + triple-mediator SEM diagram (with fit indices) |
| Fig. 10 | Boundary conditions — technology proficiency moderation + simple-slope decomposition |
| Fig. 11 | Long-term retention forgetting curves + cumulative learning effect |

All figures are output at 300 DPI to `results/figures/`.

---

## Key Results

### Main Effects (Table 3)

| Measure | Virtual Idol | Traditional | Control |
|---------|-------------|-------------|---------|
| Moral Cognition gain (%) | **+42.5** | +26.1 | +8.9 |
| Moral Emotion gain (%) | **+43.7** | +28.4 | +11.6 |
| Behavioral Intention gain (%) | **+39.9** | +23.6 | +6.7 |
| Cohen's d | **0.92** | 0.54 | 0.18 |

### Mediation Decomposition (Table 7)

| Path | Mediation % |
|------|-------------|
| VI → Social Presence → Moral Emotion | **60.3%** |
| VI → Engagement → Moral Cognition | 36.7% |
| VI → Emotional Response → Behavioral Intention | 52.7% |
| VI → Multiple Mediators → Learning Outcomes | **76.2%** |

### SEM Model Fit

| Index | Value | Threshold | Assessment |
|-------|-------|-----------|------------|
| χ²/df | 2.18 | < 3 | ✓ Good |
| CFI | 0.963 | > 0.90 | ✓ Good |
| TLI | 0.952 | > 0.90 | ✓ Good |
| RMSEA | 0.059 | < 0.08 | ✓ Good |

### 8-Week Retention (Table 10)

| Group | Cognition | Emotion | Behavior |
|-------|-----------|---------|----------|
| Virtual Idol | **86.2%** | **81.5%** | **77.3%** |
| Traditional | 71.8% | 65.4% | 58.9% |
| Control | 60.3% | 52.7% | 45.1% |

### MEEM-MARL Pipeline Outputs

- **Attention weights**: α_eye=0.196, α_face=0.198, α_voice=0.207, α_phys=0.199, α_log=0.200
- **RL convergence**: Final average reward = 4.72 (100 episodes)
- **Optimal policy distribution**: Content (53.8%), Expression (31.7%), Tone (8.6%), Pacing (5.9%)

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{meem_marl_2024,
  title   = {A Quantitative Evaluation Framework for Emotional Computing of the
             Moral Role Model Effect of Virtual Idols Based on
             Attention-Weighted Multimodal Fusion},
  journal = {[Journal Name]},
  year    = {2024},
  note    = {Manuscript under review}
}
```

---

## Ethics Statement

This study was conducted in accordance with the Declaration of Helsinki and approved by the Institutional Review Board of Central South University (approval number: CSU-PSY-2024-007). All participants provided written informed consent prior to enrollment. Participation was voluntary, and participants could withdraw at any time without consequence. All data are de-identified; participant records use coded identifiers (P001–P186) and no personally identifiable information is stored in this repository.

---

## Data Availability

The experimental dataset (n=186 participants, 11 measurement sheets) is provided as a companion data package. It contains:

- `虚拟偶像道德榜样效应实验_原始数据.xlsx` — Full raw data (11 sheets: demographics, moral cognition, moral emotion, behavioral intention, eye tracking, physiology, facial expression, engagement logs, subjective measures, item bank)
- `数据字典.pdf` — Variable codebook describing all 200+ variables, coding conventions, and measurement instruments
- `实验材料汇编.docx` — Full intervention materials: 4-week virtual idol narrative scripts and 10 moral dilemma scenarios used in the experimental condition

Place the contents of the data package into the `data/` folder to reproduce all analyses.

---

## Dependencies

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
statsmodels>=0.14
scikit-learn>=1.3
imbalanced-learn>=0.11
torch>=2.0
semopy>=2.3
pingouin>=0.5
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
```

---

## License

This code is released under the MIT License. The dataset is released for academic research use only.
