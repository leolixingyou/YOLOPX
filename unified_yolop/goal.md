# Project Goal & Research Roadmap

**Document Version:** 1.0
**Last Updated:** 2025-08-05

---

## 1. The Ultimate Goal: Beyond a Collection of Tricks

Our primary objective is not merely to apply existing Multi-Task Learning (MTL) techniques to a YOLOP model. Instead, the goal is to produce a **novel, elegant, and extensible panoptic perception model**. This research aims to move beyond a simple aggregation of optimization methods and propose a more fundamental solution to the task conflict problem.

For a doctoral dissertation, the core contribution must be a **unified and coherent design philosophy**. We aim to develop a new model architecture or training paradigm that inherently minimizes task conflict, incorporates principles of rapid learning, and demonstrates clear, extensible design principles. This stands in contrast to creating a "Frankenstein" model that, while potentially effective, is a complex patchwork of disparate techniques.

---

## 2. Where We Are Now: The End of the Preparatory Phase

We are currently at a critical inflection point. We have just completed all necessary preparatory work and are about to begin the core experimental phase of our research.

**Completed Stages:**

1.  **Phase 0: Foundational Work (✅ Completed)**
    -   **Literature Review**: Systematically analyzed the YOLOP family of models (v1, v2, v3, X) and key MTL optimization papers (PCGrad, CAGrad, GradNorm, etc.). All summaries are documented in `Ref/papers.md`.
    -   **Code Unification**: Established `unified_yolop` as the single, authoritative codebase.
    -   **Flexible Model Loading**: Engineered the `builder.py` to be capable of constructing different YOLOP variants (`v1`, `v3`, `X`) from their official, isolated modules, ensuring faithful replication.

2.  **Phase 1: Technical Framework Implementation (✅ Completed)**
    -   **MTL Strategy Framework**: Designed and implemented a modular, plug-and-play framework for gradient-based MTL strategies located in `mtl_strategies/`.
    -   **High-Quality Implementations**: Created faithful and efficient implementations of `Original`, `PCGrad`, `CAGrad`, and a true, dual-optimizer `GradNorm`, which are superior to legacy versions.

In short, we have built a powerful and flexible experimental platform. We have the models to test, and we have the tools (MTL strategies) to test them with.

---

## 3. The Path Forward: From Analysis to Innovation

Our research plan is structured in three major upcoming stages:

### **Stage A: Systematic Experimentation & Analysis (Current Stage)**

This is our immediate focus. The goal is to generate the foundational data that will underpin all subsequent theoretical work.

-   **Objective**: To rigorously evaluate how different MTL gradient strategies perform across architecturally distinct models (specifically, high-conflict anchor-free vs. low-conflict anchor-based).
-   **Key Tasks**:
    1.  **Execute the Experiment Matrix**: Run a series of `8` core experiments, crossing 2 models (`YOLOPX`, `YOLOPv1`) with 4 MTL strategies (`Original`, `PCGrad`, `CAGrad`, `GradNorm`).
    2.  **Integrate Strategies**: Modify the training script (`run_experiment_v2.py`) to accept the `--mtl-strategy` flag and correctly use our new framework.
    3.  **Analyze Results**: Collect performance metrics (mAP, mIoU) and conflict metrics (TCI) to produce a comprehensive comparative report.

-   **Challenges & Difficulties in this Stage**:
    -   **Computational Cost**: Running 8 full training cycles is computationally expensive and time-consuming. We must ensure experiments are well-managed.
    -   **Hyperparameter Sensitivity**: `GradNorm` in particular has its own learning rate and `alpha` parameter. We may need to perform small-scale preliminary runs to find stable settings, to ensure the comparison is fair.
    -   **Data Interpretation**: The results may not be simple. An MTL strategy might reduce conflict but not improve performance, or vice-versa. We must be prepared to analyze these nuances deeply.

### **Stage B: Insight Generation & Hypothesis Formulation**

-   **Objective**: To transform the raw data from Stage A into a deep, theoretical understanding of the problem.
-   **Key Tasks**:
    1.  Analyze the quantitative results to find patterns.
    2.  Formulate a concrete, testable hypothesis that explains *why* anchor-based models exhibit lower conflict.

### **Stage C: Innovative Model Design & Validation**

-   **Objective**: To design, implement, and validate our novel solution, which will be the centerpiece of the dissertation.
-   **Key Tasks**:
    1.  Based on the hypothesis from Stage B, design a new model or training method.
    2.  Implement this new design.
    3.  Benchmark it against all previous results from Stage A to prove its superiority.
    4.  Validate its extensibility, for example, by testing on a multi-class detection problem.

By following this roadmap, we ensure our work is systematic, builds logically upon previous results, and culminates in the creation of a novel, well-defended scientific contribution.