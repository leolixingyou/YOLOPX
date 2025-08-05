# Guide to Implementing MTL Gradient Strategies

This document provides a critical analysis of legacy code, a clear distinction between different MTL approaches, and a concrete plan for implementing gradient-based strategies within this directory. This plan is based on a synthesis of the academic papers in `Ref/papers.md` and the analysis in `yolop_series/mtl/IMPLEMENTATION_SUMMARY.md`.

---

## 1. Critical Assessment: Gradient vs. Architectural Strategies

A multi-task learning system can be improved in two fundamentally different ways:

1.  **Architectural Strategies**: These methods modify the network's `forward()` pass. They introduce new layers or modules (e.g., attention, feature fusion) to create better, more task-specific feature representations *before* the loss is even calculated. **TAG** and **MDO** fall squarely into this category. Their core contributions are `nn.Module`s that change how features flow and are transformed.

2.  **Gradient Strategies**: These methods operate during the `backward()` pass. They take the losses or gradients from standard task heads and manipulate them to mitigate conflict *before* updating the shared weights. **GradNorm**, **PCGrad**, and **CAGrad** are pure gradient strategies.

**Crucial Insight**: The legacy `yolop_series/mtl` code conflated these two concepts. It had architectural ideas (like TAG) mixed with gradient solvers. This is a flawed design. Architectural changes belong in the model's definition (`/models/yolop_unified.py`), while this `mtl_strategies` directory should **exclusively** contain gradient strategies that plug into the training loop.

**Your decision to separate MDO was correct.** It, along with TAG, should be treated as a separate experiment in modifying the YOLOP model architecture, not as a plug-and-play gradient solver.

---

## 2. From Paper to Code: A Plan for True Implementations

Our goal is to implement each gradient strategy as a class inheriting from `MTLStrategy`, ensuring they are faithful to their respective papers.

### 2.1. `PCGradStrategy`

-   **Paper Concept**: For any pair of conflicting gradients, project one onto the normal plane of the other to remove the conflicting component.
-   **Legacy Code (`grad_solvers.py`)**: The implementation was functionally correct but looped through each parameter's gradients individually. This is inefficient.
-   **Implementation Plan (`pcgrad.py`)**:
    1.  Define `class PCGradStrategy(MTLStrategy)`.
    2.  In `backward(losses, shared_params, **kwargs)`:
        a. Calculate the per-task gradients for all `shared_params`.
        b. **Flatten** the gradients for each task into a single vector `g_i_flat`.
        c. Create a buffer of these flattened gradients.
        d. Iterate through the buffer. For each pair `(g_i_flat, g_j_flat)`, if `cosine_similarity < 0`, perform the projection: `g_i_flat -= (dot_product / norm_sq) * g_j_flat`.
        e. After the loop, the buffer contains the modified, non-conflicting flattened gradients.
        f. Sum the gradients in the buffer: `g_final_flat = torch.sum(torch.stack(buffer), dim=0)`.
        g. **Apply the final gradient**: Un-flatten `g_final_flat` and apply it to the `.grad` attribute of each corresponding parameter in `shared_params`.

### 2.2. `CAGradStrategy`

-   **Paper Concept**: Find a new gradient `g` that is a minimal perturbation from the average gradient `g_avg`, under the constraint that `g` has a non-negative cosine similarity with all individual task gradients `g_i`.
-   **Legacy Code (`grad_solvers.py`)**: The implementation correctly identified the dual optimization problem and used a simple gradient descent solver. This is a viable approach.
-   **Implementation Plan (`cagrad.py`)**:
    1.  Define `class CAGradStrategy(MTLStrategy)`.
    2.  The `__init__` method should accept the hyperparameter `c` (defaulting to `0.5` as per the paper).
    3.  In `backward(losses, shared_params, **kwargs)`:
        a. Get the flattened per-task gradients `g_i_flat`.
        b. The logic from the legacy `_cagrad_grads` is a strong starting point and can be adapted directly. It correctly computes the average gradient `g0`, forms the matrix `A = G @ G.T`, and solves for the `alphas` that determine the final gradient weighting.
        c. Compute the final gradient `g_final_flat = alphas @ G_matrix`.
        d. Apply this final flattened gradient to the shared parameters as in PCGrad.

### 2.3. `GradNormStrategy`

-   **Paper Concept**: Dynamically learn loss weights `w_i` by performing gradient descent on a special loss function `L_grad` that penalizes imbalances in both gradient magnitudes and task training rates.
-   **Legacy Code (`grad_solvers.py`)**: The implementation was **not a true representation of GradNorm**. It was a heuristic that periodically set weights based on loss ratios, omitting the core concept of `L_grad` and the gradient-based learning of weights.
-   **Implementation Plan (`gradnorm.py`)**:
    1.  Define `class GradNormStrategy(MTLStrategy)`.
    2.  The `__init__` method must:
        a. Define `self.weights = nn.Parameter(torch.ones(num_tasks))`. These are the learnable loss weights.
        b. Define a separate optimizer for these weights: `self.weights_optimizer = torch.optim.Adam([self.weights], lr=weight_lr)`.
        c. Store the hyperparameter `alpha`.
        d. Store the initial losses `L_i(0)` after the first batch.
    3.  The `backward` method is a **two-step process**:
        a. **Step 1: Update Model Weights**. Compute the main weighted loss `L_main = sum(self.weights * losses)` and call `L_main.backward(retain_graph=True)`. The `retain_graph=True` is essential for the next step.
        b. **Step 2: Update Loss Weights**. 
           i. Get the L2 norm of the gradients on the **last shared layer** for each task. This requires accessing `.grad` on that layer's weights from the previous backward call.
           ii. Calculate the loss ratios `r_i(t)`.
           iii. Calculate the average gradient norm `G_bar_W(t)`.
           iv. Formulate the GradNorm loss: `L_grad = Σ | G_W_i(t) - G_bar_W(t) * [r_i(t)]^α |`.
           v. Zero the gradients of the weights optimizer: `self.weights_optimizer.zero_grad()`.
           vi. Backpropagate the GradNorm loss: `L_grad.backward()`.
           vii. Update the weights: `self.weights_optimizer.step()`.
           viii. After the step, renormalize `self.weights.data` so they sum to `T`.
    4.  The main `step()` method of the strategy will only call `self.optimizer.step()`, which updates the main model parameters. The weight update happens inside `backward`.

This structured approach ensures our implementations are modular, faithful to the original papers, and clearly separated by their underlying philosophy (gradient vs. architecture).