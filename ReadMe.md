# Physics-Informed Learning

A personal Scientific Machine Learning (SciML) learning and implementation library focused on Physics-Informed Neural Networks (PINNs), Operator Learning, and their application to engineering problems.

* Current focus: Physics-Informed Neural Networks (PINNs)
* Operator Learning modules will be added as the repository expands.

The goal of this repository is not simply to reproduce benchmark examples, but to develop a framework-independent understanding of modern Scientific Machine Learning through implementations, derivations, and study notes.

Every topic is accompanied by study notes and is built from first principles rather than relying on high-level abstractions alone.

While my long-term goal is to apply these techniques to geotechnical and environmental engineering problems, the implementations and notes are written to be framework-agnostic and accessible to anyone learning PINNs and Scientific Machine Learning.

---

## Motivation

Physics-Informed Machine Learning sits at the intersection of:

* Machine Learning
* Scientific Computing
* Numerical Methods
* Partial Differential Equations (PDEs)
* Engineering Physics

My long-term goal is to develop scientifically consistent surrogate models for engineering systems.

I plan to begin with geo-environmental engineering applications before expanding to broader engineering problems.

To reach that goal, I am systematically studying and implementing the major ideas in Scientific ML from first principles.

---

## Repository Structure

```text
Physics-Informed-Learning/
├── ReadMe.md
├── requirements.txt
└── pinns/
    ├── pinns_notes.pdf
    │
    ├── 01_diffusion ────────────── pytorch · jax · deepxde
    ├── 02_burgers ──────────────── pytorch · jax · deepxde
    ├── 03_inverse_diffusion ────── pytorch · jax · deepxde
    ├── 04_poisson_2d ───────────── pytorch · jax · deepxde
    ├── 05_wave_1d ──────────────── pytorch · jax · deepxde
    │
    └── terzaghi_consolidation ──── nnx · deepxde    ← showcase
```

### Notes

Handwritten study notes created while working through foundational literature and implementations.

The notes are intentionally included to document the learning process alongside the code.

---

### PINNs

Current implementations:

| PINNs implemented for       | Concept                              |
| --------------------------- | ------------------------------------ |
| 01 - Diffusion Equation     | Basic PINN formulation               |
| 02 - Burgers Equation       | Nonlinear PDEs                       |
| 03 - Inverse Diffusion      | Parameter discovery                  |
| 04 - 2D Poisson Equation    | Multi-dimensional spatial domains    |
| 05 - Wave Equation          | Second-order time derivatives        |
| Terzaghi Consolidation      | Geotechnical engineering application |

Frameworks explored throughout the implementations include:

* PyTorch
* JAX
* Flax Linen
* Flax NNX
* DeepXDE

---

## Learning Philosophy

The objective of this repository is framework-independent understanding.

Every implementation is built around the same mathematical ideas:

* PDE residuals
* Boundary conditions
* Initial conditions
* Automatic differentiation
* Physics-informed loss functions

while exploring how different frameworks express those concepts.

---

## Current Areas of Study

* Physics-Informed Neural Networks
* Scientific Machine Learning foundations
* Geotechnical engineering applications
* JAX ecosystem (Flax, Optax, NNX)
* DeepXDE

---

## Why Multiple Frameworks?

Implementations are intentionally developed across multiple frameworks including PyTorch, JAX, Flax, and DeepXDE.

The objective is not framework specialization, but understanding the underlying mathematical concepts independently of any particular software stack.

---

## Planned Topics

### Advanced PINNs

* Spectral Bias
* Adaptive Collocation
* Residual-Based Adaptive Refinement (RAR)
* Loss Balancing Strategies
* Multi-Scale PINNs
* XPINNs / Domain Decomposition

### Operator Learning

* DeepONet
* Physics-Informed DeepONet
* Fourier Neural Operators (FNO)
* Transformer-Based Operator Learning

### Engineering Applications I am currently considering, given my background

* HELP-based surrogate modeling
* Landfill leachate prediction
* Contaminant transport modeling
* Geotechnical digital twins

---

## References

This repository is heavily inspired by foundational work in Scientific Machine Learning, including:

* Raissi et al. — Physics-Informed Neural Networks

Additional references are listed within individual modules as they are implemented.

---

## Status

This repository is actively developed as part of a long-term effort to build expertise in Scientific Machine Learning and apply it to real-world engineering research.
