# Terzaghi 1D Consolidation PINN

This implementation represents the first domain-specific application within this repository and serves as a transition from benchmark PINN problems to engineering-focused Scientific Machine Learning.

The objective is to model **Terzaghi's One-Dimensional Consolidation Theory** using Physics-Informed Neural Networks (PINNs).

---

## Why Terzaghi Consolidation?

Terzaghi's consolidation theory is a foundational concept in geotechnical engineering.

When a load is applied to a saturated clay layer, the load is initially carried by excess pore water pressure. As water gradually drains from the soil, excess pore pressure dissipates and the soil skeleton begins to carry the load, resulting in settlement.

This process is governed by the normalized consolidation equation:

\[
u_t = c_v u_{zz}
\]

where:

* (u) = excess pore water pressure
* (z) = depth
* (t) = time factor
* (c_v) = coefficient of consolidation

Although mathematically similar to the diffusion equation, the physical interpretation is directly related to pore pressure dissipation and soil settlement.

---

## Problem Setup

### Domain

* Depth: \(z \in [0,1]\)
* Time: \(t \in [0,1]\)

### Initial Condition

Immediately after loading:

\[
u(z,0)=1
\]

The entire soil layer initially carries the applied load as excess pore pressure.

### Boundary Conditions

For a double-drained clay layer:

\[
u(0,t)=0
\]

\[
u(1,t)=0
\]

representing highly permeable drainage layers at the top and bottom boundaries.

---

## Corner Discontinuity and Training Challenges

This problem contains a discontinuity at the corners of the domain:

\[
(z,t)=(0,0), (1,0)
\]

At (t=0), the Initial Condition requires:

\[
u=1
\]

while the Boundary Conditions require:

\[
u=0
\]

at the same locations.

This creates a sharp transition that neural networks often struggle to represent accurately. During training, the optimizer may partially violate the boundary conditions in order to reduce the overall loss.

To improve enforcement of the physical constraints, weighted loss terms were introduced:

\[
Loss_{total}
============

Loss_{PDE}
+
100,Loss_{IC}
+
100,Loss_{BC}
\]

This places stronger emphasis on satisfying the initial and boundary conditions while still minimizing the PDE residual.

### How to run

| Stack | Command |
| ----- | ------- |
| Flax NNX | `cd nnx && python run.py` |
| DeepXDE | `cd deepxde && python model.py` |

### Results

* `nnx/terzaghi_isochrones_without_penalty_terms.png` — NNX without IC/BC loss weighting.
* `nnx/terzaghi_isochrones.png` — NNX with weighted IC and BC losses.
* `deepxde/deepxde_terzaghi_isochrones.png` — DeepXDE (`loss_weights=[1, 100, 100]`).

---

## Frameworks Implemented

### Flax NNX

Implementation using Flax NNX and JAX.

Topics explored:

* NNX Modules
* NNX Optimizers
* Automatic Differentiation with `jax.grad`
* Vectorization with `jax.vmap`
* Custom PINN loss construction

### DeepXDE

Implementation using DeepXDE's PDE abstractions.

Topics explored:

* Geometry and time-domain definitions
* Initial and boundary conditions
* Automatic Jacobian and Hessian computation
* Weighted PINN losses

---

## Repository Context

This implementation completes the foundational PINN benchmark sequence within this repository:

1. Diffusion Equation
2. Burgers Equation
3. Inverse Diffusion
4. 2D Poisson Equation
5. Wave Equation
6. Terzaghi Consolidation

The next stage of the repository focuses on advanced PINN techniques, including:

* Spectral Bias
* Adaptive Collocation
* Residual-Based Adaptive Refinement (RAR)
* Loss Balancing Strategies
* Domain Decomposition Methods

---

## Future Direction

The long-term objective is to extend these ideas toward geo-environmental engineering applications, including:

* Landfill leachate generation and seepage estimation
* Contaminant transport modeling
* Physics-informed surrogate models
* Operator learning methods for engineering systems

This implementation serves as the first geotechnical application within the repository and provides a foundation for future research-oriented work.
