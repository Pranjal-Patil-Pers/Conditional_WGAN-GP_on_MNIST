## Conditional WGAN-GP on MNIST

---

## Overview
This project implements a **class-conditional Wasserstein GAN with Gradient Penalty (WGAN-GP)** trained on the **MNIST dataset (32×32)**.  
The goal is to generate high-quality, label-conditioned images using a **projection discriminator** and **label-conditional generator**, and to explore training stability, latent interpolation, and truncation effects.

---

## Learning Objectives
- Implement and train a conditional WGAN-GP on MNIST.
- Incorporate a projection discriminator and label-conditional generator.
- Apply gradient penalty for Lipschitz continuity.
- Maintain a Generator EMA (Exponential Moving Average) for cleaner samples.
- Visualize loss curves, critic diagnostics, conditional samples, latent interpolations, and truncation sweeps.

---

## Model Architectures

### 1. Conditional Generator \( G(z, y) \)
**Inputs:**  
Noise vector \( z \in \mathbb{R}^{64} \) and label embedding \( e_g(y) \in \mathbb{R}^{32} \).

**Architecture:**
1. Concatenate \([z, e_g(y)] \rightarrow\) FC: \((64 + 32) \rightarrow 4 \times 4 \times 128\)
2. Reshape to \((128, 4, 4)\)
3. Three upsampling blocks with BatchNorm and ReLU:
   - Upsample ×2 → Conv(128→64) → BN → ReLU  
   - Upsample ×2 → Conv(64→32) → BN → ReLU  
   - Upsample ×2 → Conv(32→16) → BN → ReLU
4. Output: Conv(16→1, 3×3) → Tanh

**Notes:**  
- Kaiming (He) initialization for all Conv and Linear layers.  
- BatchNorm is applied only in the generator.

---

### 2. Projection Critic \( D(x, y) \)
**Inputs:**  
Image \( x \in \mathbb{R}^{1 \times 32 \times 32} \) (normalized to [−1,1]) and label \( y \).

**Architecture:**
1. Conv(1→32, 4×4, s=2, p=1) → LeakyReLU(0.2)  
2. Conv(32→64, 4×4, s=2, p=1) → LeakyReLU(0.2)  
3. Conv(64→128, 4×4, s=2, p=1) → LeakyReLU(0.2)  
4. Global sum pooling → \( f(x) \in \mathbb{R}^{128} \)  
5. Projection head:  
   \[
   D(x, y) = w^\top f(x) + \langle f(x), e(y) \rangle
   \]

**Notes:**  
- No sigmoid activation (raw Wasserstein scores).  
- No normalization layers (spectral/weight) are used.

---

## Loss Functions

### Critic Loss (with Gradient Penalty)
\[
L_C = - \mathbb{E}[D(x,y)] + \mathbb{E}[D(G(z,y),y)] + \lambda_{gp} \, \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x}, y)\|_2 - 1)^2]
\]
where  
\[
\hat{x} = \epsilon x + (1 - \epsilon)G(z,y), \quad \epsilon \sim \mathcal{U}(0,1)
\]  
and \( \lambda_{gp} = 10 \).

### Generator Loss
\[
L_G = - \mathbb{E}[D(G(z,y),y)]
\]

---

## Generator EMA (Exponential Moving Average)
After every generator update:
\[
\theta_{ema} \leftarrow \tau \theta_{ema} + (1 - \tau)\theta, \quad \tau = 0.999
\]

EMA improves sampling stability and image quality.

---

## Training Details

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 2 × 10⁻⁴ |
| Betas | (0.0, 0.9) |
| λgp | 10 |
| ncritic | 2–3 |
| Epochs | 10–20 |
| Batch Size | 128 |
| Device | GPU (recommended) |

---

## Plots and Visualizations

During training, plot:
- Losses: \( L_C, L_G \)
- Critic metrics: \( \mathbb{E}[D(x,y)] \), \( \mathbb{E}[D(G(z,y),y)] \), Gradient Penalty (GP)

At the end of training:
1. **Conditional Samples Grid:** 10×N (rows = digits 0–9)  
2. **Latent Interpolations:** Smooth transitions between two latent vectors per class  
3. **Truncation Sweep:** Scaling \( z \leftarrow \psi z \) with \( \psi \in \{3.0, 2.5, \ldots, 0.1\} \)

---

## Results Summary

| Visualization | Description |
|----------------|-------------|
| Conditional Grid | High-quality digit samples per class using EMA generator. |
| Interpolation Panel | Smooth morphing between latent endpoints, class-consistent. |
| Truncation Sweep | Demonstrates diversity (high ψ) vs. fidelity (low ψ). |

---




