---
layout: post
title: "Z-Image and LADD: A First-Principles Guide to Fast Diffusion Distillation"
date: 2026-03-30
categories: engineering
math: true
---

**Target audience:** ML practitioners familiar with transformers and diffusion basics who want to understand how Z-Image generates images and how LADD distills multi-step diffusion models into few-step ones.

---

## Table of Contents

1. [Overview](#overview)
2. [Timeline & Evolution](#timeline--evolution)
3. [The Core Problem: Why Diffusion Models Are Slow](#1-the-core-problem-why-diffusion-models-are-slow)
4. [Flow Matching: Straightening the Path](#2-flow-matching-straightening-the-path)
5. [Z-Image Architecture: The Single-Stream DiT](#3-z-image-architecture-the-single-stream-dit)
6. [The Distillation Landscape: From Progressive to Adversarial](#4-the-distillation-landscape-from-progressive-to-adversarial)
7. [LADD: Latent Adversarial Diffusion Distillation](#5-ladd-latent-adversarial-diffusion-distillation)
8. [The LADD Training Loop in Detail](#6-the-ladd-training-loop-in-detail)
9. [Applying LADD to Z-Image](#7-applying-ladd-to-z-image)
10. [Summary](#summary)
11. [Key References](#key-references)

---

## Overview

Diffusion models generate stunning images but pay a steep price: they require 20--50 sequential denoising steps at inference, each a full forward pass through a billion-parameter network. A single 1024x1024 image can take several seconds even on an A100.

**Distillation** compresses that multi-step process into 1--4 steps. The idea is simple: train a **student** model to shortcut the teacher's iterative trajectory, producing comparable quality in a fraction of the time.

This post covers two systems that sit at the frontier of this problem:

- **Z-Image** --- Alibaba's 6B-parameter text-to-image model built on a **Scalable Single-Stream Diffusion Transformer (S3-DiT)** with flow matching. It represents the current state-of-the-art in open-weight image generation.
- **LADD (Latent Adversarial Diffusion Distillation)** --- Stability AI's method for distilling large diffusion models into few-step generators by training entirely in latent space with adversarial feedback from the teacher model itself.

Understanding both is essential for anyone building fast, high-quality image generation systems.

---

## Timeline & Evolution

| Year | Method | Key Innovation |
|------|--------|---------------|
| 2020 | DDPM (Ho et al.) | Denoising diffusion as a practical generative model |
| 2022 | Latent Diffusion / Stable Diffusion | Move diffusion to VAE latent space --- 64x cheaper |
| 2022 | Progressive Distillation (Salimans & Ho) | Halve steps iteratively: student matches two teacher steps in one |
| 2022 | Rectified Flow (Liu et al.) | Straight ODE trajectories reduce discretization error |
| 2023 | DiT (Peebles & Xie) | Replace UNet with a transformer backbone |
| 2023 | Consistency Models (Song et al.) | Self-consistency constraint enables single-step generation |
| 2023 | ADD / SDXL Turbo (Sauer et al.) | Adversarial loss + DINOv2 discriminator in pixel space |
| 2024 | **LADD** / SD3 Turbo (Sauer et al.) | Adversarial distillation entirely in latent space |
| 2024 | Stable Diffusion 3 (Esser et al.) | MMDiT with flow matching at scale |
| 2025 | **Z-Image** (Alibaba Tongyi) | 6B single-stream DiT with Qwen3 text encoder |

---

## 1. The Core Problem: Why Diffusion Models Are Slow

A diffusion model learns to reverse a noising process. During training, it sees images corrupted with increasing amounts of Gaussian noise and learns to predict and remove that noise. At inference, it starts from pure noise and iteratively denoises --- each step removing a small amount of noise until a clean image emerges.

The mathematical framework defines a forward process that adds noise:

$$x_t = \alpha_t \, x_0 + \sigma_t \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $x_0$: the clean image (or its latent representation)
- $x_t$: the noised version at timestep $t$
- $\alpha_t, \sigma_t$: noise schedule parameters controlling the signal-to-noise ratio
- $\epsilon$: random Gaussian noise

The model $F_\theta$ (where $\theta$ denotes the learnable parameters) learns to reverse this process. At inference, you start at $x_T$ (pure noise) and solve the reverse ODE or SDE step by step. Each step requires a full forward pass through $F_\theta$.

The problem: with 50 steps and a 6B-parameter model, generating one image means 300 billion multiply-accumulate operations. Cutting steps from 50 to 4 gives a ~12x speedup --- but naively skipping steps produces blurry, incoherent outputs because the ODE solver accumulates discretization error. Flow matching offers a way to reduce this error at its source.

---

## 2. Flow Matching: Straightening the Path

Both Z-Image and the models LADD was designed for (SD3) use **flow matching** instead of classical DDPM. The key insight: if the path from noise to data is a straight line, you can traverse it in a single Euler step with zero discretization error.

![Two panels comparing DDPM curved trajectories requiring many steps versus flow matching straight trajectories requiring few steps, with noise and data distributions shown as point clusters](/images/zimage-ladd/flow_matching.svg)

### The Flow Matching Formulation

Flow matching can be seen as a special case of the general forward process above, where $\alpha_t = 1-t$ and $\sigma_t = t$. This simplifies to a linear interpolation between noise and data:

$$x_t = (1 - t) \cdot x_0 + t \cdot \epsilon$$

- $t \in [0, 1]$: interpolation parameter (0 = clean data, 1 = pure noise)
- The trajectory from $x_0$ to $\epsilon$ is a straight line in latent space

The model learns the **velocity** --- the direction and magnitude of the flow:

$$v_\theta(x_t, t) = F_\theta(x_t, t)$$

The training target is simply:

$$\text{target} = \epsilon - x_0$$

- This is the constant velocity vector pointing from data to noise along the straight path.

At inference, the denoised output is recovered via:

$$\hat{x}_0 = x_t - t \cdot v_\theta(x_t, t)$$

- $v_\theta(x_t, t)$: the model's predicted velocity at the current point and time

### Why Straight Paths Matter for Distillation

With curved DDPM trajectories, each Euler step introduces error that compounds across steps. With straight flow-matching trajectories, even a coarse 4-step or 8-step Euler solver closely tracks the true path. This means the base model is already easier to distill --- distillation methods like LADD start from a better foundation.

Z-Image uses the `FlowMatchEulerDiscreteScheduler` and trains with velocity prediction. The noise schedule is further adapted via **dynamic time shifting**: a resolution-dependent shift adjusts the noise distribution so that larger images spend more training time at higher noise levels, where the denoising task requires understanding global structure. With the generative framework established, we turn to Z-Image's architecture.

---

## 3. Z-Image Architecture: The Single-Stream DiT

Z-Image is a 6-billion parameter text-to-image model that generates images at up to 2048x2048 resolution. Its architecture is a **Scalable Single-Stream Diffusion Transformer (S3-DiT)** --- a design that departs from both the UNet tradition (Stable Diffusion) and the dual-stream MMDiT design (Flux, SD3).

![Z-Image S3-DiT architecture showing the forward pass from inputs through refiners, concatenation, main transformer, to output, with color-coded components for text, image, transformer, and timestep](/images/zimage-ladd/zimage_architecture.svg)

### The Single-Stream Design

The defining choice in Z-Image is how text and image information interact. Consider three approaches:

| Design | How Text Meets Image | Example |
|--------|---------------------|---------|
| **Cross-attention** (UNet) | Text tokens attend to image features via separate cross-attention layers | Stable Diffusion 1/2, SDXL |
| **Dual-stream MMDiT** | Separate text and image streams with cross-attention bridges | Flux, SD3 |
| **Single-stream (S3-DiT)** | All tokens concatenated into one sequence; full self-attention | **Z-Image** |

Z-Image concatenates text embeddings and image latent tokens into a single sequence and processes them through unified self-attention. Every text token can attend to every image token and vice versa, with no architectural separation.

The advantage: maximum parameter efficiency. Every parameter in every attention layer is used for both modalities. The dual-stream approach in Flux duplicates parameters across streams --- Flux uses 12B parameters where Z-Image targets comparable quality with 6B.

### Text Encoder: Qwen3

Where most diffusion models use CLIP or T5 for text encoding, Z-Image uses **Qwen3** --- a full causal language model. The text encoding pipeline:

1. Format the prompt using Qwen3's chat template
2. Tokenize (max 512 tokens)
3. Extract the **second-to-last hidden state** (dimension 2560)
4. Keep only non-padding tokens (variable-length sequences)
5. Project from 2560 to 3840 via a learned `cap_embedder` (RMSNorm + Linear)

Using an LLM as the text encoder gives Z-Image better understanding of complex, compositional prompts compared to CLIP's contrastive embeddings. The model can reason about spatial relationships, negations, and multi-object scenes.

### VAE: Latent Space Encoding

Z-Image uses a convolutional **AutoencoderKL** (from the Flux/SD lineage) that compresses images into a compact latent representation:

- **Spatial downscale:** 8x (three downsample stages)
- **Latent channels:** 16 (matching Flux/SD3, up from 4 in SDXL)
- **Effective downscale with patching:** 16x (8x VAE $\times$ 2x patch)

The encoding formula normalizes the latent distribution:

$$z = (z_{\text{raw}} - \mu_{\text{shift}}) \times s_{\text{scale}}$$

- $z_{\text{raw}}$: raw VAE encoder output
- $\mu_{\text{shift}}$: shift factor (centers the distribution)
- $s_{\text{scale}}$: scaling factor (normalizes variance)

A 1024x1024 RGB image becomes a 64x64x16 latent tensor, which after 2x2 patchification becomes a sequence of $32 \times 32 = 1024$ tokens, each of dimension 3840.

### The Transformer Blocks

Each of the 30 S3-DiT blocks contains:

**Adaptive Layer Norm (adaLN):** The timestep $t$ is embedded and projected to produce per-layer scale ($\gamma$) and gate ($g$) parameters via a tanh-gated modulation:

$$h' = \gamma \cdot \text{RMSNorm}(h), \quad h_{\text{out}} = g \cdot h'$$

- $h$: input hidden state from the previous sub-layer
- $\gamma$: learned scale from timestep embedding
- $g$: learned gate (tanh activation) controlling information flow

**Self-Attention with 3D RoPE:** Queries and keys receive **Rotary Position Embeddings** across three axes:

- Temporal axis (dim 32): encodes frame position (text tokens use this axis for sequence position)
- Height axis (dim 48): spatial row position of image patches
- Width axis (dim 48): spatial column position of image patches

The attention uses 30 heads with dimension 128 each, and applies QK-Norm (RMSNorm on queries and keys before the dot product) for training stability.

**SwiGLU Feed-Forward:** A gated linear unit with SiLU activation:

$$\text{FFN}(x) = w_2 \cdot (\text{SiLU}(w_1 \cdot x) \odot w_3 \cdot x)$$

- $w_1, w_3$: parallel projections to hidden dim 10240
- $\odot$: element-wise multiplication (the "gate")
- $w_2$: projection back to 3840

### Refiners: Preprocessing Before the Main Transformer

Before concatenation, Z-Image applies specialized preprocessing:

- **Context Refiner** (2 transformer layers): processes text embeddings with self-attention only --- no timestep conditioning. This gives the text tokens a chance to build internal representations before mixing with image tokens.
- **Noise Refiner** (2 transformer layers): processes image tokens with adaLN timestep conditioning. This allows the model to adjust image token representations based on the current noise level before they meet text tokens.

### Inference

At inference, Z-Image generates images in 28--50 steps using Euler integration with classifier-free guidance (CFG). The CFG formula interpolates between conditional and unconditional predictions:

$$v_{\text{guided}} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

- $s$: guidance scale (typically 3.0--5.0)
- $v_{\text{cond}}$: velocity predicted with the text prompt
- $v_{\text{uncond}}$: velocity predicted with an empty/null prompt

Z-Image also supports **CFG truncation**: guidance is only applied at high noise levels (early steps) and disabled at low noise levels (later steps), reducing inference cost and avoiding over-saturation artifacts. Even with these optimizations, 28+ steps remain too slow for interactive use --- motivating distillation.

---

## 4. The Distillation Landscape: From Progressive to Adversarial

Before diving into LADD, it helps to understand why earlier distillation methods fell short and what problems LADD was designed to solve.

### Progressive Distillation

The simplest approach: train a student to collapse two teacher steps into one. Repeat recursively --- 50 steps become 25, then 12, then 6, then 3. Each halving requires a full training run, and quality degrades at very low step counts. Below 4 steps, outputs become noticeably blurry.

### Consistency Models

Enforce a self-consistency property: any point along the same ODE trajectory should map to the same output. This enables single-step generation in principle, but outputs lack the perceptual sharpness of multi-step models --- there's no signal pushing the model toward realistic high-frequency detail.

### Adversarial Diffusion Distillation (ADD)

ADD (used to create SDXL Turbo) introduced the key insight: use a **GAN-style discriminator** to force the student's outputs to be perceptually realistic, even in a single step. The discriminator uses DINOv2 features to distinguish real images from student outputs.

ADD produced a breakthrough in single-step quality, but had critical limitations:

1. **Pixel-space bottleneck:** The discriminator operated on RGB images, requiring the student's latent output to be decoded through the VAE. This consumed enormous VRAM.
2. **Fixed resolution:** DINOv2 accepts inputs up to 518x518, capping the training resolution.
3. **Dual losses required:** Both an adversarial loss ($L_{\text{adv}}$) and a distillation loss ($L_{\text{distill}}$) were necessary for stable training.

These limitations made ADD impractical for distilling the next generation of large models (SD3 at 8B parameters, Z-Image at 6B) at megapixel resolutions.

---

## 5. LADD: Latent Adversarial Diffusion Distillation

LADD solves ADD's limitations with one architectural insight: **use the teacher diffusion model itself as the discriminator backbone, operating entirely in latent space**.

![Side-by-side comparison of ADD requiring VAE decoder and pixel-space DINOv2 discriminator versus LADD operating entirely in latent space with the teacher model as discriminator backbone](/images/zimage-ladd/add_vs_ladd.svg)

### The Three Roles of the Teacher

In LADD, the pretrained teacher model serves triple duty:

| Role | What It Does |
|------|-------------|
| **Data Generator** | Generates synthetic training latents via multi-step sampling with CFG |
| **Feature Extractor** | Processes re-noised samples and provides intermediate features for discrimination |
| **Quality Anchor** | Its generative features encode what "real" latents look like at every noise level |

This unification is elegant: the same model that knows how to generate good images also knows how to judge them --- and it does both without ever touching pixel space.

### Generative vs. Discriminative Features

A central claim of LADD: **generative features** (from a diffusion model) are better discriminator backbones than **discriminative features** (from DINOv2, CLIP, etc.).

Why? Discriminative models have a **texture bias** --- they classify based on local texture patterns. Generative models have a **shape bias** closer to human perception --- they understand global structure because they must reconstruct it during denoising.

This matters because the discriminator's feature space determines what the student optimizes for. Texture-biased features push the student toward sharp textures but can miss structural coherence (wrong number of fingers, objects merging). Shape-biased features push toward globally coherent outputs.

### Noise-Level-Specific Feedback

LADD introduces a powerful control mechanism absent in ADD: by choosing the noise level $\hat{t}$ at which the discriminator processes samples, you control the **granularity** of its feedback.

- **High noise** $\hat{t}$ (near 1.0): the teacher processes heavily noised samples, so only global structure survives. The discriminator gives **structural feedback** --- object layout, composition, overall coherence.
- **Low noise** $\hat{t}$ (near 0.0): the teacher processes nearly clean samples, preserving fine detail. The discriminator gives **textural feedback** --- edges, textures, local realism.

The noise level is sampled from a **logit-normal distribution** $\pi(\hat{t}; m, s)$:

$$\hat{t} \sim \text{LogitNormal}(m, s)$$

- $m$: location parameter controlling the bias toward high or low noise
- $s$: scale parameter controlling the spread
- Sweet spot: $m = 1, s = 1$ balances structural and textural feedback

This replaces the ad-hoc loss weighting, CLIP guidance, and multi-scale discriminators that traditional GANs require to balance global coherence with local detail.

---

## 6. The LADD Training Loop in Detail

![The LADD training pipeline showing data generation by the teacher, the student denoising path, re-noising, feature extraction through the teacher as discriminator backbone, and adversarial loss flowing back to the student](/images/zimage-ladd/ladd_training_pipeline.svg)

Here is the complete training loop, step by step:

### Step 1: Generate Synthetic Training Data

The teacher model generates synthetic latents from text prompts using multi-step sampling with classifier-free guidance. These serve as the "real" data for the discriminator.

Key insight: **using synthetic data from the teacher eliminates the need for a separate distillation loss.** The teacher's distribution is already encoded in the data --- the adversarial loss alone is sufficient.

### Step 2: Noise the Synthetic Latent

Sample a student timestep $t$ from a discrete set. For multi-step distillation:

$$t \in \{1.0, \; 0.75, \; 0.5, \; 0.25\}$$

Add noise to the synthetic latent:

$$x_t = (1 - t) \cdot x_0 + t \cdot \epsilon$$

- $x_0$: clean synthetic latent from the teacher
- $\epsilon \sim \mathcal{N}(0, I)$: fresh Gaussian noise

### Step 3: Student Denoises

The student model predicts the clean latent from the noised input:

$$\hat{x}_0 = x_t - t \cdot v_\theta(x_t, t)$$

- $v_\theta$: the student's velocity prediction network (same architecture as the teacher, initialized from the teacher's weights)

### Step 4: Re-noise for Discrimination

Both the student's output $\hat{x}_0$ and the "real" synthetic latent $x_0$ are re-noised at a fresh timestep $\hat{t}$:

$$\hat{x}_{\hat{t}} = (1 - \hat{t}) \cdot \hat{x}_0 + \hat{t} \cdot \epsilon', \quad x_{\hat{t}} = (1 - \hat{t}) \cdot x_0 + \hat{t} \cdot \epsilon'$$

- $\hat{t} \sim \text{LogitNormal}(1, 1)$: sampled from the discriminator noise distribution
- $\epsilon'$: fresh noise (same noise used for both, so the only difference is $\hat{x}_0$ vs $x_0$)

### Step 5: Extract Teacher Features

Pass both re-noised samples through the **frozen teacher model**. After each attention block, extract the full token sequence. These token sequences are reshaped to their 2D spatial layout (preserving height and width structure).

### Step 6: Discriminator Heads Classify

Independent **2D convolutional discriminator heads** process the features from each attention block. Each head outputs a real/fake prediction, conditioned on:

- The noise level $\hat{t}$ (so the discriminator knows what detail level to expect)
- Pooled text embeddings (so it can assess prompt alignment)

The use of 2D convolutions (vs. ADD's 1D) is essential for supporting multi-aspect-ratio training --- 1D convolutions would conflate spatial dimensions when image tokens have varying height/width strides.

### Step 7: Compute Adversarial Loss

The student is trained with a standard adversarial objective --- fool the discriminator heads into classifying its outputs as real:

$$L_{\text{adv}} = \mathbb{E}\left[\sum_{l} D_l\left(\phi_l(\hat{x}_{\hat{t}}), \hat{t}, c\right)\right]$$

- $\phi_l(\cdot)$: features extracted from the teacher's $l$-th attention block
- $D_l$: discriminator head at layer $l$
- $c$: conditioning (text embeddings)
- Sum over all layers provides multi-scale feedback

The discriminator itself is trained with the standard GAN objective to correctly classify real vs. fake.

### Step 8: Update

- **Student:** updated via gradient descent on $L_{\text{adv}}$
- **Discriminator heads:** updated to better distinguish real from fake
- **Teacher:** frozen (no gradient updates)

### Multi-Step Training Schedule

For high-resolution training (above 512x512), LADD uses a **warm-up schedule** for the student timesteps:

| Phase | Iterations | Timestep Distribution | Purpose |
|-------|-----------|----------------------|---------|
| Warm-up | 0--500 | $p = [0, 0, 0.5, 0.5]$ for $t \in \{1, 0.75, 0.5, 0.25\}$ | Train on low-noise (easy) steps first |
| Full training | 500+ | $p = [0.7, 0.1, 0.1, 0.1]$ | Shift focus to high-noise (hard) steps |

This prevents early training instability at high noise levels where the denoising task is hardest.

### Inference with the Distilled Student

The student uses a fixed set of timesteps matching its training:

- **4-step inference:** $t \in \{1.0, 0.75, 0.5, 0.25\}$
- **2-step inference:** $t \in \{1.0, 0.5\}$
- **1-step inference:** $t = 1.0$ only

No classifier-free guidance is needed --- the student learns to produce guided-quality outputs directly, since it was trained on CFG-generated synthetic data. With the general LADD pipeline established, we can now map it onto Z-Image's architecture.

---

## 7. Applying LADD to Z-Image

Z-Image and LADD were developed independently (by Alibaba and Stability AI, respectively), but LADD's design is architecture-agnostic. Applying LADD to Z-Image requires mapping LADD's components onto Z-Image's S3-DiT architecture.

### Teacher and Student Setup

- **Teacher:** A frozen copy of the pretrained Z-Image model (6B parameters). Generates synthetic latents via 28--50 step sampling with CFG, and provides intermediate features for discrimination.
- **Student:** Initialized from the same Z-Image checkpoint. Trained to denoise in 1--4 steps.

Both share the same S3-DiT architecture, so the teacher's attention block features have the same dimensionality and structure as the student's.

### Feature Extraction Points

Z-Image's 30-layer transformer provides natural feature extraction points for the discriminator. After each S3-DiT block, the token sequence (containing both text and image tokens) can be reshaped to extract the image tokens in their 2D spatial layout. Independent discriminator heads are attached at a subset of layers to provide multi-scale feedback:

- **Early layers** (1--10): capture low-level features, textures, local structure
- **Middle layers** (11--20): capture mid-level composition, object relationships
- **Late layers** (21--30): capture high-level semantics, prompt alignment

### The Discriminator Heads

Each head is a lightweight 2D convolutional network:

1. Extract image tokens from the concatenated sequence
2. Reshape to 2D spatial layout (e.g., 32x32 for 1024x1024 images)
3. Apply 2D convolutions conditioned on $\hat{t}$ and pooled text embeddings
4. Output real/fake logits

The heads are small relative to the 6B teacher --- typically a few million parameters each --- so they add minimal training overhead.

### Compatibility with Flow Matching

LADD's re-noising procedure naturally aligns with Z-Image's flow matching formulation. Both use the same linear interpolation:

$$x_t = (1 - t) \cdot x_0 + t \cdot \epsilon$$

The teacher processes re-noised samples at timestep $\hat{t}$ exactly as it would during its own denoising --- no adaptation needed. The timestep conditioning through adaLN ensures the teacher's features are noise-level-aware.

### What Changes vs. Baseline Training

| Component | Baseline Z-Image Training | Z-Image + LADD |
|-----------|--------------------------|----------------|
| Training data | Real image-text pairs, encoded to latents | Synthetic latents from teacher (no real images needed) |
| Loss | MSE on predicted velocity | Adversarial loss from discriminator heads |
| Gradient flow | Through student only | Through student + discriminator heads |
| Teacher model | Not used | Frozen; generates data + extracts features |
| Timesteps | Continuous sampling | Discrete set: $\{1, 0.75, 0.5, 0.25\}$ |
| CFG at inference | Required ($s = 3$--$5$) | Not needed (baked into synthetic data) |
| Inference steps | 28--50 | 1--4 |

---

## Summary

Z-Image and LADD represent two complementary advances in image generation:

**Z-Image** introduces the single-stream DiT paradigm (S3-DiT), where text and image tokens share a unified attention mechanism. Combined with a Qwen3 LLM text encoder and flow matching, it achieves state-of-the-art quality at 6B parameters --- half the size of comparable dual-stream models like Flux.

**LADD** solves the distillation problem by making three key choices:
1. **Latent-only operation** --- never decode to pixel space, enabling megapixel training
2. **Teacher as discriminator** --- the pretrained model provides both synthetic data and discriminative features
3. **Noise-level feedback control** --- the re-noising timestep $\hat{t}$ controls whether the discriminator focuses on structure or texture

Together, they enable a pipeline where Z-Image's 50-step generation can be compressed to 4 steps while maintaining quality --- a ~12x speedup that makes real-time, high-resolution image generation practical.

The core lesson from both systems: **reuse what you already have.** Z-Image reuses a single parameter set for both modalities. LADD reuses the teacher model as the discriminator. Elegance in architecture often comes from finding multiple uses for the same components.

---

## Key References

| Year | Paper | Contribution |
|------|-------|-------------|
| 2020 | Ho et al., "Denoising Diffusion Probabilistic Models" | Established DDPM as a practical generative framework |
| 2022 | Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" | Moved diffusion to latent space (Stable Diffusion) |
| 2022 | Salimans & Ho, "Progressive Distillation for Fast Sampling" | First practical diffusion distillation method |
| 2022 | Liu et al., "Flow Straight and Fast" | Rectified flow / straight ODE trajectories |
| 2023 | Peebles & Xie, "Scalable Diffusion Models with Transformers" | DiT: transformer backbone for diffusion |
| 2023 | Song et al., "Consistency Models" | Self-consistency for single-step generation |
| 2023 | Sauer et al., "Adversarial Diffusion Distillation" | ADD: adversarial + distillation loss (SDXL Turbo) |
| 2024 | Sauer et al., "Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation" | **LADD**: latent-space adversarial distillation (SD3 Turbo) |
| 2024 | Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" | SD3: MMDiT + flow matching at scale |
| 2025 | Alibaba Tongyi, "Z-Image: An Efficient Image Generation Foundation Model with S3-DiT" | **Z-Image**: single-stream DiT with Qwen3 encoder |
