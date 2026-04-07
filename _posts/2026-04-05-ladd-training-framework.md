---
layout: post
title: "Distilling a 6B Image Model with LADD: Architecture, Experiments & Hard Lessons"
date: 2026-04-05
categories: engineering
math: true
---

**Target audience:** ML engineers familiar with diffusion models who want to understand how adversarial distillation works in practice — including the parts that break.

This is **Part 2** of our series on distilling [Z-Image](https://github.com/vionwinnie/Z-Image-LADD-distillation), a 6.15B parameter text-to-image model. [Part 1 covered data curation](/engineering/ladd-data-pipeline/) — how we assembled 500K diverse prompts from 9 sources. This post covers the training framework: the architecture, the loss function, hyperparameter tuning, scaling to 8 GPUs with FSDP, and the blunders that taught us the most.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The LADD Architecture](#2-the-ladd-architecture)
3. [The Discriminator Design](#3-the-discriminator-design)
4. [Loss Function & Training Loop](#4-loss-function--training-loop)
5. [Training Metrics: What to Watch and What Breaks](#5-training-metrics-what-to-watch-and-what-breaks)
6. [Hyperparameter Experiments](#6-hyperparameter-experiments)
7. [Results: 4 Steps vs 50 Steps](#7-results-4-steps-vs-50-steps)
8. [Scaling Up: The FSDP Journey](#8-scaling-up-the-fsdp-journey)
9. [The First Full Run: Mode Collapse at Scale](#9-the-first-full-run-mode-collapse-at-scale)
10. [The Hard Lessons: Blunders & Principles](#10-the-hard-lessons-blunders--principles)
11. [Summary & Next Steps](#11-summary--next-steps)
12. [Key References](#12-key-references)

---

## 1. Overview

The goal is simple to state: take a 50-step diffusion model and make it generate images in 4 steps, with minimal quality loss. The method is **LADD** — Latent Adversarial Diffusion Distillation (Sauer et al., 2024).

Unlike traditional knowledge distillation that minimizes MSE between teacher and student outputs, LADD uses **adversarial training** — a GAN (Generative Adversarial Network) — **in the teacher's latent feature space**. A lightweight discriminator learns to distinguish teacher features from student features, and the student learns to fool it. No pixel-space losses, no perceptual networks, no FID-optimizing tricks — just a GAN operating on frozen teacher representations.

The setup involves three models cooperating in a delicate balance:

| Component | Parameters | Role | Trainable |
|-----------|-----------|------|-----------|
| **Student** (S3-DiT) | 6.15B | Denoises in fewer steps | Yes |
| **Teacher** (S3-DiT) | 6.15B | Provides feature representations | No (frozen) |
| **Discriminator** (Conv heads) | 14M | Multi-scale adversarial feedback | Yes |
| Text Encoder (Qwen3) | ~3B | Encodes prompts | No (frozen) |
| VAE (AutoencoderKL) | ~0.5B | Latent ↔ pixel conversion | No (frozen) |

The student is initialized as an exact copy of the teacher. Training teaches it to skip steps — to produce in 4 steps what the teacher produces in 50.

---

## 2. The LADD Architecture

![Full LADD architecture showing the student, teacher, and discriminator models with data flow paths and gradient flow through the frozen teacher](/images/ladd-training-framework/ladd_architecture.svg)

The architecture has three key ideas that separate it from simpler distillation approaches.

### Idea 1: The student predicts velocity, not noise

Z-Image uses **flow matching** (Lipman et al., 2023) — a framework where, unlike noise-prediction diffusion (DDPM), the forward process interpolates _linearly_ between data and noise, and the model predicts the velocity of that interpolation:

$$x_t = (1 - t) \cdot x_0 + t \cdot \varepsilon$$

- $x_0$: clean latent (teacher-generated with CFG=5, 50 steps)
- $\varepsilon$: Gaussian noise
- $t$: timestep in $[0, 1]$ — higher means noisier

The student predicts a **velocity** $v_\theta$ and we recover the denoised latent:

$$\hat{x}_0 = x_t - t \cdot v_\theta(x_t, t, c)$$

- $v_\theta$: student's velocity prediction
- $c$: text conditioning from Qwen3

This is implemented in [`train_ladd.py:824`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.py#L824):

```python
# Convert velocity to denoised latent: x̂_0 = x_t - t * v
student_pred = x_t - t_bc * student_velocity
```

### Idea 2: Re-noising creates a shared comparison space

The student's denoised prediction $\hat{x}_0$ and the teacher's clean latent $x_0$ can't be compared directly by the discriminator — they might be at different quality levels for trivial reasons (the student just started training). Instead, both are **re-noised** to a shared noise level $\hat{t}$, sampled from a **logit-normal distribution** — a distribution that applies a sigmoid to a Gaussian sample, concentrating values in $(0, 1)$ away from the extremes:

$$\text{fake: } (1 - \hat{t}) \cdot \hat{x}_0 + \hat{t} \cdot \varepsilon_1$$

$$\text{real: } (1 - \hat{t}) \cdot x_0 + \hat{t} \cdot \varepsilon_2$$

This ensures the discriminator sees a smooth mix of noise levels rather than specializing on one scale.

From [`ladd_utils.py:59-76`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_utils.py#L59):

```python
def logit_normal_sample(batch_size, m=1.0, s=1.0, device="cpu", generator=None):
    """Sample from logit-normal: u ~ Normal(m, s²), t = sigmoid(u)."""
    u = torch.normal(mean=m, std=s, size=(batch_size,), generator=generator, device=device)
    t = torch.sigmoid(u)
    t = t.clamp(0.001, 0.999)
    return t
```

### Idea 3: Gradients flow through the frozen teacher

This is the subtlest part. The teacher's weights are frozen (`requires_grad_(False)`), but on the **fake path**, the computation graph is kept alive — no `torch.no_grad()`. This means gradients flow backward through the teacher's operations to reach the student:

```
student → x̂₀ → re-noise → teacher(forward, no_grad on weights) → disc → g_loss
                                    ↑ gradients flow through here
```

The real path uses `torch.no_grad()` because no gradient is needed — it's just providing a reference.

From [`train_ladd.py:914-920`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.py#L914):

```python
# Teacher forward WITH gradient graph (frozen weights, live graph)
_, fake_extras_grad = teacher(
    fake_input_grad,
    t_hat,
    prompt_embeds,
    return_hidden_states=True,
)
```

---

## 3. The Discriminator Design

![Multi-scale discriminator with 6 heads tapping teacher transformer layers at different depths, showing FiLM conditioning and logit summation](/images/ladd-training-framework/discriminator_design.svg)

The discriminator is not a full model — it's **6 independent lightweight heads**, each attached to a different layer of the teacher transformer. This multi-scale design is what makes LADD work with only 14M parameters (0.2% of the student).

### Why multiple layers?

Each teacher transformer block captures different abstractions:

| Layers | What they capture | Why it matters |
|--------|-------------------|----------------|
| 5, 10 | Texture, local patterns | Catches blurriness, color artifacts |
| 15, 20 | Object composition, spatial relationships | Catches structural errors |
| 25, 29 | Semantics, prompt alignment | Catches meaning drift |

If the discriminator only watched one layer, the student could learn to fool that specific abstraction level while degrading at others. Six layers make gaming the system much harder.

### Head architecture

Each head is a small convolutional network with **FiLM conditioning** (Feature-wise Linear Modulation) from the timestep and text embedding. FiLM works by learning a per-channel **scale** and **shift** from the conditioning signal, so the discriminator can adjust which features matter depending on the noise level and prompt:

From [`ladd_discriminator.py:67-72`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_discriminator.py#L67):

```python
cond = torch.cat([t_embed, text_embed], dim=-1)   # conditioning signal
film_params = self.cond_mlp(cond)                  # MLP predicts scale+shift
scale, shift = film_params.chunk(2, dim=-1)        # split in half along last dim
h = h * (1.0 + scale) + shift                     # modulate features
```

The MLP outputs a vector of size `hidden_dim * 2` (512), and `.chunk(2, dim=-1)` splits it into two halves of 256 each — one for scale, one for shift. Each feature channel gets its own modulation: a scale near 0 suppresses that channel, a large scale amplifies it.

The full head architecture from [`ladd_discriminator.py:18-85`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_discriminator.py#L18):

```python
class LADDDiscriminatorHead(nn.Module):
    def __init__(self, feature_dim=3840, hidden_dim=256, cond_dim=256):
        super().__init__()
        # FiLM conditioning: timestep + text → scale, shift
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # scale + shift
        )
        self.proj = nn.Linear(feature_dim, hidden_dim)

        # 2D conv layers (applied after reshaping tokens to spatial layout)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1)
        self.gn2 = nn.GroupNorm(16, hidden_dim // 2)
        self.conv_out = nn.Conv2d(hidden_dim // 2, 1, 1)
```

The pipeline per head: **Linear projection** (3840→256) → **FiLM modulation** (scale + shift from timestep and text) → **reshape to 2D spatial** → **two conv blocks** with GroupNorm → **1×1 conv** → **global mean pool** → scalar logit.

All 6 head logits are summed into `total_logit` for the final real/fake decision:

```python
for layer_idx in self.layer_indices:
    head_logits = self.heads[str(layer_idx)](img_feats, spatial_size, t_embed, text_embed)
    total_logit = total_logit + head_logits
```

---

## 4. Loss Function & Training Loop

![Single training step pipeline showing timestep sampling, student forward, re-noising, teacher feature extraction, and discriminator classification with asymmetric update schedule](/images/ladd-training-framework/training_step.svg)

### The hinge loss

LADD uses the **hinge loss** variant of adversarial training — the same loss used in spectral normalization GANs. It has a nice property: the discriminator loss saturates once it's confident, preventing it from becoming arbitrarily strong.

From [`ladd_discriminator.py:202-216`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_discriminator.py#L202):

```python
@staticmethod
def compute_loss(real_logits, fake_logits):
    """Hinge loss for GAN training."""
    d_loss = torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))
    g_loss = -torch.mean(fake_logits)
    return d_loss, g_loss
```

**Discriminator loss** pushes real logits above +1 and fake logits below -1. Once confident, the ReLU clips to zero — no further gradient signal. This prevents the discriminator from running away.

**Generator (student) loss** is simply $-\mathbb{E}[\text{fake logits}]$ — maximize the discriminator's score on student outputs.

### The asymmetric update schedule

The discriminator and student don't update at the same frequency. The discriminator updates **every step**, while the student updates only every $N$ steps (`gen_update_interval`):

From [`train_ladd.py:802, 887-944`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.py#L802):

```python
is_gen_step = (global_step % args.gen_update_interval == 0)

# Discriminator: always update
accelerator.backward(d_loss)
disc_optimizer.step()

# Student: update every N steps
if is_gen_step:
    accelerator.backward(g_loss_update)
    student_optimizer.step()
```

Why? The discriminator needs to stay ahead of the student to provide useful gradient signal. If both update every step, they oscillate. The discriminator uses a 10× higher learning rate (5e-5 vs 5e-6) for the same reason.

### Timestep curriculum

The student sees a mix of denoising difficulties, with a curriculum that shifts from easy to hard:

From [`train_ladd.py:408-440`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.py#L408):

```python
student_timesteps = [1.0, 0.75, 0.5, 0.25]

if global_step < warmup_steps:
    # Warmup: only easy tasks (shown for n=4; source computes dynamically)
    probs = [0.0, 0.0, 0.5, 0.5]
else:
    # Main phase: heavily favor t=1.0 (full denoising — the hard case)
    probs = [0.7, 0.1, 0.1, 0.1]
```

At $t = 1.0$, the student starts from pure noise — this is the hardest case and what matters most for few-step inference. At $t = 0.25$, it starts from a mostly-clean input. The curriculum warms up on easy cases before emphasizing the hard one.

---

## 5. Training Metrics: What to Watch and What Breaks

Adversarial training is fragile. Unlike supervised training where the loss monotonically decreases, GAN losses oscillate by design — the question is whether they oscillate _healthily_. Here are the metrics we logged to W&B every step and how to read them.

### d_loss and g_loss

**How they're computed** (from [`ladd_discriminator.py:202-216`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_discriminator.py#L202)):

$$d\_loss = \mathbb{E}[\text{ReLU}(1 - \text{real\_logits})] + \mathbb{E}[\text{ReLU}(1 + \text{fake\_logits})]$$

$$g\_loss = -\mathbb{E}[\text{fake\_logits}]$$

The discriminator loss pushes real logits above +1 and fake logits below -1. Once confident, the ReLU clips to zero — the loss saturates. The generator loss simply wants fake logits to be as high as possible (fool the discriminator).

**Healthy signs:**
- Both oscillate in the 0-4 range without trending to extremes
- Neither stays pinned at 0 for extended periods
- g_loss gradually trends downward (student improving)

**Danger signs:**
- **NaN** — training has diverged. Gradients exploded, likely from a broken FSDP config or incompatible optimizer settings. Example: [`eager-smoke-151`](https://wandb.ai/yeun-yeungs/ladd/runs/6nt3oo8k) crashed at step 303 with NaN losses after we experimented with FSDP settings that broke gradient flow. Once NaN appears, the run is unrecoverable — kill it.

![d_loss and g_loss going NaN immediately after step 1 due to broken FSDP gradient flow configuration](/images/ladd-training-framework/chart_nan_crash.png)
- **d_loss pinned at 0** — discriminator is too confident on every sample. With batch_size=1, this is expected (hinge loss trivially saturates on a single sample). With batch_size≥2, it means the discriminator is dominating.
- **g_loss flat and high** — student isn't improving despite non-zero d_loss. Check `grad_norm/student` — if it's zero on gen steps, the gradient path is broken.

**Batch size effect on loss noise:**

The per-step loss is computed on the micro-batch (per-GPU batch size), not the effective batch size. This has a dramatic effect on signal quality:

- **bs=1** ([`vulcan-tanagra-110`](https://wandb.ai/yeun-yeungs/ladd/runs/8h1tukl6)): Losses are extremely noisy — d_loss spikes between 0 and 4 every step, g_loss swings between -2 and 6. The hinge loss on a single sample is either fully saturated (0) or fully active — there's no middle ground. This makes it very hard to tell from the loss curves alone whether training is progressing.

- **bs=2** ([`q1ft7t1z`](https://wandb.ai/yeun-yeungs/ladd/runs/q1ft7t1z)): Noticeably smoother. With 2 samples, the loss can take intermediate values (e.g. one sample saturated, one not = loss of ~1 instead of 0 or 4). The trends become readable.

![Side-by-side d_loss and g_loss comparison showing bs=1 with sharp binary spikes versus bs=2 with smoother intermediate values](/images/ladd-training-framework/chart_bs1_vs_bs2_loss.png)

### disc/accuracy_real and disc/accuracy_fake

**How they're computed** (from [`ladd_eval.py:58-60`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/ladd_eval.py#L58)):

```python
accuracy_real = (real_logits > 0).float().mean()   # % of real samples classified as real
accuracy_fake = (fake_logits < 0).float().mean()   # % of fake samples classified as fake
```

These use a threshold of 0 (not the hinge margin of ±1). They measure whether the discriminator is _directionally_ correct, even if not confident enough to produce non-zero loss.

**Healthy range:** Both between 0.6-0.9. The discriminator is right more often than not, but the student fools it sometimes.

**Danger signs:**
- **Both pinned at 1.0** — discriminator dominance. Every sample is correctly classified with high confidence. The student gets no signal. This is what we saw in the [collapsed full run](#9-the-first-full-run-mode-collapse-at-scale) with bs=1.
- **Both below 0.5** — discriminator has collapsed and can't tell real from fake. The student gets random gradient directions. Several Phase 2 sweep runs showed `disc_accuracy_fake=0%` when the disc learning rate was too low.
- **accuracy_real high but accuracy_fake low** — discriminator learned to say "real" for everything. It correctly identifies real samples but can't catch fakes.

**Important caveat:** These are computed on the per-GPU micro-batch. With bs=1, accuracy can only be 0 or 1 — there are no intermediate values. With bs=2, it can be 0, 0.5, or 1. Don't over-interpret oscillating accuracy at small batch sizes.

![Side-by-side discriminator accuracy comparison showing bs=1 pinned at 0 or 1 versus bs=2 with intermediate 0.5 values possible](/images/ladd-training-framework/chart_bs1_vs_bs2_accuracy.png)

### disc/logit_gap

**How it's computed:**

```python
logit_gap = real_logits.mean() - fake_logits.mean()
```

The raw difference between the discriminator's average score on real vs fake samples. This is the most direct measure of how well the discriminator separates real from fake.

**Healthy range:** Positive and stable (2-6). The discriminator can tell them apart but isn't overwhelmingly confident.

**Danger signs:**
- **Logit gap > 15** — discriminator diverging, gradients likely exploding
- **Logit gap < 0.1** — discriminator provides no learning signal, real and fake look identical to it
- **Logit gap is NaN** — training has diverged (same as NaN loss)

### Per-layer logits (disc/layer_N_real, disc/layer_N_fake)

Each of the 6 discriminator heads (layers 5, 10, 15, 20, 25, 29) logs its own real/fake logit means. When all layers show nearly identical real and fake logits (as we saw in the collapsed run — `layer_10_fake ≈ layer_10_real`), it confirms the student outputs are indistinguishable from noise at every abstraction level.

Healthy training shows gaps at multiple layers, with late layers (25, 29) typically showing the largest gap (semantic-level discrimination is hardest to fool).

---

## 6. Hyperparameter Experiments

### The autoresearch setup

For hyperparameter tuning, we took inspiration from Andrej Karpathy's [autoresearch](https://x.com/karpathy/status/1884338155553030590) concept — have an AI agent run experiments autonomously overnight. We built a lightweight framework around this idea:

- [`research/experiment.py`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/research/experiment.py) — the **only file the agent modifies**. Hyperparameters are constants at the top; the script generates a bash command, trains, evaluates, and logs to W&B.
- [`research/program.md`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/research/program.md) — agent instructions: read prior results, propose the next experiment, modify `experiment.py`, run it, record the result, repeat.
- [`research/results.tsv`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/research/results.tsv) — experiment log (commit hash, KID, VRAM, status, description).

Each experiment ran 500 steps on a single A100 80GB (the debug split: 98 prompts, 512px). The agent ran autonomously for ~8 hours overnight, completing 21 experiments across two rounds.

### What we tuned (and in what order)

LADD has several hyperparameters that interact in non-obvious ways. Here's what each one controls:

| Hyperparameter | What it controls | Default | Why it matters |
|---------------|-----------------|:-------:|----------------|
| `student_lr` | Student optimizer learning rate | 5e-6 | Too high → divergence. Too low → slow learning. |
| `disc_lr` | Discriminator learning rate | 5e-5 | Must stay ahead of student (typically 10× higher). |
| `gen_update_interval` (GI) | Discriminator steps per student update | 5 | Controls D/G balance — the most sensitive knob. |
| `renoise_m` | Logit-normal mean for re-noising $\hat{t}$ | 1.0 | Controls what noise level the discriminator mostly sees. Higher → more noise → harder to distinguish. |
| `renoise_s` | Logit-normal std for re-noising $\hat{t}$ | 1.0 | Controls spread of noise levels. Wider → more diversity in discriminator feedback. |
| `disc_layer_indices` | Which teacher layers get discriminator heads | [5,10,15,20,25,29] | Determines which abstraction levels are supervised. |
| `disc_hidden_dim` | Hidden dimension per discriminator head | 256 | Controls head capacity — too large → overfitting, too small → underfitting. |

We tuned them in this order, each time fixing the best value and moving on:

1. **Learning rates** (student_lr, disc_lr) — establish stable training dynamics first
2. **Generator update interval** (GI) — the D/G balance knob with the most impact
3. **Noise schedule** (renoise_m, renoise_s) — controls discriminator's operating point
4. **Discriminator architecture** (layer_indices, hidden_dim) — structural choices, tuned last

### Round 1: Pre-fix sweep (broken pipeline)

These results used KID (Kernel Inception Distance) — an unbiased metric preferred over FID for small sample sizes. Lower is better.

**Noise schedule exploration** — tuning the logit-normal distribution parameters for re-noising:

| `renoise_m` | `renoise_s` | KID | vs baseline |
|:-----------:|:-----------:|:---:|:-----------:|
| 1.0 (default) | 1.0 | 0.00804 | — |
| 0.0 | 1.0 | 0.00551 | -31% |
| **0.5** | **1.0** | **0.00461** | **-43%** |
| -0.5 | 1.0 | 0.00508 | -37% |
| 0.5 | 0.5 | 0.00564 | -30% |
| 0.5 | 1.5 | 0.00559 | -30% |

The default $m = 1.0$ was too high — `sigmoid(1.0) ≈ 0.73`, meaning the discriminator mostly saw high-noise samples where real and fake are hard to distinguish. $m = 0.5$ (`sigmoid(0.5) ≈ 0.62`) shifted the distribution toward moderate noise levels where the discriminator gets more useful signal.

**Generator update interval** — how many discriminator steps per student update:

| GI | KID | vs baseline |
|:--:|:---:|:-----------:|
| 2 | 0.01102 | +37% (worse) |
| 3 | 0.00804 | baseline |
| 4 | 0.00241 | -70% |
| 6 | 0.00202 | -75% |
| **8** | **0.00087** | **-89%** |
| 10 | 0.01290 | +60% (worse) |

GI=8 was a dramatic win — 89% better than baseline. But this turned out to be an artifact of the broken pipeline (see [Section 8](#theme-1-validate-your-baseline-before-tuning-anything)).

**Discriminator architecture** — we also tested head configurations:

| Config | KID | Notes |
|--------|:---:|-------|
| 6 layers [5,10,15,20,25,29], dim=256 | 0.000869 | **best** |
| 3 layers [10,20,29] | 0.001469 | 69% worse |
| 8 layers [3,7,11,15,19,23,27,29] | 0.001513 | 74% worse |
| dim=128 | 0.001184 | 36% worse |
| dim=512 | 0.006385 | 635% worse |

The original 6-layer, 256-dim config was already optimal. More layers added noise without new signal; larger hidden dims made the heads harder to train.

Full results tracked in [`research/results.tsv`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/research/results.tsv).

### Round 2: Post-fix sweep (corrected pipeline)

After fixing 5 critical bugs, we re-ran the sweep. The results shifted significantly:

| Experiment | Config change | KID | vs baseline |
|:----------:|:-------------|:---:|:-----------:|
| baseline | slr=5e-6, dlr=5e-5, GI=8 | 0.0637 | — |
| exp1 | dlr=1e-5 (lower disc LR) | 0.0624 | -2% |
| **exp2** | **GI=3 (was 8)** | **0.0589** | **-7.5%** |
| exp3 | slr=2e-5 (higher student LR) | 0.0792 | +24% (worse) |
| exp4 | GI=3 + dlr=1e-5 | 0.0616 | -3.3% |

The optimal GI **flipped from 8 to 3** after fixing the pipeline. Why? In the broken pipeline, "real" samples were just noise-mixed-with-noise — trivially easy for the discriminator. It needed many steps to avoid overwhelming the student. With proper teacher latents as real samples, the discrimination task became genuinely hard, and the student needed more frequent updates to keep up.

### Current best configuration

```python
STUDENT_LR        = 5e-6    # Conservative — adversarial training is fragile
DISC_LR           = 5e-5    # 10x student LR (disc needs to lead)
GEN_UPDATE_INTERVAL = 3     # Update student every 3 disc steps
RENOISE_M         = 0.5     # LogitNormal mean (moderate noise bias)
RENOISE_S         = 1.0     # LogitNormal std (wide spread)
DISC_HIDDEN_DIM   = 256     # Per-head projection dimension
DISC_LAYER_INDICES = [5, 10, 15, 20, 25, 29]  # 6 of 30 teacher layers
```

---

## 7. Results: 4 Steps vs 50 Steps

### Student vs Teacher: visual comparison

Here's what the student produces at 4 steps compared to the teacher at 50 steps (CFG=5). These images are pulled directly from our W&B eval runs.

**Prompt: "The image captures a dynamic scene at a bullfighting arena..."**

| Teacher (50 steps, CFG=5) | Student (4 steps, 500 train steps) | Student (4 steps, 2000 train steps) |
|:---:|:---:|:---:|
| ![Teacher reference — bullfighting arena](/images/ladd-training-framework/teacher_ref_row0.png) | ![Student at 500 training steps — bullfighting arena](/images/ladd-training-framework/student_500steps_row0.png) | ![Student at 2000 training steps — bullfighting arena](/images/ladd-training-framework/student_2000steps_row0.png) |

**Prompt: "cyberpunk birthday party with robots, androids and flamenco guitarist watching mars sunset..."**

| Teacher (50 steps, CFG=5) | Student (4 steps, 500 train steps) | Student (4 steps, 2000 train steps) |
|:---:|:---:|:---:|
| ![Teacher reference — cyberpunk party](/images/ladd-training-framework/teacher_ref_row10.png) | ![Student at 500 training steps — cyberpunk party](/images/ladd-training-framework/student_500steps_row10.png) | ![Student at 2000 training steps — cyberpunk party](/images/ladd-training-framework/student_2000steps_row10.png) |

**Prompt: "The image displays a promotional advertisement for a speaker system. At the top of the image, in bold red letters..."**

| Teacher (50 steps, CFG=5) | Student (4 steps, 500 train steps) | Student (4 steps, 2000 train steps) |
|:---:|:---:|:---:|
| ![Teacher reference — speaker advertisement](/images/ladd-training-framework/teacher_ref_row9.png) | ![Student at 500 training steps — speaker advertisement](/images/ladd-training-framework/student_500steps_row9.png) | ![Student at 2000 training steps — speaker advertisement](/images/ladd-training-framework/student_2000steps_row9.png) |

This last example is the most encouraging — the student at 2000 steps picks up the layout (bold text, speaker image, URL at bottom) even though the details are still muddy. It shows the student _is_ learning structure, just slowly at this tiny scale. This is expected: 500-2000 steps with batch_size=1 on 98 prompts is barely scratching the surface. The LADD paper uses 50K-200K steps with large batch sizes. The production 8-GPU run (20K steps, effective batch 64) should close this gap significantly.

### Training progression

We tracked KID against 416 teacher-generated reference images (CFG=5, corrected scheduler) at different training checkpoints:

| Training steps | KID (↓ better) | d_loss | Observation |
|:--------------:|:--------------:|:------:|:------------|
| 500 | 0.0637 ± 0.0053 | 0.0 | Coarse structure emerges |
| 2000 | 0.0702 ± 0.0058 | 0.0 | KID worsens — disc collapse |

The KID **worsening** from 500 to 2000 steps was our first signal that something was off with the training dynamics. The discriminator loss collapsing to 0 at batch_size=1 meant the hinge loss saturated — the disc could perfectly separate real from fake with just 1 sample, providing no useful gradient.

This is expected at bs=1 and not a sign of discriminator dominance — the hinge margin of ±1 is trivially achieved with a single sample. At the production batch size of 64 (8 GPUs × 4 per-GPU × 2 grad accum), this saturation should resolve.

### Overfit experiments

To verify the architecture itself works, we ran two overfit tests on just 10 prompts:

| Experiment | LR | Result |
|-----------|-----|--------|
| Aggressive (slr=1e-4, dlr=1e-3) | 20× higher | **Diverged** — pure noise output |
| Winning LR (slr=5e-6, dlr=5e-5) | Standard | **Semantically correct** but blue color shift |

The winning-LR overfit produced recognizable images matching prompts — proof that the gradient flow and architecture work. The blue color shift was mode collapse from tiny data: with 10 prompts and bs=1, the student oscillates between pleasing individual prompts instead of learning general features.

**Key scaling evidence:**
- Text conditioning works — compositions match prompts (98-prompt run)
- More data = better images (98 prompts > 10 prompts)
- More steps = better KID (500 steps: 0.00804 → 2000 steps: 0.00723, pre-fix)
- Gradient flow confirmed — weight deltas grow, grad norms are non-zero
- Discriminator active — d_loss oscillates, not collapsed

The bottleneck is compute and data scale, not architecture.

### W&B run links

All experiments are tracked on W&B under project [`yeun-yeungs/ladd`](https://wandb.ai/yeun-yeungs/ladd):

- [v2 corrected training (500 steps)](https://wandb.ai/yeun-yeungs/ladd/runs/au7vsbzy)
- [v2 eval at 500 steps](https://wandb.ai/yeun-yeungs/ladd/runs/idj1oc8i)
- [v2 eval at 2000 steps](https://wandb.ai/yeun-yeungs/ladd/runs/2389e6fo)
- [v2 training at 2000 steps](https://wandb.ai/yeun-yeungs/ladd/runs/j0n7eah3)

---

## 8. Scaling Up: The FSDP Journey

![FSDP memory layout comparing 8-GPU sharded deployment at ~26GB per GPU versus single-GPU at ~78GB causing OOM](/images/ladd-training-framework/fsdp_memory.svg)

### Why FSDP?

On a single A100 80GB, the memory budget is brutal:

| Component | Size |
|-----------|------|
| Student (bf16) | 12 GB |
| Teacher (bf16) | 12 GB |
| Student optimizer (fp32 Adam) | 24 GB |
| Activations (512px, grad ckpt) | ~30 GB |
| **Total** | **~78 GB** → OOM |

We first tried **8-bit Adam** (bitsandbytes) to cut optimizer states from 24 GB to 6 GB. This worked at 256px but still OOM'd at 512px due to activation memory. The real solution was multi-GPU.

### DeepSpeed: 9 ways to fail

Before FSDP, we tried DeepSpeed ZeRO-2 with CPU optimizer offload. It failed in 9 distinct ways — each one a lesson in why DeepSpeed assumes single-model training:

1. **Dual engine crash** — wrapping both student and discriminator caused `IndexError` in gradient reduction
2. **Two-Accelerator pattern** — failed with `mpi4py` errors on second `deepspeed.initialize()`
3. **Student LR stuck at 0** — DeepSpeed's internal WarmupLR didn't configure correctly through Accelerate's "auto" values
4. **Frozen disc broke gradient flow** — `discriminator.requires_grad_(False)` during gen step severed the computation graph, student grad norms were zero
5. **Double gradient reduction** — both `d_loss.backward()` and `g_loss.backward()` triggered ZeRO-2's reduction hooks on student params
6. **Grad norms always zero** — captured after `zero_grad()`; ZeRO-2 manages gradients in internal buffers
7. **Checkpoint write failure** — `PytorchStreamWriter failed` on the ~24GB optimizer state file
8. **`os.execv` + `tee` incompatibility** — experiment runner output not captured
9. **GPU memory leak via `fork()`** — parent process held GPU memory, child OOM'd

**Conclusion:** DeepSpeed ZeRO is designed for single-model training. The GAN setup with alternating D/G updates, cross-model gradient flow, and two optimizers is fundamentally incompatible.

### FSDP configuration

FSDP (Fully Sharded Data Parallel) worked cleanly because it operates at the module level rather than the optimizer level. Our config wraps each of the 30 transformer blocks as separate FSDP units:

From [`training/fsdp_config.yaml`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/fsdp_config.yaml):

```yaml
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: ZImageTransformerBlock
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_use_orig_params: true      # Required for per-param optimizer groups
  fsdp_state_dict_type: FULL_STATE_DICT
```

Key design choice: the **teacher is NOT wrapped in FSDP** — it's replicated on every GPU. At 12 GB in bf16, it fits easily on 80GB A100s, and it only does forward passes (no optimizer states to shard). Wrapping it would add FSDP all-gather overhead for no memory benefit.

### Memory per GPU with FSDP

| Component | Per-GPU Size |
|-----------|:------------:|
| Student (sharded 1/8) | ~1.5 GB |
| Student optimizer (sharded 1/8) | ~6 GB |
| Teacher (full replica) | ~12 GB |
| Discriminator | ~0.03 GB |
| Activations + grad checkpointing | ~4-6 GB |
| **Total** | **~26 GB / 80 GB** |

Comfortable margin. No need for 8-bit Adam, CPU offloading, or precomputed text embeddings on the cluster.

### Production launch

From [`training/train_ladd.sh`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.sh):

```bash
accelerate launch \
    --config_file training/fsdp_config.yaml \
    training/train_ladd.py \
    --train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=20000 \
    --learning_rate=5e-6 \
    --learning_rate_disc=5e-5 \
    --gen_update_interval=3 \
    --mixed_precision=bf16 \
    --gradient_checkpointing \
    --checkpointing_steps=2000 \
    --validation_steps=2000
```

Effective batch size: $4 \times 8 \times 2 = 64$. Target: 20K steps in ~2 hours.

---

## 9. The First Full Run: Mode Collapse at Scale

We launched the first 8-GPU production run ([`yeun-yeungs/ladd/stjmyjsi`](https://wandb.ai/yeun-yeungs/ladd/runs/stjmyjsi)) — 8x A100 80GB, 10K precomputed latents, 20K target steps. After all the single-GPU validation, FSDP debugging, and hyperparameter sweeps, this was supposed to be the payoff run.

The results were devastating.

### What happened

At step 0 the student (initialized from teacher weights) produces recognizable images. By step 2000, all outputs collapse into the same colorful noise pattern regardless of prompt:

**Prompt: "A row of colorful, stylized, and simplified animal figures..."**

| Teacher (50 steps) | Student step 0 | Student step 2000 | Student step 4000 |
|:---:|:---:|:---:|:---:|
| ![Teacher — animal figures](/images/ladd-training-framework/fullrun_teacher_row0.png) | ![Student step 0 — animal figures](/images/ladd-training-framework/fullrun_step_0_student_row0.png) | ![Student step 2000 — collapsed to noise](/images/ladd-training-framework/fullrun_step_2000_student_row0.png) | ![Student step 4000 — same noise pattern](/images/ladd-training-framework/fullrun_step_4000_student_row0.png) |

**Prompt: "videogame screenshot of a very psychedelic dreamy luxury flooded tropical universe..."**

| Teacher (50 steps) | Student step 0 | Student step 2000 | Student step 4000 |
|:---:|:---:|:---:|:---:|
| ![Teacher — psychedelic room](/images/ladd-training-framework/fullrun_teacher_row30.png) | ![Student step 0 — psychedelic room](/images/ladd-training-framework/fullrun_step_0_student_row30.png) | ![Student step 2000 — collapsed to noise](/images/ladd-training-framework/fullrun_step_2000_student_row30.png) | ![Student step 4000 — same noise pattern](/images/ladd-training-framework/fullrun_step_4000_student_row30.png) |

Every prompt produces the same speckled noise. The KID at step 4000: **0.593** — catastrophically high. For reference, the untrained student (teacher weights, zero LADD training) scores KID = 0.069. Training didn't just fail to improve — it made the model **8.6× worse** than not training at all.

### Diagnosing from W&B

The discriminator accuracy charts told a misleading story — `disc/accuracy_real` and `disc/accuracy_fake` both pinned at 1.0 throughout training. At first glance, this screamed "discriminator too strong, student getting no gradient."

But `d_loss` and `g_loss` were actually non-zero and oscillating (0-4 range). The losses existed — so why wasn't the student learning?

The answer: the accuracy was computed on a **per-GPU micro-batch of 1 sample**. With batch_size=1, the discriminator trivially classifies one sample. The accuracy metric was degenerate, not the training signal itself. We needed to look at the loss curves, not the accuracy.

### Root cause: two misconfigured hyperparameters

The production run diverged from our validated best config in two critical ways:

| Parameter | Production run | Validated best | Impact |
|-----------|:--------------:|:--------------:|--------|
| `train_batch_size` | **1** | 2-4 | Hinge loss on 1 sample saturates to 0 every micro-step. Gradient accumulation sums 8 zeros = zero. |
| `renoise_m` | **1.0** | 0.5 | Discriminator mostly sees heavily noised samples (sigmoid(1.0)=0.73). Gradient signal is weak even when non-zero. |

The combination was lethal: weak gradients from high-noise re-noising (`m=1.0`), further zeroed out by degenerate hinge loss on single samples. The student drifted from its teacher-initialized weights into noise with no corrective signal.

### The fix

```bash
--train_batch_size=2              # was 1 — hinge loss needs >1 sample
--gradient_accumulation_steps=4   # was 8 — keeps effective BS=64 (2×4×8)
--renoise_m=0.5                   # was 1.0 — 43% better in sweep
--warmup_schedule_steps=0         # was 10 — no benefit in sweep
```

> **Lesson:** The total effective batch size is not the only thing that matters — the **per-micro-step batch size** determines whether the loss function produces meaningful gradients. Hinge loss with bs=1 is degenerate. And always double-check that production configs match your validated sweep winners.

Eval images from [`yeun-yeungs/ladd-eval`](https://wandb.ai/yeun-yeungs/ladd-eval?nw=nwuserdcvionwinnie).

---

## 10. The Hard Lessons: Blunders & Principles

This section is the most valuable part of this post. We made mistakes that cost days of debugging and invalidated entire experiment rounds. They cluster into four themes, each with a principle we now follow.

### Theme 1: Validate your baseline before tuning anything

We ran 21 hyperparameter experiments and found a configuration that scored KID = 0.000869 — an 89% improvement over baseline. Then we discovered **five critical bugs** in the training pipeline, and every single KID number became meaningless.

#### Bug 1: Scheduler linspace off-by-one

Our custom `FlowMatchEulerDiscreteScheduler.set_timesteps()` had a subtle indexing error:

```python
# BROKEN — leaves 11% residual noise at the final step
timesteps = np.linspace(sigma_max_t, sigma_min_t, num_inference_steps + 1)[:-1]

# CORRECT — matches diffusers exactly
timesteps = np.linspace(sigma_max_t, sigma_min_t, num_inference_steps)
```

With 50 inference steps, the scheduler's smallest sigma was 0.109 instead of reaching 0.0. Teacher images had 11% residual noise — they looked blurry, and we were evaluating KID against blurry references.

#### Bug 2: Teacher images generated without CFG

The [`precompute_fid_reference.py`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/scripts/precompute_fid_reference.py) script generated teacher reference images with `guidance_scale=0`. The Z-Image model requires CFG (~5.0) for sharp, well-composed images. Without it, outputs were unconditional-like and lacked detail. On top of that, TeaCache was enabled (`teacache_thresh=0.5`), skipping 75% of transformer computations for speed at the cost of further quality degradation.

The difference is stark — here's the same prompt rendered with CFG=0 vs CFG=5 (from our W&B [`debug-teacher-cfg-sweep`](https://wandb.ai/yeun-yeungs/ladd/runs/mg6h6a9f) run):

**Prompt: "Two wine bottles with green glass and white labels..."**

| CFG=0 (no guidance) | CFG=3 | CFG=5 (selected) |
|:---:|:---:|:---:|
| ![Teacher output with no CFG — washed out, missing detail](/images/ladd-training-framework/teacher_cfg0.0_row5.png) | ![Teacher output with CFG=3 — improved but still soft](/images/ladd-training-framework/teacher_cfg3.0_row5.png) | ![Teacher output with CFG=5 — sharp labels, correct colors](/images/ladd-training-framework/teacher_cfg5.0_row5.png) |

Without CFG, the labels are illegible smudges. With CFG=5, the text is crisp and the bottle shapes are well-defined. We were evaluating our student against the left image — no wonder the KID numbers looked deceptively good.

#### Bug 3: Student input was pure noise regardless of timestep

The student at timestep $t = 0.25$ should see a mostly-clean input: $x_t = 0.75 \cdot x_0 + 0.25 \cdot \varepsilon$. Instead, it received pure noise at every timestep. At $t = 0.25$, the student was told "you're almost done denoising" while looking at pure static — it had no signal about what to reconstruct.

#### Bug 4: No velocity-to-latent conversion

The student predicts velocity $v_\theta$, but the code used raw velocity as the denoised prediction. The correct conversion is $\hat{x}_0 = x_t - t \cdot v_\theta$ — without it, the discriminator received meaningless inputs.

#### Bug 5: "Real" samples were noise vs. noise

The LADD paper (Section 3.2) specifies that "real" samples should be teacher-generated images re-noised to the discriminator's timestep. Our code used `add_noise(noise1, noise2, t_hat)` — random noise mixed with random noise. The discriminator was learning to distinguish two flavors of Gaussian noise, which is a trivially learnable but useless task.

The corrected training flow:

```
OFFLINE:
  teacher_x0 = teacher.generate(prompt, cfg=5, steps=50, output_type="latent")

ONLINE per step:
  1. x_t = (1-t) * teacher_x0 + t * ε           ← student input (Bug 3 fix)
  2. v = student(x_t, t, prompt)                  ← velocity prediction
  3. x̂_0 = x_t - t * v                           ← denoised latent (Bug 4 fix)
  4. fake_noisy = (1-t̂) * x̂_0 + t̂ * ε₁           ← re-noise for disc
  5. real_noisy = (1-t̂) * teacher_x0 + t̂ * ε₂     ← real path (Bug 5 fix)
```

**The worst part:** the relative ordering of hyperparameter configs from Round 1 _likely still holds_ (all experiments used the same broken pipeline), but the optimal GI flipped from 8 to 3 once the discriminator faced a genuinely hard discrimination task. We had to re-run the entire sweep.

> **Principle: Never tune hyperparameters on an unvalidated baseline.** Before any sweep, verify: (1) the teacher produces sharp images independently, (2) the training loop's math matches the paper step-by-step, (3) a single training step produces finite, non-trivial gradients. Log teacher outputs to W&B as a sanity check _before_ starting experiments.

### Theme 2: Budget compute realistically — and plan for sacrifices

We had 500K curated prompts but needed teacher latents (50-step generation with CFG=5) for every one. We benchmarked:

| Batch size | Time per image | Peak VRAM |
|:----------:|:--------------:|:---------:|
| 1 | 8.9s | 21.9 GB |
| 4 | 7.5s | 25.2 GB |
| 8 | 7.3s | 29.5 GB |

At 7.3s/image with embarrassingly parallel sharding (each GPU processes an independent subset of prompts, no communication needed), precomputing all 500K latents across 8 A100s would take:

$$\frac{500{,}000 \times 7.3}{8} = 456{,}250 \text{ seconds} \approx 127 \text{ hours}$$

We had 6 hours of compute. The math was brutal: we could precompute roughly **10K latents in 4 hours** — 2% of our dataset. Training would repeat each latent ~128 times over 20K steps.

This was a hard trade-off we should have modeled _before_ curating 500K prompts. The curation pipeline (Part 1) was optimized for diversity and prompt quality, which is still valuable for future runs. But for the first training run, 10K stratified-sampled prompts with heavy repetition was the reality.

> **Principle: Model the full compute pipeline end-to-end before committing.** Teacher latent precomputation is not a minor preprocessing step — for a 6B model at 50 steps with CFG, it dominates the compute budget. Calculate wall-clock time for every stage, including precomputation, and work backward from your time budget to determine feasible dataset size.

### Theme 3: Distributed training requires whole-system thinking

FSDP debugging consumed significant time because distributed primitives interact in non-obvious ways with GAN training patterns.

#### The `get_state_dict` hang

FSDP's `accelerator.get_state_dict()` is a **collective operation** — all ranks must call it, not just rank 0. Our validation code called it inside an `if accelerator.is_main_process:` guard. Rank 0 entered the gather; ranks 1-7 skipped it. Deadlock.

From the fix in [`train_ladd.py:263`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/training/train_ladd.py#L263):

```python
# get_state_dict is a collective op under FSDP — ALL ranks must call it.
state_dict = accelerator.get_state_dict(student_model)

# Only rank 0 does file I/O
if not accelerator.is_main_process:
    return
```

#### Checkpoint format issues

`accelerator.save_state()` is incompatible with 8-bit Adam under FSDP. PyTorch's `.pt` format had temp file issues with large state dicts. The fix: save student weights as safetensors, gate `save_state` behind `if not is_fsdp`.

#### Validation subprocess can't log to parent's W&B

Our async validation spawns a subprocess on a separate GPU. The child process can't resume the parent's W&B run context. Fix: the parent process logs the eval table after the subprocess completes.

#### The costly `get_state_dict` double-call

`get_state_dict()` takes ~10 minutes for FSDP gather on a 6B model. We were calling it separately for checkpointing and validation — 20 minutes wasted when they coincide at the same step. Fix: one call, shared between both.

```python
if need_checkpoint or need_validation:
    state_dict = accelerator.get_state_dict(student)  # Single gather
    if need_checkpoint:
        save_checkpoint(state_dict)
    if need_validation:
        launch_validation(state_dict)
```

> **Principle: Distributed ops are collective — think in terms of what _all_ ranks do, not just rank 0.** Before adding any distributed code: (1) identify which operations are collective (all-gather, all-reduce, broadcast), (2) ensure every rank hits every collective in the same order, (3) run a 2-GPU smoke test before scaling to 8. We built [`scripts/smoke_test_fsdp.sh`](https://github.com/vionwinnie/Z-Image-LADD-distillation/blob/main/scripts/smoke_test_fsdp.sh) to catch these issues early.

### Theme 4: Instrument first, debug later

Several of our bugs were only caught because we logged the right things to W&B — and several persisted because we _didn't_ log the right things early enough.

**What caught the scheduler bug:** We logged teacher images to W&B as part of a CFG sweep (`debug-teacher-cfg-sweep`). The images looked blurry at all CFG values. This prompted investigation of the scheduler, which revealed the linspace off-by-one. Without visual inspection, we would have continued tuning hyperparameters against a broken baseline.

**What we should have logged earlier:**
- Teacher-generated images at step 0 (before any training)
- The exact timestep and sigma values the scheduler produces
- A side-by-side of student input $x_t$ at different timesteps (would have caught Bug 3 immediately — "why does t=0.25 look like pure noise?")
- Decoded student predictions $\hat{x}_0$ (would have caught Bug 4 — raw velocity doesn't look like an image)

**The pixelated artifact investigation:** Student outputs showed pixelated grid artifacts. The investigation led us to the scheduler parameters, which led to the linspace bug. The fix required regenerating all teacher reference images and latents, then re-running every experiment.

> **Principle: Log intermediate representations, not just scalar metrics.** Scalars like KID and d_loss tell you _that_ something is wrong; images and tensors tell you _what_. At minimum, log: teacher outputs at step 0, student predictions every N steps, input $x_t$ at different timesteps, and scheduler sigma schedules. A 5-minute W&B setup saves days of blind debugging.

### Theme 5: Debug slices lie — what works at small scale can break at full scale

All 21 hyperparameter experiments ran on a 98-prompt debug slice with `train_batch_size=1` on a single GPU. The sweep converged, KID improved, the architecture was validated. We were confident.

Then we launched the full 8-GPU run with 10K prompts — and the student collapsed into noise within 2000 steps ([Section 8](#9-the-first-full-run-mode-collapse-at-scale)).

The debug slice succeeded for the wrong reasons:
- **bs=1 with 98 prompts**: each prompt was seen every ~98 steps. The student effectively memorized the discriminator's feedback on specific prompts. The hinge loss saturated on individual samples, but the rapid prompt cycling created enough gradient diversity to mask the problem.
- **Scaling to 10K prompts**: each prompt seen every ~10K steps. The student no longer cycles through familiar prompts fast enough to maintain that implicit diversity. The degenerate bs=1 hinge loss becomes the dominant dynamic, and the student drifts.
- **Config drift**: the production run accidentally used `renoise_m=1.0` and `warmup_schedule_steps=10` instead of the validated `0.5` and `0`. The debug sweep validated one config; the production run launched a different one. No automated check caught the mismatch.

The fix required increasing `train_batch_size` from 1 to 2 (so the hinge loss is non-degenerate per micro-step) and ensuring the production config exactly matched the sweep winners.

> **Principle: Your debug slice is not a miniature version of your full run — it's a different problem.** Validate on the debug slice to confirm the architecture works, but expect hyperparameters to shift at full scale. At minimum: (1) test with a realistic per-GPU batch size before launching, (2) diff your production launch command against your best sweep config, and (3) add validation image logging from step 0 so you catch collapse immediately instead of discovering it hours later.

---

## 11. Summary & Next Steps

We built a LADD training framework that distills a 6.15B image model from 50 steps to 4. The key components:

- **Architecture**: Student (6.15B, trainable) + Teacher (6.15B, frozen) + Discriminator (14M, 6 multi-scale conv heads on teacher features)
- **Loss**: Adversarial hinge loss only — no distillation loss, no pixel-space comparison
- **Training**: Asymmetric D/G updates (3:1), logit-normal re-noising ($m=0.5$, $s=1.0$), timestep curriculum
- **Infrastructure**: FSDP across 8× A100 80GB, ~26 GB per GPU, 20K steps in ~2 hours

Single-GPU experiments confirmed the architecture works: the student produces semantically correct images that match prompts, quality improves with more data and steps, and gradient flow is healthy. The remaining gap is scale — batch size 64 (vs 1), 10K+ prompts (vs 98), and 20K steps (vs 2000).

### What's next

1. Run the production 8-GPU training (20K steps, effective batch 64, 10K precomputed latents)
2. Compare 4-step student outputs against 50-step teacher at scale
3. If time permits: precompute more latents (50K+) for a second run with less repetition
4. Evaluate on held-out test set (13K prompts) with KID and visual quality assessment

The code is open source at [github.com/vionwinnie/Z-Image-LADD-distillation](https://github.com/vionwinnie/Z-Image-LADD-distillation).

---

## Appendix: Anti-Mode-Collapse Hyperparameter Sweep

After the first full run collapsed ([Section 8](#9-the-first-full-run-mode-collapse-at-scale)), we ran a second round of experiments specifically targeting discriminator dominance. The goal: find hyperparameters that prevent mode collapse on the full 10K dataset. Reference untrained KID: **0.0689** (anything above means training made things worse).

The Phase 1 sweep results (debug split, 98 prompts) are covered in [Section 5](#5-hyperparameter-experiments). For reference, the **untrained student** (teacher weights, no LADD training at all) has **KID = 0.0689 ± 0.0067** at 4 inference steps. Any KID above this means training actively made things worse.

This appendix covers **Phase 2** — the anti-collapse sweep run after the production failure, using a fresh evaluation setup with corrected teacher images:

| Run | Config | KID | Verdict |
|:----|:-------|:---:|:--------|
| GI=2, dlr=1e-5 | weaker disc | 0.0666 | Best in this phase (below untrained baseline) |
| **GI=2, dlr=1e-5, dim=128** | **even weaker** | **0.0664** | **Best overall in Phase 2** |
| GI=2, dlr=2e-5 | | 0.0684 | Slightly worse |
| GI=3, dlr=1e-5 | | 0.0728 | Worse |
| GI=2, dlr=1e-5, dim=128, layers=[10,20,29] | two changes at once | 0.0791 | Worse |
| Last run | slr=5e-6, dlr=1e-5, gi=2 | 0.0788 | disc_acc_fake=0%, disc not learning |

### Takeaways from the sweep

1. **GI is the dominant knob.** Phase 1: GI=3→8 gave 89% improvement. Phase 2: GI=2 with low disc LR was the sweet spot. The optimal value depends on the evaluation regime.

2. **Lower disc learning rate helps.** `dlr=1e-5` consistently outperformed `dlr=5e-5` at preventing discriminator dominance.

3. **Smaller disc hidden dim has diminishing returns.** `dim=128` helped marginally; `dim=512` hurt badly. The default 256 is a reasonable middle ground.

4. **Noise schedule (renoise_m) matters less than GI.** `M=0.5` was best, but the effect was modest compared to GI tuning.

5. **Disc layer indices didn't help.** Reducing from 6 to 3 or expanding to 8 layers always made things worse.

6. **disc_accuracy_fake=0% is a warning sign.** Several Phase 2 runs show the discriminator can't classify fakes at all — the disc is too weak to provide useful signal. This is the opposite failure mode from the original collapse (disc too strong). The sweet spot is narrow.

The final run in the log (KID=0.0788) showed a regression with `disc_accuracy_fake=0%` — the discriminator was too weak to guide the student. This highlights the fundamental tension in adversarial distillation: the discriminator must be strong enough to provide signal but not so strong that it overwhelms the student. Finding this balance is the central challenge.

---

## 12. Key References

| Year | Paper | Contribution |
|------|-------|-------------|
| 2023 | Lipman et al., *Flow Matching for Generative Modeling* | Flow matching framework — linear interpolation between noise and data, velocity prediction |
| 2023 | Sauer et al., *Adversarial Diffusion Distillation* (ADD) | First adversarial distillation for diffusion — SDXL-Turbo, 1-4 step generation |
| 2024 | Sauer et al., *LADD: Latent Adversarial Diffusion Distillation* | Moves discrimination to teacher's latent features — scalable, no pixel losses, 14M discriminator for 6B+ models |
| 2018 | Miyato et al., *Spectral Normalization for GANs* | Hinge loss and spectral norm — stabilizes adversarial training |
| 2018 | Perez et al., *FiLM: Visual Reasoning with Feature-wise Linear Modulation* | FiLM conditioning — scale/shift modulation used in discriminator heads |
