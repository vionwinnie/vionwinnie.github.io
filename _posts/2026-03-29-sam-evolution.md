---
title: "From SAM to SAM 2: How Segment Anything Learned to Track Objects in Video"
category: engineering
excerpt: "First-principles walkthrough of SAM's promptable segmentation, SAM 2's streaming memory architecture for video tracking, and the future of the Segment Anything lineage."
tags: computer-vision segmentation sam foundation-models video
comments: true
math: true
---

# From SAM to SAM 2: How Segment Anything Learned to Track Objects in Video

**Target audience:** ML practitioners familiar with vision transformers and attention mechanisms who want to understand how promptable segmentation extends from images to video.

---

## Table of Contents

1. [Overview](#overview)
2. [Timeline & Evolution](#timeline--evolution)
3. [The Problem: Why Segmentation Needs a Foundation Model](#1-the-problem-why-segmentation-needs-a-foundation-model)
4. [SAM: Promptable Segmentation for Images](#2-sam-promptable-segmentation-for-images)
5. [The Gap: Why SAM Cannot Handle Video](#3-the-gap-why-sam-cannot-handle-video)
6. [SAM 2: Extending to Video with Memory](#4-sam-2-extending-to-video-with-memory)
7. [The Memory Mechanism in Detail](#5-the-memory-mechanism-in-detail)
8. [Practical Differences: Speed, Data, and Deployment](#6-practical-differences-speed-data-and-deployment)
9. [What About SAM 3?](#7-what-about-sam-3)
10. [Summary](#summary)
11. [Key References](#key-references)

---

## Overview

Image segmentation -- identifying which pixels belong to which objects -- has been a core computer vision problem for decades. Traditionally, every segmentation task required a purpose-built model: one for medical imaging, another for autonomous driving, another for satellite imagery. Each model was trained on a narrow, task-specific dataset and could only segment the categories it had seen during training.

The **Segment Anything Model (SAM)**, released by Meta AI in April 2023, changed this by introducing **promptable segmentation**: give the model any image and a simple prompt (a click, a bounding box, or text), and it segments the indicated object -- any object, in any image, without task-specific training. SAM achieved this through massive scale (1 billion masks on 11 million images) and an elegant architecture that decouples heavy image encoding from lightweight prompt-conditioned decoding.

But SAM operates on single images. It has no concept of time, no memory of what it saw in the previous frame, no ability to track an object as it moves. Each frame is an island.

**SAM 2**, released in July 2024, extends the paradigm to video. The key innovation is a **memory mechanism** that lets the model condition its predictions for the current frame on what it has seen in past frames -- without relying on optical flow or explicit motion estimation. You click on an object in one frame, and SAM 2 tracks it through the entire video, streaming frame-by-frame and maintaining a memory bank of spatial features and object pointers.

This post walks through both architectures from first principles: how SAM works, why it can't do video, what SAM 2 adds to solve that, and how the memory mechanism enables efficient video tracking. We'll also address the question of "SAM 3" -- whether it exists, and what the likely next steps for this line of work are.

---

## Timeline & Evolution

| Year | Model / Method | Key Innovation | Scale |
|------|---------------|----------------|-------|
| 2020 | DETR | End-to-end object detection with transformers | COCO |
| 2021 | ViT, MAE | Vision Transformer; masked autoencoder pretraining | ImageNet |
| 2022 | Masked-attention Mask Transformer (Mask2Former) | Unified architecture for panoptic/instance/semantic segmentation | COCO, ADE20K |
| 2023 | **SAM** (Segment Anything Model) | Promptable foundation model for image segmentation | SA-1B (1.1B masks, 11M images) |
| 2023 | SEEM, Grounded-SAM | Open-vocabulary extensions to SAM | Various |
| 2024 | **SAM 2** (Segment Anything Model 2) | Streaming memory architecture for video + image segmentation | SA-V (50.9K videos, 642.6K masklets) |
| 2024 | SAM 2.1 | Improved training recipe, better small-object handling | SA-V + extended data |
| 2024-25 | EfficientSAM, FastSAM, MobileSAM | Distilled / efficient variants for edge deployment | Various |

---

## 1. The Problem: Why Segmentation Needs a Foundation Model

Before SAM, building a segmentation system meant choosing a specific task:

- **Semantic segmentation** assigns a class label to every pixel ("this pixel is road, this pixel is car"), but doesn't distinguish between two cars.
- **Instance segmentation** detects individual objects and assigns a mask to each, but only for predefined categories.
- **Panoptic segmentation** combines both, but still requires a fixed category set.

Every one of these approaches requires a labeled training set for the specific categories you care about. Want to segment surgical instruments in endoscopy videos? You need thousands of labeled frames of surgical instruments. Want to segment coral in underwater footage? Same story.

The core insight behind SAM is that segmentation itself -- the act of separating figure from ground -- is a general skill that doesn't inherently require category knowledge. A human can segment any object they've never seen before, given a simple prompt like "that thing right there." SAM aims to replicate this: a single model that segments *anything*, guided by a prompt.

---

## 2. SAM: Promptable Segmentation for Images

SAM's architecture has three components, designed around one key principle: **amortize the expensive part, make the interactive part cheap**.

![SAM architecture showing the three-component design: a heavy image encoder (ViT-H) that runs once per image, a prompt encoder that processes user input, and a lightweight mask decoder that produces segmentation masks in real time.](/assets/img/blog/sam-evolution/sam1_architecture.svg)

### Image Encoder

The image encoder is a **ViT-H** (Vision Transformer, Huge variant) pretrained with **MAE** (Masked Autoencoder). It takes a 1024x1024 image and produces a 64x64 grid of feature embeddings (256-dimensional each). This is the expensive step: the ViT-H has 632 million parameters and processes the image in ~0.15 seconds on an A100 GPU. Crucially, **it runs only once per image**, regardless of how many prompts the user provides.

### Prompt Encoder

The prompt encoder converts the user's input into tokens that the decoder can consume:

- **Point prompts** (positive/negative clicks): encoded as learned embeddings plus positional encodings at the click coordinates
- **Box prompts**: encoded as two corner points (top-left, bottom-right) with positional encodings
- **Mask prompts** (from a previous iteration): downscaled via convolutions and added element-wise to the image embedding
- **Text prompts**: encoded via a CLIP text encoder (used in some variants)

The prompt encoder is extremely lightweight -- it adds negligible latency.

### Mask Decoder

The decoder is a modified **two-layer transformer** that performs cross-attention between prompt tokens and image embeddings. It outputs:

- **Three valid masks** at different granularities (subpart, part, whole object) -- because a single point prompt can be ambiguous (does a click on a person's hand mean "the hand" or "the whole person"?)
- **An IoU (Intersection over Union) confidence score** for each mask

The decoder runs in ~50ms -- fast enough for real-time interactive use. This is the key architectural trick: the heavy encoder runs once, and users can click repeatedly, getting near-instant mask predictions each time.

### The SA-1B Dataset

SAM's generalization comes from its training data: **SA-1B**, the largest segmentation dataset ever created at the time. It contains **1.1 billion masks** on **11 million images**, built through a three-stage data engine:

1. **Assisted-manual**: Human annotators used an early SAM model to speed up labeling
2. **Semi-automatic**: SAM proposed masks, annotators labeled remaining objects
3. **Fully-automatic**: SAM generated masks for all objects with no human input; annotators only performed quality checks

This data flywheel -- where the model helps generate the training data for its next version -- is one of SAM's most important contributions beyond the architecture itself.

---

## 3. The Gap: Why SAM Cannot Handle Video

SAM processes each image independently. If you run SAM on 30 consecutive video frames, you get 30 independent sets of masks with no correspondence between them. Frame 1 might segment a dog perfectly, but there's no mechanism to say "that mask in frame 1 corresponds to this mask in frame 2."

The fundamental problems for video segmentation that SAM cannot solve:

1. **Object persistence**: An object that appears in frame 1 should maintain its identity throughout the video. SAM has no concept of identity across frames.

2. **Prompt efficiency**: You shouldn't need to click on the object in every frame. Ideally, you click once and the model tracks the object.

3. **Occlusion handling**: Objects disappear behind other objects and reappear. The model needs to "remember" what the object looked like before occlusion and re-identify it.

4. **Appearance change**: Objects rotate, deform, change lighting. The model needs a representation robust to these variations.

5. **Efficiency at video scale**: A 30-second video at 30 FPS has 900 frames. Running the full SAM encoder independently on each frame would be slow and wasteful -- consecutive frames share most of their visual content.

The naive approach of "run SAM per frame and match masks" fails because it has no temporal context. Existing video object segmentation (VOS) methods like XMem and DeAOT address these problems, but they use specialized architectures that don't generalize across domains the way SAM does for images.

---

## 4. SAM 2: Extending to Video with Memory

SAM 2 solves the image-to-video gap with one core idea: **add a memory mechanism to SAM's architecture so the model can condition its current-frame predictions on what it saw in past frames.**

![Side-by-side comparison of SAM and SAM 2: SAM processes single images with no temporal reasoning while SAM 2 adds memory attention and a memory bank to enable object tracking across video frames.](/assets/img/blog/sam-evolution/sam_to_sam2_comparison.svg)

### Design Philosophy

SAM 2's design follows three principles:

1. **Strict superset of SAM**: SAM 2 handles both images *and* video. For images, the memory mechanism is simply disabled, and the model reduces to a SAM-like architecture. This means SAM 2 replaces SAM entirely -- you never need both.

2. **Streaming architecture**: SAM 2 processes video frames one at a time, left to right, with no look-ahead. This is essential for real-time and interactive applications where future frames don't exist yet.

3. **Promptable at any frame**: The user can provide prompts (clicks, boxes, masks) at any frame in the video, not just the first one. This enables interactive correction -- if the model loses track, the user clicks to re-identify the object.

### Architecture Overview

SAM 2 has these components:

![SAM 2's streaming architecture showing the image encoder (Hiera), memory attention module, prompt encoder and mask decoder, memory encoder, and the memory bank containing recent memories, prompted memories, and object pointers.](/assets/img/blog/sam-evolution/sam2_streaming_architecture.svg)

**Image Encoder: Hiera (replacing ViT-H)**

SAM 2 replaces SAM's plain ViT-H with **Hiera**, a hierarchical vision transformer also pretrained with MAE. Hiera is both faster and more efficient than ViT-H because it uses a multi-scale architecture (similar to Swin Transformer) that processes high-resolution features at early layers and lower-resolution features at later layers. On images alone, SAM 2 with Hiera is **6x faster** than SAM with ViT-H while achieving better segmentation quality.

**Memory Attention (new in SAM 2)**

This is the key new module. Before the current frame's features reach the mask decoder, they pass through a stack of transformer layers that perform:

1. **Self-attention** over the current frame's features
2. **Cross-attention** where the current frame's features (queries) attend to the memory bank (keys and values)

This cross-attention is where temporal reasoning happens. The current frame literally "looks at" past frames to understand where the tracked object is and what it looks like.

**Prompt Encoder + Mask Decoder (same as SAM)**

The decoder architecture is essentially unchanged from SAM. It receives memory-conditioned features (instead of raw image features) and any optional user prompts, and produces masks + IoU scores.

**Memory Encoder (new in SAM 2)**

After producing a mask for the current frame, the memory encoder takes the mask decoder's output and combines it with the image encoder's features (via lightweight convolutions) to produce a **spatial memory** -- a compact representation of "what this frame's object looks like and where it is." This spatial memory is added to the memory bank.

**Memory Bank (new in SAM 2)**

The memory bank stores three types of information:

| Memory Type | What It Stores | Retention Policy |
|------------|---------------|-----------------|
| **Recent memories** | Spatial features + predicted masks from the last N frames | FIFO queue (oldest dropped when full) |
| **Prompted memories** | Spatial features + masks from frames where the user provided a prompt | Kept permanently (highest priority) |
| **Object pointers** | Lightweight token vectors summarizing each frame's object representation | Kept for all past frames |

---

## 5. The Memory Mechanism in Detail

The memory mechanism is the heart of SAM 2's video capability. Let's trace exactly how it enables tracking.

![Detailed view of SAM 2's memory mechanism showing how past frame features and masks are encoded into spatial memories, stored in the memory bank, and consumed via cross-attention by the current frame to enable tracking.](/assets/img/blog/sam-evolution/memory_mechanism_detail.svg)

### Step 1: Encode the Current Frame

The image encoder (Hiera) processes frame `t` and produces spatial features `F_t` -- a grid of feature vectors describing the visual content at each spatial location.

### Step 2: Condition on Memory

The memory attention module takes `F_t` as input and transforms it by attending to the memory bank. Concretely, the cross-attention computes:

```
Q = W_q * F_t          (current frame features as queries)
K = W_k * M_bank       (memory bank entries as keys)
V = W_v * M_bank       (memory bank entries as values)

F_t' = softmax(Q * K^T / sqrt(d)) * V
```

- `F_t`: current frame features (spatial grid)
- `M_bank`: concatenation of all spatial memories + object pointers in the memory bank
- `F_t'`: memory-conditioned features
- `d`: dimension of the key vectors

The softmax attention means the model learns to match spatial locations in the current frame to corresponding locations in past frames. If the tracked object was at position (x=30, y=40) in the previous frame and has moved to (x=35, y=42) in the current frame, the cross-attention naturally aligns these -- the query at (35, 42) will have high attention weight to the key at (30, 40) in the previous frame's memory, because the object features are similar.

This is why SAM 2 **doesn't need optical flow**: the cross-attention mechanism implicitly computes a soft correspondence between current and past frames. It's more flexible than optical flow because it can handle non-rigid deformation, appearance changes, and even occlusion-and-reappearance (the memory bank stores frames from the distant past, not just the immediately previous frame).

### Step 3: Decode the Mask

The memory-conditioned features `F_t'` are passed to the mask decoder along with any user prompts. The decoder produces the mask for frame `t`.

### Step 4: Update the Memory Bank

The memory encoder takes:
- The predicted mask for frame `t`
- The image encoder features `F_t`

It fuses them via convolution to produce a spatial memory representation and an object pointer token. These are added to the memory bank:
- The spatial memory goes into the recent memories queue (FIFO, with capacity N -- typically N=6)
- If frame `t` had a user prompt, its spatial memory also goes into the prompted memories store (never evicted)
- The object pointer token is always stored

### Why This Is Efficient

The memory mechanism adds minimal overhead per frame:

1. **Memory attention** is a standard transformer cross-attention -- the computational cost scales linearly with the number of memory entries times the spatial resolution of the current frame
2. **Memory bank size is bounded**: recent memories have a fixed FIFO capacity, and object pointers are lightweight tokens (not full spatial grids)
3. **No backward passes through the video**: the model processes strictly left-to-right, producing a mask for each frame in a single forward pass
4. **The memory encoder is a few convolution layers** -- much cheaper than running the image encoder again

The result: SAM 2 runs at approximately **44 FPS** on an A100 GPU for video segmentation, which is close to real-time for 30 FPS video.

### Handling Occlusion

When an object becomes fully occluded, SAM 2's mask decoder will predict a low-confidence (low IoU) mask or no mask at all. The model has an **occlusion head** that predicts whether the object is visible in the current frame. If the object is predicted as occluded:

- No memory is added for that frame (to avoid corrupting the memory bank with bad predictions)
- The memory bank retains the last good observations
- When the object reappears, the cross-attention can match the reappearing features against the stored memories from before occlusion

This is a significant advantage over optical-flow-based trackers, which lose the object entirely during occlusion because there's no visible motion to track.

### Multi-Object Tracking

SAM 2 can track multiple objects simultaneously. Each object gets its own set of object pointer tokens in the memory bank. The memory attention processes all objects in parallel, and the mask decoder produces separate masks for each tracked object. This is efficient because the expensive image encoder and memory attention run once per frame regardless of how many objects are being tracked.

---

## 6. Practical Differences: Speed, Data, and Deployment

### SA-V Dataset

Just as SAM's generalization came from the SA-1B dataset, SAM 2's video capability comes from **SA-V** (Segment Anything in Video):

- **50.9K videos** spanning diverse real-world scenarios
- **642.6K masklets** (mask trajectories across frames)
- Built using a similar data-engine approach: annotators used an interactive SAM 2 prototype to label videos, and the resulting data trained the next version

SA-V is the largest video object segmentation dataset, roughly **53x larger** than the previous largest (YouTubeVOS at ~4.5K videos). This scale is critical -- prior VOS models trained on small datasets struggle with rare objects and unusual scenarios.

### Speed Comparison

| Metric | SAM (ViT-H) | SAM 2 (Hiera-L) |
|--------|-------------|-----------------|
| Image encoder | ViT-H (632M params) | Hiera-L (smaller, hierarchical) |
| Image segmentation speed | ~0.15s per image | ~0.025s per image (~6x faster) |
| Video segmentation | N/A (image only) | ~44 FPS streaming |
| Interactive latency | ~50ms per prompt | ~50ms per prompt |
| Multi-object overhead | N/A | Minimal (shared encoder) |

### Model Sizes

SAM 2 comes in multiple sizes:

| Variant | Image Encoder | Total Params | Use Case |
|---------|--------------|-------------|----------|
| SAM 2 Tiny | Hiera-T | ~39M | Mobile / edge |
| SAM 2 Small | Hiera-S | ~46M | Lightweight |
| SAM 2 Base+ | Hiera-B+ | ~81M | Balanced |
| SAM 2 Large | Hiera-L | ~224M | Maximum quality |

Even SAM 2 Large is significantly smaller than SAM's ViT-H (632M), while achieving better segmentation quality on both images and video.

---

## 7. What About SAM 3?

As of March 2026, **there is no official "SAM 3" model from Meta AI**. The naming convention suggests a natural progression, and the question comes up frequently, but no paper or model by that name has been published.

What has happened since SAM 2:

### SAM 2.1 (Late 2024)

Meta released **SAM 2.1**, an incremental update that improved the training recipe without changing the architecture. Key improvements included better handling of small objects, improved mask quality on edge cases, and extended training data. This is an engineering refinement, not an architectural shift.

### The Broader Ecosystem

The SAM lineage has spawned a rich ecosystem of derivative and complementary work:

- **EfficientSAM** and **MobileSAM**: Distilled, smaller versions of SAM for edge deployment
- **Grounded-SAM**: Combines SAM with Grounding DINO for text-prompted segmentation
- **SAM-Track**: Combines SAM with dedicated tracking models
- **Medical SAM variants**: Fine-tuned for medical imaging (MedSAM, SAM-Med2D)

### Where the Field Is Heading

Based on the trajectory from SAM to SAM 2, the likely directions for a future SAM iteration (whether called "SAM 3" or otherwise) include:

1. **3D understanding**: Extending from 2D video to 3D scene segmentation, possibly incorporating depth or multi-view geometry. There are already papers exploring SAM for 3D tasks (SA3D, SAM3D), but these are community extensions, not official Meta models.

2. **Open-vocabulary integration**: Deeper integration with language models so that text prompts become as robust as point prompts. Current text-prompt support in SAM is limited.

3. **Longer temporal reasoning**: SAM 2's memory bank has a fixed window. Future work may explore hierarchical memory or retrieval-based memory for very long videos (hours, not minutes).

4. **Action-aware segmentation**: Understanding not just where objects are but what they're doing -- linking segmentation with activity recognition and prediction.

5. **Further efficiency**: Pushing toward real-time performance on consumer hardware, not just A100 GPUs.

It is important to be honest here: **if you've encountered references to "SAM 3," they likely refer to either community speculation, unofficial extensions, or papers like "SA3D" / "SAM3D" that use SAM for 3D tasks but are not official successors in the SAM lineage.**

---

## Summary

The evolution from SAM to SAM 2 represents a clean architectural progression:

**SAM (2023)** solved promptable image segmentation by decoupling a heavy image encoder (ViT-H, run once) from a lightweight prompt-conditioned decoder (run per interaction). The SA-1B dataset (1.1 billion masks) gave it the generalization to segment any object in any image.

**SAM 2 (2024)** extended this to video by adding three components:
- A **memory encoder** that compresses each frame's features and predicted mask into a compact spatial memory
- A **memory bank** that stores recent memories, prompted memories, and object pointer tokens
- A **memory attention module** that lets the current frame cross-attend to the memory bank, implicitly computing temporal correspondences without optical flow

This memory mechanism is what enables efficient video tracking: you prompt the object once, and the model propagates the segmentation forward by continuously conditioning on its memory of past frames. Occlusion handling comes naturally -- the memory bank retains pre-occlusion observations, and cross-attention can match the object when it reappears.

SAM 2 also replaced ViT-H with the hierarchical Hiera encoder, achieving 6x faster image segmentation while improving quality. It is a strict superset of SAM: images are segmented with memory disabled, and the same model handles video with memory enabled.

As of March 2026, there is no "SAM 3." The most recent official release is SAM 2.1, a training-recipe improvement over SAM 2. The field continues to evolve through community extensions (3D, medical, mobile), and a future official successor will likely push toward 3D scene understanding and tighter language integration.

---

## Key References

| Year | Paper | Contribution |
|------|-------|-------------|
| 2023 | Kirillov et al., "Segment Anything" (arXiv:2304.02643) | Introduced promptable segmentation, SAM architecture, and the SA-1B dataset |
| 2024 | Ravi et al., "SAM 2: Segment Anything in Images and Videos" (arXiv:2408.00714) | Extended SAM to video with streaming memory architecture and SA-V dataset |
| 2024 | Ryali et al., "Hiera: A Hierarchical Vision Transformer without Bells and Whistles" | The image encoder used in SAM 2, replacing ViT-H with a faster hierarchical design |
| 2021 | He et al., "Masked Autoencoders Are Scalable Vision Learners" | MAE pretraining strategy used for both SAM and SAM 2 encoders |
| 2020 | Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT) | The Vision Transformer foundation that SAM builds on |
| 2022 | Cheng et al., "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model" | Prior work on memory-based VOS that influenced SAM 2's memory design |
| 2022 | Yang et al., "Decoupling Features in Hierarchical Propagation for Video Object Segmentation" (DeAOT) | Another influential memory-based VOS method |
