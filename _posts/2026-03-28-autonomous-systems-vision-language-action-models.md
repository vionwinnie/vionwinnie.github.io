---
title: "Autonomous Systems: Vision-Language-Action Models"
category: engineering
excerpt: "First-principles guide to VLAs — architecture, RT-2, Octo, OpenVLA, action tokenisation, pretraining, and connections to autonomous driving."
tags: autonomous-driving vla self-driving tech-interview
comments: true
---

# Vision-Language-Action Models: A First-Principles Guide

*A bottom-up explanation from visual perception to embodied action*
*Last updated: March 2026*

---

## 1. From Vision-Language to Action

### The progression

![VLM to VLA Progression](/assets/img/blog/autonomous-systems/vlm_to_vla_progression.svg)

Computer vision has evolved through a clear sequence of capability expansions:

| Era | Models | What they do | Output |
|-----|--------|-------------|--------|
| Vision-only | ResNet (2015), ViT (2020) | Classify images, detect objects | Class labels, bounding boxes |
| Vision-Language | CLIP (2021), LLaVA (2023) | Connect images to natural language | Text descriptions, visual Q&A |
| Vision-Language-Action | RT-2 (2023), Octo (2024) | Perceive, reason, *and act* in the physical world | Motor commands, trajectories |

Each step adds a new modality. A **vision model** like ViT takes a photograph and outputs "there is a red cup on the table." A **vision-language model (VLM)** like LLaVA can answer "where is the red cup relative to the plate?" A **vision-language-action model (VLA)** can take the instruction "pick up the red cup" and output the sequence of motor commands to actually do it.

### Grounding: from words to physics

**Grounding** is the process of connecting abstract language descriptions to concrete physical referents. The instruction "pick up the red cup" requires two kinds of grounding:

1. **Visual grounding** — identify which pixels in the camera image correspond to the red cup (object recognition + spatial localisation)
2. **Action grounding** — translate the concept of "pick up" into a sequence of motor commands: move end-effector above the cup → lower → close gripper → lift

VLMs already solve visual grounding — they can locate objects, describe spatial relations, and follow complex instructions. The missing piece is action grounding: mapping that understanding to physical control signals. This is exactly the gap that VLAs fill.

### Why VLMs are a natural starting point

VLMs pre-trained on internet-scale data (billions of image-text pairs) arrive with an enormous library of **world knowledge**: they know what thousands of objects look like, how they relate spatially ("the fork is to the left of the plate"), and can parse compositional instructions ("put the green block on top of the red block, then move both to the corner"). This knowledge transfers directly to robotics. A VLA doesn't need to learn what a "cup" is from scratch — it inherits that from the VLM backbone and only needs to learn the mapping from perception to motor output.

---

## 2. VLA Architecture

### The canonical pipeline

![VLA Architecture](/assets/img/blog/autonomous-systems/vla_architecture.svg)

Nearly all modern VLAs follow a three-stage pipeline:

```
Camera Image(s)  ──→  [Vision Encoder]  ──→  Visual Tokens
                                                    ↓
Language Instruction  ──→  [Tokenizer]  ──→  Text Tokens  ──→  [Language Model Backbone]  ──→  Hidden States
                                                                                                    ↓
                                                                                              [Action Head]  ──→  Robot Actions
```

Each component has a distinct role:

### Vision encoder

The **vision encoder** converts raw camera images into a sequence of **visual tokens** — dense vector representations that the language model can process alongside text. Most VLAs use a pre-trained **Vision Transformer (ViT)**, typically frozen or lightly fine-tuned:

- **SigLIP** (used in OpenVLA, pi0): a sigmoid-loss variant of CLIP's vision encoder that produces better-calibrated visual features
- **DINOv2** (used in some Octo variants): self-supervised ViT that captures rich spatial features without requiring text supervision

Concretely, a 224×224 image is split into 14×14 patches, each embedded as a 1024-dimensional vector, yielding 196 visual tokens. These tokens are projected into the language model's embedding space via a learned linear layer or MLP.

### Language model backbone

The **language model backbone** is typically a pre-trained large language model (LLM) — PaLM-E, Llama 2, Gemma, or similar. It processes a combined sequence of visual tokens interleaved with tokenised language instructions. The LLM provides:

- **Instruction following**: parsing "put the apple in the bowl" into a structured plan
- **Reasoning**: determining which object is the apple, where the bowl is, what grasp strategy to use
- **World knowledge**: knowing that apples are graspable, bowls are containers, and the apple should go *inside* the bowl

The LLM outputs a sequence of hidden state vectors, one per token position. The hidden states at designated output positions are passed to the action head.

### Action head

The **action head** maps the LLM's hidden states to robot actions. This is where the VLA's output becomes physical. There are three main design choices:

| Approach | How it works | Pros | Cons |
|----------|-------------|------|------|
| **Direct regression** | MLP maps hidden states to continuous action values (e.g., 7 floats for 7-DOF arm) | Simple, precise | Unimodal — can only predict one action, problematic when multiple valid actions exist |
| **Discrete action bins** | Each action dimension is quantised into bins (e.g., 256); LLM outputs bin indices as tokens | Leverages LLM's token prediction; simple training | Quantisation error; resolution limited by bin count |
| **Diffusion-based** | Iteratively denoises random noise into action sequences, conditioned on hidden states | Handles multi-modal distributions; smooth trajectories | Slower inference (requires multiple denoising steps) |

### Concrete walkthrough

Consider a robot arm facing a table with several objects. The input is:

- **Image**: camera view showing an apple, a banana, and a bowl on a table
- **Instruction**: "put the apple in the bowl"

The forward pass proceeds as:

1. **Vision encoder** (SigLIP) converts the camera image into 196 visual tokens capturing the spatial layout of all objects
2. **Tokenizer** converts "put the apple in the bowl" into text tokens
3. **LLM backbone** (e.g., Llama 2 7B) processes the concatenated sequence [visual tokens, text tokens]. Its internal attention identifies the apple (red, round, left side of image), the bowl (concave, right side), and infers the required motion direction (left → right, then down)
4. **Action head** takes the LLM's final hidden states and outputs a sequence of end-effector poses: `[(x₁, y₁, z₁, roll₁, pitch₁, yaw₁, gripper₁), ..., (xₙ, yₙ, zₙ, rollₙ, pitchₙ, yawₙ, gripperₙ)]`

The robot executes these actions, physically moving the apple into the bowl.

---

## 3. Key Models

### RT-2 (Robotics Transformer 2, Google DeepMind, 2023)

**RT-2** was the first model to demonstrate that a large VLM can be directly fine-tuned into an effective robot controller.

| Aspect | Detail |
|--------|--------|
| **Base model** | PaLM-E (12B) or PaLI-X (55B) |
| **Key insight** | Represent robot actions as text tokens in the LLM's existing vocabulary |
| **Action format** | 7-DOF actions encoded as integer strings: `"1 128 91 241 1 128 0"` where each number is a discretised action dimension |
| **Training data** | Robot episodes from Google's fleet + internet-scale image-text data |

The elegance of RT-2 is that it requires **no architectural changes** to the VLM — actions are just another kind of text output. The model learns to "speak robot" the same way it learned to speak English.

**Results**: RT-2 achieves **2× improvement on unseen objects** over its predecessor RT-1 (62% vs 32% success rate), demonstrating that VLM pre-training enables strong generalisation. It can manipulate objects it has never seen in robot training (e.g., picking up a specific toy figure) because the VLM backbone recognises those objects from internet data.

### Octo (UC Berkeley, 2024)

**Octo** is an open-source generalist robot policy designed for multi-embodiment control.

| Aspect | Detail |
|--------|--------|
| **Architecture** | Transformer-based (not derived from an LLM) |
| **Training data** | 800K episodes from the Open X-Embodiment dataset |
| **Action head** | Diffusion-based — generates action chunks via iterative denoising |
| **Multi-embodiment** | Supports different robots via task-specific readout heads |

Octo's **diffusion action head** is particularly important: it naturally handles **multi-modal action distributions** — situations where multiple valid actions exist (e.g., you can reach around either side of an obstacle). The readout heads are swappable, allowing the same backbone to control a WidowX arm, a Franka Panda, or other robots by changing only the final output layer.

### OpenVLA (Stanford/Berkeley, 2024)

**OpenVLA** scales the VLA concept to 7 billion parameters, showing that bigger VLM backbones yield better robot controllers.

| Aspect | Detail |
|--------|--------|
| **Base model** | Llama 2 7B + SigLIP + DINOv2 dual vision encoders (via the Prismatic VLM backbone) |
| **Training data** | Open X-Embodiment dataset (970K robot episodes) |
| **Action format** | Each of 7 action dimensions discretised into 256 bins |
| **Key finding** | Scaling the VLM backbone improves manipulation success rate |

OpenVLA demonstrates a **16.5% absolute improvement** over RT-2-X on WidowX manipulation tasks, providing evidence that the scaling laws observed in language models apply to robotic control as well. As an open-source release (weights, code, and data), it established a common baseline for VLA research.

### pi0 (Physical Intelligence, 2024)

**pi0** targets dexterous manipulation — tasks requiring fine motor control like folding laundry or assembling objects.

| Aspect | Detail |
|--------|--------|
| **Architecture** | VLM backbone + flow matching action head |
| **Action generation** | Flow matching (a continuous-time variant of diffusion) for smooth, precise trajectories |
| **Pre-training** | Internet-scale image-text data + large-scale robot data |
| **Fine-tuning** | Task-specific fine-tuning for dexterous manipulation |

The **flow matching** action head is a key distinction: instead of the iterative denoising steps of diffusion, flow matching learns a continuous vector field that transports samples from noise to actions along straight(er) paths. This yields faster inference and smoother generated trajectories — critical for dexterous tasks where jittery motions would cause failures.

---

## 4. Action Tokenisation

### The fundamental challenge

![Action Tokenisation Approaches](/assets/img/blog/autonomous-systems/action_tokenisation.svg)

Robot actions are **continuous** — joint torques, end-effector velocities, and gripper apertures are real-valued quantities with arbitrary precision. Language models operate on **discrete tokens** from a finite vocabulary. Bridging this gap is the core technical challenge of VLA design.

### Discretising continuous actions

The simplest approach, pioneered by RT-2: **bin each action dimension independently**.

For a 7-DOF robot arm, each action is a vector $\mathbf{a} = (a_1, a_2, \ldots, a_7)$ where each $a_i \in [a_i^{\min}, a_i^{\max}]$. Discretisation maps each continuous value to one of $K$ bins:

$$\text{bin}(a_i) = \left\lfloor \frac{a_i - a_i^{\min}}{a_i^{\max} - a_i^{\min}} \cdot (K-1) \right\rfloor$$

With $K = 256$ bins (OpenVLA's choice), the quantisation error is at most $\frac{a_i^{\max} - a_i^{\min}}{2 \times 255}$ per dimension — typically sub-millimetre for typical robot workspace ranges.

RT-2 maps these bin indices to existing integer tokens in the LLM vocabulary (tokens for "1", "128", "91", etc.), requiring no vocabulary modification.

### Action chunking (ACT)

**Action Chunking with Transformers (ACT)** (Zhao et al., 2023) addresses a different problem: **compounding errors**.

When a policy predicts one action at a time, small errors accumulate. A 1° joint angle error per step becomes 10° after 10 steps. ACT predicts a **chunk** of $H$ future actions simultaneously:

$$\pi(\mathbf{o}_t) \rightarrow (\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1})$$

where $\mathbf{o}_t$ is the observation at time $t$ and $H$ is the chunk size (typically 10–100 steps).

Benefits:
- **Reduced compounding error**: the model plans further ahead, producing temporally coherent motion
- **Smoother trajectories**: consecutive actions within a chunk are jointly optimised
- **Multi-modality via CVAE**: ACT uses a **Conditional Variational Autoencoder** to sample different action chunks for the same observation, handling situations where multiple valid motions exist

### Diffusion-based action generation

**Diffusion Policy** (Chi et al., 2023) models the action distribution as a diffusion process. Starting from pure Gaussian noise $\mathbf{a}^{(T)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the model iteratively denoises to produce an action sequence:

$$\mathbf{a}^{(t-1)} = \text{denoise}_\theta(\mathbf{a}^{(t)}, \mathbf{o}, t) \quad \text{for } t = T, T-1, \ldots, 1$$

where $\mathbf{o}$ is the observation (visual features + instruction embedding) and $\theta$ are learned parameters.

Concrete example: a robot needs to place a block that could go in either of two valid locations. A regression head would average the two locations (placing the block *between* them — a failure). A diffusion head naturally samples from the bimodal distribution, committing to one valid location per rollout.

### Tradeoffs

| Method | Precision | Multi-modality | Speed | Training simplicity |
|--------|-----------|----------------|-------|-------------------|
| Discrete bins | Limited by bin count | No (argmax over bins) | Fast (single forward pass) | Leverages LLM cross-entropy loss |
| Direct regression | High (continuous output) | No (unimodal) | Fast | Simple MSE loss |
| Diffusion / flow matching | High | Yes (samples from distribution) | Slow (10–100 denoising steps) | Requires noise schedule tuning |
| ACT (CVAE) | High | Yes (latent sampling) | Fast | Requires KL balancing |

---

## 5. Pretraining Recipes

VLAs are not trained from scratch. They follow a multi-stage recipe that reuses as much existing knowledge as possible.

![Pretraining Stages](/assets/img/blog/autonomous-systems/pretraining_stages.svg)

### Stage 1: Vision-language pre-training

The VLM backbone (e.g., Llama 2 + SigLIP) is pre-trained on **billions of image-text pairs** scraped from the internet. This stage teaches:

- **Object recognition**: what a cup, apple, screwdriver, etc. look like
- **Spatial reasoning**: understanding "on top of," "to the left of," "inside"
- **Instruction parsing**: following compositional natural language commands
- **Common-sense physics**: glasses are fragile, liquids spill, heavy objects require firm grasps

This is the most computationally expensive stage but is amortised across all downstream applications (not just robotics).

### Stage 2: Robot data fine-tuning

The pre-trained VLM is fine-tuned on robot manipulation datasets. The primary dataset is **Open X-Embodiment** (2023):

| Statistic | Value |
|-----------|-------|
| Total episodes | 1M+ |
| Robot embodiments | 22 |
| Tasks | 500+ manipulation skills |
| Data sources | 21 institutions worldwide |

During fine-tuning, the model learns to map visual understanding and instruction parsing to physical actions. Crucially, **not all pre-trained knowledge needs to be re-learned** — the VLM already understands "apple" and "bowl"; fine-tuning only teaches the action mapping.

### The key insight: knowledge transfer

A VLA fine-tuned on Open X-Embodiment data has manipulated perhaps 100 distinct object categories in robot training. But through its VLM pre-training, it recognises thousands of objects. When encountering an object it has never physically manipulated (say, a rubber duck), the VLM backbone still identifies it correctly, and the action head can generalise the grasping strategy from similar objects seen during robot training.

### Preventing catastrophic forgetting

A major risk during Stage 2 is **catastrophic forgetting** — the model "forgets" its VLM capabilities as it learns robot control. Mitigation strategies:

- **Co-training**: mix internet image-text data with robot data during fine-tuning (see Section 7)
- **Low learning rate**: fine-tune with 10–100× lower learning rate than pre-training
- **Frozen encoder**: keep the vision encoder weights frozen, only fine-tuning the LLM and action head
- **LoRA**: use parameter-efficient fine-tuning that modifies only a small fraction of weights

---

## 6. Generalisation

Generalisation is the central promise of VLAs. A robot that only works on objects, environments, and instructions it has seen during training is not useful. VLAs generalise along three axes:

### Unseen objects

The VLA can manipulate objects absent from its robot training data because the VLM backbone recognises them from internet pre-training.

**Example**: RT-2 was never trained to pick up a specific action figure in robot data, but the VLM backbone has seen millions of images of action figures online. When given the instruction "pick up the action figure," RT-2 correctly identifies and grasps it.

**Quantitative evidence**:
- RT-2: **62%** success on unseen objects vs **32%** for RT-1 (which lacks VLM pre-training)
- OpenVLA: **16.5% absolute improvement** over RT-2-X on WidowX manipulation tasks

### Unseen environments

Real deployment means different backgrounds, lighting conditions, table heights, and camera angles from training. VLAs handle this through:

- **VLM robustness**: VLMs trained on diverse internet images are inherently robust to visual domain shift
- **Domain randomisation**: randomising colours, textures, lighting, and camera poses during training
- **Spatial generalisation**: the VLM's spatial reasoning transfers across environments (understanding "left of" doesn't depend on the specific table)

### Unseen instructions

VLAs can follow novel natural language commands that recombine known concepts in new ways:

| Training instructions | Novel test instruction |
|----------------------|----------------------|
| "pick up the red block" | "stack the blocks by colour" |
| "put X in the bowl" | "sort the fruits into the two bowls" |
| "move X to the left" | "arrange the objects in a line from smallest to largest" |

This **compositional generalisation** comes from the LLM backbone, which understands language compositionality from pre-training. The model has never executed "stack by colour" as a single skill, but it can decompose it into known primitives.

---

## 7. Co-training with Internet Data and Robot Data

### The data scarcity problem

Robot data is expensive. Even the largest robot dataset (Open X-Embodiment) contains roughly 1M episodes — orders of magnitude less than the billions of image-text pairs used for VLM pre-training. If you fine-tune a VLM exclusively on robot data, it tends to lose its broad visual and linguistic capabilities.

### The co-training solution

**Co-training** interleaves robot episodes with internet image-text pairs during fine-tuning. In each training batch, some examples are robot manipulation sequences (image → action), and others are standard VLM tasks (image → text description, visual question answering, etc.).

```
Training batch:
  [Robot] Image of table → "pick up cup" → action tokens: 1 128 91 241 1 128 0
  [Internet] Photo of park → "A golden retriever catching a frisbee in a sunny park"
  [Robot] Image of drawer → "open the top drawer" → action tokens: 0 64 180 128 1 90 1
  [Internet] Diagram of solar system → "The third planet from the sun is Earth"
  ...
```

### Data ratio and loss weighting

Typical co-training mixes:

| Model | Robot data fraction | Internet data fraction |
|-------|-------------------|----------------------|
| RT-2 | 50% | 50% |
| OpenVLA | ~100% robot (no explicit co-training) | Pre-trained backbone frozen |
| pi0 | Varies by stage | Internet data in pre-training, robot data in fine-tuning |

The data ratio matters: too much internet data and the model under-fits robot skills; too much robot data and it loses VLM knowledge. RT-2 found that **equal mixing** worked well, and critically, **keeping internet data improved robot task success** — not just language understanding. The hypothesis: internet data acts as a regulariser, preventing the model from over-fitting to the limited robot training distribution.

---

## 8. Embodiment-Agnostic Models

### The vision

The ultimate goal is a **foundation model for robotics** — a single model that can control any physical embodiment (robot arms, quadrupeds, drones, humanoids) by learning shared representations of physical interaction. Just as GPT-4 handles English, French, and code with one model, an embodiment-agnostic VLA would handle a Franka Panda, a Boston Dynamics Spot, and a quadrotor with one model.

### Octo's approach

Octo addresses multi-embodiment control through **modular tokenisation**:

- **Shared backbone**: a single transformer processes all observations and generates shared hidden representations
- **Embodiment-specific observation tokenisers**: convert each robot's sensor data (different camera configurations, proprioceptive states) into a common token format
- **Embodiment-specific action readout heads**: decode the shared hidden states into each robot's native action space

```
Franka Panda (7-DOF arm)  ──→  [Obs Tokeniser A]  ──→                          ──→  [Readout A]  ──→  7-DOF joint torques
                                                        [Shared Transformer]
WidowX (5-DOF arm)        ──→  [Obs Tokeniser B]  ──→                          ──→  [Readout B]  ──→  5-DOF joint velocities
```

### Open X-Embodiment dataset

The enabling dataset for embodiment-agnostic models is **Open X-Embodiment** (2023), a collaborative effort across 21 institutions:

| Property | Value |
|----------|-------|
| Robot types | 22 distinct embodiments |
| Data format | Standardised RLDS (Reinforcement Learning Datasets) format |
| Tasks | 500+ manipulation skills |
| Scale | 1M+ episodes |

Standardising the data format across robots was itself a significant engineering effort — different labs use different coordinate frames, control frequencies, camera setups, and action representations.

### Current limitations

Embodiment-agnostic models remain far from matching specialist models:

1. **Action space mismatch**: a 7-DOF arm and a quadruped have fundamentally different action spaces. Shared representations must abstract over these differences, losing embodiment-specific structure
2. **Negative transfer**: training on data from dissimilar robots can hurt performance on any single robot. A quadruped's locomotion data may not help (and may hinder) a tabletop manipulation model
3. **Observation heterogeneity**: robots have different camera configurations (monocular, stereo, wrist-mounted), proprioceptive sensors, and tactile feedback. Unifying these into a common input format inevitably loses information
4. **Performance gap**: Octo fine-tuned for a specific robot still underperforms specialist policies trained only on that robot's data

---

## 9. Benchmarks

### Simulation benchmarks

| Benchmark | Focus | Key features |
|-----------|-------|-------------|
| **SIMPLER** (Li et al., 2024) | VLA evaluation in simulation | Google Robot + WidowX simulation with real-world-matched visual rendering; designed to predict real-world VLA performance |
| **Language-Table** | Language-conditioned tabletop manipulation | 2D pushing tasks with natural language instructions; 442K human demonstrations |
| **CALVIN** | Long-horizon manipulation | Sequences of 5+ sub-tasks; tests compositional task execution |
| **RLBench** | Diverse manipulation | 100 tasks with language variations; supports multiple observation types |

**SIMPLER** deserves special attention: it was specifically designed to evaluate VLAs by using visually realistic rendering that matches real-world conditions. Its key contribution is showing high correlation between simulation and real-world VLA performance, enabling cheaper evaluation loops.

### Real-world evaluation

Real-world robot evaluation remains the gold standard but is fundamentally challenging:

- **Protocol**: typically 20–50 trials per task, measuring binary success rate
- **Non-reproducible**: exact conditions (lighting, object placement, calibration) vary between trials
- **Expensive**: each trial requires physical setup, execution, and reset — often manual
- **Small sample sizes**: statistical significance is hard to establish with 20-50 trials; a "5% improvement" may be within noise

Most VLA papers report real-world success rates across a curated set of tasks (e.g., "pick up X," "put X in Y," "open drawer"), with separate categories for seen vs unseen objects/instructions.

---

## 10. Connection to Autonomous Driving

The VLA framework extends naturally beyond tabletop manipulation to **autonomous driving**, where the "action" is a planned trajectory rather than a robot arm command.

### EMMA as a driving VLA

**EMMA** (Waymo, 2024) is a direct instance of the VLA paradigm applied to driving:

| VLA Component | EMMA Implementation |
|---------------|-------------------|
| Vision encoder | Gemini's built-in visual encoder processing multi-camera images |
| Language backbone | Gemini 1.0 Nano-1 |
| Action output | Future trajectory waypoints $(x_t, y_t)$ in BEV space, encoded as text |
| Instruction input | Task-specific text prompts (e.g., "predict the ego trajectory") |

EMMA uses **chain-of-thought reasoning** before action output: it first describes the scene, identifies critical objects, and states a high-level driving decision, then outputs trajectory waypoints. This mirrors VLA designs where the LLM "reasons" before generating actions.

### Other driving VLAs

| Model | Approach | Key Feature |
|-------|----------|------------|
| **DriveVLM** (Tsinghua, 2024) | VLM + chain-of-thought + hierarchical planner | Scene description → critical object identification → decision → planning |
| **LMDrive** (2024) | Closed-loop LLM driving with language instructions | Takes natural language navigation instructions as input |
| **DriveLM** (OpenDriveLab, 2024) | Graph-structured visual QA for driving | Structures driving reasoning as a graph of perception → prediction → planning QA pairs |

### The VLA4AD survey taxonomy

The **VLA for Autonomous Driving (VLA4AD)** survey organises driving VLAs into two paradigms:

1. **End-to-end VLA**: a single model maps sensor inputs directly to driving actions (EMMA, LMDrive). The VLA handles perception, prediction, and planning in one forward pass
2. **Dual-system VLA**: the VLM handles high-level reasoning and scene understanding, while a separate classical or learned planner handles trajectory optimisation (DriveVLM-Dual). The VLM acts as a "copilot" providing structured scene descriptions to a conventional planner

### Action tokenisation in driving

The same action tokenisation challenges from robotics appear in driving, with domain-specific solutions:

| Approach | Model | How it works |
|----------|-------|-------------|
| **Text-based trajectory** | EMMA | Waypoint coordinates written as floating-point text: "(2.1, 0.3), (4.5, 0.8), ..." |
| **Discrete motion tokens** | MotionLM (Waymo, 2023) | Quantise trajectory space into discrete tokens; model multi-agent motion forecasting as next-token prediction |
| **Continuous regression** | UniAD, VAD | Direct regression of BEV trajectory waypoints via MLP head |

MotionLM is particularly elegant: it treats multi-agent trajectory prediction as a **language modelling** problem, tokenising the 2D position space into a discrete vocabulary and predicting future positions autoregressively. This directly parallels RT-2's approach of representing robot actions as text tokens.

### Robotics vs driving: key differences

| Dimension | Robotic Manipulation | Autonomous Driving |
|-----------|--------------------|--------------------|
| Action space | 6-7 DOF end-effector pose + gripper | 2D trajectory waypoints (x, y) or steering + acceleration |
| Temporal horizon | 1–10 seconds | 5–15 seconds (trajectory prediction) |
| Safety criticality | Low (tabletop tasks) | Extremely high (human lives at stake) |
| Multi-agent | Usually single-agent | Must predict and react to dozens of agents |
| Evaluation | Real-world trials (20–50) | Closed-loop simulation + real-world drives |
| Data scale | ~1M episodes | Millions of driving hours (Waymo, Tesla) |

Despite these differences, the core VLA insight transfers: **pre-trained VLMs provide world knowledge that improves generalisation to novel scenarios**, whether those scenarios involve unseen objects on a table or rare driving situations at unusual intersections.

---

## Summary

Vision-Language-Action models represent the convergence of three research threads: visual perception (ViT, DINO), language understanding (LLMs), and embodied control (robot learning). The key architectural pattern — vision encoder → language model → action head — is simple, but its power comes from inheriting internet-scale knowledge through pre-trained VLM backbones.

The field's central tensions remain:

- **Discrete vs continuous actions**: tokenising actions as text is elegant but lossy; diffusion/flow matching is expressive but slow
- **Generalist vs specialist**: embodiment-agnostic models are appealing but still underperform specialist policies
- **Simulation vs real-world evaluation**: simulation is cheap and reproducible; real-world is expensive but trustworthy
- **Data scarcity**: robot data remains orders of magnitude smaller than internet data, making co-training strategies essential

The extension to autonomous driving (EMMA, DriveVLM, MotionLM) shows that the VLA paradigm is not limited to tabletop manipulation — it is a general framework for any system that must perceive, reason, and act in the physical world.

---

## References

- Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv:2307.15818, 2023
- Octo Model Team, "Octo: An Open-Source Generalist Robot Policy," arXiv:2405.12213, 2024
- Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024
- Black, Nakamoto et al., "pi0: A Vision-Language-Action Flow Model for General Robot Control," Physical Intelligence, 2024
- Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT), RSS 2023
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Space Diffusion," RSS 2023
- Hwang et al., "EMMA: End-to-End Multimodal Model for Autonomous Driving," arXiv 2024 (accepted at TMLR), 2024
- Seff et al., "MotionLM: Multi-Agent Motion Forecasting as Language Modeling," ICCV 2023
- Tian et al., "DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models," arXiv:2402.12289, 2024
- Shao et al., "LMDrive: Closed-Loop End-to-End Driving with Large Language Models," CVPR 2024
- Ma et al., "DriveLM: Driving with Graph Visual Question Answering," ECCV 2024
- Li et al., "SIMPLER: Simulated Manipulation Policy Evaluation for Real Robot Setups," arXiv:2405.05941, 2024
- Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," arXiv:2310.08864, 2023
