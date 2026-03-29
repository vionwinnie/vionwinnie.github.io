---
title: "Survey: Motion Planning & End-to-End VLM-Based Driving"
category: engineering
excerpt: "Paper survey covering EMMA, UniAD, VAD, ChauffeurNet, MotionLM, GPT-Driver, DriveVLM, S4-Driver, scaling laws, and Waymo Foundation Model — prepared for Waymo interview."
tags: autonomous-driving planning vlm tech-interview
comments: true
---

# Quick Survey: Motion Planning, Control, and End-to-End VLM-Based Reasoning for Autonomous Driving

*Prepared for Waymo Visual Reasoning team interview with Wei-Chih Hung*
*Last updated: March 2026*

---

## Overview

Autonomous driving is undergoing a paradigm shift from modular pipelines (perception -> prediction -> planning -> control) toward **end-to-end learned systems** that map sensor inputs directly to driving actions. This shift is accelerated by the emergence of **Vision-Language Models (VLMs)** and **Vision-Language-Action (VLA)** architectures that unify visual perception, natural language reasoning, and trajectory generation within a single framework. The promise is twofold: better generalization to long-tail scenarios through pre-trained world knowledge, and improved interpretability through chain-of-thought reasoning expressed in natural language.

The field has evolved through several phases: (1) classical trajectory optimization and rule-based planners (pre-2018); (2) imitation learning from human demonstrations (ChauffeurNet, 2018); (3) modular end-to-end models with differentiable intermediate representations (UniAD, VAD, 2023); (4) LLM/VLM-augmented driving systems (GPT-Driver, DriveVLM, 2023-2024); and (5) fully end-to-end multimodal models that represent all outputs as language tokens (EMMA, 2024). Each phase did not replace the previous one -- rather, the field maintains active research across all paradigms, with the frontier now focused on scaling VLA models, closing the sim-to-real gap, and establishing reliable closed-loop evaluation.

Waymo has been a consistent contributor across this entire trajectory, from ChauffeurNet and MultiPath to MotionLM and EMMA. The Visual Reasoning team, led by researchers including Wei-Chih Hung, sits at the intersection of perception, scene understanding, and end-to-end planning -- making EMMA a natural convergence point of their research directions in open-vocabulary panoptic segmentation (ECCV 2024) and VLM-based driving.

---

## Timeline & Evolution

| Year | Paper/System | Key Innovation | Venue |
|------|-------------|----------------|-------|
| 2018 | ChauffeurNet (Waymo) | Imitation learning with synthesized perturbations for robust driving | RSS 2019 |
| 2019 | MultiPath (Waymo) | Anchor-based multi-modal trajectory prediction with GMMs | CoRL 2019 |
| 2021 | MultiPath++ (Waymo) | Efficient polyline encoding + trajectory aggregation | ICRA 2022 |
| 2021 | nuPlan (Motional) | First closed-loop ML planning benchmark | arXiv |
| 2023 | **UniAD** (SenseTime/OpenDriveLab) | Unified perception-prediction-planning with query-based transformers | **CVPR 2023 Best Paper** |
| 2023 | VAD | Vectorized scene representation for efficient end-to-end planning | ICCV 2023 |
| 2023 | GameFormer (NVIDIA) | Game-theoretic interactive prediction + planning | ICCV 2023 |
| 2023 | MotionLM (Waymo) | Multi-agent motion forecasting as language modeling | ICCV 2023 |
| 2023 | GPT-Driver | Motion planning reformulated as LLM language generation | arXiv |
| 2023 | GAIA-1 (Wayve) | Generative world model for driving video synthesis | arXiv |
| 2024 | DriveVLM (Tsinghua) | VLM with CoT for scene understanding + hierarchical planning | arXiv |
| 2024 | LMDrive | Closed-loop LLM driving with language instructions | CVPR 2024 |
| 2024 | DriveLM (OpenDriveLab) | Graph VQA for structured driving reasoning | ECCV 2024 Oral |
| 2024 | VADv2 | Probabilistic planning, closed-loop CARLA SOTA | arXiv |
| 2024 | DTPP (NVIDIA) | Differentiable joint conditional prediction + cost evaluation | ICRA 2024 |
| 2024 | **EMMA (Waymo)** | **End-to-end multimodal model: all outputs as text via Gemini** | **TMLR** |
| 2024 | Tesla FSD v12 | Full end-to-end neural net replacing 300K lines of C++ | Production |
| 2025 | **S4-Driver (Waymo/UC Berkeley)** | **Self-supervised E2E driving MLLM with no human annotations; sparse volume 3D lifting** | **CVPR 2025** |
| 2025 | VLA Survey papers | Systematization of VLA4AD into end-to-end vs dual-system paradigms | ICCV 2025 Workshop |
| 2025 | **Scaling Laws for Driving (Waymo)** | **First empirical scaling laws for joint motion forecasting and planning** | **arXiv** |
| 2025 | **WOD-E2E (Waymo)** | **Long-tail E2E driving benchmark with Rater Feedback Score metric** | **arXiv** |
| 2025 | **Waymo Foundation Model** | **"Think Fast / Think Slow" dual-system production architecture with Driving VLM** | **Waymo Blog** |
| 2026 | FROST-Drive (Dong et al.) | Frozen VLM encoder + adapter for E2E driving; optimizes for RFS on WOD-E2E | WACV 2026 Workshop |
| 2026 | Waymo World Model | Genie 3-based photorealistic 3D simulation | Waymo Blog |

---

## 1. EMMA Deep-Dive

**EMMA: End-to-End Multimodal Model for Autonomous Driving**
Hwang, Xu, Lin, **Hung**, Ji, Choi, Huang, He, Covington, Sapp, Zhou, Guo, Anguelov, Tan (Waymo, 2024)
[arXiv:2410.23262](https://arxiv.org/abs/2410.23262) | Published in TMLR

### Architecture

| Component | Detail |
|-----------|--------|
| **Backbone** | Gemini 1.0 Nano-1 (smallest Gemini variant) |
| **Input** | Raw multi-camera images (up to 4 frames) + text prompts |
| **Output** | Natural language text encoding trajectories, 3D detections, road graphs |
| **Training** | End-to-end fine-tuning of the pre-trained MLLM on driving tasks |

### Key Design Decisions

**Text-based output representation.** All waypoint coordinates are represented as plain text floating-point numbers (not specialized tokens). Future trajectories are expressed as waypoint sets in BEV space: O_trajectory = {(x_t, y_t)} for t=1..T_f. This allows the model to leverage the pre-trained language model's numerical reasoning capabilities without custom tokenizers.

**Task-specific prompts.** The same model handles multiple tasks by switching prompts:
- Motion planning: predict ego trajectory waypoints
- 3D object detection: detect every object with 3D bounding boxes
- Road graph estimation: estimate lane boundaries and road structure

**Chain-of-thought reasoning.** EMMA generates a structured reasoning chain before outputting trajectories:
- R1: Scene description (weather, road conditions, traffic density)
- R2: Critical objects with precise 3D/BEV coordinates
- R3: Behavior descriptions of identified objects
- R4: Meta driving decision (one of 12 high-level action categories)

This CoT approach improves planning performance by **6.7%** over the baseline without reasoning.

**Multi-task co-training.** Joint training across planning, detection, and road graph tasks yields improvements in all three domains -- a key finding supporting the unified architecture thesis.

### Quantitative Results

| Benchmark | Metric | EMMA | EMMA+ | Previous SOTA |
|-----------|--------|------|-------|---------------|
| nuScenes Planning | Avg L2 (m) | 0.32 | 0.29 | 0.39 (DriveVLM-Dual) |
| WOMD | ADE@1s (m) | -- | 0.030 | -- |
| WOMD | ADE@5s (m) | -- | 0.610 | -- |
| WOD 3D Detection | Vehicle Precision | +16.3% relative | -- | -- |

*EMMA+ uses additional internal pre-training data*

### Key Limitations (from the paper)

1. **Limited temporal context**: Only processes up to 4 frames; cannot capture long-term dependencies
2. **No 3D sensor fusion**: Cannot integrate LiDAR/radar due to MLLM architecture constraints
3. **Consistency gaps**: No guarantee that planning and perception outputs are mutually consistent
4. **Expensive closed-loop eval**: Sensor simulation costs several times more than behavior simulation
5. **Deployment latency**: Large model requires distillation or optimization for real-time inference

### Why EMMA Matters

EMMA represents a bet that **foundation model pre-training** (via Gemini) provides enough world knowledge to compensate for limited driving-specific training data, and that **natural language as a universal interface** can unify the fragmented autonomous driving stack. If the approach scales with larger models and more data, it could fundamentally change how AV systems are built.

---

## 2. End-to-End Autonomous Driving Models

### ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst
Bansal et al. (Waymo, 2018) | [arXiv:1812.03079](https://arxiv.org/abs/1812.03079) | RSS 2019

- **Contribution:** First major learned driving policy at Waymo, demonstrating that imitation learning can produce an urban driving system — and identifying the key failure modes (distribution shift, rare events) that shaped all subsequent work.
- **Method:** A mid-level imitation learning approach that operates on a top-down rendered representation of the scene rather than raw sensor data:
  1. **Input representation:** The driving scene is rendered as a set of rasterized BEV images encoding: road map, traffic lights, speed limit, route, dynamic objects (as bounding boxes with heading), and past agent trajectories. This mid-level representation abstracts away raw perception.
  2. **Architecture:** A convolutional neural network (ChauffeurNet) takes the stacked BEV images and outputs a future trajectory for the ego vehicle, represented as a sequence of waypoints. A separate control module converts waypoints to steering/throttle/brake.
  3. **Key innovation — synthesized perturbations:** Pure behavioral cloning suffers from distribution shift (the model never sees recovery from mistakes during training). ChauffeurNet addresses this by synthesizing training perturbations: the ego vehicle's position/heading is artificially perturbed from the expert trajectory, and the model is trained to recover back to the expert path. This exposes the model to off-distribution states during training.
  4. **Loss augmentation:** Beyond L2 trajectory loss, the system adds losses for collision avoidance, on-road driving, and following the route — acting as learned safety constraints.
- **Results:** Demonstrated autonomous urban driving on real Waymo vehicles. Perturbation training reduced collisions by ~60% compared to pure behavioral cloning. Successfully handled complex scenarios including unprotected turns, yielding, and nudging around double-parked vehicles.
- **Significance:** Established the IL-for-driving paradigm at Waymo and identified the core challenges (distribution shift, long-tail events, causal confusion) that motivated subsequent work. The mid-level representation approach influenced MultiPath and later Waymo prediction models. The perturbation augmentation idea became standard practice.

### MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction
Chai et al. (Waymo, 2019) | [arXiv:1910.05449](https://arxiv.org/abs/1910.05449) | CoRL 2019

- **Contribution:** Introduced anchor-based multi-modal trajectory prediction using a fixed set of trajectory anchors combined with Gaussian Mixture Models (GMMs), enabling efficient and diverse prediction of agent futures.
- **Method:**
  1. **Anchor trajectories:** A fixed set of K trajectory anchors is pre-computed by clustering trajectories from the training data (e.g., K=64). These anchors capture common motion patterns (straight, left turn, right turn, lane change, etc.) and serve as "templates" for prediction.
  2. **Architecture:** A CNN processes the rasterized BEV scene representation (similar to ChauffeurNet's input). For each agent, the model outputs: (a) a probability distribution over the K anchors (which motion mode is most likely), and (b) per-anchor residual offsets and uncertainty estimates (Gaussian parameters) that refine each anchor to the specific situation.
  3. **Output:** The final prediction is a GMM: K Gaussian components, each centered on an anchor + residual, weighted by the predicted mode probabilities. This naturally captures multi-modality — a vehicle at an intersection might have high probability on both "go straight" and "turn left" anchors.
- **Results:** On internal Waymo prediction benchmarks: significant improvements in multi-modal prediction accuracy over single-trajectory baselines. The anchor-based approach achieves better coverage of the true future distribution while maintaining computational efficiency. Later evolved into MultiPath++ (ICRA 2022) which replaced rasterized inputs with polyline encoders for 2x efficiency gains.
- **Significance:** Established the anchor-based prediction paradigm used widely in industry (including Waymo's production system). The key insight — decomposing prediction into mode selection + residual regression — influenced subsequent work including MotionLM. Demonstrated that prediction should be fundamentally multi-modal, not single-trajectory.

### UniAD: Planning-Oriented Autonomous Driving
Hu et al. (SenseTime / Shanghai AI Lab, 2023) | [arXiv:2212.10156](https://arxiv.org/abs/2212.10156) | **CVPR 2023 Best Paper**

- **Contribution:** First unified framework connecting perception, prediction, and planning through query-based transformers, optimized end-to-end toward the planning objective.
- **Method:** Multi-camera images -> BEV features -> five cascaded modules: TrackFormer (tracking), MapFormer (online mapping), MotionFormer (trajectory prediction), OccFormer (occupancy prediction), and Planner. Unified query design lets information flow across all tasks. Two-stage training (perception first, then full E2E).
- **Results:** +20% tracking accuracy, +30% mapping accuracy, -38% motion forecasting error, -28% planning error vs prior SOTA on nuScenes.
- **Significance:** Proved that planning-oriented joint optimization outperforms independently optimized modules. Set the template for subsequent E2E models.

### VAD / VADv2: Vectorized Autonomous Driving
Jiang et al. (HUST, 2023-2024) | [arXiv:2303.12077](https://arxiv.org/abs/2303.12077) (ICCV 2023) | [arXiv:2402.13243](https://arxiv.org/abs/2402.13243) (VADv2)

- **Contribution:** Replaced dense rasterized scene representations with fully vectorized representations (polylines for map, trajectories for agents), enabling faster and safer planning with explicit structural constraints.
- **Method:**
  - **Rasterized BEV (prior work like UniAD):** The scene is represented as a dense grid of pixels in bird's-eye view — essentially a top-down image where each cell encodes occupancy, semantic class, or feature vectors. This is memory-intensive (e.g., a 200x200 grid at 0.5m resolution), loses instance-level structure, and requires the planner to re-extract object boundaries from the grid.
  - **Vectorized representation (VAD):** Instead of a dense grid, the scene is represented as sets of polylines and trajectories:
    - **Map elements** are ordered sequences of control points defining lane boundaries, crosswalks, and road edges (e.g., a lane boundary = [(x1,y1), (x2,y2), ..., (xN,yN)]).
    - **Agent motions** are represented as trajectory sequences of future positions for each detected agent.
    - These vectorized elements serve as explicit instance-level planning constraints: the ego trajectory is optimized to avoid agent trajectories and stay within map boundaries through vectorized attention mechanisms (ego queries attend to agent/map vectors).
  - **VADv2:** Extends VAD to probabilistic planning — instead of regressing a single trajectory, it models a discrete distribution over a large vocabulary of trajectory candidates (4096 action tokens). At each timestep, the model samples from this distribution, enabling stochastic rollouts and multi-modal planning. Trained with a classification objective rather than regression.
- **Results:** VAD-Base: L2 error 0.54/1.15/1.98m (1/2/3s), -29% collision rate, 2.5x faster than UniAD on nuScenes. VAD-Tiny: up to 9.3x faster. VADv2: SOTA closed-loop performance on CARLA Town05 Long — driving score of 64.3 (vs. 31.0 for UniAD, 38.2 for VAD).
- **Significance:** Demonstrated that vectorized representations are more efficient (less memory, faster inference) and preserve structural information better than rasterized BEV grids. The vectorized design influenced subsequent work including EMMA's approach to representing outputs as structured sequences rather than dense grids.

### Tesla FSD v12-v13 (Production, 2024-2025)

- **Contribution:** First production deployment of a fully end-to-end neural network for autonomous driving, replacing ~300,000 lines of C++ with learned models.
- **Method:** 8 cameras -> neural network -> direct control outputs (steering, acceleration, braking). Trained on billions of miles of human driving data using massive GPU clusters (35,000+ H100s).
- **Results:** 100x improvement in miles between critical interventions (v12.5). Over 8.3 billion FSD miles driven by Feb 2026.
- **Significance:** Largest-scale validation of end-to-end driving, though details remain proprietary. Demonstrates the approach is viable at production scale.

### S4-Driver: Scalable Self-Supervised Driving Multimodal Large Language Model with Spatio-Temporal Visual Representation
Xie, Xu, He, Hwang, Luo, Ji, Lin, Chen, Lu, Leng, Anguelov, Tan (UC Berkeley, Waymo, Cornell, Georgia Tech, 2025) | [arXiv:2505.24139](https://arxiv.org/abs/2505.24139) | CVPR 2025

- **Contribution:** First self-supervised end-to-end driving MLLM that requires NO human annotations. Built on PaLI-3 5B. Removes the annotation bottleneck that limits prior supervised E2E approaches.
- **Method:**
  1. **Sparse volume strategy:** Lifts 2D MLLM visual features into 3D space without fine-tuning the vision encoder, enabling spatio-temporal reasoning from camera images alone.
  2. **Hierarchical planning with free CoT:** Uses meta-decisions (keep stationary, keep speed, accelerate, decelerate) as chain-of-thought reasoning derived automatically from driving logs — no human annotation needed.
  3. **Multi-decoding aggregation:** Employs nucleus sampling to generate diverse trajectory candidates, then aggregates them for robust planning.
  4. **Self-supervised training:** All supervision signals come from raw driving logs (ego trajectories, vehicle dynamics) rather than human-labeled perception annotations.
- **Results:** SOTA on nuScenes planning (Avg L2: 0.31m self-supervised, vs 0.37m for VAD supervised — a self-supervised model beating supervised methods). On WOMD-Planning-ADE (103K scenarios, 100x larger than nuScenes): ADE@5s 0.655, bADE@5s 0.830. Outperforms MotionLM on behavior-wise metrics despite using only raw camera images.
- **Significance:** EMMA's successor direction at Waymo — removes the annotation bottleneck entirely. Shows self-supervised training scales with data (performance improves monotonically with more driving logs). Critical insight: you don't need expensive human labels to train competitive E2E driving models.

### FROST-Drive: Scalable and Efficient End-to-End Driving with a Frozen Vision Encoder
Dong, Zhu, Wu, Sun (2026) | [arXiv:2601.03460](https://arxiv.org/abs/2601.03460) | WACV 2026 LLVM-AD Workshop

- **Contribution:** Demonstrates that keeping a VLM vision encoder **frozen** (no fine-tuning) outperforms domain-adapted encoders on long-tail driving scenarios — challenging the conventional fine-tuning paradigm.
- **Method:** Three-component architecture: (1) a **frozen vision encoder** from a pre-trained VLM that preserves broad world knowledge, (2) a **transformer-based multimodal fusion adapter** that learns to extract driving-relevant features without modifying encoder weights, and (3) a **GRU-based waypoint decoder** for trajectory generation. Introduces a custom loss function that directly optimizes for the **Rater Feedback Score (RFS)** metric from WOD-E2E.
- **Results:** On Waymo Open E2E Dataset, the frozen-encoder approach outperforms fully fine-tuned models, particularly on rare long-tail scenarios. The broad generalization from pre-training transfers more effectively than specialized fine-tuning.
- **Significance:** Independent validation of S4-Driver's frozen-encoder insight. The fact that *not* fine-tuning the vision encoder works better is counterintuitive but consistent with the emerging understanding that VLM pre-training provides world knowledge that domain-specific fine-tuning destroys. Directly relevant to Waymo's production stack — the Foundation Model's Driving VLM is fine-tuned from Gemini, but FROST-Drive suggests the vision encoder portion should remain frozen.

---

## 3. VLMs/LLMs for Driving Reasoning

### GPT-Driver: Learning to Drive with GPT
Mao et al. (2023) | [arXiv:2310.01415](https://arxiv.org/abs/2310.01415)

- **Contribution:** First to reformulate motion planning as a language modeling problem, using GPT-3.5 as a motion planner that outputs trajectory waypoints as text tokens.
- **Method:** A three-stage prompting-reasoning-finetuning pipeline:
  1. **Prompting:** Heterogeneous planner inputs (ego state, nearby agent positions/velocities, HD map elements, route waypoints) are serialized into structured natural language prompts. Coordinates are encoded as comma-separated floating-point values in the ego vehicle's coordinate frame (e.g., "(2.31, 0.05)").
  2. **Reasoning:** GPT-3.5 performs chain-of-thought reasoning over the prompt — describing the scene context, identifying critical agents, and explaining its driving intention before outputting a trajectory.
  3. **Finetuning:** The model is fine-tuned on a curated set of human-driving demonstrations from nuScenes to calibrate its numerical outputs to realistic driving behavior.
  The output is a sequence of future BEV waypoints: {(x_t, y_t)} for t=1..6 at 0.5s intervals (3s horizon).
- **Results:** On nuScenes open-loop planning: L2 error of 0.71m (1s), 1.38m (2s), 2.05m (3s) — competitive with specialized planners like UniAD (0.48/0.96/1.65m). Collision rate of 0.31% (comparable to specialized models). Notably strong zero-shot generalization to out-of-distribution driving scenarios.
- **Significance:** Proof-of-concept that LLMs can reason about driving geometry and produce numerically precise trajectories, opening the VLM-for-driving research direction. Directly inspired EMMA's approach of representing all outputs as text. Key insight: LLM pre-training provides implicit world knowledge about physical dynamics and traffic conventions.

### DriveVLM: Convergence of Autonomous Driving and Large VLMs
Tian et al. (Tsinghua / BYD, 2024) | [arXiv:2402.12289](https://arxiv.org/abs/2402.12289)

- **Contribution:** VLM-based system with chain-of-thought reasoning for scene description, analysis, and hierarchical planning. Also proposes DriveVLM-Dual, a practical hybrid combining VLM reasoning with traditional AV pipeline.
- **Method:** Multi-view images -> VLM processes via CoT: scene description -> scene analysis -> hierarchical planning (meta-actions, decision descriptions, waypoints). DriveVLM-Dual runs VLM reasoning in parallel with a fast traditional planner.
- **Results:** Strong performance on nuScenes and their SUP-AD dataset. DriveVLM-Dual deployed on production vehicle.
- **Significance:** First VLM driving system deployed on a real vehicle. The dual-system architecture addresses the latency problem of large VLMs.

### LMDrive: Closed-Loop End-to-End Driving with Large Language Models
Shao et al. (Shanghai AI Lab, 2024) | [arXiv:2312.07488](https://arxiv.org/abs/2312.07488) | CVPR 2024

- **Contribution:** First work to use LLMs for closed-loop end-to-end driving with natural language instruction following, demonstrating that language models can directly output vehicle control signals in a reactive driving loop.
- **Method:** A multi-modal architecture with three key components:
  1. **Vision encoder:** Processes multi-view camera images and LiDAR BEV projections through separate encoders, producing visual tokens for each sensor modality and each timestep. Multiple frames of history are retained (typically the last 4 frames at 2Hz).
  2. **Q-Former adapter:** A Querying Transformer (Q-Former, adapted from BLIP-2) bridges the visual encoders and the LLM. It uses a set of learnable query tokens that cross-attend to the visual features, compressing the high-dimensional visual information into a fixed number of tokens that the LLM can process. This is critical for efficiency — raw visual features would overwhelm the LLM's context window.
  3. **LLM decoder (LLaMA-based):** Takes the Q-Former output tokens concatenated with tokenized language instructions (e.g., "Turn left at the next intersection") and predicts: (a) control signals — waypoints in ego frame that are converted to steering, throttle, and brake via a PID controller, and (b) a binary instruction completion flag indicating whether the current instruction has been fulfilled.
  The system operates in closed-loop: at each timestep, new sensor observations are encoded and the LLM generates updated control outputs, allowing the vehicle to react to dynamic changes.
- **Results:** Introduced LangAuto benchmark with 64K instruction-following clips in CARLA. Achieves driving score of 35.3 on CARLA Town05 Long benchmark (competitive with prior E2E methods). On LangAuto: 79.2% instruction completion rate with 0.22 collision rate per km. Outperforms TransFuser and InterFuser baselines on instruction-conditioned driving.
- **Significance:** Showed LLMs can handle the full closed-loop driving cycle including instruction understanding, visual reasoning, and continuous control generation. The Q-Former adapter design became influential for subsequent VLM-driving systems needing to compress visual features for language models.

### DriveLM: Driving with Graph Visual Question Answering
Sima et al. (OpenDriveLab, 2024) | [arXiv:2312.14150](https://arxiv.org/abs/2312.14150) | ECCV 2024 Oral

- **Contribution:** Proposed Graph VQA — a structured multi-step QA framework that mirrors human driving reasoning as a directed acyclic graph, with explicit dependencies between perception, prediction, and planning stages.
- **Method:** The core idea is Graph VQA, where driving reasoning is structured as a graph:
  - **Nodes** are individual QA pairs, each belonging to one of three stages: (1) **Perception** nodes (e.g., "What objects are in the left lane?" -> "A sedan at (12.3, -3.1) moving at 8 m/s"), (2) **Prediction** nodes (e.g., "Will the sedan change lanes?" -> "Likely yes, its trajectory curves toward ego lane"), (3) **Planning** nodes (e.g., "What should ego do?" -> "Decelerate and maintain following distance").
  - **Edges** represent logical dependencies between QA pairs — a prediction node has directed edges from the perception nodes it depends on, and planning nodes have edges from relevant prediction nodes. This encodes the causal structure of driving decisions.
  - The full graph for a driving scene typically contains 10-20 QA nodes with edges encoding which perceptual facts feed into which predictions and plans.
  DriveLM-Agent is a baseline model that jointly performs Graph VQA and end-to-end driving: a VLM (BLIP-2 based) answers the graph-structured questions sequentially and produces final trajectory waypoints. DriveLM-Data provides human-annotated graph QA pairs on nuScenes (17K keyframes) and CARLA.
- **Results:** On nuScenes planning: L2 error of 0.45m (1s) and 1.48m (3s), competitive with UniAD (0.48/1.65m). On DriveLM-Challenge: achieves 42.8 language score and 37.2 driving score. Strong zero-shot transfer to unseen sensor configurations (e.g., different camera layouts) with only ~5% performance drop, thanks to the language-based intermediate representation.
- **Significance:** Established a principled framework for structured, interpretable driving reasoning. The graph structure makes it possible to trace exactly which perceptual facts led to a planning decision — a concrete improvement over flat CoT approaches. Provides a challenging benchmark (DriveLM-Data) for the community.

---

## 4. Classical vs. Learned Motion Planning

| Approach | Method | Strengths | Weaknesses |
|----------|--------|-----------|------------|
| **Trajectory Optimization** | Minimize cost function over trajectory space (comfort, safety, progress) subject to vehicle dynamics constraints | Interpretable, safety guarantees, handles constraints explicitly | Requires hand-designed cost functions; struggles with complex multi-agent interactions |
| **Sampling-based** | Generate candidate trajectories, evaluate and select (e.g., lattice planners, RRT variants) | Handles non-convex constraints; Waymo's production system uses elements of this | Combinatorial explosion; quality depends on sampling strategy |
| **Imitation Learning** | Learn policy from expert demonstrations (behavioral cloning, DAgger, ChauffeurNet) | Learns complex behaviors from data; no reward engineering | Distribution shift; causal confusion; struggles with rare events |
| **Reinforcement Learning** | Learn policy by maximizing reward in simulation (PPO, SAC applied to driving) | Can discover novel strategies; handles multi-agent interaction | Reward shaping is difficult; sim-to-real gap; safety during training |
| **End-to-End Learned** | Map sensors directly to trajectories/controls (UniAD, VAD, EMMA) | Jointly optimized; no information bottleneck between modules | Black-box; harder to verify safety; requires massive data |
| **VLM/VLA-based** | Use pre-trained language models for reasoning + planning (EMMA, DriveVLM) | World knowledge transfer; interpretable reasoning; instruction following | High latency; limited spatial precision; hallucination risk |

### Key Insight for the Interview
Waymo's trajectory shows a deliberate evolution: ChauffeurNet (pure IL) -> MultiPath (learned prediction, classical planning) -> MotionLM (language modeling for prediction) -> EMMA (language modeling for everything). Each step incorporates more learning while the production system likely maintains safety-critical classical components as fallbacks.

---

## 5. Joint Prediction + Planning Models

### MotionLM: Multi-Agent Motion Forecasting as Language Modeling
Seff et al. (Waymo, 2023) | [arXiv:2309.16534](https://arxiv.org/abs/2309.16534) | ICCV 2023

- **Contribution:** Cast multi-agent motion prediction as autoregressive language modeling over discrete motion tokens. No anchors or latent variables needed.
- **Method:** Continuous trajectories discretized into motion tokens. Standard language modeling objective (maximize log probability over sequence). Joint distributions over multiple agents produced in a single autoregressive pass.
- **Results:** SOTA on WOMD motion prediction and interaction prediction benchmarks. #1 on the interactive challenge leaderboard.
- **Significance:** Direct precursor to EMMA's philosophy of "everything as language." Proved that the language modeling paradigm works for structured prediction in driving.

### GameFormer: Game-Theoretic Interactive Prediction and Planning
Huang et al. (NVIDIA, 2023) | [arXiv:2303.05760](https://arxiv.org/abs/2303.05760) | ICCV 2023

- **Contribution:** Formulated multi-agent interaction prediction as a hierarchical game-theoretic process, where agents iteratively refine their predicted strategies by reasoning about each other's plans — analogous to levels of strategic reasoning in game theory (level-k planning).
- **Method:** A Transformer encoder first processes the scene context (agent histories, map polylines) into agent feature embeddings. Then a hierarchical game-theoretic decoder operates over K levels:
  - **Level 0:** Each agent independently predicts M candidate future trajectories (initial "best responses" ignoring others).
  - **Level k (k>0):** Each agent updates its predicted trajectories conditioned on all other agents' level-(k-1) predictions plus the scene context. This is implemented via cross-attention between agent queries, creating an implicit game where agents iteratively best-respond to each other.
  - The ego agent's final plan is selected from its level-K trajectory candidates using a learned scoring function.
  This hierarchy captures escalating levels of strategic interaction without explicitly solving a game equilibrium.
- **Results:** On WOMD interactive prediction: minADE of 0.517m (single agent), significantly outperforming non-interactive baselines. On nuPlan closed-loop planning: highest overall score among learning-based planners, competitive with the expert rule-based planner (PDM-Closed). Specifically, achieves 92.0 overall score on nuPlan Val14 closed-loop reactive benchmark.
- **Significance:** Addresses the chicken-and-egg problem of joint prediction-planning: ego plan depends on others' predictions, which depend on the ego plan. The level-k hierarchy provides a principled approximation. Directly influenced DTPP's approach to ego-conditioned prediction.

### DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation
Huang et al. (NVIDIA, 2024) | [arXiv:2310.05885](https://arxiv.org/abs/2310.05885) | ICRA 2024

- **Contribution:** Differentiable framework for jointly training ego-conditioned prediction and a learned cost function, enabling tree-structured policy planning that evaluates branching future scenarios.
- **Method:** Three tightly coupled components trained end-to-end:
  1. **Ego-conditioned predictor:** A query-centric Transformer takes candidate ego plan proposals and predicts how surrounding agents would react to each proposal (conditional motion forecasts). This produces different predicted futures for each ego plan.
  2. **Learned cost function:** Instead of hand-designing a cost function, DTPP learns a context-aware cost model that takes as input the ego plan, the conditional predictions of other agents, and latent interaction features from the Transformer. The cost captures safety (collision risk), comfort (jerk, lateral acceleration), and progress, but the weights and feature interactions are learned from data.
  3. **Tree policy planner:** At inference, the planner constructs a tree of ego action sequences — branching at each time step into multiple candidate maneuvers. Each branch is scored by the learned cost function evaluated on the predicted reactions of other agents. The optimal plan is the minimum-cost path through the tree, found via beam search.
  The key insight is that because all three components are differentiable, gradients from the planning cost flow back through the predictor, teaching it to produce predictions that are most useful for planning (not just most accurate in isolation).
- **Results:** On nuPlan closed-loop reactive benchmark: joint training of prediction + cost function improves planning score by 4.2% over separately trained modules. Outperforms rule-based IDM planner and matches PDM-Closed on key safety metrics while producing smoother trajectories.
- **Significance:** Shows that tight coupling of prediction and planning through differentiable training yields better decisions than pipeline approaches. The tree policy planner is a practical middle ground between greedy single-trajectory planning and intractable full contingency planning.

### Scaling Laws of Motion Forecasting and Planning
Baniodeh, Goel, Ettinger, Fuertes, Seff, Shen, Gulino, et al. (Waymo, 2025) | [arXiv:2506.08228](https://arxiv.org/abs/2506.08228)

- **Contribution:** First empirical scaling laws study for joint motion forecasting and planning in autonomous driving. Uses an encoder-decoder autoregressive transformer (MotionLM architecture) on ~500K hours / 59.8M run segments / 5.6M miles of driving data.
- **Key Findings:**
  1. **Power-law scaling:** Cross-entropy loss follows power-law scaling with compute, mirroring LLM scaling behavior (Chinchilla-style laws apply to driving).
  2. **Optimal compute allocation:** Model size should grow 1.5x faster than dataset size (N_opt is proportional to C^0.63, D_opt is proportional to C^0.44). At the same compute budget, the optimal motion forecasting model is ~50x smaller than the optimal LLM — driving needs proportionally more data than language.
  3. **Closed-loop correlation:** Closed-loop driving metrics (safety, comfort) correlate with pre-training loss — bigger models produce safer drivers.
  4. **Inference-time scaling:** Sampling + clustering from smaller models can match larger models' performance up to a crossover point, providing a practical compute-performance tradeoff.
  5. **Cross-agent transfer:** Training on other agents' driving logs improves ego-agent planning — skills transfer across agents.
- **Significance:** Validates that "scaling is all you need" for autonomous driving. Provides principled guidance for compute allocation between model size and data. Shows Waymo's research direction: scale the MotionLM paradigm with massive data. Critical complement to EMMA — while EMMA explores VLM-based E2E driving, this paper shows the fundamental scaling properties of the autoregressive approach to driving.

---

## 6. Safety and Interpretability

### How VLM-Based Approaches Improve Explainability

| Aspect | Traditional E2E (UniAD, VAD) | VLM-Based (EMMA, DriveVLM) |
|--------|------------------------------|---------------------------|
| **Decision transparency** | Intermediate representations (BEV, heatmaps) provide some insight but require expert interpretation | Natural language reasoning chains explain *why* a decision was made in human-readable form |
| **Failure analysis** | Requires probing internal activations | Can inspect the textual CoT to identify reasoning errors |
| **Human communication** | Cannot naturally explain behavior to passengers or operators | Can generate explanations: "Slowing down because pedestrian stepping into crosswalk" |
| **Instruction following** | Fixed behavior policy | Can accept and act on natural language instructions |
| **Regulatory compliance** | Difficult to audit internal decision process | Text-based reasoning provides audit trail |

### Key Challenges

- **Hallucination risk:** VLMs can generate plausible-sounding but factually incorrect reasoning (e.g., detecting phantom objects). This is safety-critical in driving.
- **Consistency:** EMMA's paper explicitly notes there is "no guarantee that [planning and perception] outputs will be always consistent." A model might reason correctly but plan incorrectly, or vice versa.
- **Latency vs. safety:** Thorough CoT reasoning takes time. Dual-system approaches (DriveVLM-Dual) address this by running slow VLM reasoning alongside a fast reactive planner.
- **Verification:** Unlike rule-based systems, it is difficult to formally verify that a VLM-based planner will never make unsafe decisions.

### EMMA's Specific Approach to Interpretability
EMMA's four-stage CoT (R1-R4) provides structured interpretability:
1. R1 (scene description) shows what the model perceives
2. R2 (critical objects) shows what the model attends to
3. R3 (behavior descriptions) shows the model's predictions of others
4. R4 (meta driving decision) shows the chosen action category

This improves planning by 6.7% while providing an inspection point at each reasoning stage.

### WOD-E2E: Waymo Open Dataset for End-to-End Driving
Xu, Lin, Jeon, Feng, Zou, Sun, Gorman, et al. (Waymo, 2025) | [arXiv:2510.26125](https://arxiv.org/abs/2510.26125)

- **Contribution:** New benchmark specifically designed for end-to-end driving evaluation on long-tail scenarios (events occurring at <0.03% frequency). Contains 4,021 segments (~12 hours), 8 cameras with 360-degree coverage, routing info, and ego trajectory ground truth.
- **Scenario Coverage:** 11 challenging scenario categories mined from 6.4M miles of driving data using rule-based heuristics + MLLM scoring (Gemini 2.5 Pro for rarity assessment):
  - Construction zones, complex intersections, pedestrians, cyclists, cut-ins, foreign object debris, special vehicles, and more.
- **Novel Metric — Rater Feedback Score (RFS):** Expert raters score 3 trajectory candidates (0-10) at critical moments. Unlike ADE/L2 which compare to a single ground truth, RFS captures multi-modal acceptability — multiple safe trajectories can score well even if they differ from the recorded ground truth. This addresses a fundamental limitation: ADE penalizes safe evasive maneuvers that diverge from what the driver happened to do.
- **Results:** WOD-E2E has significantly higher rarity scores than nuScenes and WOMD across all percentiles. Already used for the 2025 Waymo Open Dataset Challenge.
- **Significance:** Directly addresses EMMA's evaluation limitation (EMMA was evaluated on nuScenes, which is dominated by nominal driving). Represents the field's shift from nominal-driving benchmarks to long-tail benchmarking. The RFS metric is a step toward human-aligned evaluation of planning quality.

---

## 7. Key Waymo Research Contributions

| Paper | Year | Contribution | arXiv |
|-------|------|-------------|-------|
| **ChauffeurNet** | 2018 | IL for urban driving with synthesized perturbations; first major learned planner at Waymo | [1812.03079](https://arxiv.org/abs/1812.03079) |
| **MultiPath** | 2019 | Anchor-based multi-modal trajectory prediction using GMMs | [1910.05449](https://arxiv.org/abs/1910.05449) |
| **MultiPath++** | 2021 | Efficient polyline scene encoding + trajectory aggregation | [2111.14973](https://arxiv.org/abs/2111.14973) |
| **Waymo Open Dataset** | 2019 | One of the largest AV datasets; used by 36K+ researchers worldwide | [1912.04838](https://arxiv.org/abs/1912.04838) |
| **WOMD** | 2021 | Waymo Open Motion Dataset for behavior prediction benchmarking | [2104.10133](https://arxiv.org/abs/2104.10133) |
| **LET-3D-AP** | 2022 | Longitudinal error tolerant 3D detection metric | [2206.07705](https://arxiv.org/abs/2206.07705) |
| **MotionLM** | 2023 | Motion forecasting as language modeling (discrete tokens, autoregressive) | [2309.16534](https://arxiv.org/abs/2309.16534) |
| **SceneDiffuser** | 2024 | Diffusion-based scene initialization + rollout for traffic simulation | Waymo Research |
| **3D OV Panoptic Seg** (Hung et al.) | 2024 | Open-vocabulary 3D panoptic segmentation for driving | [2401.02402](https://arxiv.org/abs/2401.02402) |
| **EMMA** | 2024 | End-to-end multimodal model: Gemini backbone, text-based output | [2410.23262](https://arxiv.org/abs/2410.23262) |
| **WOMD-Reasoning** | 2024 | 3M Q&A pairs for map recognition, motion narratives, interaction reasoning | Waymo Open Dataset |
| **S4-Driver** | 2025 | Self-supervised E2E driving MLLM; no human annotations; sparse volume 3D lifting | [2505.24139](https://arxiv.org/abs/2505.24139) |
| **Scaling Laws for Driving** | 2025 | First empirical scaling laws for joint motion forecasting and planning | [2506.08228](https://arxiv.org/abs/2506.08228) |
| **WOD-E2E** | 2025 | Long-tail E2E driving benchmark with Rater Feedback Score metric | [2510.26125](https://arxiv.org/abs/2510.26125) |
| **Waymo Foundation Model** | 2025 | "Think Fast / Think Slow" dual-system production architecture with Driving VLM | Waymo Blog |
| **Waymo World Model** | 2026 | Genie 3-based photorealistic 3D simulation for rare event testing | Waymo Blog |

### Wei-Chih Hung's Research Trajectory at Waymo

Wei-Chih Hung's work traces a clear path toward EMMA:
1. **Semi-supervised segmentation** (BMVC 2018) -- learning from limited labels
2. **SCOPS: Self-supervised co-part segmentation** (CVPR 2019) -- unsupervised part discovery
3. **LET-3D-AP metrics** (2022) -- improving 3D detection evaluation for AV
4. **3D Open-Vocabulary Panoptic Segmentation** (ECCV 2024) -- open-vocab understanding using VLMs
5. **EMMA** (2024) -- unifying perception + planning via VLMs

The through-line is: *using large pre-trained models (CLIP, Gemini) to improve generalization in autonomous driving perception and planning, especially for open-world / long-tail scenarios.*

### Waymo Foundation Model: Demonstrably Safe AI (Blog, December 2025)

- **Source:** Waymo Blog, December 2025
- **Contribution:** Reveals Waymo's production architecture built around a unified foundation model with a "Think Fast / Think Slow" dual-system design inspired by dual-process theory:
  1. **Sensor Fusion Encoder (System 1 — "Think Fast"):** Fuses camera + LiDAR + radar for fast, reactive perception and responses. Handles time-critical situations requiring immediate action.
  2. **Driving VLM (System 2 — "Think Slow"):** Camera-based vision-language model, fine-tuned from Gemini, for complex semantic reasoning about novel or ambiguous situations (e.g., understanding construction zone signage, unusual road configurations).
  3. **World Decoder:** Predicts agent behaviors, generates HD maps, creates candidate trajectories, and validates outputs for safety. Serves as the shared output stage.
- **Unified Foundation:** The same foundation model architecture powers three key applications:
  - **Driver:** On-vehicle autonomous driving
  - **Simulator:** Generating realistic scenarios for testing
  - **Critic:** Evaluating and scoring driving performance for continuous improvement
- **Deployment Strategy:**
  - Teacher-to-student distillation compresses the large foundation model for on-vehicle deployment
  - Inner loop: RL-based optimization in simulation to improve driving policy
  - Outer loop: Real-world Critic feedback from actual driving to identify failure modes and update training
- **Significance:** Shows how EMMA's ideas (VLM for driving) evolved into Waymo's production system. The dual-system architecture is the practical resolution of the "VLM latency problem" — the VLM handles complex reasoning while the sensor fusion encoder handles time-critical reactions. This is the clearest public statement of Waymo's production AI architecture.

---

## 8. Open Problems and Trends

### Active Research Frontiers

| Problem | Current State | Key Challenge |
|---------|--------------|---------------|
| **Closed-loop evaluation** | nuPlan provides first real ML benchmark; CARLA widely used but unrealistic | Real-world closed-loop testing is expensive; sim-to-real gap remains large |
| **Scalability** | EMMA uses Gemini Nano (smallest); Tesla uses 35K+ H100s | How to scale VLM-based planners to real-time on vehicle hardware? |
| **Sim-to-real transfer** | World models (GAIA-2, Waymo World Model) generate photorealistic scenarios | Generated scenarios may not cover the true distribution of rare events |
| **Multi-sensor fusion in VLMs** | EMMA is camera-only; cannot integrate LiDAR | VLM architectures not designed for 3D point cloud inputs |
| **Consistency guarantees** | No current method guarantees consistent perception + planning outputs | Formal verification of neural networks remains intractable at scale |
| **Regulatory frameworks** | EU AI Act, NHTSA guidelines emerging | How to certify a system whose reasoning is a neural network? |
| **Long-tail scenarios** | WOD-E2E dataset targets <0.03% frequency events | Requires either massive data or effective simulation of rare events |
| **Model distillation** | Active research area | Compress large VLMs to deploy on vehicle hardware without losing capability |

### Emerging Trends (2025-2026)

1. **VLA (Vision-Language-Action) unification:** Two paradigms crystallizing -- (a) End-to-End VLA integrating everything in one model, (b) Dual-System VLA with slow VLM reasoning + fast reactive controller. EMMA exemplifies (a); DriveVLM-Dual exemplifies (b).

2. **World models for simulation:** Waymo's Genie 3-based world model and Wayve's GAIA-2 can generate photorealistic, interactive driving scenarios including rare events (tornados, animals). These could transform closed-loop evaluation.

3. **Instruction-following driving:** LMDrive, DriveLM show that natural language can serve as the interface between human intent and vehicle behavior. This has implications for ride-hailing UX.

4. **Tokenization of everything:** MotionLM showed trajectories can be discretized into tokens; EMMA showed all outputs can be text. The trend is toward universal tokenization of driving primitives.

5. **Scaling laws for driving:** Does more pre-training data + larger models reliably improve driving performance? EMMA's use of Gemini Nano suggests Waymo is exploring this axis; EMMA+ (with more data) shows consistent gains.

---

## Key Concepts & Terminology

| Term | Definition |
|------|-----------|
| **BEV (Bird's Eye View)** | Top-down 2D representation of the 3D scene, commonly used as the intermediate representation in E2E driving models |
| **End-to-End (E2E)** | Systems that learn the full pipeline from raw sensors to control outputs, without hand-designed intermediate modules |
| **VLA (Vision-Language-Action)** | Models that unify visual perception, language reasoning, and action generation |
| **Chain-of-Thought (CoT)** | Technique where the model generates intermediate reasoning steps before producing a final answer |
| **Open-loop evaluation** | Testing model outputs against recorded ground truth without simulating the effect of the model's actions on the environment |
| **Closed-loop evaluation** | Testing where the model's actions affect the simulated environment, enabling interaction with other agents |
| **Imitation Learning (IL)** | Learning a policy by mimicking expert demonstrations (e.g., human driving) |
| **Behavioral Cloning (BC)** | Simplest form of IL: supervised learning on (state, action) pairs from expert data |
| **Causal confusion** | When the model learns spurious correlations (e.g., brake lights -> slow down) instead of true causal relationships |
| **Distribution shift** | Gap between training data distribution and deployment distribution, particularly problematic for IL |
| **Motion tokens** | Discrete representation of continuous trajectory segments, used in MotionLM and related work |
| **Anchor trajectories** | Pre-defined trajectory templates used to initialize multi-modal prediction (MultiPath) |
| **nuScenes** | Large-scale AV dataset from Motional with 1000 scenes, widely used for perception and planning benchmarks |
| **WOMD** | Waymo Open Motion Dataset, focused on behavior prediction with 100K+ scenes |
| **nuPlan** | Closed-loop planning benchmark with 1500 hours of driving data from 4 cities |
| **Dual-system architecture** | Inspired by dual-process theory: fast reactive system + slow deliberative system operating in parallel |

---

## Recommended Reading Order

For maximum understanding, read in this sequence:

### Phase 1: Foundations (start here)
1. **ChauffeurNet** (2018) -- understand IL for driving and its limitations
   [arXiv:1812.03079](https://arxiv.org/abs/1812.03079)

2. **MultiPath** (2019) -- anchor-based multi-modal prediction
   [arXiv:1910.05449](https://arxiv.org/abs/1910.05449)

### Phase 2: End-to-End Revolution
3. **UniAD** (2023) -- the CVPR Best Paper that defined E2E driving
   [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)

4. **VAD** (2023) -- vectorized alternative, more efficient
   [arXiv:2303.12077](https://arxiv.org/abs/2303.12077)

### Phase 3: Language Meets Driving
5. **MotionLM** (2023) -- Waymo's bridge from prediction to language modeling
   [arXiv:2309.16534](https://arxiv.org/abs/2309.16534)

6. **GPT-Driver** (2023) -- first LLM-as-planner proof-of-concept
   [arXiv:2310.01415](https://arxiv.org/abs/2310.01415)

### Phase 4: VLM-Based Driving Systems
7. **DriveVLM** (2024) -- CoT reasoning for driving + practical dual-system design
   [arXiv:2402.12289](https://arxiv.org/abs/2402.12289)

8. **DriveLM** (2024) -- structured Graph VQA for driving reasoning
   [arXiv:2312.14150](https://arxiv.org/abs/2312.14150)

9. **LMDrive** (2024) -- instruction-following closed-loop driving
   [arXiv:2312.07488](https://arxiv.org/abs/2312.07488)

### Phase 5: EMMA and Beyond (read most carefully)
10. **EMMA** (2024) -- the paper your interviewer co-authored; know this cold
    [arXiv:2410.23262](https://arxiv.org/abs/2410.23262)

11. **S4-Driver** (2025) -- EMMA's successor direction; self-supervised E2E driving without annotations
    [arXiv:2505.24139](https://arxiv.org/abs/2505.24139)

12. **Scaling Laws for Driving** (2025) -- validates scaling for motion forecasting/planning; compute allocation guidance
    [arXiv:2506.08228](https://arxiv.org/abs/2506.08228)

13. **WOD-E2E** (2025) -- long-tail benchmark that addresses EMMA's evaluation limitations
    [arXiv:2510.26125](https://arxiv.org/abs/2510.26125)

14. **Waymo Foundation Model Blog** (2025) -- how EMMA's ideas evolved into Waymo's production architecture
    Waymo Blog, December 2025

15. **VLA4AD Survey** (2025) -- systematic overview of the field EMMA sits in
    [arXiv:2512.16760](https://arxiv.org/abs/2512.16760)

### Bonus: Wei-Chih Hung's Other Work
16. **3D Open-Vocabulary Panoptic Segmentation** (ECCV 2024)
    [arXiv:2401.02402](https://arxiv.org/abs/2401.02402)

---

## Interview Preparation Notes

### Questions You Should Be Ready to Discuss

1. **"How does EMMA compare to UniAD?"**
   UniAD uses specialized transformer modules with intermediate BEV representations; EMMA unifies everything through a pre-trained MLLM with text outputs. EMMA trades architectural inductive bias for pre-trained world knowledge. UniAD may be more data-efficient for driving-specific tasks; EMMA may generalize better to novel scenarios.

2. **"What are the limitations of representing trajectories as text?"**
   Precision loss from tokenization; no explicit geometric constraints; no guarantee of physically feasible trajectories; higher inference latency than direct regression. EMMA addresses precision with floating-point text representation but cannot enforce kinematic constraints.

3. **"How would you improve EMMA?"**
   Potential directions: longer temporal context (more than 4 frames); multi-sensor fusion (LiDAR integration); consistency losses between perception and planning outputs; model distillation for deployment; reinforcement learning fine-tuning for closed-loop improvement.

4. **"Why use Gemini Nano instead of a larger model?"**
   Likely latency constraints for real-time driving. An interesting research question is whether scaling to larger Gemini variants yields proportional gains, or whether driving-specific fine-tuning matters more than model size.

5. **"How do you evaluate E2E driving models fairly?"**
   Open-loop (L2 on nuScenes) is insufficient -- it cannot capture compounding errors or interaction effects. Closed-loop (CARLA, nuPlan) is better but sim-to-real gap is significant. Real-world closed-loop testing is gold standard but expensive. WOMD Sim Agents challenge is Waymo's attempt at scalable closed-loop eval.

---

## Related YouTube Videos

| Topic | Video | Channel | Link |
|-------|-------|---------|------|
| E2E AD Tutorial | End-to-end Autonomous Driving: Past, Current and Onwards | OpenDriveLab | https://youtu.be/Z4n1vlAYqRw |
| E2E AD Misconceptions | Common Misconceptions in Autonomous Driving (Andreas Geiger) | WAD at CVPR | https://www.youtube.com/watch?v=x_42Fji1Z2M |
| DriveVLM | DriveVLM Demo Video | MARS Lab | https://www.youtube.com/watch?v=mt-SdHTTZzA |
| Motion Planning | Autonomous Driving: The Way Forward (Vladlen Koltun) | WAD at CVPR | https://youtu.be/rj7A2OP7KO4 |
| Motion Forecasting | Boris Ivanovic — CVPR 2025 OpenDriveLab Tutorial | OpenDriveLab | https://youtu.be/EWfdgvSd5b0 |
| Tesla FSD | AI for Full Self-Driving (Andrej Karpathy, CVPR 2021) | WAD at CVPR | https://www.youtube.com/watch?v=g6bOwQdCJrc |
| Tesla FSD | Foundation Models for Autonomy (Ashok Elluswamy, CVPR 2023) | WAD at CVPR | https://www.youtube.com/watch?v=6x-Xb_uT7ts |
| Imitation Learning | Feedback in Imitation Learning (ICML 2020 Workshop) | ICML | https://www.youtube.com/watch?v=4VAwdCIBTG8 |
| Raquel Urtasun Keynote | Self-Driving Keynote (CVPR 2021 WAD) | WAD at CVPR | https://youtu.be/PSZ2Px9PrHg |

---

*Survey compiled from web research. All paper details verified through arXiv and published sources. Where exact details could not be confirmed, this is noted explicitly.*
