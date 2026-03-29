---
title: "Autonomous Systems: Self-Driving Cars"
category: engineering
excerpt: "First-principles guide to self-driving — AV stack, HD maps, motion prediction, planning, end-to-end driving, simulation, safety, and SAE levels."
tags: autonomous-driving self-driving tech-interview
comments: true
---

# Self-Driving Cars: A First-Principles Guide

*Target audience: ML engineer preparing for autonomous driving interviews (e.g., Waymo)*
*Last updated: March 2026*

---

## Table of Contents

1. [The Autonomous Driving Stack](#1-the-autonomous-driving-stack)
2. [HD Maps vs Mapless Driving](#2-hd-maps-vs-mapless-driving)
3. [Motion Prediction](#3-motion-prediction)
4. [Planning](#4-planning)
5. [End-to-End Driving](#5-end-to-end-driving)
6. [Simulation and Evaluation](#6-simulation-and-evaluation)
7. [Safety](#7-safety)
8. [Levels of Autonomy (SAE J3016)](#8-levels-of-autonomy-sae-j3016)

---

## 1. The Autonomous Driving Stack

An autonomous vehicle must answer four questions in sequence, tens of times per second:

1. **What is around me?** (Perception)
2. **What will happen next?** (Prediction)
3. **What should I do?** (Planning)
4. **How do I execute it?** (Control)

Each question maps to a module. Together they form the **autonomous driving stack** — the software architecture that converts raw sensor data into physical vehicle motion.

### 1.1 The Four Modules

**Perception** takes raw sensor inputs — camera images, LiDAR point clouds (a rotating laser scanner that measures distances to surrounding surfaces, producing a 3D "cloud" of points), and radar returns — and extracts a structured understanding of the scene. This includes:

- **3D object detection**: locating vehicles, pedestrians, and cyclists as 3D bounding boxes with position, size, heading, and velocity
- **Semantic segmentation**: classifying every pixel or point into categories (road, sidewalk, vegetation, building)
- **Lane and road graph estimation**: identifying lane boundaries, crosswalks, traffic signs, and signal states

The output is a structured scene representation: a list of detected objects with their properties, a map of drivable surface, and the state of traffic controls.

**Prediction** takes the perception output — detected objects and their observed trajectories over the last few seconds — and forecasts where each agent will be in the future (typically 3–8 seconds ahead). A car approaching an intersection might go straight, turn left, or turn right, so prediction must be **multi-modal**: it outputs multiple possible future trajectories with associated probabilities, not a single deterministic path.

**Planning** takes the predicted futures of all agents and decides what the ego vehicle (the self-driving car itself) should do. It produces a **trajectory**: a sequence of future positions, headings, and velocities for the ego vehicle over the next few seconds. The trajectory must balance competing objectives — safety (avoid collisions), comfort (smooth acceleration and steering), progress (make forward progress toward the destination), and legality (obey traffic rules).

**Control** converts the planned trajectory into physical actuator commands: steering angle, throttle position, and brake pressure. A common approach is a **PID controller** (proportional-integral-derivative — a feedback controller that adjusts outputs based on the error between desired and actual state) that tracks the planned trajectory by correcting for deviations in real time.

### 1.2 Information Flow

The pipeline flows in one direction:

![The classical modular AV stack](/assets/img/blog/autonomous-systems/av_stack.svg)

```
Sensors (cameras, LiDAR, radar)
    → Perception (structured scene: objects, lanes, signals)
        → Prediction (future trajectories of all agents)
            → Planning (ego trajectory)
                → Control (steering, throttle, brake)
```

Each module consumes the output of the previous one and produces a more abstract, decision-relevant representation. Raw camera pixels (millions of values) are compressed into a handful of detected objects; those objects' futures are distilled into trajectory distributions; and the optimal ego response is reduced to a single trajectory, which is finally realized as three scalar actuator commands.

### 1.3 Concrete Walkthrough: "A Pedestrian Steps Off the Curb"

Trace through the stack:

1. **Perception**: The camera detects a person at the road edge. The LiDAR confirms the 3D position at (15m ahead, 3m to the right). The object tracker assigns a consistent ID and estimates the pedestrian is moving laterally at 1.2 m/s toward the ego lane.

2. **Prediction**: Given the pedestrian's position, velocity, and the nearby crosswalk, the prediction module outputs two modes:
   - Mode A (70% probability): pedestrian continues crossing into the ego lane, reaching the ego's path in ~2 seconds
   - Mode B (30% probability): pedestrian stops at the lane edge and waits

3. **Planning**: The planner evaluates candidate ego trajectories against the predicted futures. Maintaining current speed would bring the ego vehicle to the pedestrian's predicted crossing point in 1.5 seconds — a collision under Mode A. The planner selects a trajectory that decelerates from 30 km/h to 10 km/h, yielding to the pedestrian.

4. **Control**: The controller converts the deceleration trajectory into brake pressure commands, smoothly reducing speed.

### 1.4 Modular vs End-to-End

![Modular vs End-to-End architectures](/assets/img/blog/autonomous-systems/modular_vs_e2e.svg)

The stack described above is the **modular** approach: each module is developed, tested, and debugged independently.

| Property | Modular | End-to-End |
|----------|---------|------------|
| **Interpretability** | High — can inspect each module's output (did perception miss the pedestrian? did prediction assign wrong probabilities?) | Low — a single neural network maps sensors to trajectories; internal reasoning is opaque |
| **Debuggability** | Can isolate failures to a specific module | Failures are diffuse; hard to attribute errors |
| **Information flow** | Each module's output is a **bottleneck** — downstream modules can only use the information explicitly passed to them. If perception discards a subtle cue (e.g., a pedestrian's gaze direction), prediction can never recover it | No bottleneck — the network can learn to preserve any task-relevant information from raw inputs |
| **Optimization** | Each module optimized for its own loss, not for final driving quality | Jointly optimized for the planning objective — the system learns what perception features matter for driving |

Two landmark papers represent these paradigms:

- **UniAD** (Hu et al., CVPR 2023 Best Paper) is a structured end-to-end model: it retains five cascaded modules (tracking, mapping, motion prediction, occupancy prediction, planning) but connects them with transformer queries and trains them jointly. It demonstrated that planning-oriented joint optimization beats independently optimized modules by large margins (-28% planning error on nuScenes). See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for a detailed breakdown.

- **EMMA** (Hwang, Hung et al., Waymo, 2024) takes unification further: it feeds raw camera images into Gemini (a multimodal large language model) and represents *all* outputs — trajectories, 3D detections, road graphs — as natural language text. There are no task-specific modules at all. See both the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) and the [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}) for deep dives on EMMA's architecture and results.

The field has not converged on one paradigm. Waymo revealed their production architecture in December 2025 with the **Waymo Foundation Model** blog post, which describes a "Think Fast / Think Slow" dual-system design:

- **Sensor Fusion Encoder (System 1 — "Think Fast")**: fuses camera + LiDAR + radar for fast, reactive perception and responses to time-critical situations
- **Driving VLM (System 2 — "Think Slow")**: a camera-based vision-language model fine-tuned from Gemini for complex semantic reasoning about novel or ambiguous situations (construction zone signage, unusual road configurations)
- **World Decoder**: predicts agent behaviors, generates HD maps, creates candidate trajectories, and validates outputs for safety

The same foundation model architecture powers driving, simulation, and evaluation (a "Critic" that scores driving performance). Teacher-to-student distillation compresses the large model for on-vehicle deployment. This dual-system architecture is the practical resolution of the VLM latency problem — the VLM handles complex reasoning while the sensor fusion encoder handles time-critical reactions.

---

## 2. HD Maps vs Mapless Driving

### 2.1 What is an HD Map?

An **HD map** (high-definition map) is a pre-built, centimeter-accurate digital representation of the road environment. Unlike consumer navigation maps (Google Maps, Waze), which store road connectivity and turn-by-turn directions, HD maps encode:

- **Lane geometry**: exact polyline positions of every lane boundary, lane center, and lane type (solid, dashed, double yellow)
- **Road topology**: which lanes connect to which, merge/split points, intersection structure
- **Traffic controls**: position and type of every traffic sign, signal, speed limit, stop line, crosswalk
- **Static infrastructure**: curbs, medians, guardrails, poles

HD maps are created by driving specialized **survey vehicles** equipped with high-precision LiDAR and GPS through every road to be mapped. The raw data is processed using **LiDAR SLAM** (Simultaneous Localization and Mapping — an algorithm that builds a map while simultaneously tracking the vehicle's position within it) and then manually annotated by human labelers who trace lane lines, mark signs, and verify topology.

Companies like **Waymo** and **Cruise** historically relied heavily on HD maps. When the self-driving car localizes itself within the HD map (using LiDAR scan matching), it already "knows" where the lanes are, what the speed limit is, and where the crosswalks are — even before perceiving anything. This dramatically simplifies perception and planning.

### 2.2 Why HD Maps Are Expensive

| Cost factor | Description |
|-------------|-------------|
| **Survey fleet** | Specialized vehicles with high-precision sensors must drive every road |
| **Manual annotation** | Human labelers trace lane lines and label signs; weeks of effort per city |
| **Constant updates** | Construction, new signs, re-striped lanes — maps go stale in weeks |
| **Geographic scalability** | Mapping a new city costs millions; Waymo operates in only 4 metro areas (San Francisco, Phoenix, Los Angeles, Austin) |

A frequently cited figure: mapping a single city for autonomous driving costs $5–10M and must be refreshed quarterly.

### 2.3 Online Map Construction

Rather than pre-building maps, recent research learns to **predict map elements from sensor data in real time**:

- **MapTR** (Liao et al., ICLR 2023) predicts vectorized map elements (lane boundaries, road edges, pedestrian crossings) as ordered point sequences directly from multi-camera images. It uses a transformer decoder with learned map queries, producing a structured map in a single forward pass (~25 FPS).
- **VectorMapNet** (Liu et al., ICML 2023) predicts vectorized map elements as polylines, including their semantic class and connectivity.

These methods produce maps that are "good enough" for planning in most driving scenarios, but lack the guaranteed accuracy of pre-built HD maps — particularly for fine details like lane widths or signal timing.

### 2.4 Mapless Driving

**Tesla's approach** is the most prominent example of mapless driving: no pre-built HD maps at all. The vehicle relies entirely on:

- Real-time perception to detect lanes, signs, and road boundaries from camera images
- Learned priors from billions of miles of driving data (the neural network has seen enough roads to have strong expectations about lane structure)
- Coarse navigation maps (OpenStreetMap-level) for routing only

The tradeoff:

| | HD Maps | Mapless |
|---|---------|---------|
| **Accuracy** | Centimeter-precise, verified | Perception-dependent, can fail in poor visibility |
| **Scalability** | Expensive per-city | Works anywhere there are roads |
| **Freshness** | Can be stale | Always reflects current conditions |
| **Failure mode** | Outdated map vs reality (e.g., moved construction zone) | Perception failure (e.g., missing lane lines in snow) |

### 2.5 The Trend

The industry is converging toward **lighter maps**: a coarse routing-level map (road connectivity, approximate speed limits) combined with real-time perception for fine-grained scene understanding. EMMA exemplifies this — it receives navigation instructions as text and perceives lane structure from cameras, without relying on pre-built HD maps. UniAD's MapFormer module similarly predicts online map elements from BEV features.

---

## 3. Motion Prediction

### 3.1 The Task

Given:
- **Observed trajectories**: the past T_obs seconds of each agent's position (typically 2–5 seconds of history)
- **Scene context**: road geometry (lanes, boundaries), traffic signals, and spatial relationships between agents

Predict:
- **Future trajectories**: K possible future paths for each agent over the next T_pred seconds (typically 3–8 seconds), each with an associated probability

The output is a set of trajectory hypotheses: {(p_k, τ_k)} for k = 1..K, where p_k is the probability of mode k and τ_k = {(x_t, y_t)}_{t=1}^{T_pred} is the trajectory.

### 3.2 Why Multi-Modal?

Consider a car stopped at a T-intersection. It could:
- Turn left (30% probable)
- Turn right (50% probable)
- Make a U-turn (5% probable)
- Remain stopped (15% probable)

A single "average" prediction (splitting the difference between left and right) would place the car in the middle of the intersection — a location it will almost certainly *not* occupy. Multi-modal prediction avoids this **mode averaging** problem by maintaining distinct hypotheses.

The planner must reason over all modes: if the car turns left, the ego vehicle must yield; if it turns right, the ego vehicle can proceed. Planning against the wrong single prediction could be dangerous.

### 3.3 Classical Approach: Social Forces

The **social forces model** (Helbing & Molnár, 1995) treats each agent as a particle subject to forces:

- **Goal attraction**: a force pulling the agent toward its destination (e.g., the end of the lane)
- **Repulsion from others**: a force pushing agents apart to avoid collisions, exponentially increasing as distance decreases
- **Repulsion from boundaries**: walls, curbs, and lane edges exert repulsive forces

The resulting trajectory is the solution to a system of differential equations:

$$m_i \frac{d\mathbf{v}_i}{dt} = \mathbf{F}_{\text{goal}} + \sum_{j \neq i} \mathbf{F}_{\text{repel}}(i,j) + \mathbf{F}_{\text{boundary}}$$

This is intuitive and interpretable, but limited: it cannot capture complex behaviors like yielding at an intersection, responding to traffic signals, or negotiating merges.

### 3.4 Graph Neural Networks for Agent Interaction

Modern prediction models agent interactions using **graph neural networks (GNNs)**: each agent is a node, and interactions are edges.

**LaneGCN** (Li et al., CVPR 2020) introduced a key architecture:
- **Actor-to-lane attention**: each agent attends to nearby lane segments to understand its spatial context
- **Lane-to-lane propagation**: lane graph connectivity is encoded through graph convolutions, propagating information along road topology
- **Actor-to-actor attention**: agents attend to each other to model interactions (e.g., following, yielding)

This graph structure naturally encodes that an agent's behavior depends on both the road it is on (lane geometry, traffic rules) and the agents it is interacting with.

### 3.5 Trajectory Forecasting Approaches

Three major paradigms have emerged:

#### Anchor-Based: MultiPath (Waymo, 2019)

**MultiPath** (Chai et al., CoRL 2019) pre-computes a fixed set of K **anchor trajectories** by clustering trajectories from training data (e.g., K = 64 clusters capturing patterns like "go straight," "turn left," "lane change right"). For each agent, the model predicts:
1. A probability distribution over the K anchors (which motion mode?)
2. Per-anchor residual offsets that adjust the anchor to the specific situation

The output is a **Gaussian Mixture Model (GMM)**: K Gaussian components, each centered on an anchor + residual, weighted by mode probabilities. This decomposition of prediction into *mode selection* + *residual regression* proved highly effective and influenced Waymo's production prediction system. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for details on MultiPath and its successor MultiPath++.

#### Autoregressive: MotionLM (Waymo, ICCV 2023)

**MotionLM** (Seff et al., 2023) reframes prediction as **language modeling**: continuous trajectories are discretized into a vocabulary of motion tokens (short trajectory segments), and a transformer decoder autoregressively generates the sequence of tokens that composes each agent's future trajectory.

Key insight: a single autoregressive pass can jointly model *multiple agents* by interleaving their tokens in a shared sequence. This naturally captures interactions — Agent A's next step depends on Agent B's last step. MotionLM achieved #1 on the WOMD interaction prediction benchmark. It is also a direct precursor to EMMA's philosophy of "everything as language." See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for the detailed MotionLM summary.

#### Scaling Laws for Driving (Waymo, 2025)

**Scaling Laws of Motion Forecasting and Planning** (Baniodeh, Goel et al., Waymo, 2025) provides the first empirical scaling laws study for joint motion forecasting and planning, using the MotionLM architecture on ~500K hours of driving data:

- **Power-law scaling**: cross-entropy loss follows power-law scaling with compute, mirroring LLM scaling behavior (Chinchilla-style laws apply to driving)
- **Optimal compute allocation**: model size should grow 1.5x faster than dataset size — at the same compute budget, the optimal driving model is ~50x smaller than the optimal LLM, meaning driving needs proportionally more data than language
- **Closed-loop correlation**: closed-loop driving metrics (safety, comfort) correlate with pre-training loss — bigger models produce safer drivers
- **Cross-agent transfer**: training on other agents' driving logs improves ego-agent planning

This validates that the autoregressive approach to driving scales predictably with compute and data, providing principled guidance for resource allocation. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for the complete analysis.

#### Diffusion-Based

**Diffusion models** generate diverse trajectory predictions via iterative denoising: start with random noise, then progressively denoise it into a plausible trajectory. The stochastic generation process naturally produces diverse, multi-modal outputs. Methods like MotionDiffuser (Jiang et al., 2023) apply this to multi-agent prediction, producing high-quality diverse samples but at higher computational cost than feed-forward approaches.

### 3.6 Metrics

| Metric | Definition | Intuition |
|--------|-----------|-----------|
| **minADE_K** | Minimum Average Displacement Error over K predictions: for each ground-truth trajectory, find the closest of the K predicted trajectories and compute the average L2 error over all timesteps | "How close is the best prediction to reality, on average?" |
| **minFDE_K** | Minimum Final Displacement Error: same as minADE but only at the final timestep | "How close is the best prediction's endpoint?" |
| **Miss Rate** | Fraction of scenarios where no prediction is within a threshold of the ground truth at the final timestep | "How often do we completely miss the agent's future location?" |
| **mAP** | Mean Average Precision: evaluates both the spatial accuracy and the probability calibration of multi-modal predictions | Used by Waymo's motion prediction benchmark |

minADE_K with K = 6 is the most commonly reported metric. Lower is better for all metrics.

---

## 4. Planning

Planning answers: given what I perceive and predict, what trajectory should the ego vehicle follow? This is the core decision-making problem in autonomous driving.

### 4.1 Rule-Based Planners

The simplest approach: a **state machine** with hand-coded rules.

```
IF gap_to_leading_vehicle > safe_following_distance:
    maintain_speed()
ELIF gap_to_leading_vehicle > emergency_distance:
    decelerate(rate=comfortable)
ELSE:
    emergency_brake()
```

State machines encode driving behavior as transitions between discrete states (cruising, following, lane changing, stopping) with explicit conditions for each transition. Tesla's FSD before v12 was reportedly implemented as ~300,000 lines of C++ encoding such rules.

**Strengths**: predictable, verifiable, easy to debug specific failures.
**Weaknesses**: brittle in complex scenarios (unprotected left turns with multiple interacting agents), difficult to scale (exponential growth in rules for edge cases), cannot learn from data.

### 4.2 Optimization-Based Planning

Define a **cost function** J(τ) over candidate ego trajectories τ, and find the trajectory that minimizes cost subject to constraints:

$$\tau^* = \arg\min_{\tau} J(\tau) \quad \text{s.t.} \quad \tau \text{ satisfies kinematic constraints}$$

The cost function typically includes:

| Term | What it penalizes | Example |
|------|-------------------|---------|
| **Collision risk** | Proximity to predicted agent trajectories | $\sum_i \exp(-d(\tau, \hat{\tau}_i) / \sigma)$ |
| **Comfort** | Jerk (rate of change of acceleration), lateral acceleration | $\int \|\dddot{\mathbf{x}}(t)\|^2 dt$ |
| **Progress** | Deviation from desired speed, time to reach destination | $\sum_t (v_t - v_{\text{target}})^2$ |
| **Rule compliance** | Lane departure, speed limit violation, signal violation | Penalty for trajectory points outside lane boundaries |

**Kinematic constraints** ensure the trajectory is physically feasible: maximum steering angle, maximum acceleration/deceleration, and the bicycle model (a simplified model of car kinematics where the car has a front and rear axle connected by a rigid body with a fixed wheelbase).

This is the classical approach used in production by most AV companies. The core challenge is **cost function design**: balancing competing objectives (safety vs. progress, comfort vs. responsiveness) requires extensive tuning, and the right tradeoff changes with context (school zone vs. highway).

### 4.3 Learning-Based Planning

#### Imitation Learning: ChauffeurNet (Waymo, RSS 2019)

**ChauffeurNet** (Bansal et al., RSS 2019) learns a driving policy from human demonstrations. The scene is rendered as a top-down BEV image (road map, traffic lights, detected objects, past trajectories), and a CNN predicts a future ego trajectory.

The key innovation was **synthesized perturbations**: pure behavioral cloning (supervised learning on expert state-action pairs) suffers from **distribution shift** — the model only sees states along the expert's trajectory during training, so it doesn't know how to recover from mistakes. ChauffeurNet addresses this by artificially perturbing the ego position during training and requiring the model to recover back to the expert path. This reduced collisions by ~60%. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for the full ChauffeurNet analysis.

#### End-to-End Learned: UniAD, VAD

**UniAD** and **VAD** (Jiang et al., ICCV 2023) jointly optimize perception, prediction, and planning in a single network. The planner benefits from features learned for perception — for example, a subtle visual cue (a turning signal, a pedestrian's body orientation) that a modular pipeline might discard at the perception-planning interface can flow directly to the planner.

**VAD** introduced an important efficiency idea: representing the scene as **vectorized** polylines (lane boundaries as point sequences, agent trajectories as coordinate lists) rather than dense BEV grids. This is more memory-efficient and preserves instance-level structure. VAD achieved 2.5x faster inference than UniAD with lower planning error. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for detailed VAD results.

#### VLM-Based: EMMA, DriveVLM

The newest paradigm uses pre-trained **Vision-Language Models (VLMs)** for planning:

- **EMMA** feeds camera images into Gemini and outputs trajectory waypoints as text. Chain-of-thought reasoning (scene description → critical objects → behavior predictions → meta-decision) improves planning by 6.7%.
- **DriveVLM** (Tian et al., 2024) uses a VLM for CoT reasoning and hierarchical planning, with a practical **dual-system** variant (DriveVLM-Dual) that runs a slow VLM for strategic reasoning alongside a fast classical planner for reactive control.

See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for detailed coverage of GPT-Driver, DriveVLM, LMDrive, and DriveLM.

### 4.4 Classical vs Learned Planning Comparison

| | Classical (Optimization) | Imitation Learning | End-to-End | VLM-Based |
|---|---|---|---|---|
| **Strengths** | Interpretable, safety guarantees, handles constraints explicitly | Learns complex behaviors from data; no reward engineering | Jointly optimized; no information bottleneck | World knowledge transfer; interpretable reasoning; instruction following |
| **Weaknesses** | Hand-designed cost functions; struggles with complex multi-agent interactions | Distribution shift; causal confusion; struggles with rare events | Black-box; harder to verify safety; requires massive data | High latency; limited spatial precision; hallucination risk |
| **Production use** | Waymo, Cruise (core planner) | ChauffeurNet (historical) | Tesla FSD v12+ | DriveVLM-Dual (BYD) |

*This table is adapted from the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}), which provides a more detailed comparison.*

### 4.5 Cost Function Design

What makes a good trajectory? The planner must balance:

- **Safety**: the trajectory must not collide with any agent under any predicted future mode. In practice, safety is often a hard constraint rather than a soft cost term — the planner rejects any trajectory that violates a minimum time-to-collision threshold.
- **Comfort**: passengers should not experience sudden acceleration, hard braking, or sharp turns. This is quantified as jerk (the derivative of acceleration) and lateral acceleration.
- **Progress**: the vehicle should make forward progress toward the destination at a reasonable speed. Standing still is safe and comfortable but useless.
- **Rule compliance**: stay in lane, obey speed limits, stop at red lights, yield to pedestrians.

Balancing these objectives is the core challenge of planning. A perfectly safe planner that never moves is useless; an aggressive planner that maximizes progress is dangerous. Production systems use carefully tuned weights, often with context-dependent adjustments (e.g., lower speed targets in school zones, tighter following distances on highways).

---

## 5. End-to-End Driving

### 5.1 Definition

**End-to-end driving** maps raw sensor inputs directly to driving outputs (trajectories or control commands) using a single learned model, without hand-designed intermediate representations. The model implicitly learns its own internal representations for perception and prediction as a byproduct of being trained to produce good driving trajectories.

In practice, "end-to-end" spans a spectrum:

- **Structured end-to-end** (UniAD): retains explicit intermediate modules (detection, tracking, mapping, prediction) but trains them jointly. Information flows through defined interfaces, but all modules are optimized for the final planning loss.
- **Monolithic end-to-end** (EMMA, Tesla FSD v12): a single model with no explicit intermediate modules. The model may develop internal representations analogous to detection or prediction, but these are not architecturally enforced.

### 5.2 UniAD Architecture

**UniAD** (Hu et al., CVPR 2023 Best Paper) chains five transformer decoder modules:

```
Multi-camera images
  → BEV Feature Extractor (lift 2D features to bird's-eye view)
    → TrackFormer (detect and track objects using track queries)
      → MapFormer (predict vectorized map elements)
        → MotionFormer (predict future trajectories of tracked agents)
          → OccFormer (predict dense occupancy of the scene)
            → Planner (select ego trajectory)
```

The key architectural idea is **query-based information passing**: each module maintains a set of learned queries (e.g., TrackFormer has one query per tracked object). These queries are passed downstream — MotionFormer receives the track queries and produces motion-augmented queries, which OccFormer and the Planner consume. This allows end-to-end gradient flow while maintaining interpretable intermediate outputs.

Results on nuScenes: +20% tracking accuracy, +30% mapping accuracy, -38% motion forecasting error, -28% planning error vs prior independently-optimized modules. The [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}) and [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) both contain detailed UniAD summaries.

### 5.3 EMMA Architecture

![EMMA architecture](/assets/img/blog/autonomous-systems/emma_architecture.svg)

**EMMA** (Hwang, Hung et al., Waymo, 2024) takes a radically different approach:

```
Multi-camera images + text prompt
  → Gemini MLLM (pre-trained multimodal large language model)
    → Text output: trajectories, 3D detections, road graphs, reasoning chains
```

Everything is represented as natural language. A trajectory is output as a sequence of floating-point coordinate pairs in text: "(2.31, 0.05), (4.62, 0.08), ...". 3D bounding boxes, lane boundaries, and even the model's reasoning are all text strings.

**Chain-of-thought reasoning** is a standout feature. Before outputting a trajectory, EMMA generates a structured reasoning chain:
- R1: Scene description (weather, road conditions)
- R2: Critical objects with 3D coordinates
- R3: Behavior predictions for those objects
- R4: Meta driving decision (one of 12 action categories: e.g., "decelerate for pedestrian")

This CoT improves planning L2 error by 6.7% and provides an interpretability mechanism: engineers can inspect *why* the model made a decision. EMMA achieved SOTA on nuScenes planning (0.32m average L2, vs 0.39m for the prior best). See both survey files for extensive EMMA analysis.

### 5.4 S4-Driver: Self-Supervised End-to-End Driving

**S4-Driver** (Xie, Xu et al., Waymo/UC Berkeley, CVPR 2025) extends EMMA's direction by removing the need for human annotations entirely. Built on PaLI-3 5B, it is the first self-supervised end-to-end driving MLLM:

- **Sparse volume strategy**: lifts 2D MLLM visual features into 3D space without fine-tuning the vision encoder, enabling spatio-temporal reasoning from camera images alone
- **Hierarchical planning with free CoT**: derives meta-decisions (keep stationary, keep speed, accelerate, decelerate) automatically from driving logs — no human annotation needed
- **Multi-decoding aggregation**: uses nucleus sampling to generate diverse trajectory candidates, then aggregates them for robust planning
- **Self-supervised training**: all supervision signals come from raw driving logs (ego trajectories, vehicle dynamics) rather than human-labeled perception annotations

Results: SOTA on nuScenes planning (Avg L2: 0.31m self-supervised, vs 0.37m for VAD supervised). On WOMD-Planning-ADE (103K scenarios): ADE@5s 0.655. A self-supervised model beating supervised methods demonstrates that expensive human labels are not necessary for competitive E2E driving. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for the full S4-Driver analysis.

### 5.5 VAD: Vectorized Autonomous Driving

**VAD** (Jiang et al., ICCV 2023) sits between UniAD and EMMA in design philosophy. Its key contribution is replacing dense BEV grid representations with **vectorized** representations:

- Map elements are polylines: ordered sequences of 2D points defining lane boundaries and road edges
- Agent motions are trajectory sequences
- The ego trajectory is optimized via attention to these vectorized elements

This is more memory-efficient than dense grids and preserves instance-level structure. **VADv2** extended this to probabilistic planning with 4096 discrete trajectory tokens, achieving SOTA closed-loop performance on CARLA. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for the complete VAD/VADv2 breakdown.

### 5.6 Tesla FSD v12–v13

Tesla deployed the first production end-to-end driving system in 2024:

- **Input**: 8 cameras (no LiDAR; Tesla initially removed radar from some vehicles in 2021 but reintroduced high-resolution radar with the HW4 platform)
- **Output**: direct control commands (steering, acceleration, braking)
- **Training**: billions of miles of human driving data, trained on 35,000+ H100 GPUs
- **Scale**: over 8.3 billion FSD miles driven by February 2026
- **Impact**: replaced ~300,000 lines of C++ (the previous rule-based stack) with neural networks

Details remain proprietary, but Tesla has described using **occupancy networks** (predicting which 3D voxels in the scene are occupied, similar to the occupancy prediction work covered in the [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}) — see TPVFormer and SurroundOcc) and learned lane detection without HD maps.

### 5.7 The Core Insight

End-to-end models avoid **information bottlenecks**: in a modular pipeline, if perception represents each car as a bounding box (position, size, heading, velocity), the predictor never sees raw appearance features that might indicate intent (a driver looking over their shoulder before a lane change, a car's wheels starting to turn). End-to-end models can learn to preserve such cues.

But they sacrifice **interpretability**: when an end-to-end model makes a bad decision, it is difficult to determine whether the failure was in perception (didn't detect the car), prediction (predicted it would stay in lane), or planning (saw the risk but chose a bad trajectory). EMMA's chain-of-thought reasoning is an attempt to recover interpretability within an end-to-end architecture — structured reasoning stages make the failure point inspectable.

---

## 6. Simulation and Evaluation

### 6.1 Why Simulation Matters

A self-driving car must handle events that occur once in millions of miles: a child chasing a ball into the road, a tire blowout on the car ahead, an ambulance running a red light. Testing on public roads alone cannot encounter these events frequently enough to validate the system. **Simulation** allows engineers to generate millions of these scenarios, test the driving system's response, and iterate rapidly.

### 6.2 CARLA

**CARLA** (Dosovitskiy et al., 2017) is the standard open-source driving simulator for academic research. It provides:

- Photorealistic rendering of urban environments with dynamic weather, lighting, and traffic
- Physics simulation for vehicle dynamics, collisions, and sensor models
- Multiple pre-built towns with varying complexity
- A Python API for scripting scenarios and collecting data

The most common benchmark is **Town05 Long**: 10 predefined routes in an unseen town, evaluated on driving score (a composite of route completion and infraction frequency). VADv2 holds SOTA here with a driving score of 64.3. CARLA's limitation is a persistent **sim-to-real gap**: the rendered environments, though visually appealing, don't fully capture the visual complexity and behavioral diversity of the real world.

### 6.3 nuPlan

**nuPlan** (Motional, 2021) is the first **closed-loop ML planning benchmark** built from real driving data:

- **Scale**: 1,500 hours of real driving data from 4 cities (Las Vegas, Boston, Pittsburgh, Singapore)
- **Scenarios**: 75 scenario types covering common and edge-case driving situations
- **Evaluation**: both open-loop (trajectory comparison) and closed-loop (reactive simulation where other agents respond to the ego)

nuPlan's closed-loop evaluation revealed that models performing well on open-loop metrics can fail catastrophically in closed-loop: small errors compound over time as the ego deviates from the recorded trajectory, encountering states the model has never seen.

GameFormer (Huang et al., NVIDIA, ICCV 2023) achieved top performance on the nuPlan closed-loop reactive benchmark with a game-theoretic approach to interactive prediction and planning. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for GameFormer and DTPP details.

### 6.4 Open-Loop vs Closed-Loop Evaluation

![Open-loop vs closed-loop evaluation](/assets/img/blog/autonomous-systems/open_vs_closed_loop.svg)

| | Open-Loop | Closed-Loop |
|---|-----------|------------|
| **How it works** | Compare predicted trajectory to recorded ground truth | Model's actions affect the simulated environment; other agents react |
| **Metric example** | L2 displacement error at 1/2/3 seconds | Driving score (route completion × safety) |
| **Strengths** | Simple, fast, deterministic | Captures compounding errors and interaction effects |
| **Weaknesses** | Doesn't capture how errors compound; a 0.5m error at t=1 may cause a 5m error at t=3 in reality | Requires a simulator with realistic agent behavior; computationally expensive |
| **Use case** | Quick model comparison, development iteration | Final validation, safety assessment |

The critical insight: **open-loop metrics can be misleading**. A model that always predicts "go straight at current speed" achieves reasonable open-loop L2 error (because most driving is straight-line cruising) but would crash at the first curve. Closed-loop evaluation forces the model to handle the consequences of its own actions.

### 6.5 Waymo Open Dataset (WOD)

One of the largest autonomous driving datasets, used by 36,000+ researchers worldwide:

- **Perception benchmark**: 3D object detection (LiDAR and camera), semantic segmentation, panoramic video panoptic segmentation (100K images from 5 cameras, 28 classes — see the [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}) for details on Waymo's panoramic video panoptic segmentation dataset)
- **Motion prediction benchmark (WOMD)**: 100K+ scenes with agent trajectories, enabling behavior prediction research. Includes the WOMD Sim Agents challenge for scalable closed-loop evaluation
- **WOMD-Reasoning**: 3M question-answer pairs for map recognition, motion narratives, and interaction reasoning — bridging the gap between trajectory data and language understanding
- **2025 challenges**: vision-based end-to-end driving, long-tail scenario handling, interaction prediction

### 6.6 WOD-E2E: Long-Tail End-to-End Driving Benchmark

**WOD-E2E** (Xu, Lin et al., Waymo, 2025) is a new benchmark specifically designed for evaluating end-to-end driving on long-tail scenarios — events occurring at <0.03% frequency. It contains 4,021 segments (~12 hours) from 8 cameras with 360-degree coverage, mined from 6.4M miles of driving data across 11 challenging scenario categories (construction zones, complex intersections, cut-ins, foreign object debris, and more).

WOD-E2E introduces the **Rater Feedback Score (RFS)**: expert raters score trajectory candidates (0–10) at critical moments. Unlike ADE/L2 which compare to a single ground truth, RFS captures multi-modal acceptability — multiple safe trajectories can score well even if they differ from the recorded ground truth. This addresses a fundamental limitation of standard metrics: ADE penalizes safe evasive maneuvers that diverge from what the human driver happened to do.

WOD-E2E has significantly higher rarity scores than nuScenes and WOMD across all percentiles, representing the field's shift from nominal-driving benchmarks to long-tail evaluation. Already used for the 2025 Waymo Open Dataset Challenge. See the [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}) for details.

### 6.7 World Models for Simulation

The newest frontier in simulation: **world models** that generate photorealistic driving scenarios from learned models rather than hand-built graphics engines.

- **GAIA-1** (Wayve, 2023): a generative world model that produces driving video by predicting future frames conditioned on past frames and an action (steering, acceleration). It can generate novel scenarios including weather changes and rare events. See the [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}) for context on world models and simulation.
- **Waymo's Genie 3-based world model** (2026): uses Google DeepMind's Genie 3 architecture to generate photorealistic 3D driving scenarios. Can simulate rare events (animals on road, unusual weather) that are difficult to encounter in real-world testing.

World models promise to solve the sim-to-real gap by generating data that is visually indistinguishable from real driving footage, while also enabling controllable generation of rare and dangerous scenarios.

---

## 7. Safety

Safety in autonomous driving operates at multiple levels: the hardware must not fail, the software must not have bugs, and the AI models must not make wrong decisions. Standards exist for each level.

### 7.1 Functional Safety: ISO 26262

**ISO 26262** is the international standard for functional safety of electrical/electronic systems in road vehicles. It addresses failures caused by hardware malfunctions or software bugs — for example, a sensor producing incorrect readings due to a hardware fault, or a software crash in the planning module.

The standard defines **ASIL levels** (Automotive Safety Integrity Level) from A (lowest) to D (highest), based on three factors:

| Factor | Description |
|--------|-------------|
| **Severity** | How bad is the potential harm? (minor injury → fatal) |
| **Exposure** | How likely is the driving situation? (rare → frequent) |
| **Controllability** | Can the driver (or system) prevent the harm? (easily → not at all) |

A steering system failure at highway speed has high severity, high exposure, and low controllability → **ASIL D** (most stringent). An infotainment glitch has low severity → **ASIL A** or no safety requirement.

Each ASIL level prescribes engineering requirements: redundant hardware, software testing coverage, design review processes, and documentation standards. Higher ASIL = more stringent requirements = more expensive to certify.

### 7.2 SOTIF: ISO 21448

**SOTIF** (Safety of the Intended Functionality) addresses a different class of failures: the system works exactly as designed, but its intended functionality is insufficient for safe operation. This is directly relevant to ML-based autonomous driving:

- A perception model correctly follows its training but fails to detect a dark-clothed pedestrian at night → the car hits the pedestrian
- The prediction model correctly predicts the most likely future but assigns insufficient probability to a rare but critical behavior → the planner doesn't account for it

ISO 26262 cannot address these failures because there is no hardware malfunction or software bug — the system behaves as designed, but the design is insufficient. SOTIF provides a framework for:

1. Identifying **triggering conditions** (situations that cause the intended functionality to fail, e.g., sun glare, unusual objects)
2. Analyzing the **functional insufficiencies** that lead to unsafe behavior
3. Validating that residual risk is acceptable through testing and analysis

### 7.3 Operational Design Domain (ODD)

The **Operational Design Domain** defines the specific conditions under which the autonomous driving system is designed to operate:

| ODD Parameter | Example Constraint |
|---------------|-------------------|
| **Geography** | Specific mapped cities (Waymo: SF, Phoenix, LA, Austin) |
| **Road type** | Surface streets only, or including highways |
| **Speed range** | Up to 45 mph (urban) or 65 mph (highway) |
| **Weather** | Clear, light rain, or also heavy rain and snow |
| **Time of day** | Daytime only, or 24/7 |
| **Connectivity** | Requires cellular connection for remote assistance |

Waymo's ODD is currently: specific pre-mapped areas in San Francisco, Phoenix, Los Angeles, and Austin. Within these areas, the system operates fully driverlessly (no safety driver) at urban speeds, in most weather conditions. Outside the ODD, the system is not designed to operate — this is a deliberate safety boundary.

The ODD concept is fundamental to the L4 autonomy approach: rather than building a system that works everywhere (L5, which does not exist), build one that works perfectly within a well-defined operating envelope and refuse to operate outside it.

### 7.4 Verification Challenges for Neural Networks

Classical software can be verified through code review, formal proofs, and exhaustive testing of defined input/output specifications. Neural networks resist these approaches:

- **Black-box reasoning**: a deep neural network's decision process involves millions of learned parameters; there is no human-readable "rule" that explains why it made a specific prediction
- **Infinite input space**: the space of possible camera images is effectively infinite; exhaustive testing is impossible
- **Adversarial vulnerability**: small, imperceptible perturbations to sensor inputs can cause dramatic changes in network outputs
- **Distribution shift**: the network may encounter driving scenarios absent from its training data

Approaches to mitigating these challenges:

- **VLM-based interpretability**: EMMA's chain-of-thought reasoning provides partial transparency — engineers can inspect the textual reasoning to identify failure modes. This does not constitute formal verification but enables more effective debugging than pure black-box models.
- **Redundancy**: Waymo's production system runs multiple sensor modalities (cameras + LiDAR + radar) and maintains classical safety checks (hard-coded collision avoidance, speed limiters) alongside learned components. If the neural planner proposes a dangerous trajectory, a rule-based safety layer can override it.
- **Statistical validation**: demonstrating safety through massive-scale testing — Waymo publishes safety reports comparing their crash and injury rates to human drivers across millions of autonomous miles.

### 7.5 Redundancy and Fallback

Production AV systems employ defense-in-depth:

1. **Sensor redundancy**: multiple sensor types (camera, LiDAR, radar) provide overlapping coverage. If one sensor fails or is occluded, others compensate. BEVFusion-style architectures (see [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %})) enable systematic multi-modal fusion.
2. **Computational redundancy**: dual compute platforms can take over if one fails.
3. **Algorithmic redundancy**: a learned planner may produce the primary trajectory, but a classical safety checker verifies it against hard constraints (minimum following distance, maximum deceleration, lane boundaries) before execution.
4. **Operational fallback**: if the system detects a condition it cannot handle (sensor failure, ODD violation), it executes a **minimal risk condition** — typically pulling over safely and stopping.

---

## 8. Levels of Autonomy (SAE J3016)

The **SAE J3016** standard defines six levels of driving automation, from no automation to full automation. These levels describe who — the human or the system — performs the driving task and who serves as the fallback.

![SAE levels of driving automation](/assets/img/blog/autonomous-systems/sae_levels.svg)

### 8.1 The Six Levels

| Level | Name | Who Drives | Who Monitors | Who Falls Back | Real-World Example |
|-------|------|-----------|--------------|---------------|-------------------|
| **L0** | No automation | Human | Human | Human | Basic car with no driver aids |
| **L1** | Driver assistance | Human + system (one function) | Human | Human | Adaptive cruise control OR lane keeping (one at a time) |
| **L2** | Partial automation | System (multiple functions) | **Human** | Human | Tesla Autopilot, GM SuperCruise — system steers AND accelerates/brakes, but human must monitor at all times |
| **L3** | Conditional automation | System | **System** | **Human** (on request) | Mercedes Drive Pilot — system drives on highways under 40 mph; can request human takeover with ~10 second warning |
| **L4** | High automation | System | System | **System** | **Waymo** — fully driverless in defined ODD; no human needed; system handles all fallback scenarios within its ODD |
| **L5** | Full automation | System | System | System | Does not exist — would operate in all conditions, all geographies, all weather |

### 8.2 The Critical Transitions

**L1 → L2**: The jump from automating one function (e.g., cruise control adjusts speed) to automating multiple functions simultaneously (e.g., the car steers and adjusts speed together). The human is still responsible for monitoring and intervening. This is where most consumer vehicles sit today.

**L2 → L3**: The fundamental shift. At L2, the human must continuously monitor the driving environment. At L3, the **system** monitors the environment, and the human is only required to take over when the system requests it (a "takeover request"). The human can theoretically look away from the road during L3 operation.

**L3 → L4**: At L3, the human must be ready to resume control within a few seconds of a takeover request. This creates the **handoff problem**: research shows humans are poor at re-engaging with a driving task after minutes or hours of passive monitoring. Reaction times of 10–15 seconds are common, during which the vehicle may travel 100+ meters. At L4, there is no expectation of human takeover — the system must handle all situations within its ODD, including achieving a minimal risk condition if it encounters something it cannot handle.

### 8.3 The L2 → L4 Gap

L3 is the most contentious level because it relies on a human fallback that may be unreliable:

- A human who has not been actively driving for 30 minutes may take 10+ seconds to understand the situation and resume control
- Ambiguity about who is responsible (human or system) creates legal and ethical complexity
- Mercedes Drive Pilot (the only certified L3 system as of 2026) is limited to highway driving under 40 mph, where the consequences of slow human re-engagement are lower

Waymo deliberately **skipped L3**: rather than building a system that requires unreliable human fallback, they built an L4 system that operates fully driverlessly within a constrained ODD. The bet is that a system reliable enough for L3 is almost reliable enough for L4, and the safety benefit of removing the human fallback problem outweighs the cost of a more restricted ODD.

### 8.4 Current Industry State (March 2026)

| Company | Level | Status | Approach |
|---------|-------|--------|----------|
| **Waymo** | L4 | Commercial robotaxi in SF, Phoenix, LA, Austin | Foundation Model: dual-system (Sensor Fusion Encoder + Driving VLM + World Decoder); S4-Driver, Scaling Laws research |
| **Cruise** (GM) | L4 | Operations paused since Oct 2023 incident | Was modular + HD maps |
| **Tesla** | L2 (marketed as "FSD") | Active, 8.3B+ miles | End-to-end neural net, camera-only, no HD maps |
| **Mercedes** | L3 | Certified Drive Pilot on German/US highways | Rule-based + LiDAR, limited to <40 mph highway |
| **Baidu Apollo** | L4 | Commercial robotaxi in 10+ Chinese cities | Modular stack, HD maps, partnerships with automakers |
| **Pony.ai** | L4 | Robotaxi in China, public IPO (Nov 2024) | Multi-sensor, pre-mapped areas |
| **Zoox** (Amazon) | L4 | Testing purpose-built robotaxi (bidirectional vehicle) | Custom vehicle design, urban L4 |
| **Mobileye** | L2+ / L4 dev | SuperVision (L2+) shipping in production vehicles | Camera-first (EyeQ chips), REM crowd-sourced mapping |

The gap between L2 and L4 remains the defining divide: L2 systems (Tesla) scale geographically but require human supervision; L4 systems (Waymo) operate autonomously but in limited areas. No company has bridged this gap at scale.

---

## Summary: How It All Connects

The autonomous driving stack is an information pipeline: raw sensor data flows through increasingly abstract representations until it becomes a physical vehicle action. The traditional modular approach — perception → prediction → planning → control — provides interpretability and debuggability at the cost of information bottlenecks between modules.

The field is evolving along three axes simultaneously:

1. **Toward end-to-end**: UniAD showed joint optimization beats independent modules; EMMA showed a single VLM can replace the entire stack; S4-Driver showed annotation-free self-supervised training can match supervised methods; Tesla FSD v12 deployed end-to-end at production scale; and the Waymo Foundation Model reveals a dual-system architecture that resolves the VLM latency problem for production deployment. But interpretability and safety verification remain unsolved.

2. **Toward mapless driving**: HD maps are accurate but expensive and geographically limited. Online map construction (MapTR), learned lane prediction, and VLM-based scene understanding reduce dependence on pre-built maps.

3. **Toward simulation-driven development**: CARLA and nuPlan enable closed-loop evaluation; world models (GAIA-1, Waymo's Genie 3-based model) promise photorealistic simulation of rare events, potentially solving the long-tail testing problem.

The unifying trend is **learning from data at scale**: more data, larger models, and joint optimization are consistently improving driving quality. The open question is whether these learned systems can be made safe and verifiable enough for full deployment — a question that sits at the intersection of ML research, engineering, and regulation.

---

## Further Reading

This guide provides a breadth-first overview. For deeper dives into specific areas:

- **Segmentation, BEV fusion, occupancy prediction, open-vocabulary perception, and foundation models for driving**: see [segmentation survey]({% post_url 2026-03-28-segmentation-autonomous-driving-survey %}), which covers panoptic segmentation (Kirillov et al.), Mask2Former, BEVFusion, SAM/SAM 2, occupancy prediction (TPVFormer, SurroundOcc), and open-vocabulary 3D panoptic segmentation (Hung et al., ECCV 2024).

- **Motion planning, imitation learning, end-to-end models, VLM-based driving, and game-theoretic planning**: see [motion planning survey]({% post_url 2026-03-28-motion-planning-vlm-survey %}), which covers ChauffeurNet, MultiPath, UniAD, VAD, MotionLM, GPT-Driver, DriveVLM, LMDrive, DriveLM, GameFormer, DTPP, EMMA, S4-Driver, Scaling Laws for Driving, WOD-E2E, and the Waymo Foundation Model in detail.
