---
title: "Autonomous Systems: Robot Learning"
category: engineering
excerpt: "First-principles guide to robot learning — kinematics, control, imitation learning, sim-to-real transfer, manipulation, locomotion, and safety."
tags: autonomous-driving robotics self-driving tech-interview
comments: true
---

# Robot Learning: A First-Principles Guide

> Target audience: ML engineers preparing for robotics interviews. This guide builds from foundational robotics concepts to modern learning-based methods, defining every term with concrete examples before building on it.

---

## 1. Robot Kinematics

Kinematics is the study of motion without considering forces. Before a robot can learn anything, we need to describe *where its body is* and *how it moves*.

### 1.1 Configuration Space vs Task Space

A **configuration** is the complete specification of every joint in a robot. For a robot arm with 6 revolute joints, the configuration is a vector of 6 angles: **q** = (θ₁, θ₂, θ₃, θ₄, θ₅, θ₆). The set of all valid configurations is the **configuration space** (C-space). A 6-joint arm lives in a 6-dimensional C-space.

**Task space** (also called **operational space** or **Cartesian space**) is the space we actually care about — the position and orientation of the robot's hand (the **end-effector**). For a robot placing a cup on a table, task space is the 3D position (x, y, z) and 3D orientation (roll, pitch, yaw) of the gripper — 6 dimensions total, called a **pose**.

Concrete example: imagine a desk lamp with two hinged segments. The configuration is two angles (θ₁, θ₂). The task space is where the lamp head points in 2D. Many different (θ₁, θ₂) pairs can aim the lamp at the same spot — this many-to-one mapping is the root of why inverse kinematics is hard.

### 1.2 Forward Kinematics

**Forward kinematics** (FK) answers: *given joint angles, where is the end-effector?*

For a chain of rigid links, each joint applies a transformation to the next link. We compose these transformations using 4×4 **homogeneous transformation matrices**:

```
T = T₁ · T₂ · ... · Tₙ
```

Each Tᵢ encodes a rotation and translation:

```
Tᵢ = | Rᵢ  dᵢ |
     | 0    1  |
```

where Rᵢ is a 3×3 rotation matrix and dᵢ is a 3×1 translation vector.

**2-link planar arm example.** Two links of lengths L₁ and L₂ with joint angles θ₁ and θ₂ (both revolute, rotating in the plane):

```
x = L₁·cos(θ₁) + L₂·cos(θ₁ + θ₂)
y = L₁·sin(θ₁) + L₂·sin(θ₁ + θ₂)
```

This is a direct geometric calculation — FK always has a unique, closed-form solution.

![Forward and inverse kinematics of a 2-link planar robot arm](/assets/img/blog/autonomous-systems/forward_inverse_kinematics.svg)

### 1.3 DH (Denavit-Hartenberg) Parameters

Rather than deriving transforms from scratch for every robot, the **Denavit-Hartenberg convention** provides a systematic 4-parameter description of each link-joint pair:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Link length | a | Distance between joint axes along the common normal |
| Link twist | α | Angle between joint axes about the common normal |
| Link offset | d | Distance along the joint axis (variable for prismatic joints) |
| Joint angle | θ | Rotation about the joint axis (variable for revolute joints) |

Each joint's transform is:

```
Tᵢ = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

**2-link planar arm DH table:**

| Joint | θ      | d | a   | α |
|-------|--------|---|-----|---|
| 1     | θ₁     | 0 | L₁  | 0 |
| 2     | θ₂     | 0 | L₂  | 0 |

Both joints are revolute (θ is the variable), planar (no twist α = 0, no offset d = 0). DH parameters are the standard way to specify robot geometry in textbooks and manufacturer datasheets. Given a DH table, you can mechanically compute the full FK — no geometric intuition needed.

### 1.4 Inverse Kinematics

**Inverse kinematics** (IK) answers the opposite question: *given a desired end-effector pose, what joint angles achieve it?*

Why it's harder than FK:

- **Non-uniqueness:** Multiple joint configurations can reach the same pose. A human arm can touch a doorknob with the elbow pointing up or down.
- **No solution:** The desired pose may be outside the robot's **workspace** (the reachable region of task space).
- **Singularities:** Configurations where the robot loses a degree of freedom — the **Jacobian** becomes rank-deficient. Near a singularity, tiny task-space motions require enormous joint velocities.

**Jacobian-based iterative methods** solve IK numerically. The **Jacobian** J relates small joint-space motions to small task-space motions:

```
Δx = J(q) · Δq
```

where Δx is the task-space error and Δq is the joint correction. To find Δq:

```
Δq = J⁻¹ · Δx        (if J is square and non-singular)
Δq = J† · Δx          (pseudoinverse, if J is non-square)
Δq = Jᵀ · Δx          (transpose method, simpler but slower convergence)
```

The algorithm: start from an initial guess q₀, compute the task-space error Δx = x_desired - FK(q₀), compute Δq, update q ← q + Δq, repeat until convergence. This is essentially Newton's method applied to the FK equations.

**Damped least squares** (Levenberg-Marquardt) adds regularization to handle singularities:

```
Δq = Jᵀ(J·Jᵀ + λ²I)⁻¹ · Δx
```

The damping factor λ trades off accuracy for stability near singular configurations.

### 1.5 Joint Spaces

Robots use two fundamental joint types:

- **Revolute joints** rotate about an axis. Most robot arms are all-revolute. Configuration variable: angle θ ∈ [θ_min, θ_max].
- **Prismatic joints** slide along an axis (like a linear rail). Configuration variable: displacement d ∈ [d_min, d_max].

**Joint limits** constrain the C-space to a subset. A 6-DOF arm with joint limits has a C-space that's a hyper-rectangle in ℝ⁶ rather than all of ℝ⁶. The **workspace** is the image of this constrained C-space through FK — an irregularly shaped region in task space. Workspace analysis determines what a robot can reach, and is critical for cell layout and task feasibility.

**Degrees of freedom (DOF):** The number of independent joint variables. A task in 3D (position + orientation) requires 6 DOF. Robots with more than 6 joints are **redundant** — they have extra freedom that can be used for obstacle avoidance, singularity avoidance, or optimizing secondary objectives (e.g., minimizing joint torques).

---

## 2. Dynamics and Control

Kinematics tells us *where* the robot goes. **Dynamics** tells us *what forces* make it go there. **Control** is the art of computing those forces in real time.

The equation of motion for a robot arm is:

```
M(q)·q̈ + C(q, q̇)·q̇ + g(q) = τ
```

where M is the inertia matrix, C captures Coriolis and centrifugal effects, g is gravity, and τ is the vector of joint torques we command. This is the **manipulator equation** — it's nonlinear, coupled, and configuration-dependent.

### 2.1 PID Control

**PID control** is the most widely deployed control law in industrial robotics. It computes the control signal u(t) as a sum of three terms:

```
u(t) = Kp·e(t) + Ki·∫e(t')dt' + Kd·(de/dt)
```

where e(t) = q_desired - q_actual is the tracking error.

| Term | Role | Concrete example (controlling a joint angle) |
|------|------|-----------------------------------------------|
| **P** (proportional) | Pushes toward the target proportionally to how far away you are | Joint is 10° off → apply torque proportional to 10° |
| **I** (integral) | Eliminates steady-state error by accumulating past error | Gravity pulls the arm down 0.5° persistently → I term builds up and compensates |
| **D** (derivative) | Damps oscillation by opposing rapid error changes | Joint is approaching target quickly → D term brakes to prevent overshoot |

**Tuning tradeoffs:**
- High Kp → fast response but oscillation
- High Ki → eliminates steady-state error but can cause integral windup and instability
- High Kd → good damping but amplifies sensor noise

In practice, each joint runs an independent PID loop at 1 kHz+. PID works well when the dynamics are mild (slow motions, light payloads) but struggles with fast, dynamic motions where joint coupling matters — that's where model-based control helps.

![PID control loop block diagram](/assets/img/blog/autonomous-systems/pid_control.svg)

### 2.2 Model Predictive Control (MPC)

**Model Predictive Control** solves an optimization problem at every timestep:

```
minimize  Σ_{t=0}^{H} [ (x_t - x_ref)ᵀ Q (x_t - x_ref) + u_tᵀ R u_t ]
subject to  x_{t+1} = f(x_t, u_t)        (dynamics model)
            u_min ≤ u_t ≤ u_max           (actuator limits)
            x_t ∈ X_safe                   (state constraints)
```

The key idea: **optimize over a finite horizon H**, but **only execute the first action**, then re-solve at the next timestep with updated state. This **receding horizon** approach provides feedback and handles model error.

Why MPC is powerful for robotics:
- **Constraints are first-class citizens.** Joint limits, torque limits, obstacle avoidance — all enter as explicit constraints rather than being hacked into the cost function.
- **Preview.** MPC can anticipate future reference changes (e.g., an upcoming corner in a trajectory).
- **Nonlinear dynamics.** Nonlinear MPC (NMPC) uses the full nonlinear dynamics model f(x, u).

The downside: solving an optimization problem at every timestep is computationally expensive. Real-time MPC on robots typically runs at 50–500 Hz, depending on the horizon length and solver.

### 2.3 Impedance Control

Classical control tracks a trajectory: "be at position x_d at time t." But what if the robot touches something? A stiff position controller will either push with enormous force (dangerous) or lose contact.

**Impedance control** specifies a desired *relationship* between force and motion, modeled as a virtual spring-damper:

```
F = K·(x_d - x) + D·(ẋ_d - ẋ)
```

where K is the stiffness matrix and D is the damping matrix. The robot behaves like a spring attached to the desired position:
- Far from x_d → large restoring force
- Moving away from ẋ_d → damped resistance

Concrete example: a robot polishing a surface. In the normal direction, you want low stiffness (comply with surface shape) and moderate damping (smooth motion). In the tangential directions, you want the robot to track the polishing trajectory. Impedance control achieves this by setting different K and D values per axis.

**Admittance control** is the dual: given a force input (from a force/torque sensor), compute a motion output. Used when the robot has stiff position-controlled joints but needs compliant behavior.

Impedance control is essential for **contact-rich tasks**: assembly, insertion, collaborative robots (cobots) working alongside humans, and any manipulation that involves sustained contact.

### 2.4 Computed Torque Control

**Computed torque** (also called **inverse dynamics control**) uses the full dynamics model to cancel nonlinearities:

```
τ = M(q)·(q̈_d + Kp·e + Kd·ė) + C(q, q̇)·q̇ + g(q)
```

Substituting into the manipulator equation, the nonlinear terms cancel, leaving:

```
ë + Kd·ė + Kp·e = 0
```

This is a linear second-order system — easily tuned with standard linear control theory.

The catch: computed torque requires an accurate dynamics model (M, C, g). Model errors lead to imperfect cancellation and degraded performance. In practice, it's combined with robust or adaptive control to handle uncertainties.

---

## 3. Imitation Learning

Now we enter the learning-based approaches. Instead of manually programming control laws, can the robot learn from watching an expert?

### 3.1 The Core Idea

**Imitation learning** (IL) learns a policy π(a|s) from a dataset of expert demonstrations:

```
D = {(s₁, a₁), (s₂, a₂), ..., (sₙ, aₙ)}
```

where sᵢ is a state (e.g., joint angles + camera image) and aᵢ is the action the expert took (e.g., joint velocity command). The goal: find π that mimics the expert.

This avoids the reward-engineering problem of RL — the expert *shows* what to do rather than specifying a scalar reward.

### 3.2 Behavioral Cloning (BC)

**Behavioral cloning** is the simplest approach: treat imitation as supervised learning.

```
minimize  Σᵢ L(π(sᵢ), aᵢ)
```

Train a neural network to predict the expert's action given the state. For continuous actions, L is often MSE; for discrete actions, cross-entropy.

**The distribution shift problem.** BC works on the expert's state distribution. But at test time, the learned policy makes small errors → visits slightly different states → makes bigger errors → cascades into states the expert never demonstrated. Over a trajectory of length T, the error compounds quadratically: total error ∝ O(T²).

Concrete example: a self-driving car trained with BC drifts slightly left. The expert data never shows recovery from this situation. The car drifts further left. This is why naive BC fails on long-horizon tasks.

Despite this, BC with modern architectures and large datasets can work surprisingly well. The key is **data diversity** — if the training data covers enough of the state space (including recovery behaviors), the compounding effect is mitigated. Diffusion Policy (Chi et al., 2023) uses a diffusion model as the policy class for BC, generating multi-modal action distributions that handle the one-to-many nature of demonstrations.

### 3.3 DAgger (Dataset Aggregation)

**DAgger** (Ross et al., 2011) directly addresses distribution shift:

1. Train initial policy π₁ on expert data D
2. Roll out π₁ in the environment, collecting states {s'₁, s'₂, ...}
3. Query the expert for the correct actions at these new states: {(s'₁, a*₁), (s'₂, a*₂), ...}
4. Aggregate: D ← D ∪ {new data}
5. Retrain π₂ on the aggregated dataset
6. Repeat

By training on states the *learned policy* actually visits (not just the expert's states), DAgger breaks the distribution shift loop. Theoretical guarantee: after N rounds, the policy's performance approaches the expert's.

The practical limitation: DAgger requires an interactive expert who can provide corrections during rollouts. This is feasible in simulation or with a human teleoperator, but expensive in general.

![Behavioral Cloning vs DAgger imitation learning approaches](/assets/img/blog/autonomous-systems/imitation_learning.svg)

### 3.4 Beyond Behavioral Cloning

**Inverse reinforcement learning** (IRL) takes a different angle: instead of learning the policy directly, infer the expert's *reward function* R(s, a) from demonstrations, then optimize a policy for that reward. The idea is that the reward function is a more compact, transferable representation of the task than raw state-action pairs. MaxEntropy IRL (Ziebart et al., 2008) is a foundational algorithm. Adversarial IRL methods like GAIL (Ho & Ermon, 2016) train a discriminator to distinguish expert from policy behavior — the policy is rewarded for fooling the discriminator.

**One-shot imitation learning** aims to learn a new task from a single demonstration by leveraging prior experience across many tasks. The model learns a task-conditioned policy: given one demo of a new task, generalize to new initial conditions (Duan et al., 2017).

### 3.5 Connection to Autonomous Driving

Imitation learning has found major real-world application in self-driving:

- **ChauffeurNet** (Bansal et al., 2019, Waymo): trained a driving policy via imitation learning on expert driving logs. Key insight: synthesized **perturbations** during training — artificially shifted the car off-trajectory and trained it to recover. This is a data-augmentation approach to distribution shift, complementary to DAgger.
- **Tesla FSD v12+**: trained an end-to-end neural network from human driving video demonstrations. The network maps camera images directly to vehicle controls, bypassing hand-coded rules. This represents the largest-scale deployment of imitation learning, trained on millions of miles of human driving data.

Both systems demonstrate that with sufficient data scale and augmentation, BC-style approaches can work on long-horizon, safety-critical tasks.

---

## 4. Sim-to-Real Transfer

Training robots in the real world is slow (wall-clock time), expensive (hardware damage), and dangerous. Simulation is fast, cheap, and safe. But policies trained in simulation often fail when deployed on real hardware.

### 4.1 The Reality Gap

The **reality gap** has three main components:

| Gap type | Example |
|----------|---------|
| **Visual** | Simulated images look different (textures, lighting, reflections, noise) |
| **Physics** | Simulated contact dynamics are approximate (friction, deformation, soft contacts) |
| **Dynamics** | Simulated motor response differs from real actuators (latency, backlash, gear flexibility) |

A policy that achieves 95% success in simulation might achieve 5% success on the real robot — not because the algorithm is wrong, but because the policy has overfit to simulator-specific artifacts.

![Sim-to-real transfer pipeline and approaches to bridge the reality gap](/assets/img/blog/autonomous-systems/sim_to_real.svg)

### 4.2 Domain Randomization

**Domain randomization** (DR) takes a brute-force approach: if you don't know the exact real-world parameters, randomize them over a wide range during training. The policy must succeed across all randomizations, so it learns to be robust to variation.

What gets randomized:
- **Visual:** textures, lighting direction and color, camera position, background, object colors
- **Physics:** friction coefficients, object masses, joint damping, contact stiffness
- **Dynamics:** motor gains, action delay, sensor noise

**Canonical example: OpenAI's Rubik's Cube** (OpenAI, 2019). A dexterous Shadow Hand solved a Rubik's Cube in the real world after training entirely in simulation with massive domain randomization — thousands of physics parameters randomized simultaneously. The policy learned to be robust to any specific parameter setting because it had to succeed across all of them. This is sometimes called **Automatic Domain Randomization (ADR)**: start with narrow randomization ranges and automatically widen them as the policy improves.

### 4.3 System Identification

**System identification** (sysid) takes the opposite approach: instead of randomizing, *measure* the real-world parameters and tune the simulator to match.

Methods:
- **Manual measurement:** Weigh objects, measure friction with force sensors, characterize motor response curves.
- **Learned sysid:** Collect real-world trajectories, optimize simulator parameters to minimize the trajectory mismatch between sim and real.
- **Online adaptation:** Identify parameters on the fly from a few seconds of real-world interaction, then select or fine-tune the policy accordingly (e.g., RMA — Rapid Motor Adaptation, Kumar et al., 2021).

Sysid works well when the simulation is fundamentally capable of matching reality (the sim has the right *structure* but wrong *parameters*). It breaks down when the sim is missing important phenomena entirely (e.g., cable dynamics, deformable contacts).

### 4.4 Domain Adaptation

**Domain adaptation** uses representation learning to bridge the sim-real gap:

- **Adversarial domain adaptation:** Train an encoder to produce features that a discriminator cannot classify as "sim" or "real." The policy operates on these domain-invariant features.
- **Cycle-consistency (CycleGAN):** Learn to translate simulated images to look realistic (and vice versa). The policy trains on translated images that resemble real camera input.
- **Canonical representations:** Map both sim and real observations to an intermediate representation (e.g., semantic segmentation, depth maps) that's consistent across domains.

### 4.5 Progressive Transfer

Rather than a binary sim → real transfer, **progressive approaches** bridge the gap gradually:

1. Train base policy in simulation with domain randomization
2. Deploy on real robot, collect a small dataset (tens to hundreds of trajectories)
3. Fine-tune the policy on real data, using sim data as regularization

This combines the sample efficiency of sim training with the accuracy of real-world fine-tuning. **Residual policy learning** is a related idea: train a base policy in sim, then learn a small residual correction on the real robot that compensates for the sim-real gap.

---

## 5. Reward Shaping and Curriculum Learning

RL for robotics faces a fundamental challenge: designing reward functions that are both informative enough for learning and aligned with the true objective.

### 5.1 The Sparse Reward Problem

Many robotics tasks have natural **sparse rewards**: the robot gets +1 if it completes the task and 0 otherwise. Picking up an object: reward = 1 if object is grasped, 0 otherwise.

The problem: a random policy almost never achieves the goal, so it receives reward 0 on virtually every trajectory. There's no learning signal to improve from. For a task like inserting a peg into a hole with <1mm tolerance, a random policy might succeed once in 10⁸ trials.

### 5.2 Reward Shaping

**Reward shaping** adds intermediate reward signals to guide learning:

```
r_shaped(s, a, s') = r_sparse(s, a, s') + F(s, s')
```

Examples of shaping terms F:
- Distance to goal: F = -(||x - x_goal||)
- Reaching toward object: F ∝ -||hand - object||
- Contact detection: F = +δ when gripper contacts object
- Orientation alignment: F ∝ cos(θ_current, θ_desired)

The risk: poorly designed shaping can create **local optima**. A robot rewarded for moving its hand close to an object might learn to hover near it without ever grasping — the shaped reward is maximized, but the actual task isn't completed.

### 5.3 Potential-Based Reward Shaping

**Potential-based reward shaping** (Ng et al., 1999) provides a theoretical guarantee: if the shaping function has the form:

```
F(s, s') = γ·Φ(s') - Φ(s)
```

for some potential function Φ(s), then the **optimal policy is invariant** — shaping doesn't change *what* the optimal policy is, only *how fast* the agent finds it. This is the safe way to add shaping: define a potential function (e.g., Φ(s) = -||x - x_goal||) and derive F from it.

### 5.4 Curriculum Learning

**Curriculum learning** starts with easy task variants and progressively increases difficulty:

- **Manual curriculum:** Start with the peg near the hole, gradually increase the starting distance.
- **Automatic curriculum (self-play):** Two agents compete or cooperate, naturally generating tasks at the frontier of the learner's ability (e.g., one agent places objects, another learns to pick them up).
- **Goal-conditioned policies:** Train a policy π(a|s, g) conditioned on a goal g. Sample goals of increasing difficulty as the policy improves.

### 5.5 Hindsight Experience Replay (HER)

**Hindsight Experience Replay** (Andrychowicz et al., 2017) is an elegant solution to sparse rewards in goal-conditioned settings.

The insight: a failed trajectory still reaches *some* final state. Instead of discarding it, **relabel** the trajectory with the achieved state as the intended goal.

```
Original:  goal = "stack block on red block"  →  failed  →  reward = 0
Relabeled: goal = "place block at position (0.3, 0.2, 0.1)"  →  succeeded!  →  reward = 1
```

Every trajectory becomes a successful demonstration of *something*. The policy learns the structure of the task (how to reach arbitrary positions) and gradually generalizes to harder goals. HER converts a sparse-reward problem into a dense-reward one without manual shaping.

---

## 6. Manipulation

Manipulation — grasping, placing, assembling, tool use — is the central capability that makes robots useful in unstructured environments.

### 6.1 Grasping

**Analytical grasping** reasons about physics:
- **Force closure:** A grasp achieves force closure if the contact forces can resist any external wrench (force + torque). Requires contacts that span the wrench space — typically 3+ contacts for 2D, 7+ for 3D without friction, fewer with friction.
- **Form closure:** The object is geometrically constrained — it cannot move regardless of forces. Stricter than force closure.

**Data-driven grasping** trains neural networks to predict grasp success:
- Input: RGB or depth image of the scene
- Output: grasp quality score for candidate grasp poses
- **6-DOF grasp pose prediction:** Predict the full SE(3) pose of the gripper (position + orientation). Models like **GraspNet** (Fang et al., 2020) and **Contact-GraspNet** (Sundermeyer et al., 2021) predict dense grasp proposals from point clouds.
- **Planar grasping:** Simplification where the gripper approaches from above — reduces the problem to predicting (x, y, θ) in the image plane. Effective for bin picking.

Modern grasping pipelines combine perception (object detection/segmentation), grasp prediction, motion planning (collision-free path to the grasp pose), and control (compliant approach and force-controlled closure).

### 6.2 Dexterous Manipulation

**Dexterous manipulation** uses multi-fingered hands (3–5 fingers, 15–25 DOF) to manipulate objects with human-like skill.

Why it's hard:
- **High-dimensional action space:** A 24-DOF hand has an enormous space of possible finger configurations.
- **Contact dynamics are discontinuous:** Sliding → sticking, contact → no contact. These discontinuities make gradient-based optimization difficult and simulation inaccurate.
- **Underactuation:** Many dexterous hands have fewer actuators than joints (tendons drive multiple joints), adding constraints.
- **Sensing:** Tactile sensing (fingertip pressure, contact location) is critical but challenging to simulate and integrate.

**In-hand reorientation** — rotating a held object to a target orientation without putting it down — is a benchmark task. OpenAI's Rubik's cube work (2019) and NVIDIA's DextremeNet (2023) demonstrated that RL with sim-to-real transfer can achieve dexterous manipulation, though reliability remains below human level.

### 6.3 Contact-Rich Tasks

**Contact-rich manipulation** involves sustained, controlled contact: peg insertion, screwing, polishing, surface wiping.

Key challenges:
- **Contact state estimation:** Is the peg touching the hole edge? Which surface is in contact?
- **Force control:** Pure position control fails — small position errors create large forces. Impedance/admittance control (Section 2.3) is essential.
- **Hybrid position-force control:** Control position in free-space directions and force in constrained directions. Requires identifying the constraint frame.

Industrial assembly often combines:
1. Vision for coarse alignment (get the peg near the hole)
2. Force-guided search (spiral search pattern with force feedback to find the hole)
3. Impedance control for insertion (compliant in the insertion direction)

RL approaches to assembly (e.g., Luo et al., 2021) learn these strategies end-to-end from force and proprioceptive feedback, often outperforming hand-coded search patterns.

---

## 7. Locomotion

While manipulation concerns what a robot does with its hands, **locomotion** concerns how it moves through the world.

### 7.1 Why Legs?

Wheels are simple and efficient on flat ground. Legs exist for **terrain traversal**: stairs, rubble, gaps, uneven ground. Biological systems overwhelmingly use legs for terrestrial locomotion. The tradeoff: legs are mechanically complex, dynamically unstable, and harder to control — but they can go almost anywhere.

The fundamental locomotion challenge: **maintain balance while moving.** A wheeled robot is statically stable (it won't fall over if you stop the motors). A legged robot must actively balance — it's an inherently dynamic system.

### 7.2 Quadruped Locomotion

**Quadruped robots** (Boston Dynamics Spot, ANYmal, Unitree Go1/Go2) have become the most commercially successful legged platforms.

**Gait patterns** describe which legs move when:

| Gait | Pattern | Speed | Stability |
|------|---------|-------|-----------|
| **Walk** | 3 legs on ground at all times, move one at a time | Slow | Statically stable |
| **Trot** | Diagonal pairs move together (front-left + rear-right, then front-right + rear-left) | Medium | Dynamically stable |
| **Gallop** | Front pair then rear pair (or rotary variant) | Fast | Dynamically stable, brief flight phases |

**Central Pattern Generators (CPGs)** are bio-inspired controllers: networks of coupled oscillators that produce rhythmic patterns without sensory feedback. Each oscillator drives one leg, and coupling between oscillators determines the gait. By modulating oscillator frequency and phase relationships, a CPG can transition between gaits. CPGs provide a low-dimensional parameterization of locomotion that RL can learn to modulate.

### 7.3 Humanoid Balance

**Bipedal locomotion** is fundamentally harder: two legs provide a smaller support polygon, and walking involves controlled falling.

**Inverted pendulum model:** The simplest model of bipedal balance. The robot's center of mass (CoM) is a point mass on a massless rigid leg pivoting at the foot. The **Linear Inverted Pendulum Model (LIPM)** constrains the CoM to a horizontal plane, yielding analytically tractable dynamics.

**Zero Moment Point (ZMP)** is the point on the ground where the net moment of all forces (gravity + inertia) is zero. The **ZMP stability criterion**: if the ZMP stays within the **support polygon** (convex hull of foot contacts), the robot won't tip over. Traditional humanoid walking (Honda ASIMO, early Atlas) planned footsteps to keep the ZMP inside the support polygon.

Limitation: ZMP-based walking is conservative — it can't handle running, jumping, or dynamic motions where the ZMP leaves the support polygon. Modern approaches use full-body dynamics and optimization-based control (MPC) to plan through dynamic maneuvers.

### 7.4 Modern RL-Based Locomotion

The dominant modern approach: **train locomotion policies with RL in simulation, transfer to real hardware.**

Workflow:
1. Simulate the robot with a physics engine (MuJoCo, Isaac Sim)
2. Train an RL policy (typically PPO) to walk, using reward = forward velocity - energy penalty - stability penalty
3. Apply domain randomization to terrain, dynamics, sensor noise
4. Deploy on the real robot

Landmark results:
- **ANYmal** (ETH Zurich / RSL): RL-trained quadruped locomotion that traverses stairs, slopes, and rough terrain. Teacher-student framework: a privileged teacher policy (with ground-truth terrain info) is distilled into a student policy that uses only proprioception and a learned terrain estimator (Lee et al., 2020; Miki et al., 2022).
- **Boston Dynamics Atlas:** Combined model-based control with RL for dynamic parkour — running, jumping, backflips. Uses MPC for high-level trajectory optimization and RL for low-level whole-body control.
- **Agility Robotics Digit:** Bipedal robot for logistics, using RL-trained walking policies.

Key insight from locomotion research: **proprioceptive policies** (using only joint angles, velocities, and IMU — no vision) can handle remarkably complex terrain by learning to "feel" the ground through foot contacts. Vision is added for *anticipatory* behavior (seeing a staircase before reaching it).

---

## 8. Safety

A robot that learns by trial and error can break itself, damage its environment, or injure people. Safety is not optional — it's the primary constraint.

### 8.1 Safe Exploration

**Safe exploration** asks: how can an agent explore and learn without visiting catastrophic states?

Approaches:
- **Constrained action spaces:** Limit joint velocities, accelerations, and torques to mechanically safe ranges. Simple but effective — prevents most self-damage.
- **Safety filters / shielding:** Run a safety monitor in parallel with the learning policy. If the proposed action would violate a safety constraint (e.g., collision with the table), override it with a safe action. The policy learns within the filtered action space.
- **Cautious initialization:** Start from a known-safe policy (e.g., from imitation learning) and explore incrementally.
- **Sim-then-real:** Do all exploratory learning in simulation. Only deploy mature policies on hardware.

### 8.2 Constrained RL (CMDPs)

A **Constrained Markov Decision Process** (CMDP) adds safety constraints to the standard MDP:

```
maximize  E[Σ γᵗ r(sₜ, aₜ)]          (return)
subject to E[Σ γᵗ cᵢ(sₜ, aₜ)] ≤ dᵢ    (constraint i)
```

where cᵢ is a cost function (e.g., collision indicator) and dᵢ is the maximum allowable cumulative cost.

**Lagrangian relaxation** is the standard solution: convert the constrained problem to an unconstrained one by introducing Lagrange multipliers:

```
L(π, λ) = E[Σ γᵗ r] - Σᵢ λᵢ · (E[Σ γᵗ cᵢ] - dᵢ)
```

Optimize π to maximize L (the primal), and optimize λ to minimize L (the dual). The multipliers λᵢ are learned alongside the policy — they automatically increase when constraints are violated and decrease when satisfied. Algorithms: CPO (Constrained Policy Optimization, Achiam et al., 2017), PPO-Lagrangian.

### 8.3 Risk-Aware Planning

Standard RL optimizes *expected* return. But for safety, we care about worst-case or tail outcomes.

**CVaR (Conditional Value at Risk):** Instead of optimizing the mean return, optimize the expected return in the worst α% of outcomes. CVaR₀.₀₅ optimizes the average return in the bottom 5% of trajectories — focusing on catastrophic scenarios.

**Worst-case optimization:** Robust MDPs model uncertainty as an adversary that chooses the worst-case transition dynamics within a set. The policy is optimized against this adversary, guaranteeing performance under any dynamics in the uncertainty set.

**Chance constraints:** Require that safety violations occur with probability less than ε:

```
P(unsafe event) ≤ ε
```

This is a probabilistic version of hard constraints, appropriate when deterministic guarantees are impossible but statistical guarantees are sufficient.

### 8.4 Connection to Autonomous Driving Safety

Autonomous driving is the highest-stakes application of robot learning, with mature safety frameworks:

- **ISO 26262** (Functional Safety): Defines Automotive Safety Integrity Levels (ASIL A–D) for hardware and software. Requires fault detection, redundancy, and systematic development processes. Primarily addresses *component failures* (sensor dies, software crashes).

- **ISO 21448 / SOTIF** (Safety of the Intended Functionality): Addresses failures that occur even when all components work correctly — the system misinterprets a scene, makes a wrong decision. Directly relevant to ML-based systems: a perception model that misclassifies a pedestrian is a SOTIF issue, not an ISO 26262 issue.

- **Operational Design Domain (ODD):** The specific conditions under which the system is designed to operate (geographic area, weather, speed range, road type). Safety claims are always relative to the ODD — a system safe on highways in clear weather makes no safety claims about snowy mountain roads. ODD specification is how the industry scopes the safety problem to something tractable.

- **Safety validation:** Billions of miles of testing (physical + simulation) are needed for statistical confidence. Waymo and Cruise publish safety reports comparing autonomous vs human crash rates per mile. Scenario-based testing and adversarial testing complement statistical approaches.

---

## Summary: Key Concepts for Interviews

| Topic | Core question | Key concepts |
|-------|---------------|--------------|
| Kinematics | Where is the robot? | FK (always solvable), IK (hard: non-unique, singularities), Jacobian |
| Control | How to move it? | PID (simple, per-joint), MPC (optimal, handles constraints), impedance (for contact) |
| Imitation learning | Learn from demos? | BC (simple, distribution shift), DAgger (interactive correction), IRL (infer reward) |
| Sim-to-real | Sim → real? | Domain randomization (brute-force robustness), sysid (match the sim), domain adaptation (invariant features) |
| Reward shaping | Sparse reward? | Potential-based shaping (preserves optimality), HER (relabel goals), curriculum (progressive difficulty) |
| Manipulation | Grasp and use? | Force/form closure, data-driven grasping, dexterous hands (high DOF, contact discontinuities) |
| Locomotion | Walk and balance? | Gaits, CPGs, ZMP (stability criterion), RL + sim-to-real (modern approach) |
| Safety | Don't break things? | CMDPs (constraints in RL), CVaR (worst-case), safety filters, SOTIF (system-level) |

---

## References

- Siciliano, B., et al. *Robotics: Modelling, Planning and Control.* Springer, 2009. (The comprehensive reference for Sections 1–2.)
- Ross, S., et al. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS, 2011. (DAgger)
- Andrychowicz, M., et al. "Hindsight Experience Replay." NeurIPS, 2017. (HER)
- OpenAI. "Solving Rubik's Cube with a Robot Hand." 2019. (Domain randomization, dexterous manipulation)
- Bansal, M., et al. "ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst." RSS, 2019. (Imitation learning for driving)
- Ho, J. & Ermon, S. "Generative Adversarial Imitation Learning." NeurIPS, 2016. (GAIL)
- Ng, A., et al. "Policy Invariance Under Reward Transformations." ICML, 1999. (Potential-based shaping)
- Achiam, J., et al. "Constrained Policy Optimization." ICML, 2017. (Safe RL)
- Miki, T., et al. "Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild." Science Robotics, 2022. (ANYmal)
- Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS, 2023. (Diffusion-based BC)
- Kumar, A., et al. "RMA: Rapid Motor Adaptation for Legged Robots." RSS, 2021. (Online system identification)
- Handa, A., et al. "DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality." ICRA, 2023. (DextremeNet)
- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. "Learning Quadrupedal Locomotion over Challenging Terrain." Science Robotics, 2020.
- Duan, Y., Schulman, J., Chen, X., Bartlett, P., Sutskever, I., Abbeel, P. "One-Shot Imitation Learning." NeurIPS, 2017.
- Ziebart, B., Maas, A., Bagnell, J.A., Dey, A. "Maximum Entropy Inverse Reinforcement Learning." AAAI, 2008.
- Luo, J., Xu, D., Cai, Y., et al. "Robust Multi-Modal Policies for Industrial Assembly via Reinforcement Learning and Demonstrations." 2021.
