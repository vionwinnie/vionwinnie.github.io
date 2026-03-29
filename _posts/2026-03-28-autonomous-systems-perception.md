---
title: "Autonomous Systems: Perception"
category: engineering
excerpt: "First-principles guide to perception for autonomous driving — sensors, fusion, 3D detection, depth estimation, occupancy networks, and segmentation."
tags: autonomous-driving perception self-driving tech-interview
comments: true
---

# Perception in Autonomous Systems: A First-Principles Guide

**Target audience:** ML engineers with general ML knowledge preparing for autonomous driving interviews.
**Date:** March 2026

---

## Table of Contents

1. [Sensor Modalities](#1-sensor-modalities)
2. [Sensor Fusion](#2-sensor-fusion)
3. [3D Object Detection](#3-3d-object-detection)
4. [Depth Estimation](#4-depth-estimation)
5. [Occupancy Networks](#5-occupancy-networks)
6. [Lane Detection and Road Topology](#6-lane-detection-and-road-topology)
7. [Semantic Mapping](#7-semantic-mapping)
8. [Segmentation for Driving](#8-segmentation-for-driving)

---

## 1. Sensor Modalities

An autonomous vehicle perceives the world through multiple physical sensors, each exploiting a different part of the electromagnetic spectrum (or mechanical vibration). No single sensor is sufficient -- each has fundamental physical limitations. Understanding these limitations is the first step to understanding why modern AV stacks look the way they do.

### 1.1 Cameras

A camera captures light reflected off surfaces and projects a 3D scene onto a 2D image plane. Cameras are the richest source of *semantic* information (color, texture, lane markings, traffic signs) but fundamentally lose depth during projection.

#### Monocular Camera (Pinhole Model)

The simplest camera model is the **pinhole model**: light from a 3D point passes through an infinitely small aperture and lands on an image plane behind it.

A 3D world point `P = (X, Y, Z)` projects to a 2D pixel `p = (u, v)` via:

```
[u]     [fx  0  cx] [r11 r12 r13 | tx] [X]
[v]  =  [ 0 fy  cy] [r21 r22 r23 | ty] [Y]
[1]     [ 0  0   1] [r31 r32 r33 | tz] [Z]
                                        [1]
         ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
         Intrinsics K  Extrinsics [R|t]
```

- **Intrinsic parameters** (`K`): properties of the camera itself -- focal lengths `(fx, fy)` in pixels and the principal point `(cx, cy)` where the optical axis hits the sensor. *Example:* a 1920x1080 camera might have `fx = fy = 1000`, `cx = 960`, `cy = 540`.
- **Extrinsic parameters** (`[R|t]`): the camera's pose (rotation `R` and translation `t`) relative to a world coordinate frame. *Example:* a front-facing camera mounted 1.5m above the ground, tilted 5 degrees downward.

The key loss: division by `Z` (depth) makes projection irreversible. A pixel could correspond to any point along a ray.

#### Stereo Camera

A **stereo camera** is two cameras separated by a known **baseline** `B` (e.g., 12 cm). The same 3D point appears at slightly different horizontal positions in the left and right images.

```
Left image:   point at pixel u_L
Right image:  point at pixel u_R
Disparity:    d = u_L - u_R
Depth:        Z = f * B / d
```

**Epipolar geometry** constrains the search: for a point in the left image, its match in the right image must lie on the same horizontal line (after **rectification**, which aligns the two images so epipolar lines are horizontal). This reduces matching from a 2D search to a 1D search.

*Limitation:* depth accuracy degrades quadratically with distance (`delta_Z ~ Z^2 / (fB)`), so stereo works well at short range (<50m) but poorly at long range.

#### Fisheye Camera

A **fisheye lens** trades geometric accuracy for a wide **field of view (FoV)**, typically 180-220 degrees vs ~60 degrees for a standard lens. Autonomous vehicles use them for surround-view coverage, especially for near-field perception (parking, curb detection).

Fisheye images exhibit severe **radial distortion** -- straight lines in the world appear curved. Common distortion models:

- **Equidistant:** `r = f * theta` (angle-proportional)
- **Kannala-Brandt:** polynomial model `r = f * (theta + k1*theta^3 + k2*theta^5 + ...)`

You must undistort fisheye images before applying standard computer vision algorithms, or use distortion-aware architectures.

### 1.2 LiDAR (Light Detection and Ranging)

**LiDAR** measures distance by emitting laser pulses and timing their return. It produces a **point cloud** -- a set of 3D points, each with attributes `(x, y, z, intensity, ring)`, where:

- `(x, y, z)`: 3D position relative to the sensor
- **intensity**: reflectance of the surface (metal is highly reflective; dark clothing is not)
- **ring**: which laser channel produced this point (useful for organizing the cloud)

#### How Time-of-Flight Works

```
1. Emit laser pulse at time t0
2. Pulse hits object, reflects back
3. Sensor detects return pulse at time t1
4. Distance = c * (t1 - t0) / 2     (c = speed of light)
```

At the speed of light, a 1-nanosecond timing error corresponds to ~15 cm range error. Modern LiDARs achieve centimeter-level accuracy.

#### Spinning LiDAR (Velodyne-style)

A rotating assembly spins 360 degrees, firing multiple laser beams at different vertical angles. The Velodyne VLP-64, for example, has 64 laser channels, generating ~130,000 points per revolution at 10-20 Hz.

```
Top-down view of spinning LiDAR output:

          * * * * *
        *           *        <- dense ring of points
      *     LiDAR     *         at each vertical angle
      *    [sensor]    *
      *               *
        *           *
          * * * * *

Points are densest near the sensor, sparser at range.
```

*Pros:* 360-degree coverage, proven reliability (Waymo, Cruise).
*Cons:* Expensive ($10k-$75k), mechanical wear, bulky rooftop form factor.

#### Solid-State LiDAR

No moving parts. Two main technologies:

- **Flash LiDAR:** illuminates the entire FoV at once with a broad laser pulse; a 2D sensor array captures returns. Like a depth camera. Limited range (~30m) but very fast.
- **MEMS mirrors:** a tiny vibrating mirror steers the beam across the FoV. Lower cost, more compact (Luminar, Innoviz). Narrower FoV (~120 degrees horizontal) than spinning, so multiple units are needed for full coverage.

*Tradeoffs:* Solid-state is cheaper and more reliable (no mechanical rotation) but typically covers a smaller FoV and may have lower point density at range.

### 1.3 Radar

**Radar** (Radio Detection and Ranging) uses radio waves (typically 77 GHz in automotive). Radio waves are much longer wavelength than laser light, giving radar unique properties:

- **Doppler velocity:** radar directly measures the radial velocity of objects via the **Doppler effect** (frequency shift of the return signal). *Example:* a car approaching at 30 m/s shifts the 77 GHz signal by ~15 kHz. No other AV sensor measures velocity this directly.
- **Range-azimuth:** radar determines distance (from time-of-flight) and horizontal angle (from antenna array beamforming). Modern **4D imaging radars** also resolve elevation.
- **Weather robustness:** radio waves penetrate rain, fog, snow, and dust -- conditions that blind cameras and degrade LiDAR. This makes radar the most reliable sensor in adverse weather.

*Limitation:* Low angular resolution compared to LiDAR or cameras. A radar might resolve ~1-2 degrees in azimuth, making it hard to distinguish nearby objects. Radar also produces noisy returns from guardrails, overpasses, and other metal structures (clutter).

*Why radar is complementary:* Camera provides semantics but no depth or velocity. LiDAR provides depth but no velocity and degrades in weather. Radar provides velocity and works in all weather but has poor resolution. Together, they cover each other's blind spots.

### 1.4 Ultrasonic

**Ultrasonic sensors** emit high-frequency sound pulses (40-50 kHz) and measure the echo return time. Range is very short (typically 0.2-5 meters).

*Primary use case:* parking assist, low-speed maneuvering, and close-range obstacle detection. Nearly every production car has 8-12 ultrasonic sensors around the bumpers.

*Why not used for driving perception:* too short range, too slow update rate, no directional resolution.

### 1.5 IMU (Inertial Measurement Unit)

An **IMU** combines:
- **Accelerometer:** measures linear acceleration along 3 axes (X, Y, Z)
- **Gyroscope:** measures angular velocity (roll, pitch, yaw rates)

By integrating accelerometer readings over time, you can estimate velocity and position -- a process called **dead reckoning**. By integrating gyroscope readings, you can estimate orientation changes.

*The drift problem:* integration accumulates errors. A small accelerometer bias of 0.01 m/s^2 produces 1.8m of position error after just 60 seconds. IMUs are therefore never used alone -- they are fused with GPS, wheel odometry, and visual/LiDAR odometry to provide accurate, high-rate (100-1000 Hz) localization. The IMU fills in the gaps between slower sensors (GPS at 10 Hz, LiDAR at 10-20 Hz).

### 1.6 Sensor Comparison Table

| Sensor | Resolution | Range | Cost | Weather Robustness | Key Strength | Key Weakness |
|--------|-----------|-------|------|-------------------|-------------|-------------|
| **Camera** | Very high (~2-8 MP) | Unlimited (limited by optics) | Low ($10-50) | Poor (glare, rain, night) | Rich semantics, color, texture | No direct depth |
| **Spinning LiDAR** | High (~64-128 channels, ~100k pts/frame) | 200-300m | Very high ($10k-75k) | Moderate (degraded in heavy rain/snow) | Precise 3D geometry | Expensive, sparse at range |
| **Solid-state LiDAR** | Medium (~100k pts/frame) | 100-300m | Medium ($500-3k) | Moderate | Lower cost, compact | Narrower FoV |
| **Radar** | Low (1-2 deg angular) | 300m+ | Low ($50-200) | Excellent | Velocity, all-weather | Low resolution, clutter |
| **Ultrasonic** | Very low | 0.2-5m | Very low ($2-5) | Good | Cheap, close-range | Very short range |
| **IMU** | N/A (inertial) | N/A | Low ($10-100) | Excellent (internal) | High rate, orientation | Drift over time |

![Sensor Modalities](/assets/img/blog/autonomous-systems/sensor_modalities.svg)

---

## 2. Sensor Fusion

![Sensor Fusion Levels](/assets/img/blog/autonomous-systems/sensor_fusion_levels.svg)

### 2.1 Why Fuse?

Each sensor has fundamental blind spots that cannot be overcome by better algorithms alone:

| Sensor | What it lacks |
|--------|--------------|
| Camera | No direct depth measurement; struggles in low light and adverse weather |
| LiDAR | Sparse at long range; no color/texture information; expensive |
| Radar | Low spatial resolution; cannot distinguish object types; clutter |

Fusion combines complementary strengths. *Example:* a camera sees a pedestrian in a dark jacket (semantic recognition) but cannot determine distance. LiDAR returns a cluster of points at 45m (precise depth) but cannot tell if it is a person or a post. Fusing both gives: "pedestrian at 45m, walking left at 1.2 m/s."

### 2.2 Early Fusion (Raw Data Level)

**Early fusion** combines raw or minimally processed data from multiple sensors before any learning-based processing.

*Classic example:* **projecting LiDAR points onto camera images.** Each LiDAR point `(x, y, z)` is projected to pixel `(u, v)` using the known camera intrinsics `K` and the LiDAR-to-camera extrinsic transform `T`:

```
p = K * T * P_lidar
(u, v) = (p_x / p_z,  p_y / p_z)
```

This "paints" each pixel with a depth value, creating an RGBD image. A single network then processes this enriched input.

*Pros:* The network sees all raw information; no information is discarded.
*Cons:* Different modalities have very different data structures (dense 2D images vs sparse 3D point clouds), making it hard to design a single input representation. Misalignment from calibration errors is amplified.

### 2.3 Mid-Level Fusion (Feature Level)

**Mid-level fusion** (also called **deep fusion**) first processes each modality through its own encoder to produce learned feature representations, then fuses these feature maps.

*Modern standard:* **BEVFusion** (Liu et al., NeurIPS 2022) is the canonical example:

```
Camera images ──> Image backbone ──> Lift to BEV ──> ┐
                                                      ├── Concatenate ──> BEV Encoder ──> Task heads
LiDAR points ───> Point backbone ──> BEV features ──> ┘
```

Step by step:
1. **Camera branch:** Process multi-camera images with a 2D backbone (e.g., Swin Transformer). Use **Lift-Splat-Shoot (LSS)** to project 2D image features into 3D space by predicting a depth distribution per pixel, then scatter features into a **Bird's-Eye-View (BEV)** grid.
2. **LiDAR branch:** Encode point clouds using a 3D sparse convolution backbone (e.g., VoxelNet), then collapse the height dimension to produce BEV features.
3. **Fusion:** Concatenate camera-BEV and LiDAR-BEV features along the channel dimension.
4. **Task heads:** Apply task-specific heads (3D detection, segmentation, etc.) on the fused BEV features.

*Why BEVFusion is the modern standard:* It achieved state-of-the-art on nuScenes 3D detection (72.9 **nuScenes Detection Score (NDS)**) while being task-agnostic -- the same fusion backbone supports detection, segmentation, and mapping. The key insight is that BEV is the natural coordinate frame for fusion because both camera and LiDAR features can be projected into it.

### 2.4 Late Fusion (Decision Level)

**Late fusion** runs independent perception pipelines per sensor, then merges their outputs (detections, tracks, etc.) at the decision level.

```
Camera ──> Camera detector ──> Detections ──> ┐
                                               ├── **Non-Maximum Suppression (NMS)** / matching ──> Final detections
LiDAR ───> LiDAR detector ──> Detections ──> ┘
```

*Pros:* Modularity -- each detector can be developed, tested, and debugged independently. Graceful degradation if one sensor fails.
*Cons:* Loses cross-modal information. If a camera sees a partially occluded car and LiDAR sees its visible side, late fusion cannot combine these partial views -- each detector must independently decide.

### 2.5 Multi-Sensor Calibration

Fusion only works if sensors are precisely aligned. Three calibration problems:

1. **Extrinsic calibration** -- the relative pose (rotation + translation) between sensors. *Example:* knowing that the front LiDAR is 30 cm above and 10 cm behind the left camera, rotated 2 degrees to the right. Typically estimated using checkerboard targets or automatic methods (find corresponding features across modalities).

2. **Intrinsic calibration** -- the internal parameters of each sensor. For cameras: focal length, principal point, distortion coefficients. For LiDAR: beam angles, timing offsets per channel. Typically done in a factory or with calibration targets.

3. **Temporal synchronization** -- sensors run at different rates (cameras at 30 Hz, LiDAR at 10 Hz, radar at 13 Hz, IMU at 200 Hz). Their timestamps must be aligned to a common clock (typically GPS time or PTP). A 50ms synchronization error at highway speed (30 m/s) means 1.5m of positional misalignment.

---

## 3. 3D Object Detection

### 3.1 The Task

Given sensor data (point clouds, images, or both), predict a set of **3D bounding boxes** for objects in the scene. Each box is parameterized by:

```
(x, y, z, w, h, l, yaw, class, confidence)
 ^^^^^^   ^^^^^^^  ^^^   ^^^^^  ^^^^^^^^^^
 center   size     heading  category  score
```

- `(x, y, z)`: center position in the ego vehicle's coordinate frame
- `(w, h, l)`: width, height, length of the box (in meters)
- `yaw`: heading angle around the vertical axis (which direction the object faces)
- `class`: vehicle, pedestrian, cyclist, etc.
- `confidence`: detection score

This is the backbone task of AV perception. Downstream modules (tracking, prediction, planning) all consume 3D detections.

### 3.2 Point Cloud-Based Methods

![3D Detection Pipeline](/assets/img/blog/autonomous-systems/3d_detection_pipeline.svg)

#### PointPillars (Lang et al., CVPR 2019)

**PointPillars** introduced an elegant trick to make LiDAR 3D detection fast: convert the irregular 3D point cloud into a structured 2D representation that standard 2D CNNs can process.

```
Step 1: Divide the ground plane (x-y) into a grid of vertical columns ("pillars")
        Each pillar is a tall, thin volume (e.g., 0.16m x 0.16m x 4m)

        Top-down view:
        ┌───┬───┬───┬───┐
        │ . │   │ . │   │    . = pillar contains points
        ├───┼───┼───┼───┤
        │   │...│ . │   │    ... = pillar with many points (car?)
        ├───┼───┼───┼───┤
        │ . │ . │   │   │
        └───┴───┴───┴───┘

Step 2: For each non-empty pillar, encode its points using a PointNet:
        - Input: (x, y, z, intensity, x_offset, y_offset, z_offset, x_p, y_p)
        - PointNet produces a fixed-size feature vector per pillar

Step 3: Scatter pillar features back onto the 2D grid → "pseudo-image"

Step 4: Apply a standard 2D detection backbone (SSD-style) → 3D bounding boxes
```

*Why it matters:* PointPillars runs at **62 Hz** on a single GPU -- fast enough for real-time deployment. The insight that collapsing 3D to 2D via pillars (instead of **voxels** -- small 3D grid cells that discretize a volume) avoids expensive 3D convolutions while retaining enough information for accurate detection. It set the template for efficient LiDAR detectors.

#### CenterPoint (Yin et al., CVPR 2021)

**CenterPoint** is an **anchor-free** detector that treats 3D object detection as center point detection in BEV.

```
Step 1: Voxelize the point cloud, process with 3D sparse conv backbone

Step 2: Collapse to BEV → 2D feature map

Step 3: Predict a class-specific heatmap: each pixel represents the probability
        that an object center exists at that BEV location

        Heatmap example (vehicle class):
        ┌─────────────────────┐
        │ 0.0  0.0  0.0  0.0 │
        │ 0.0  0.9  0.1  0.0 │  ← peak at (1,1) = vehicle center
        │ 0.0  0.2  0.0  0.0 │
        │ 0.0  0.0  0.0  0.0 │
        └─────────────────────┘

Step 4: Extract peaks (local maxima) from the heatmap

Step 5: At each peak, regress box attributes:
        - sub-voxel center refinement (x_offset, y_offset)
        - height above ground (z)
        - box size (w, h, l)
        - heading angle (sin(yaw), cos(yaw))
        - velocity (vx, vy) — for tracking
```

*Why anchor-free matters:* Earlier detectors (SECOND, PointPillars) used pre-defined **anchor boxes** -- fixed templates of expected object sizes and orientations placed at every grid cell. This is wasteful (most anchors are negative) and requires manual tuning of anchor sizes. CenterPoint eliminates anchors entirely by directly predicting where object centers are.

CenterPoint also naturally extends to **tracking**: by predicting velocity at each detection, you can associate detections across frames using simple nearest-center matching without complex Hungarian assignment.

### 3.3 Camera-Based Methods

Detecting 3D objects from cameras alone is fundamentally harder because cameras lose depth information during projection. The core challenge is **depth ambiguity**: a small nearby car and a large distant truck can produce identical image patches.

#### Lift-Splat-Shoot (LSS) — Camera-to-BEV Lifting

**LSS** (Philion & Fidler, ECCV 2020) solves the camera-to-BEV projection problem:

```
For each pixel in each camera image:
  1. Extract image features with a 2D backbone
  2. Predict a discrete depth distribution (e.g., 112 depth bins from 1m to 60m)
  3. Outer product: feature_vector × depth_distribution → a "frustum" of features
     scattered along the camera ray at each depth bin
  4. Splat all frustum features into a common BEV grid using known camera geometry
  5. Sum overlapping contributions from all cameras

Result: a dense BEV feature map from cameras alone
```

This is the mechanism used by BEVFusion's camera branch, Tesla's BEV stack, and most modern camera-based 3D detectors.

*Scale ambiguity:* Monocular 3D detection struggles with absolute scale. A network trained on one camera setup may systematically mis-estimate depth when deployed on a different vehicle with different camera mounting heights. This is a key challenge for camera-only AV systems.

### 3.4 Fusion Methods: BEVFusion Architecture

BEVFusion (Liu et al., NeurIPS 2022) combines the best of both worlds. The full architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                     Multi-Camera Images                          │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │
│  │Front│ │F-L  │ │F-R  │ │Back │ │B-L  │ │B-R  │              │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘              │
│     └───────┼───────┼───────┼───────┼───────┘                   │
│             ▼                                                    │
│      Image Backbone (Swin-T)                                     │
│             ▼                                                    │
│      LSS: Depth Prediction + BEV Pooling                        │
│             ▼                                                    │
│      Camera BEV Features                                         │
│             │                                                    │
│             ├──── Concatenate ────┐                              │
│             │                     │                              │
│      LiDAR BEV Features          │                              │
│             ▲                     ▼                              │
│      VoxelNet + BEV Scatter    Fused BEV                        │
│             ▲                     ▼                              │
│      LiDAR Point Cloud     Conv BEV Encoder                     │
│                                   ▼                              │
│                          Task-Specific Heads                     │
│                         ┌────────┬────────┐                     │
│                     3D Det   BEV Seg   Map Seg                  │
└──────────────────────────────────────────────────────────────────┘
```

*Key engineering detail:* The BEV pooling operation (scattering camera features from frustums to BEV grid) was the bottleneck -- it was 500ms in naive implementations. BEVFusion introduced an efficient BEV pooling kernel that reduced this to 2ms using precomputed point-to-voxel mappings, making the whole pipeline practical for real-time use.

![BEV Projection](/assets/img/blog/autonomous-systems/bev_projection.svg)

### 3.5 Bird's-Eye-View (BEV) Representation

**BEV** (looking straight down at the scene from above) has become the dominant coordinate frame for AV perception. Why:

1. **Scale-invariant:** objects don't shrink with distance (unlike perspective images). A car is 4.5m x 1.8m in BEV regardless of whether it is 10m or 100m away.
2. **Natural for driving:** lane boundaries, routes, and traffic rules are defined on the ground plane. Planning happens in BEV.
3. **Easy to fuse sensors:** both LiDAR points and camera features can be projected to the same BEV grid, enabling simple concatenation-based fusion.
4. **Consistent across cameras:** surround-view cameras have different perspectives, but all map to the same BEV space.

Two main approaches to project camera features to BEV:
- **LSS (depth-based):** predict depth per pixel, scatter features to BEV (described above)
- **Cross-attention (BEVFormer):** place learnable BEV queries at each grid cell, cross-attend to image features using known camera geometry. More flexible but computationally heavier.

---

## 4. Depth Estimation

### 4.1 Why Depth Matters for Driving

Driving is inherently a 3D task -- you need to know *how far* objects are to plan safe trajectories. Cameras are cheap and information-rich, but they lose depth during projection. Recovering depth from cameras enables camera-only AV systems (lower cost, no LiDAR required) and improves camera-LiDAR fusion by densifying sparse LiDAR.

### 4.2 Stereo Matching

Given a rectified stereo pair (left and right images aligned so epipolar lines are horizontal):

```
Depth recovery pipeline:
  1. Rectification: warp images so corresponding points lie on the same row
  2. Disparity computation: for each pixel in the left image, find its match
     in the right image along the same row
  3. Depth: Z = f * B / d

  where:
    f = focal length (pixels)
    B = baseline (meters between cameras)
    d = disparity (pixel offset between left and right matches)
```

**Matching methods:**
- **Block matching:** compare small patches (e.g., 9x9) using Sum of Absolute Differences (SAD). Fast but noisy.
- **Semi-Global Matching (SGM):** adds smoothness constraints along multiple directions (8 or 16 paths). The standard for real-time stereo (used in production by Daimler, Subaru EyeSight).
- **Learned stereo:** deep networks (RAFT-Stereo, LEAStereo) predict disparity end-to-end, achieving higher accuracy at the cost of compute.

*Limitations:* Stereo depth accuracy depends on baseline. Depth error grows as `delta_Z proportional to Z^2 / (f*B)`. For a typical automotive stereo setup (B=12cm, f=700px), depth error at 50m is already ~3.5m. This is why LiDAR remains necessary for long-range perception.

### 4.3 Monocular Depth Networks

Estimating depth from a single image requires the network to learn **geometric priors** -- the expected sizes of objects, perspective cues, texture gradients, vertical position in the image.

#### Self-Supervised: Monodepth2 (Godard et al., ICCV 2019)

**Monodepth2** learns depth without any depth labels. The training signal comes from **photometric consistency** across consecutive video frames:

```
Training setup:
  - Input: frame I_t
  - Network predicts: depth map D_t and ego-motion T_{t→t+1}
  - Synthesize I_t from I_{t+1} using D_t and T:
      I'_t = warp(I_{t+1}, D_t, T_{t→t+1})
  - Loss: ||I_t - I'_t||  (photometric reconstruction error)
```

If the predicted depth and ego-motion are correct, warping the next frame should perfectly reconstruct the current frame. The network learns depth to minimize this reconstruction error.

*Scale ambiguity:* Self-supervised monocular depth is only accurate up to an unknown scale factor. The network cannot distinguish a scene that is 2x larger and 2x farther away. In practice, you resolve this by providing a single known measurement (e.g., camera height above ground, or a sparse LiDAR point).

#### Supervised Monocular Depth

When ground-truth depth is available (from LiDAR projection -- see below), you can train a network with direct depth supervision. Modern architectures (DPT, ZoeDepth, Depth Anything) use Vision Transformers and achieve remarkable zero-shot generalization.

### 4.4 LiDAR-Camera Projection

LiDAR provides sparse but accurate depth. By projecting LiDAR points onto camera images, you get **pseudo-ground-truth depth maps** for training supervised depth networks, or sparse depth inputs for **depth completion** networks.

```
Projection:
  1. Transform LiDAR point P_L to camera frame: P_C = T_{L→C} * P_L
  2. Project to pixel: (u, v) = K * P_C / P_C.z
  3. Store depth: depth_map[v, u] = P_C.z

Result: sparse depth map (typically only ~5% of pixels have depth values)

Depth completion: a network takes (RGB image, sparse depth) and predicts dense depth
```

This pipeline is how most datasets (nuScenes, KITTI, Waymo) provide depth supervision for training camera-based 3D detectors.

### 4.5 Depth Errors and Evaluation: LET-3D-AP

**LET-3D-AP** (Longitudinal Error Tolerant 3D Average Precision; Hung et al., Waymo / ECCV 2022) addresses a practical problem: camera-only 3D detectors have systematic depth errors (because depth from cameras is inherently noisy), but the standard 3D AP metric penalizes depth errors as harshly as lateral errors.

The insight: in driving, lateral errors (detecting a car one lane to the side) are far more dangerous than longitudinal errors (detecting it 2m too far or too close). LET-3D-AP modifies the IoU computation to be more tolerant of depth errors while remaining strict on lateral and size accuracy. This gives a fairer evaluation of camera-only detectors and better correlates with actual driving safety impact.

---

## 5. Occupancy Networks

### 5.1 The Problem with Bounding Boxes

3D bounding boxes are great for regular objects (cars, pedestrians, cyclists) but fail for:

- **Irregular shapes:** construction barriers, fallen debris, overturned vehicles
- **Amorphous obstacles:** snow piles, open car doors, low-hanging branches
- **Fine-grained geometry:** curb heights, road surface variations

A bounding box around a curved barrier wastes most of its volume on empty space. **Occupancy networks** provide a denser alternative: discretize 3D space into a voxel grid and predict, for each voxel, whether it is occupied and what semantic class it belongs to.

### 5.2 3D Occupancy Prediction

```
Voxel grid (simplified 2D slice showing front view):

     sky    sky    sky    sky    sky
     sky    sky    sky    sky    sky
     sky    sky    car    car    sky     ← car occupies these voxels
     road   road   car    car   road
     road   road   road   road  road
     ground ground ground ground ground

Each voxel: (occupied?, class) → e.g., (True, vehicle) or (True, road) or (False, free)
```

A typical voxel grid covers the area around the ego vehicle (e.g., 200m x 200m x 16m) at resolutions of 0.2-0.5m per voxel. This produces a dense 3D understanding of the scene.

The computational challenge: a grid of 400 x 400 x 32 voxels = 5.12 million voxels. Predicting a semantic label for each is expensive.

### 5.3 TPVFormer (Huang et al., CVPR 2023)

**TPVFormer** (Tri-Perspective View Former) addresses the computational cost of dense 3D occupancy by factorizing the 3D volume into **three orthogonal planes**:

```
Three planes:

  XY plane (top-down / BEV):     H x W features
  XZ plane (front view):         H x D features
  YZ plane (side view):          W x D features

  Total: O(HW + HD + WD)  vs  O(HWD) for dense voxels
```

For a grid of size H=W=200, D=16:
- Dense voxels: 200 x 200 x 16 = 640,000 elements
- TPV: 200x200 + 200x16 + 200x16 = 46,400 elements (~14x reduction)

To predict the occupancy of voxel `(i, j, k)`, TPVFormer looks up the corresponding features from all three planes and aggregates them:

```
feature(i,j,k) = f_XY(i,j) + f_XZ(i,k) + f_YZ(j,k)
```

Each plane is generated from multi-camera images via cross-attention (similar to BEVFormer but extended to three views). This achieves competitive occupancy prediction quality at a fraction of the compute cost.

### 5.4 SurroundOcc (Wei et al., ICCV 2023)

A key challenge for occupancy networks is obtaining ground-truth labels. Manual 3D voxel annotation is prohibitively expensive. **SurroundOcc** introduced a practical auto-labeling pipeline:

```
Auto-labeling pipeline:
  1. Aggregate multiple LiDAR sweeps over time (e.g., 10 sweeps)
     → denser point cloud covering occluded regions
  2. Apply Poisson surface reconstruction → continuous 3D surface mesh
  3. Voxelize the mesh → dense occupancy grid
  4. Transfer semantic labels from per-point annotations → labeled voxel grid
```

This pipeline converts sparse, single-frame LiDAR annotations into dense 3D occupancy labels, enabling supervised training without manual voxel labeling.

### 5.5 Connection to Tesla's Approach

Tesla's Occupancy Network (announced AI Day 2022) applies this concept at scale in a camera-only system. Their pipeline:
1. Process 8 surround cameras through a RegNet backbone
2. Lift camera features to 3D using a learned positional encoding (similar to LSS)
3. Predict a dense occupancy grid at 512 x 512 x 64 resolution
4. Temporal aggregation via a "spatial RNN" (video-module) for consistency across frames

Tesla uses occupancy for collision avoidance -- the planner treats occupied voxels as hard constraints regardless of what class they are. This is safer than relying on classification: even if the network misclassifies debris as "unknown occupied," the car still avoids it.

---

## 6. Lane Detection and Road Topology

### 6.1 Curve Fitting

At its simplest, a lane boundary can be modeled as a polynomial curve in the ego vehicle's coordinate frame:

```
y = a*x^3 + b*x^2 + c*x + d

where:
  x = longitudinal distance ahead of the ego vehicle
  y = lateral offset from the ego vehicle's center
  a, b, c, d = polynomial coefficients
```

*Why third-degree (cubic)?* A cubic polynomial can represent:
- `d`: current lateral offset
- `c`: current heading relative to lane (slope at x=0)
- `b`: curvature (e.g., highway curves)
- `a`: change in curvature (e.g., entering/exiting a curve -- a clothoid-like behavior)

Most highway geometry is well-captured by cubics. For tighter curves (urban intersections), you need either higher-degree polynomials, splines, or piecewise representations.

*Classic detection approach:*
1. Segment lane pixels in the image (semantic segmentation)
2. Apply inverse perspective mapping (IPM) to get a top-down view
3. Fit polynomials to the detected lane pixels using least squares

### 6.2 From Curves to Graphs: Road Topology

Modern driving requires more than just lane boundaries -- you need to understand the **topology** of the road: which lanes connect to which, where merges and splits occur, which lanes allow which maneuvers.

This is modeled as a **graph**:
- **Nodes:** lane centerlines (represented as polylines or Bezier curves)
- **Edges:** connectivity relationships
  - **Predecessor/successor:** which lane follows which (same direction)
  - **Left/right neighbor:** adjacent lanes for lane changes
  - **Topology at intersections:** which incoming lanes can connect to which outgoing lanes

```
Road topology graph example (intersection):

    Lane A (incoming) ──────┐
                             ├──→ Lane C (outgoing, straight)
    Lane B (incoming) ──────┤
                             └──→ Lane D (outgoing, right turn)

Nodes: {A, B, C, D}
Edges: {A→C, A→D, B→C, B→D}
```

### 6.3 Modern Approaches: Vectorized Map Prediction

Traditional AV systems rely on **pre-built HD maps** -- centimeter-accurate maps annotated with lane boundaries, traffic signs, curbs, and topology. These are expensive to create and maintain.

**MapTR** (Liao et al., ICLR 2023) pioneered **online HD map construction**: predict vectorized map elements directly from camera images in real-time.

```
MapTR architecture:
  1. Multi-camera images → backbone → BEV features (via LSS or cross-attention)
  2. Set of learnable map queries (one per potential map element)
  3. Transformer decoder: queries attend to BEV features
  4. Each query outputs:
     - A polyline (sequence of 2D points defining a lane boundary, crosswalk, etc.)
     - A class label (lane divider, road boundary, pedestrian crossing)
```

*Key innovation:* MapTR treats each map element as an *ordered set of points* and uses a permutation-invariant loss (matching predicted points to ground truth regardless of order), which handles the ambiguity of polyline direction.

**MapTRv2** extends this with lane topology prediction -- not just detecting lane lines but also predicting which lanes connect to which, directly outputting the road topology graph.

This is a major trend: moving from offline HD maps (expensive, stale) to **online map prediction** (cheap, always current, adapts to construction zones). Tesla, Waymo, and most AV companies are investing heavily in this direction.

---

## 7. Semantic Mapping

### 7.1 From Per-Frame Predictions to Persistent Maps

Per-frame perception (detecting objects, segmenting lanes) is inherently noisy and incomplete -- objects may be occluded, detections may flicker. **Semantic mapping** aggregates per-frame predictions over time into a persistent, consistent representation of the environment.

### 7.2 Temporal Aggregation

The basic pipeline:

```
For each timestep t:
  1. Run perception → per-frame 3D detections, segmentation, occupancy
  2. Estimate ego pose P_t (from GPS/IMU/LiDAR odometry)
  3. Transform per-frame predictions to a global coordinate frame using P_t
  4. Merge into the persistent map:
     - New observations confirm or update existing map elements
     - Bayesian updates weight recent observations more heavily
     - Conflicting observations are resolved (e.g., a "car" that hasn't moved
       for 100 frames might be reclassified as "parked car" or "static obstacle")
```

**Pose estimation** is critical: if you don't know where the car was when it made an observation, you can't place that observation in the global map. Pose comes from:
- GPS (coarse, ~1-3m accuracy)
- Visual odometry (track features across frames)
- LiDAR odometry (**Iterative Closest Point (ICP)**-based alignment of consecutive scans)
- IMU integration (high rate, fills gaps)
- Fusion of all the above via an Extended Kalman Filter or factor graph

### 7.3 Semantic SLAM

**SLAM** (Simultaneous Localization and Mapping) jointly estimates the vehicle's pose *and* builds a map of the environment. **Semantic SLAM** extends geometric SLAM with semantic labels:

```
Geometric SLAM map:       Semantic SLAM map:

  * * *                     [road] [road] [road]
  * . *    (points only)    [road]   .    [building]
  * * *                     [curb] [curb] [building]
```

Benefits of adding semantics:
- **Better data association:** matching "the same building corner" across frames is more robust than matching "the same 3D point"
- **Semantic priors:** roads are flat, buildings are vertical -- these constraints improve geometric estimation
- **Richer maps:** the resulting map supports not just localization but also path planning (drive on road, avoid buildings)

### 7.4 Connection to Waymo's Panoramic Video Panoptic Segmentation

Waymo released the **Panoramic Video Panoptic Segmentation** benchmark (Mei et al., ECCV 2022), which evaluates dense, temporally consistent scene understanding across all five cameras simultaneously.

The benchmark requires:
- Panoptic segmentation (every pixel classified as stuff or things-instance) across 5 cameras
- Temporal consistency: the same car must have the same instance ID across frames
- Panoramic consistency: an object visible in both the front and front-left cameras must have the same ID

This benchmark directly evaluates the capability needed for semantic mapping: can your system produce coherent, persistent scene understanding from a multi-camera video stream?

---

## 8. Segmentation for Driving

Segmentation is the dense per-pixel classification counterpart to object detection. While detection gives you bounding boxes for discrete objects, segmentation gives you precise shape boundaries and full coverage of the scene.

![Panoptic Segmentation](/assets/img/blog/autonomous-systems/panoptic_segmentation.svg)

### 8.1 Panoptic Segmentation

**Panoptic segmentation** (Kirillov et al., CVPR 2019) unifies two previously separate tasks:

- **Semantic segmentation** ("stuff"): classify every pixel into a class. Best for amorphous regions -- **road**, **sky**, **vegetation**, **building**. No notion of individual instances.
- **Instance segmentation** ("things"): detect and segment individual countable objects -- **car #1**, **car #2**, **pedestrian #3**. Each object gets a unique mask.

Panoptic segmentation assigns every pixel both a semantic class AND an instance ID (for "things" classes):

```
Input image:       Panoptic output:

 [sky][sky][sky]    (sky, -)  (sky, -)  (sky, -)
 [car][car][tree]   (car, #1) (car, #1) (veg, -)
 [road][road][road] (road, -) (road, -) (road, -)
```

#### Panoptic Quality (PQ) Metric

PQ is the standard metric, defined as:

```
PQ = (Σ IoU(p,g)) / (TP + 0.5*FP + 0.5*FN)
      matched pairs

   = Recognition Quality (RQ) × Segmentation Quality (SQ)

where:
  RQ = TP / (TP + 0.5*FP + 0.5*FN)     ← how well you detect instances
  SQ = avg IoU of matched pairs          ← how well you segment detected instances
```

PQ elegantly combines detection quality (did you find the right instances?) with segmentation quality (did you segment them accurately?).

### 8.2 Mask2Former: Universal Segmentation Architecture

**Mask2Former** (Cheng et al., CVPR 2022) showed that a single architecture can achieve state-of-the-art on all three segmentation tasks (semantic, instance, panoptic) simultaneously.

Core idea: **masked attention**. Standard cross-attention attends to the entire image, which is wasteful -- a query for "car #1" doesn't need to attend to pixels in the sky. Mask2Former restricts each query's attention to the region predicted by its current mask estimate:

```
Standard cross-attention:  query attends to ALL image features
Masked attention:          query attends ONLY to features within its predicted mask

Iteration:
  Round 1: query predicts rough mask → attend to rough region
  Round 2: refined mask → attend to refined region
  ...
  Final: precise mask
```

Architecture:
1. Image backbone + multi-scale feature pyramid
2. N learnable object queries (e.g., 100)
3. Transformer decoder with masked cross-attention
4. Each query outputs: (mask, class) pair
5. Loss: Hungarian matching between predicted and ground-truth masks

Results: 57.8 PQ on COCO (panoptic), 50.1 AP (instance), 57.7 mIoU (semantic) -- all with the same model and weights.

### 8.3 Open-Vocabulary Segmentation via CLIP Distillation

Traditional segmentation operates over a **closed vocabulary** -- a fixed set of classes defined during training (e.g., the 28 classes in Waymo). But the real world contains objects the training set never saw (unusual vehicles, animals on the road, dropped cargo).

**Open-vocabulary segmentation** uses vision-language models like **CLIP** to recognize arbitrary categories described in natural language.

**3D Open-Vocabulary Panoptic Segmentation** (Xiao, Hung et al., ECCV 2024) extends this to 3D point clouds:

```
Training pipeline:
  1. Train a standard 3D panoptic segmentation network on LiDAR
  2. For each predicted 3D segment, project it to camera images
  3. Extract CLIP features from the corresponding image region
  4. Distill CLIP features into the 3D network:
     train the 3D backbone to produce features that align with CLIP's
     text-image embedding space

Inference:
  - Given a text query (e.g., "construction cone," "shopping cart")
  - Encode text with CLIP text encoder → text embedding
  - Match against each 3D segment's distilled CLIP feature
  - Segments with high similarity are labeled with the queried class
```

This enables detecting novel object categories at test time -- critical for safety, since you cannot anticipate every possible road obstacle during training.

### 8.4 SAM and SAM 2: Foundation Models for Segmentation

**SAM** (Segment Anything Model; Kirillov et al., Meta, ICCV 2023) is a **foundation model** for image segmentation, trained on over 1 billion masks from 11 million images.

Key properties:
- **Promptable:** segment anything by providing a point click, bounding box, or text description
- **Zero-shot generalization:** works on images from domains never seen during training
- **Class-agnostic:** produces masks without semantic labels (just "this region is a coherent object")

```
SAM architecture:
  Image ──> Image Encoder (ViT-H) ──> Image Embedding (computed once)
                                            │
  Prompt (point/box/text) ──> Prompt Encoder ──> ┤
                                                  ▼
                                          Lightweight Mask Decoder
                                                  ▼
                                            Predicted Masks
```

The image encoder is expensive (ViT-H with 632M parameters) but runs once per image. The mask decoder is lightweight, enabling interactive segmentation at ~50ms per prompt.

**SAM 2** (Ravi et al., Meta, ICLR 2025) extends SAM to **video**: given a prompt on one frame, it tracks and segments the object across all subsequent frames using a memory mechanism. This is directly applicable to AV perception -- prompt "that car" in one frame, and SAM 2 tracks its mask through the video.

*Role in AV perception:* SAM/SAM 2 are not end-to-end driving perception systems, but they serve as:
- **Annotation tools:** dramatically reduce the cost of creating segmentation labels for training data
- **Foundation backbones:** fine-tune SAM features for driving-specific tasks
- **Zero-shot safety nets:** detect and segment unknown obstacles that specialized detectors miss

---

## Summary: How It All Fits Together

A modern AV perception stack integrates all of these components:

```
Sensors (cameras, LiDAR, radar, IMU)
        │
        ▼
Calibration & Synchronization
        │
        ▼
Sensor Fusion (BEVFusion-style) → Unified BEV Features
        │
        ├──→ 3D Object Detection (CenterPoint) → Bounding Boxes
        │
        ├──→ Occupancy Prediction (TPVFormer) → Dense 3D Occupancy
        │
        ├──→ Lane Detection (MapTR) → Vectorized Map + Topology
        │
        ├──→ Depth Estimation (LSS) → Dense Depth Maps
        │
        └──→ Panoptic Segmentation (Mask2Former) → Per-Pixel Labels
                │
                ▼
        Temporal Aggregation → Semantic Map
                │
                ▼
        Tracking & Prediction → Object Trajectories
                │
                ▼
        Motion Planning → Safe Trajectory for Ego Vehicle
```

The trend in the field is clear: from modular pipelines with hand-designed interfaces toward unified architectures that share representations across tasks (BEVFusion, UniAD) and ultimately toward end-to-end models (EMMA, S4-Driver) that collapse the entire stack into a single learned system. S4-Driver (Waymo/UC Berkeley, CVPR 2025) demonstrates that perception annotations are not even necessary — it lifts 2D MLLM visual features into sparse 3D volume representations, enabling self-supervised end-to-end driving that matches supervised approaches. The Waymo Foundation Model's Sensor Fusion Encoder represents the production realization of multi-modal perception research, fusing camera + LiDAR + radar for fast, reactive perception alongside a slower Driving VLM for complex semantic reasoning.

---

## Key References

| Paper | Venue | Key Contribution |
|-------|-------|-----------------|
| PointPillars (Lang et al.) | CVPR 2019 | Fast pillar-based LiDAR 3D detection (62 Hz) |
| Monodepth2 (Godard et al.) | ICCV 2019 | Self-supervised monocular depth estimation |
| LSS (Philion & Fidler) | ECCV 2020 | Camera-to-BEV lifting via depth prediction |
| CenterPoint (Yin et al.) | CVPR 2021 | Anchor-free center-based 3D detection |
| Panoptic Segmentation (Kirillov et al.) | CVPR 2019 | Unified things+stuff segmentation, PQ metric |
| Mask2Former (Cheng et al.) | CVPR 2022 | Universal segmentation architecture |
| BEVFusion (Liu et al.) | NeurIPS 2022 | Camera-LiDAR fusion in BEV space |
| LET-3D-AP (Hung et al.) | ECCV 2022 | Depth-tolerant evaluation for camera-only 3D detection |
| Waymo Panoramic Video Panoptic Seg (Mei et al.) | ECCV 2022 | Multi-camera video panoptic segmentation benchmark |
| MapTR (Liao et al.) | ICLR 2023 | Online vectorized HD map construction |
| TPVFormer (Huang et al.) | CVPR 2023 | Efficient tri-perspective occupancy prediction |
| SurroundOcc (Wei et al.) | ICCV 2023 | Auto-labeling for dense 3D occupancy |
| SAM (Kirillov et al.) | ICCV 2023 | Foundation model for promptable segmentation |
| UniAD (Hu et al.) | CVPR 2023 | Unified perception-prediction-planning |
| 3D Open-Vocab Panoptic Seg (Xiao, Hung et al.) | ECCV 2024 | Open-vocabulary 3D segmentation via CLIP distillation |
| SAM 2 (Ravi et al.) | ICLR 2025 | Video segmentation foundation model |
| EMMA (Hwang, Hung et al.) | arXiv 2024 (accepted at TMLR) | End-to-end multimodal driving via Gemini |
| S4-Driver (Xie, Xu et al.) | CVPR 2025 | Self-supervised E2E driving; sparse volume 3D lifting from 2D MLLM features |
| WOD-E2E (Xu, Lin et al.) | arXiv 2025 | Long-tail E2E driving benchmark with Rater Feedback Score metric |
| Waymo Foundation Model | Blog 2025 | Think Fast / Think Slow: Sensor Fusion Encoder + Driving VLM + World Decoder |
