---
title: "Survey: Segmentation & Scene Understanding for Autonomous Driving"
category: engineering
excerpt: "Paper survey covering panoptic segmentation, 3D detection, BEV fusion, open-vocabulary perception, and foundation models for driving — prepared for Waymo interview."
tags: autonomous-driving perception segmentation tech-interview
comments: true
---

# Quick Survey: Segmentation & Scene Understanding for Autonomous Driving

**Prepared for:** Interview with Wei-Chih Hung, Waymo Visual Reasoning Team
**Seed paper:** EMMA: End-to-End Multimodal Model for Autonomous Driving (arXiv:2410.23262)
**Date:** March 2026

---

## Overview

Segmentation and scene understanding for autonomous driving has evolved from standalone per-frame semantic labeling to unified, multi-task, multi-modal systems that jointly reason about perception, prediction, and planning. The core problem is: given raw sensor inputs (cameras, LiDAR, radar), produce a dense, structured understanding of the 3D driving scene -- classifying every pixel/point into semantic categories, distinguishing individual object instances, tracking them over time, and ideally doing so for open-vocabulary (previously unseen) categories.

The field has progressed through several paradigm shifts: (1) from separate semantic and instance segmentation to **panoptic segmentation** (Kirillov et al., 2019), which jointly handles "things" (countable objects) and "stuff" (amorphous regions like road, sky); (2) from 2D image-plane reasoning to **3D volumetric/BEV representations**, enabling direct fusion of camera and LiDAR data; (3) from closed-vocabulary fixed taxonomies to **open-vocabulary segmentation** using vision-language models like CLIP and SAM; and (4) from modular perception-then-planning pipelines to **end-to-end multimodal models** like EMMA that directly map sensor data to driving outputs via large language models.

Wei-Chih Hung's research sits at the intersection of these trends. His recent work spans **3D open-vocabulary panoptic segmentation** (ECCV 2024), **camera-only 3D detection metrics** (LET-3D-AP, ICRA 2024), **multi-object tracking** (STT, ICRA 2024), and the **EMMA end-to-end driving model** (2024). His earlier academic work focused on semi-supervised and self-supervised segmentation methods. Understanding this trajectory is key for the interview.

---

## Wei-Chih Hung: Publication Profile

| Year | Paper | Venue | Role/Notes |
|------|-------|-------|------------|
| 2018 | Adversarial Learning for Semi-Supervised Semantic Segmentation | BMVC 2018 | First author. Highly cited (~5k+ total profile citations). Used discriminator to enable semi-supervised seg. |
| 2019 | SCOPS: Self-Supervised Co-Part Segmentation | CVPR 2019 | First author. Discovers object parts without supervision using geometric priors. |
| 2020 | Mixup-CAM: Weakly-supervised Semantic Segmentation via Uncertainty Regularization | BMVC 2020 | Weakly-supervised segmentation with mixup augmentation. |
| 2020 | Weakly-Supervised Semantic Segmentation via Sub-Category Exploration | CVPR 2020 | Sub-category discovery for weakly-supervised learning. |
| 2022 | LET-3D-AP: Longitudinal Error Tolerant 3D Average Precision for Camera-Only 3D Detection | ECCV 2022 / ICRA 2024 | Co-author (Waymo). New metric addressing depth uncertainty in camera-only detectors. |
| 2022 | Waymo Open Dataset: Panoramic Video Panoptic Segmentation | ECCV 2022 | Waymo team. Introduced panoramic video panoptic segmentation benchmark. |
| 2024 | 3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation | ECCV 2024 | Co-author (Waymo). First method for 3D open-vocab panoptic seg using CLIP distillation. |
| 2024 | STT: Stateful Tracking with Transformers for Autonomous Driving | ICRA 2024 | Co-author (Waymo). Joint data association + state estimation. |
| 2024 | EMMA: End-to-End Multimodal Model for Autonomous Driving | arXiv (v3: Sep 2025) | Co-author (Waymo). End-to-end driving via Gemini-based MLLM. |

**Key themes in Hung's work:** Semi/weakly/self-supervised learning, segmentation (semantic, panoptic, open-vocabulary), evaluation metrics, end-to-end driving systems.

**Note:** The Visual Reasoning team's work on open-vocabulary perception, tracking, and evaluation metrics feeds directly into Waymo's newer systems -- S4-Driver builds on the self-supervised perception paradigm, WOD-E2E benchmarks the E2E driving stack that EMMA pioneered, and the Waymo Foundation Model's Sensor Fusion Encoder is the production realization of the multi-modal perception research.

---

## Timeline & Evolution of the Field

| Year | Paper / Milestone | Key Innovation |
|------|-------------------|----------------|
| 2018 | **Panoptic Segmentation** (Kirillov et al.) | Defined the panoptic segmentation task, unifying semantic + instance seg with PQ metric |
| 2019 | **PointPillars** (Lang et al.) | Fast LiDAR 3D detection using pillar-based point encoding; 62 Hz real-time |
| 2020 | **Panoptic-DeepLab** (Cheng et al., CVPR) | First real-time bottom-up panoptic segmentation; dual-decoder architecture |
| 2021 | **Cylinder3D** (Zhu et al., CVPR) | Cylindrical partition for LiDAR semantic seg; SOTA on SemanticKITTI |
| 2021 | **ViP-DeepLab** (Qiao et al., CVPR) | Video panoptic segmentation + monocular depth; temporal consistency |
| 2021 | **CenterPoint** (Yin et al., CVPR) | Center-based 3D detection from LiDAR; anchor-free design |
| 2022 | **BEVFusion** (Liu et al., NeurIPS / ICRA'23) | Unified camera-LiDAR fusion in BEV space; task-agnostic framework |
| 2022 | **Mask2Former** (Cheng et al., CVPR) | Universal architecture for all segmentation tasks via masked attention transformer |
| 2022 | **Waymo Panoramic Video Panoptic Seg** (Mei et al., ECCV) | Largest video panoptic seg benchmark: 100k images, 5 cameras, 28 classes |
| 2023 | **OneFormer** (Jain et al., CVPR) | Single model trained once for all three segmentation tasks; task-conditioned tokens |
| 2023 | **TPVFormer** (Zheng et al., CVPR) | Tri-perspective view for 3D occupancy prediction from cameras |
| 2023 | **SurroundOcc** (Wei et al., ICCV) | Dense multi-camera 3D occupancy prediction with auto-generated labels |
| 2023 | **UniAD** (Hu et al., CVPR Best Paper) | Unified perception-prediction-planning; end-to-end with transformer queries |
| 2023 | **SAM** (Kirillov et al., ICCV) | Foundation model for promptable segmentation; zero-shot generalization |
| 2024 | **3D Open-Vocab Panoptic Seg** (Xiao, Hung et al., ECCV) | First 3D open-vocabulary panoptic seg via CLIP-LiDAR distillation |
| 2024 | **SAM 2** (Meta, ICLR 2025) | Extends SAM to video; real-time promptable segmentation in videos |
| 2024 | **EMMA** (Hwang, Hung et al., Waymo) | End-to-end MLLM for driving; maps camera data to trajectories/objects via Gemini |
| 2025 | **OccMamba** (CVPR 2025) | State space models for efficient 3D semantic occupancy prediction |
| 2025 | **S4-Driver** (Xie, Xu et al., Waymo, CVPR) | Self-supervised E2E driving with sparse 3D volume representation from 2D MLLM features |
| 2025 | **Scaling Laws of Motion Forecasting** (Baniodeh, Goel et al., Waymo) | Power-law scaling for driving; model size grows 1.5x faster than data |
| 2025 | **WOD-E2E** (Xu, Lin et al., Waymo) | Long-tail E2E driving benchmark; Rater Feedback Score metric |
| 2025 | **Waymo Foundation Model** (Blog) | Think Fast/Think Slow: Sensor Fusion Encoder + Driving VLM + World Decoder |

---

## Detailed Paper Summaries

### Category 1: The Seed Paper and End-to-End Driving

**EMMA: End-to-End Multimodal Model for Autonomous Driving**
*Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, et al. (Waymo, 2024)* -- [arXiv:2410.23262](https://arxiv.org/abs/2410.23262)
- **Contribution:** First demonstration that a multimodal large language model (Gemini) can serve as a generalist backbone for autonomous driving, jointly handling planning, detection, and road graph estimation in a unified language space.
- **Method:** Raw camera images are fed into Gemini. All non-sensor inputs (navigation, ego status) and outputs (trajectories, 3D bounding boxes, road elements) are represented as natural language text. Task-specific prompts route the model to different outputs. Chain-of-thought reasoning improves planning by 6.7%.
- **Results:** SOTA on nuScenes planning; competitive on Waymo Open Motion Dataset and Waymo Open Dataset 3D detection.
- **Limitations:** Processes only a few frames (no long video); no LiDAR/radar input; computationally expensive. V3 (Sep 2025) notes plans for 3D sensing modality integration.
- **Why it matters for the interview:** This is the team's flagship paper. Understand the design choice of "everything as language" -- why text-based outputs for continuous quantities like trajectories? What are the scaling implications?

**UniAD: Planning-Oriented Autonomous Driving**
*Yihan Hu et al. (Shanghai AI Lab / SenseTime, CVPR 2023 Best Paper)* -- [arXiv:2212.10156](https://arxiv.org/abs/2212.10156)
- **Contribution:** Demonstrated that connecting perception, prediction, and planning in a single end-to-end network (rather than modular pipelines) significantly improves planning quality.
- **Method:** Four transformer decoder modules handle detection, tracking, mapping, and motion/occupancy prediction, connected by queries that pass information downstream. Joint optimization of all tasks.
- **Relationship to EMMA:** UniAD is a structured/modular end-to-end approach; EMMA replaces explicit task modules with a single MLLM. EMMA can be seen as pushing the "unification" further by collapsing task-specific architectures into language.

---

### Category 2: Panoptic Segmentation Foundations

**Panoptic Segmentation**
*Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, Piotr Dollar (CVPR 2019)* -- [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
- **Contribution:** Defined the panoptic segmentation task and the Panoptic Quality (PQ) metric, unifying semantic segmentation ("stuff": road, sky, vegetation) and instance segmentation ("things": cars, pedestrians) into a single coherent output.
- **Impact:** Became the standard formulation for dense scene understanding in driving, spawning hundreds of follow-up papers.

**Panoptic-DeepLab**
*Bowen Cheng, Maxwell D. Collins, Yukun Zhu, Ting Liu, Thomas S. Huang, Hartwig Adam, Liang-Chieh Chen (Google, CVPR 2020)* -- [arXiv:1911.10194](https://arxiv.org/abs/1911.10194)
- **Contribution:** First real-time bottom-up panoptic segmentation model, eliminating the need for proposal-based (top-down) instance segmentation pipelines like Mask R-CNN.
- **Method:** Uses a shared encoder with a dual-decoder architecture: (1) a **semantic branch** that produces per-pixel class predictions for both "stuff" and "things," and (2) an **instance branch** with two sub-heads -- a **center prediction** head that predicts heatmaps of object centers (class-agnostic keypoints), and an **offset regression** head that predicts the 2D offset from each "thing" pixel to its corresponding instance center. At inference, instance masks are formed by grouping "thing" pixels to their nearest predicted center using the offset vectors. Both decoders use dual-ASPP modules for multi-scale feature aggregation.
- **Results:** 35.1 PQ on COCO test-dev (competitive with top-down methods); near real-time at 15.8 FPS with MobileNetV3 backbone, making panoptic seg practical for AV deployment.
- **Why it matters:** The bottom-up center-prediction paradigm (predict centers + group pixels) became influential and was adopted by later works like CenterPoint and ViP-DeepLab.

**Mask2Former: Masked-Attention Mask Transformer for Universal Image Segmentation**
*Bowen Cheng, Ishan Misra, Alexander Schwing, Alexander Kirillov, Rohit Girdhar (Meta/UIUC, CVPR 2022)* -- [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
- **Contribution:** A single architecture that achieves SOTA across all three segmentation tasks (panoptic: 57.8 PQ on COCO, instance: 50.1 AP, semantic: 57.7 mIoU on ADE20K).
- **Method:** Masked attention constrains cross-attention to predicted mask regions, improving convergence and quality. Replaces task-specific architectures with one unified design.
- **Relationship to seed:** Mask2Former represents the "universal architecture" philosophy that EMMA extends to the entire driving stack.

**OneFormer: One Transformer to Rule Universal Image Segmentation**
*Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi (SHI-Labs, CVPR 2023)* -- [GitHub](https://github.com/SHI-Labs/OneFormer)
- **Contribution:** A single model trained **once** on a single panoptic dataset outperforms Mask2Former models that were trained separately per task.
- **Key insight:** The core novelty is **task-conditioned joint training**. During training, OneFormer samples a task token (semantic, instance, or panoptic) for each input and conditions the model on it, so a single model learns all three tasks without needing three separate training runs. At inference, the user specifies the desired task via the token.
- **Method:** Builds on Mask2Former's masked-attention transformer. Adds two components: (1) a **task-conditioned query initialization** where the task token modulates the learnable queries fed to the transformer decoder, and (2) an **inter-task contrastive loss** (query-text contrastive) that aligns the object queries with a text representation of their class derived from a text encoder, improving the discriminability of queries across tasks.
- **Results:** 68.5 PQ on Cityscapes panoptic, 83.0 mIoU on Cityscapes semantic, 46.5 AP on COCO instance -- all from a single jointly-trained model.
- **Why it matters:** Demonstrates that task specialization is unnecessary, pointing toward unified architectures that EMMA takes to the extreme across the full driving stack.

---

### Category 3: Video Panoptic Segmentation & Tracking

**ViP-DeepLab: Learning Visual Perception with Depth-Aware Video Panoptic Segmentation**
*Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen (Google/JHU, CVPR 2021)* -- [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_VIP-DeepLab_Learning_Visual_Perception_With_Depth-Aware_Video_Panoptic_Segmentation_CVPR_2021_paper.pdf)
- **Contribution:** Defines the new task of **depth-aware video panoptic segmentation (DVPS)**, which jointly solves three problems in a single model: monocular depth estimation, panoptic segmentation, and multi-frame instance tracking.
- **Method:** Extends Panoptic-DeepLab with two key additions: (1) a **depth prediction head** that shares the same encoder backbone and produces per-pixel depth estimates, and (2) a **next-frame instance branch** that predicts center offsets not just within the current frame but also to instance centers in the **next frame**, enabling temporal instance association without any post-hoc matching or tracking module. The model uses a shared encoder with separate decoder heads for semantics, instance centers/offsets, depth, and next-frame offsets.
- **Results:** 1st place on KITTI monocular depth benchmark, 1st on Cityscapes-VPS, 1st on KITTI MOTS (multi-object tracking and segmentation). Introduced Depth-aware VPQ (DVPQ) metric that jointly evaluates all three sub-tasks.
- **Why it matters:** Demonstrates that depth, segmentation, and tracking can be learned jointly with mutual benefit -- a precursor to the "unify everything" philosophy seen in UniAD and EMMA.

**Waymo Open Dataset: Panoramic Video Panoptic Segmentation**
*Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan et al. (Waymo, ECCV 2022)* -- [arXiv:2206.07704](https://arxiv.org/abs/2206.07704)
- **Contribution:** The largest video panoptic segmentation dataset: 100k images from 5 cameras, 28 semantic classes, 2,860 temporal sequences across 3 geographic locations. Instance labels are consistent across cameras and over time.
- **Why it matters:** This is Waymo's own benchmark. Knowing its structure, scale, and the STQ metric is essential for the interview.

**STT: Stateful Tracking with Transformers for Autonomous Driving**
*Waymo authors incl. Wei-Chih Hung (ICRA 2024)* -- [arXiv:2405.00236](https://arxiv.org/abs/2405.00236)
- **Contribution:** Joint data association and state estimation (velocity, acceleration) in a single transformer model, addressing a gap where most trackers optimize association but use heuristics for state estimation.
- **Method:** Consumes long-term detection history with rich appearance, geometry, and motion signals. Introduces S-MOTA and MOTPS metrics.

---

### Category 4: 3D LiDAR Segmentation

**Cylinder3D: An Effective 3D Framework for Driving-Scene LiDAR Semantic Segmentation**
*Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, Dahua Lin (CUHK, CVPR 2021)* -- [arXiv:2008.01550](https://arxiv.org/abs/2008.01550)
- **Contribution:** Cylindrical partition of 3D space for LiDAR semantic segmentation, replacing the standard Cartesian voxel grid.
- **Why cylindrical coordinates:** A rotating LiDAR scanner naturally produces points in a radial pattern -- dense nearby and sparse at distance. Cartesian voxels waste resolution: near-range voxels contain many points while far-range voxels are mostly empty. **Cylindrical coordinates** (radius, angle, height) align with the sensor's scanning geometry, producing a more balanced point distribution across voxels. This means each cylindrical partition cell contains a more uniform number of points, reducing both information loss (from overcrowded near-range voxels) and wasted computation (from empty far-range voxels).
- **Method:** Points are voxelized in cylindrical coordinates, then processed with **asymmetric 3D convolutions** -- using different kernel sizes along the height vs. horizontal dimensions to account for the non-cubic voxel shapes. A dimension-decomposition strategy further reduces the cubic computation cost of 3D convolutions.
- **Results:** 68.9 mIoU on SemanticKITTI test (1st place at the time), outperforming prior methods by 4-6 mIoU; 76.1 mIoU on nuScenes LiDAR segmentation.
- **Why it matters:** Shows that matching the data representation to the sensor geometry is critical -- a design principle relevant to any LiDAR-based perception system at Waymo.

**Instance Segmentation with Cross-Modal Consistency**
*Alex Zihao Zhu et al. (Waymo, 2022)* -- [arXiv:2210.08113](https://arxiv.org/abs/2210.08113)
- **Contribution:** Learns instance embeddings that are consistent across sensor modalities (camera + LiDAR) and stable over time, enabling unified instance segmentation without modality-specific post-processing.
- **Method:** Each sensor branch (camera, LiDAR) produces per-pixel/per-point embeddings. A **cross-modal contrastive loss** pulls embeddings of the same object instance together across cameras and LiDAR, while pushing different instances apart. A **temporal contrastive loss** does the same across consecutive frames, enforcing that the same object's embedding remains stable over time. At inference, a simple clustering on the joint embedding space yields instance masks that are consistent across views, modalities, and time -- without needing separate tracking or association steps.
- **Results:** Demonstrated on Waymo Open Dataset with improvements in cross-camera and cross-modal instance consistency.
- **Relationship to seed:** Demonstrates Waymo's investment in multi-modal consistency for segmentation. The principle of learning unified representations across modalities connects directly to EMMA's philosophy of collapsing modality-specific modules.

---

### Category 5: Camera-LiDAR Fusion

**BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation**
*Zhijian Liu et al. (MIT HAN Lab, NeurIPS 2022 / ICRA 2023)* -- [arXiv:2205.13542](https://arxiv.org/abs/2205.13542)
- **Contribution:** Unifies camera and LiDAR features in BEV space (not at the point level), preserving both geometric and semantic information. Optimized BEV pooling reduces latency by 40x.
- **Results:** +1.3% mAP/NDS for 3D detection, +13.6% mIoU for BEV map segmentation over LiDAR-only methods on nuScenes.
- **Why it matters:** BEV-based fusion is the dominant paradigm. EMMA's camera-only approach deliberately sidesteps fusion, but Waymo's production system certainly uses LiDAR. Understanding the tradeoffs is important.

---

### Category 6: Open-Vocabulary & Foundation Model Segmentation

**3D Open-Vocabulary Panoptic Segmentation with 2D-3D Vision-Language Distillation**
*Zihao Xiao, Longlong Jing, ..., Wei-Chih Hung, Thomas Funkhouser, et al. (JHU/Waymo/Google, ECCV 2024)* -- [arXiv:2401.02402](https://arxiv.org/abs/2401.02402)
- **Contribution:** The FIRST method for 3D open-vocabulary panoptic segmentation. Recognizes both base and novel (unseen) classes in 3D point clouds.
- **Method:** Fuses learnable LiDAR features with frozen CLIP visual features. A single classification head handles both base and novel classes. Two novel losses: object-level distillation and voxel-level distillation.
- **Why it matters for the interview:** This is Wei-Chih Hung's most recent and directly relevant paper. Be prepared to discuss: (a) why open-vocabulary matters for AV safety (long-tail objects), (b) the distillation approach from 2D VLMs to 3D, (c) limitations and future directions.

**SAM (Segment Anything Model)**
*Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, Ross Girshick (Meta AI, ICCV 2023 Best Paper Honorable Mention)* -- [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **Contribution:** Foundation model for **promptable segmentation** -- given any prompt (point, box, mask, or text), SAM produces a valid segmentation mask. Trained on the SA-1B dataset containing 1.1 billion masks across 11 million images.
- **Architecture:** Three components: (1) a **ViT-based image encoder** (ViT-H by default) that produces image embeddings; this runs once per image and is the computational bottleneck. (2) A **prompt encoder** that encodes sparse prompts (points, boxes) via positional encodings and dense prompts (masks) via convolutions. (3) A lightweight **mask decoder** (two transformer layers) that combines image and prompt embeddings to produce mask predictions. The decoder also predicts an IoU confidence score for each mask and outputs multiple masks (3 candidates) to handle ambiguity.
- **Results:** Strong zero-shot transfer: competitive with or better than fully supervised models on 23 diverse segmentation datasets without any fine-tuning.
- **Driving relevance:** Enables rapid annotation of driving data (reducing labeling costs); explored as a backbone for open-vocabulary driving perception. Its zero-shot capability is relevant for long-tail object segmentation.

**SAM 2: Segment Anything in Images and Videos**
*Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, Christoph Feichtenhofer (Meta AI, 2024)* -- [GitHub](https://github.com/facebookresearch/sam2)
- **Contribution:** Extends SAM from images to **video** with real-time promptable segmentation and temporal consistency, while remaining a strong image segmentation model.
- **Architecture:** Adds two key components on top of SAM's design: (1) a **memory encoder** that stores per-frame features and predicted masks into a memory bank (recent frames + prompted frames), and (2) a **memory attention** module (stacked transformer layers) that conditions the current frame's features on memories from previous frames via cross-attention. This allows the model to propagate segmentation through video without per-frame prompting. The image encoder is replaced with a hierarchical ViT (Hiera) for efficiency.
- **Results:** Trained on SA-V dataset (642k masklets across 51k videos). Achieves SOTA on video object segmentation benchmarks while running at real-time speeds (44 FPS on images). Also accepted at ICLR 2025.
- **Driving relevance:** The memory-based propagation mechanism is directly applicable to tracking objects through driving video sequences. Could enable efficient video annotation for Waymo's panoramic video datasets.

---

### Category 7: 3D Occupancy Prediction

**TPVFormer: Tri-Perspective View for 3D Semantic Scene Completion**
*Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, Jiwen Lu (THU, CVPR 2023)* -- [GitHub](https://github.com/wzzheng/TPVFormer)
- **Contribution:** Represents 3D space using three orthogonal planes instead of dense voxels, enabling camera-only 3D occupancy prediction that is both expressive and computationally tractable.
- **The three planes:** TPV decomposes the 3D volume into three axis-aligned planar feature maps: (1) **top-down (XY)** -- the BEV plane, capturing spatial layout; (2) **front (XZ)** -- capturing height and longitudinal structure; (3) **side (YZ)** -- capturing height and lateral structure. Any 3D point's feature is obtained by sampling from all three planes and summing the features. This is far more efficient than dense voxels: for a volume of H x W x D, dense voxels require O(HWD) memory, while TPV uses O(HW + HD + WD) -- effectively reducing a cubic cost to quadratic.
- **Method:** Multi-camera images are lifted to each plane via cross-attention (similar to BEVFormer), then cross-plane attention allows information exchange between the three views.
- **Results:** On nuScenes LiDAR segmentation benchmark (camera-only), achieved 27.83 mIoU with sparse LiDAR supervision, compared to MonoScene's 6.06 mIoU with dense supervision.
- **Why it matters:** Popularized the idea of 3D occupancy prediction from cameras as an alternative to bounding-box detection. Directly inspired Tesla's occupancy network approach and subsequent works like SurroundOcc.

**SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving**
*Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu (THU, ICCV 2023)* -- [GitHub](https://github.com/weiyithu/SurroundOcc)
- **Contribution:** Dense multi-camera 3D occupancy prediction with a novel **auto-labeling pipeline** that generates dense occupancy ground truth from sparse LiDAR data, solving the label scarcity problem.
- **Auto-labeling pipeline:** The key practical contribution. Existing datasets only have sparse LiDAR points, not dense voxel labels. SurroundOcc aggregates LiDAR sweeps across multiple frames (leveraging ego-motion), applies Poisson surface reconstruction to fill gaps, then voxelizes the result to produce dense occupancy labels. This avoids expensive manual 3D annotation entirely. The pipeline produces labels with ~2x the density of raw aggregated LiDAR.
- **Method:** Multi-camera images are processed with a 2D backbone, lifted to 3D via spatial cross-attention, then decoded with a coarse-to-fine 3D UNet (progressively upsampling from low to high resolution occupancy grids).
- **Results:** On nuScenes, achieved 20.30 IoU for scene completion and 34.72 mIoU for semantic occupancy with dense labels. Boosted TPVFormer's performance from 11.26 to 34.72 mIoU when using the auto-generated dense labels.
- **Why it matters:** The auto-labeling pipeline is arguably more impactful than the model itself -- it unlocked dense supervision for occupancy prediction research without requiring expensive annotation, and is now widely adopted.

---

### Category 8: Evaluation Metrics (Waymo-Specific)

**LET-3D-AP: Longitudinal Error Tolerant 3D Average Precision for Camera-Only 3D Detection**
*Wei-Chih Hung, Vincent Casser, Henrik Kretzschmar, Jyh-Jing Hwang, Dragomir Anguelov (Waymo, ECCV 2022 / ICRA 2024)* -- [arXiv:2206.07705](https://arxiv.org/abs/2206.07705)
- **Contribution:** Standard 3D AP unfairly penalizes camera-only detectors for depth errors that don't affect driving safety. LET-3D-AP tolerates longitudinal localization errors within a threshold, enabling fairer comparison of camera-only methods.
- **Why it matters:** Shows Hung's thinking about what metrics truly matter for safe driving -- not just raw accuracy but functionally relevant accuracy. This philosophy connects to EMMA's focus on planning quality.

---

## Key Concepts & Terminology

| Term | Definition |
|------|------------|
| **Panoptic Segmentation** | Joint semantic + instance segmentation. Every pixel gets a class label; "thing" pixels also get instance IDs. |
| **PQ (Panoptic Quality)** | Standard metric = SQ (Segmentation Quality) x RQ (Recognition Quality). Evaluates both segmentation accuracy and recognition. |
| **VPQ (Video Panoptic Quality)** | Extends PQ to video by evaluating temporal consistency of predictions across frames. |
| **STQ (Segmentation and Tracking Quality)** | Metric for video panoptic seg that separately evaluates segmentation and tracking quality. Used by Waymo. |
| **BEV (Bird's-Eye View)** | Top-down 2D representation of 3D scene. Common intermediate representation for multi-camera fusion. |
| **Occupancy Prediction** | Predict whether each 3D voxel in the scene is occupied and its semantic class. More general than bounding boxes. |
| **Open-Vocabulary Segmentation** | Segmenting classes not seen during training, using vision-language alignment (e.g., CLIP). |
| **Things vs. Stuff** | "Things" = countable objects (cars, people); "Stuff" = amorphous regions (road, sky, vegetation). |
| **MLLM** | Multimodal Large Language Model (e.g., Gemini, GPT-4V). EMMA is built on this foundation. |
| **Distillation** | Transferring knowledge from a large/pretrained model (teacher) to a task-specific model (student). |
| **LET-3D-AP** | Waymo's metric that tolerates depth errors in camera-only detection if they don't affect safety. |

---

## Research Frontier: Open Problems & Active Directions

### 1. Unified Perception-Planning Models
- **Open question:** Should perception remain explicit (segmentation maps, bounding boxes) or be implicit within end-to-end models like EMMA?
- **Tradeoff:** Explicit perception is interpretable and debuggable; implicit perception may learn better task-relevant representations but is a black box.
- **Waymo's position:** EMMA explores the implicit route but acknowledges limitations. The Visual Reasoning team likely works at this boundary.
- **S4-Driver update (2025):** Shows that self-supervised E2E driving approaches can match supervised ones by lifting 2D MLLM features into sparse 3D volume representations, removing the annotation bottleneck that has constrained end-to-end systems.

### 2. Open-Vocabulary & Long-Tail Perception
- **Problem:** Fixed taxonomies fail on rare objects (debris, fallen trees, unusual animals). Open-vocabulary methods using VLMs can detect novel objects but are slow and less accurate in 3D.
- **Active work:** Hung's ECCV 2024 paper is a direct contribution here. Key challenge: scaling to real-time with hundreds of potential classes.
- **Waymo's 2025 challenge** specifically targets long-tail driving scenarios.
- **WOD-E2E benchmark (2025):** The new Waymo Open Dataset E2E driving benchmark specifically targets long-tail scenarios occurring at <0.03% frequency, with a new Rater Feedback Score metric. This gives concrete evaluation infrastructure for open-vocabulary and long-tail perception research.

### 3. Temporal Consistency & Video Understanding
- **Problem:** Per-frame segmentation lacks temporal coherence. Video panoptic segmentation is expensive to annotate.
- **Direction:** SAM 2's memory-based approach for video; Waymo's panoramic video panoptic segmentation benchmark enables research here.

### 4. Camera-Only vs. Multi-Modal
- **Tension:** LiDAR provides accurate 3D geometry but is expensive; cameras are cheap but depth is ambiguous. EMMA is camera-only; Waymo's production vehicles use both.
- **LET-3D-AP** acknowledges camera depth limitations. Future EMMA versions plan LiDAR/radar integration.

### 5. 3D Occupancy as Alternative to Bounding Boxes
- **Why:** Bounding boxes can't represent irregular shapes (construction barriers, fallen cargo). Occupancy grids give dense 3D understanding.
- **Challenge:** Generating accurate occupancy labels; computational cost of dense 3D prediction. OccMamba (CVPR 2025) uses state space models for efficiency.

### 6. Foundation Models for Driving
- **Trajectory:** SAM/CLIP provide general vision capabilities; domain-specific models (EMMA, UniAD) adapt them to driving.
- **Open question:** Will driving-specific foundation models emerge, or will general VLMs + fine-tuning dominate?
- **Waymo Foundation Model (2025):** Waymo's "Think Fast / Think Slow" architecture points toward production-scale driving foundation models. The **Sensor Fusion Encoder** serves as the core perception system, fusing camera and LiDAR into a unified representation. This feeds both a fast reactive planner and a slower Driving VLM for complex reasoning, plus a World Decoder for simulation. This represents a concrete answer: driving-specific foundation models with multi-modal perception at their core.

### 7. World Models & Simulation
- **Emerging area:** GAIA-1 (Wayve), DriveDreamer, GenAD generate realistic driving scenarios. These can provide unlimited training data for segmentation models.
- **Connection to segmentation:** World models need dense scene understanding as both input and evaluation signal.

---

## Recommended Reading Order

For maximum understanding before the interview, read in this order:

1. **Panoptic Segmentation** (Kirillov et al., 2019) -- understand the foundational task definition and PQ metric
2. **Mask2Former** (Cheng et al., 2022) -- the dominant universal segmentation architecture
3. **BEVFusion** (Liu et al., 2022) -- how camera + LiDAR fusion works in practice
4. **Waymo Panoramic Video Panoptic Segmentation** (Mei et al., ECCV 2022) -- Waymo's own benchmark; know the dataset
5. **LET-3D-AP** (Hung et al., 2022) -- Wei-Chih's work on evaluation metrics; understand the philosophy
6. **UniAD** (Hu et al., CVPR 2023) -- the pre-EMMA paradigm for end-to-end driving
7. **3D Open-Vocabulary Panoptic Segmentation** (Xiao, Hung et al., ECCV 2024) -- **Wei-Chih's most relevant recent paper; read carefully**
8. **STT: Stateful Tracking with Transformers** (ICRA 2024) -- tracking side of Hung's work
9. **EMMA** (Hwang, Hung et al., 2024) -- **the seed paper; read the full paper**
10. **SAM 2** (Meta, 2024) -- understand foundation model capabilities for driving video
11. **S4-Driver** (Xie, Xu et al., Waymo, CVPR 2025) -- self-supervised E2E driving; shows annotation-free approaches closing the gap with supervised methods

---

## Interview-Specific Preparation Notes

### Questions to be ready for:
- "What are the tradeoffs between modular perception pipelines and end-to-end models like EMMA?"
- "How would you handle novel/long-tail objects that aren't in the training taxonomy?"
- "Why might camera-only approaches be preferred over LiDAR fusion in certain contexts?"
- "How do you evaluate segmentation quality in a way that's meaningful for driving safety?"
- "What's the role of foundation models (SAM, CLIP) in autonomous driving perception?"

### Questions you might ask:
- "How does the Visual Reasoning team think about the boundary between explicit perception outputs and implicit learned representations in the driving stack?"
- "The 3D open-vocab panoptic segmentation paper uses CLIP distillation -- are there plans to extend this with more recent VLMs like SigLIP or PaLI?"
- "EMMA's v3 mentions plans for LiDAR/radar integration -- what architectural challenges does that introduce for the language-based output paradigm?"
- "How does the panoramic video panoptic segmentation benchmark inform the team's research priorities?"

### Key Waymo Research themes to demonstrate awareness of:
1. **Multi-camera consistency** -- Waymo cares about consistency across 5 cameras and over time
2. **Metric design** -- LET-3D-AP shows they think carefully about what to measure
3. **Open-vocabulary generalization** -- the ECCV 2024 paper signals this is a priority
4. **End-to-end unification** -- EMMA represents a bold move toward MLLMs for driving
5. **2025 Open Dataset Challenges** -- Vision-based E2E driving, long-tail scenarios, interaction prediction

---

## Related YouTube Videos

| Topic | Video | Channel | Link |
|-------|-------|---------|------|
| BEVFusion | BEVFusion: Multi-Task Multi-Sensor Fusion | MIT HAN Lab | https://www.youtube.com/watch?v=uCAka90si9E |
| UniAD | UniAD: Planning-oriented Autonomous Driving | OpenDriveLab | https://www.youtube.com/watch?v=cyrxJJ_nnaQ |
| SAM | Segment Anything Paper Explained | No Hype AI | https://www.youtube.com/watch?v=JUMmqX-EHMY |
| E2E Autonomous Driving | Common Misconceptions in Autonomous Driving (Andreas Geiger) | WAD at CVPR | https://www.youtube.com/watch?v=x_42Fji1Z2M |
| Tesla Occupancy Networks | AI for Full Self-Driving (Andrej Karpathy, CVPR 2021) | WAD at CVPR | https://www.youtube.com/watch?v=g6bOwQdCJrc |
| E2E AD Tutorial | End-to-end Autonomous Driving: Past, Current and Onwards | OpenDriveLab | https://youtu.be/Z4n1vlAYqRw |

---

## Sources

- [EMMA Paper (arXiv)](https://arxiv.org/abs/2410.23262)
- [EMMA Waymo Research Page](https://waymo.com/research/emma/)
- [3D Open-Vocabulary Panoptic Segmentation (arXiv)](https://arxiv.org/abs/2401.02402)
- [LET-3D-AP (arXiv)](https://arxiv.org/abs/2206.07705)
- [STT: Stateful Tracking (arXiv)](https://arxiv.org/abs/2405.00236)
- [Waymo Panoramic Video Panoptic Segmentation (arXiv)](https://arxiv.org/abs/2206.07704)
- [Panoptic Segmentation (Kirillov et al., arXiv)](https://arxiv.org/abs/1801.00868)
- [Mask2Former (arXiv)](https://arxiv.org/abs/2112.01527)
- [BEVFusion (arXiv)](https://arxiv.org/abs/2205.13542)
- [UniAD (arXiv)](https://arxiv.org/abs/2212.10156)
- [Cylinder3D (arXiv)](https://arxiv.org/abs/2008.01550)
- [Instance Segmentation with Cross-Modal Consistency (arXiv)](https://arxiv.org/abs/2210.08113)
- [TPVFormer (GitHub)](https://github.com/wzzheng/TPVFormer)
- [SurroundOcc (GitHub)](https://github.com/weiyithu/SurroundOcc)
- [SAM (arXiv)](https://arxiv.org/abs/2304.02643)
- [SAM 2 (GitHub)](https://github.com/facebookresearch/sam2)
- [OneFormer (GitHub)](https://github.com/SHI-Labs/OneFormer)
- [ViP-DeepLab (GitHub)](https://github.com/joe-siyuan-qiao/ViP-DeepLab)
- [Panoptic-DeepLab (arXiv)](https://arxiv.org/abs/1911.10194)
- [Wei-Chih Hung's Website](https://hfslyc.github.io/)
- [Waymo Research Publications](https://waymo.com/research/)
- [Waymo 2025 Open Dataset Challenges](https://waymo.com/blog/2025/03/2025-waymo-open-dataset-challenges/)
- [Panoptic Perception for Autonomous Driving Survey (arXiv)](https://arxiv.org/abs/2408.15388)
- [EfficientPS (arXiv)](https://arxiv.org/abs/2004.02307)
- [GAIA-1 (arXiv)](https://arxiv.org/abs/2309.17080)
- [S4-Driver (arXiv)](https://arxiv.org/abs/2505.24139)
- [Scaling Laws of Motion Forecasting (arXiv)](https://arxiv.org/abs/2506.08228)
- [WOD-E2E (arXiv)](https://arxiv.org/abs/2510.26125)
- [Waymo Foundation Model Blog](https://waymo.com/blog/2025/12/demonstrably-safe-ai-for-autonomous-driving/)
