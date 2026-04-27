---
title: "ML Evaluation Engineer Interview: A Self-Study Question Bank"
category: engineering
excerpt: "Self-curated study questions covering production inference optimization, LLM/VLM evaluation, golden-set design, sim-to-real for robotics, and AV-flavored evaluation methodology. Each question has a toggleable suggested answer for self-quizzing."
tags: ml-evaluation interview-prep llm-as-a-judge vlm autonomous-vehicles
comments: true
---

# ML Evaluation Engineer Interview: A Self-Study Question Bank

*Last updated: April 2026*

These are study questions I drew up while preparing for a senior ML evaluation engineering interview in the autonomous-vehicles space. Sharing in case they're useful to anyone working through similar material.

The questions are written in second person — they're the questions I asked myself. Each one has a **collapsible suggested answer** so you can self-quiz first, then check.

Topics covered:

1. [Production inference optimization](#1-production-inference-optimization)
2. [VLM evaluation & self-supervised data pipelines](#2-vlm-evaluation--self-supervised-data-pipelines)
3. [Rule-based to LLM-agent migration](#3-rule-based-to-llm-agent-migration)
4. [LLM-as-a-Judge & golden sets](#4-llm-as-a-judge--golden-sets)
5. [Annotation pipelines & vendor management](#5-annotation-pipelines--vendor-management)
6. [Production ML monitoring](#6-production-ml-monitoring)
7. [VLM fine-tuning for video hazard detection](#7-vlm-fine-tuning-for-video-hazard-detection)
8. [Sim-to-real robot manipulation with VLA models](#8-sim-to-real-robot-manipulation-with-vla-models)
9. [Evaluation methodology deep dives](#9-evaluation-methodology-deep-dives)
10. [Domain curveballs](#10-domain-curveballs)

> Sections 7 and 8 reference my published project blogs — [SafetyGuardian](/safetyguardian) (VLM hazard detection) and [Sim-to-Real Robot Manipulation](/nvidia-gtc-sim2real). Read those first if a question feels unmoored from context.

---

## 1. Production Inference Optimization

**Q1.** Decomposing a P50 latency reduction across graph compilation (e.g. PyTorch AOTI), low-precision quantization (FP8), and GPU-profiling-driven kernel/batching fixes — which lever typically moves the needle most, and which is a dead end?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- The headline number is usually cumulative across all three levers; per-lever attribution depends heavily on the workload. Segmentation models, large VLMs, and diffusion transformers all respond differently.
- Profiling is the *enabler* — it tells you where AOTI compilation and FP8 quantization should be applied. AOTI/FP8 are the *executors*.
- Cost framing: latency wins translate to GPU savings only if autoscaling can capture them. A 40% P50 win at the same RPS means fewer replicas only if cold-start and tail-latency constraints permit.

</details>

**Q2.** Why **FP8** over INT8 or INT4 on a large VLM? What's the calibration set, how do you verify accuracy didn't regress, and what tends to break first when you turn it on?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- FP8 preserves dynamic range better for attention activations than INT8; INT8 PTQ tends to clip on long-tail tokens.
- Process: calibrate on a representative sample of production prompts, then re-run the visual-grounding eval suite to confirm no regression.
- Failure-mode escape hatch: mixed-precision — keep attention in FP16, FP8 the FFN.

</details>

**Q3.** **PyTorch AOTI** vs. TensorRT vs. `torch.compile` — when do you reach for AOTI? What workloads does it not help on, and which endpoints stay in eager?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- AOTI ahead-of-time compiles, removing Python overhead at serve time — wins when you have stable input shapes and want predictable latency.
- TensorRT: use where vendor kernels dominate; AOTI: use where you want flexibility in the model code.
- Stays eager: highly dynamic input shapes, control-flow-heavy preprocessing, low-volume endpoints where compile time doesn't pay back.

</details>

**Q4.** **Multi-endpoint serving stack at scale** — what's the autoscaling policy? How do you handle cold starts on a multi-billion-parameter VLM, and what's a reasonable tail-latency SLO?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Stack typically: Kubernetes + ArgoCD + Helm + Terraform.
- Cold-start on a multi-billion-param VLM is the hard problem — keep min-replica > 0; pre-warm via readiness probes.
- Tail latency: at-scale eval has the same shape — keep throughput high without paying for idle GPUs.

</details>

**Q5.** **Research-to-production parity**: what's the canonical bug class that infra-as-code (Terraform-managed infra, container-pinned envs) prevents? What goes wrong before parity is enforced?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Bug class: research notebook works, production endpoint diverges due to preprocessing/postprocessing drift, version skew on the model artifact, library version mismatch.
- Terraform-managed infra + container-pinned envs is what closes the loop.
- Same problem class shows up as research-to-eval-to-on-vehicle parity in any safety-critical deployment.

</details>

---

## 2. VLM Evaluation & Self-Supervised Data Pipelines

**Q6.** **VLM visual grounding evaluation methodology**: walk through the metric. How do you handle ambiguous prompts? What's the eval set size, how is it constructed, and what's the acceptable failure mode?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Build a structured eval set across scene types — diversity is the metric's defense against averaging away failures.
- Handle ambiguous prompts via multi-region acceptable answers + weighted scoring.
- The hard part is defining *what correct means* before measuring it. The load-bearing skill is structured eval-set construction.
- Open metric choice: IoU on bbox, pointing accuracy, top-1 region selection — depends on the use case.

</details>

**Q7.** **VLM-as-a-Judge + captioning self-supervised pipeline** — how do you prevent the judge from rubber-stamping the captioner's mistakes (judge/generator correlation)? How do you measure that the expanded data actually improved the model rather than just inflating the dataset?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Risk to anticipate: judge and captioner share blind spots → the loop confirms its own bias.
- Mitigations: judge uses a *different* base model than the captioner; periodic human spot-check of judge agreement; track data-diversity metrics, not just volume.
- Measurement: held-out eval set frozen *before* the expansion ran; compare model trained on original vs. expanded.

</details>

**Q8.** **Self-supervised expansion vs. human annotation** — where's the cost/quality crossover where you'd still pay for humans?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Self-supervised wins when judge precision is high *and* failure modes correlate with already-covered data slices.
- Humans still required for: novel scene types, edge cases, the judge calibration set itself.
- Frame as budget allocation: humans on calibration set + long-tail; pipeline on the bulk.

</details>

---

## 3. Rule-Based to LLM-Agent Migration

**Q9.** **Rule-based → LLM-agent transition** at production scale (millions of messages/day) — what's the riskiest user-facing failure mode to control during rollout, and how do you ramp traffic safely?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Gradual traffic ramp via experimentation framework; rule-based fallback retained for high-confidence intents during transition.
- Riskiest failure mode: tool-use hallucination (calling the wrong tool with confident wrong arguments) — gated behind golden-set tool-use accuracy thresholds before each ramp step.

</details>

**Q10.** **SFT with implicit feedback** (acceptance signal): what's the bias risk of training on accepted suggestions only? How would you correct for selection bias / counterfactual logging (IPS)?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Real risk: training only on accepted suggestions amplifies the existing distribution.
- Mitigations: log all suggestions (accepted + rejected); use rejection signal as negative; periodic exploration injection.
- Stronger correction: counterfactual logging with IPS reweighting.

</details>

---

## 4. LLM-as-a-Judge & Golden Sets

**Q11.** **Tool-use accuracy** — how do you decompose this metric? Per-tool precision/recall on selection vs. argument correctness vs. end-to-end task success? Where do the metrics disagree?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Decomposition: per-tool precision/recall on tool *selection* + per-tool argument correctness + end-to-end task success.
- Disagreement pattern: high per-tool precision, low end-to-end success → planning/orchestration is the bottleneck, not individual tools.
- The mental model transfers to evaluating any complex multi-step agent.

</details>

**Q12.** **LLM-as-a-Judge validation**: how do you validate each judge against the golden set — what precision/recall threshold gates deployment, and how often do judges drift and need recalibration?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Each judge validated against a golden set with precision/recall metrics before deployment.
- Recalibration trigger: judge precision on a rolling holdout slice falls below threshold.
- Cadence: event-driven on prod data drift, plus scheduled audits.

</details>

**Q13.** **Golden-set construction**: who labels it, how big, how do you handle label disagreement, and how often is it refreshed?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Sizing rule of thumb: small enough to be high-quality, large enough to give precision/recall stable to ~2 decimal places.
- Refresh: when the underlying capability set or tool catalog changes.
- Disagreement: SME adjudication; track inter-rater rate as a quality signal.

</details>

**Q14.** **Offline judge scores vs. A/B tests on adoption metrics** — how do you bridge the gap when they disagree? Which do you trust?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Disagreement is the interesting case: offline judge can be over-strict (misses user tolerance) or over-loose (misses friction).
- Trust hierarchy: A/B on adoption metric is ground truth for user impact; offline judge is the fast iteration loop. Use offline to filter candidates, A/B to confirm.

</details>

---

## 5. Annotation Pipelines & Vendor Management

**Q15.** **A large jump in F1** on a domain (e.g., 0.65 → 0.9) — was the gain from data quality, label quality, model architecture, or label-space redefinition?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Most often: data-centric improvement and label-space cleanup, not model architecture.
- Annotation pipelines that filter noisy public data against an internal product taxonomy can move the needle dramatically without touching the model.

</details>

**Q16.** **Vendor annotation pipelines**: how do you measure annotator quality and inter-rater agreement? What's your QA loop?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Vendor management: per-batch acceptance criteria; sampled audits.
- IRR: Cohen's kappa for two raters, Fleiss' kappa or Krippendorff's alpha for >2 raters; weighted kappa for ordinal labels.
- For more subjective domains (e.g., "comfort" labels in driving), formal kappa is non-optional — spot-check agreement isn't enough.

</details>

---

## 6. Production ML Monitoring

**Q17.** **Quantifying ML-driven savings** (e.g., a fraud-savings number) — how is that number computed? Counterfactual baseline, A/B holdout, or pre/post comparison?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Default framing: counterfactual based on baseline rate × volume × intervention rate.
- Acknowledge limitations: selection bias in pre-period, drift in baseline.
- Strongest method: A/B holdout if business risk allows. Pre/post is the weakest because confounders are unbounded.

</details>

**Q18.** **Time-sensitive model monitoring** — what specifically do you monitor (input drift, output drift, label delay), what fires alerts, and what's a reasonable false-alarm rate?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- "Time-sensitive" implies temporal drift: input feature drift (e.g., transaction patterns) and label arrival delay (labels confirmed days later).
- Alert design: thresholds too tight → alert fatigue; too loose → missed events. Calibrate against historical drift events.

</details>

---

## 7. VLM Fine-Tuning for Video Hazard Detection

> *Questions in this section are grounded in the [SafetyGuardian project blog](/safetyguardian) — read that first for context on dataset size, LoRA config, and the Cosmos-Reason fine-tune.*

**Q19.** **Optuna hyperparameter search**: 20 trials across LR (1e-5 to 5e-4), batch size, gradient accumulation, and LoRA rank. Why a high learning rate for LoRA fine-tuning, and how confident can you be that 20 trials is enough to claim a global optimum on a few hundred samples?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- High LR is OK for LoRA because adapter weights start at zero and the base model is frozen — effective gradient magnitude is much smaller than full fine-tuning.
- Honest: 20 trials on a small dataset is *enough to find a working config, not a global optimum*. Multi-seed runs at top-3 trial configs would estimate variance.
- The structure is right (log-scale LR + joint search over batch / grad-accum) even when trial budget is tight.

</details>

**Q20.** **LoRA targeting all 7 modules** (q/k/v/o + gate/up/down) at high rank with no dropout. Why all-modules over attention-only, why a large rank for a few hundred samples, and why no dropout?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- All 7 modules: visual-reasoning fine-tuning needs to adapt both attention (where the model attends) and FFN (what concepts it represents). Attention-only LoRA underfits multimodal tasks.
- Large rank justified by hyperparameter search; 1:1 alpha/rank ratio is conventional.
- Dropout 0: at small dataset size, regularization comes from the small effective rank itself plus early stopping. Adding dropout on top hurt val loss in trials.

</details>

**Q21.** Mean **token accuracy plateauing at ~0.52** after convergence. That's not high — how do you defend it as "good enough" for a safety-warning system, and what's the relationship between token accuracy and end-to-end hazard-classification correctness?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Token accuracy is misleading for structured output. A format like `HAZARD: <type> | SEVERITY: <level> | ACTION: <instruction>` has tokens that *don't matter* (literal "HAZARD:") and tokens that *do* (the type). 0.52 token-acc still allows correct slot extraction.
- The right metric is *slot-level F1 on hazard type and severity* — not token accuracy. Token accuracy was a training-loop diagnostic, not the deployment metric.
- For a production safety system, report end-to-end correctness on the validation set against human-rated ground truth, not training token-acc.

</details>

**Q22.** Filtering protocol with the **same model family as judge of generator** (zero-shot Cosmos-Reason as auto-reviewer for Cosmos-Predict-generated videos) plus human ratings. Doesn't this create a self-confirming loop — and how is it different from the judge/generator-correlation problem in self-supervised data expansion (Q7)?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Acknowledge the risk directly: same family means shared inductive biases. Catching the worst issues requires *human* review as the loop-breaker.
- The auto-filter is a *first pass* — rejects egregious cases (totally blank frames, wrong scene). Human is the precision filter.
- Same principle as Q7: human spot-check of judge agreement is non-optional when judge and generator share a backbone.
- Better: use a *different* VLM family (e.g., Qwen-VL or LLaVA) for the auto-filter to break the family correlation.

</details>

**Q23.** **Frame resize 1280×720 → 400×400** with `max_pixels=160000` for ~3× token efficiency. What's lost in visual fidelity, and how do you validate that the smaller frames don't destroy fine-grained hazard detection (small ice patches, distant pedestrians)?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Pixel reduction ~5.8×. Lost: distant pedestrians (small bbox), thin ice patches (texture-level), license-plate-readable text.
- Mitigated by: hazard categories being coarse-grained (pedestrian present yes/no, not "exact pose"); 5 frames per video give temporal redundancy.
- Right validation: explicit resolution ablation. If you skipped it, say so honestly.

</details>

**Q24.** **End-to-end latency around 1s** (VLM inference ~0.65s + TTS) for an *elderly pedestrian* warning use case — is 1s acceptable? What's the failure cost of a 1.5s warning vs. a 0.5s one, and how would you redesign for sub-500ms?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Honest framing: 1s is fine for *advisory warning* (notify when a hazard appears in next 5–20s window) — not fine for reactive obstacle avoidance (need <200ms).
- Use case: "look-ahead heads-up display," not "emergency brake."
- Sub-500ms redesign: smaller VLM (1B or distilled), streaming TTS overlap with inference, edge-deployed tiny detector in parallel for hard fail-safes.
- VLM inference dominates — vLLM batched serving, lower `max_pixels`, or distilled model.

</details>

**Q25.** **On-device inference impractical** (~56s/frame on a phone) — for a real product that means a network dependency. How do you reason about availability/safety when the network is the SPOF?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Real-product requirements: graceful degradation (smaller on-device classifier as fallback for the most critical hazards); buffered last-good warning; explicit "system unavailable" indicator so the user doesn't assume "no warning = no hazard."
- Honest scope: hackathon prototypes are typically cloud-only by design; production architecture is hybrid.

</details>

**Q26.** **Stopping at 20 epochs** with linear warmup and a 90/10 train/val split with no held-out test set. With ~27 val samples, your val-loss signal is noisy — how do you decide to stop at epoch 20 rather than 8 or 30, and how do you defend the no-test-set choice?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Stopping criterion: train/val loss curves plateau (visible in W&B), plus generation samples (logged every 2 epochs) showing format compliance.
- Small val set: signal *is* noisy — rely on directional trend across multiple epochs, not single-checkpoint eval.
- No held-out test set is a known weakness — production needs a frozen test set untouched during all hyperparameter search.
- Honest finish: Optuna trials selected on val loss → the "best" config is val-set-overfit by definition.

</details>

---

## 8. Sim-to-Real Robot Manipulation with VLA Models

> *Questions in this section are grounded in my [Sim-to-Real GR00T project blog](/nvidia-gtc-sim2real). Read that first for context on the policy architecture, training mix, and generalization tests.*

**Q27.** **GR00T N1.6** uses Cosmos-Reason-2B as the vision backbone with 32 DiT layers and **action chunking over 16 timesteps**. Why action chunking over autoregressive action prediction, and what's the failure mode when the chunk horizon doesn't align with task phases (e.g., grasp moment)?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Why chunk: autoregressive in action space at 30Hz means 30 sequential model calls per second of robot motion — latency-prohibitive for a 2B+ DiT model. Chunking amortizes inference.
- Failure mode: when the chunk crosses a phase boundary (approach → grasp), the predicted chunk may be inconsistent ("open gripper at step 8, close at step 12" is hard to plan if uncertainty spikes mid-chunk).
- Standard mitigation: receding-horizon — predict 16, execute first 8, replan.

</details>

**Q28.** **Synthetic > real co-training** (counterintuitive result): sim+70 Cosmos-augmented = 2/3 vials, sim+real co-training (5–50 episodes) = 1/3 vials. Why does synthetic augmentation outperform real co-training? Doesn't this contradict the conventional wisdom that real data > synthetic?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Likely explanation: with only 5–50 real episodes, real co-training adds **noise without coverage**. 70 Cosmos episodes adds **diversity at scale** that the policy needs more than fidelity.
- Honest caveat: "real > synthetic" is a function of *quantity at parity*. At low real-data quantity, high-quality synthetic wins. The crossover point is the interesting question.
- Same lesson applies to scenario coverage in any domain — synthetic-generated rare scenarios may beat sparse real captures of the same scenarios.

</details>

**Q29.** **Domain randomization** axes: lighting, HDRI, camera pose, object positions, pre-placement probability. Sim-only got high sim success but poor real success — so the sim-to-real gap was the bottleneck, not sim performance. Which DR axis tends to move real-world transfer most, and which is a placebo?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Without leave-one-out ablations, attribution is qualitative.
- Strongest qualitative signal: lighting/HDRI randomization tends to give robustness to lighting variation.
- Probable placebo: extreme camera-pose offsets — too large hurts training (forces the policy to learn invariances at the cost of accuracy at the nominal pose).
- Right experiment: leave-one-out per DR axis.

</details>

**Q30.** **Generalization tests**: OOD instruction → jerky motion; novel object (yellow rack → blue cup) → robot still targeted the original location. The model didn't actually use the language conditioning for object grounding — it memorized spatial priors. How do you diagnose that, and what would the next iteration change architecturally?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Diagnosis: novel-object swap — robot still went to original location, ignoring the new visual.
- Root cause: behavior cloning on a small dataset (75–145 episodes) can't disentangle *language → object identity* without explicit grounding supervision. The model latches onto spatial priors because they're more reliable than the language signal at that scale.
- Architectural fix: contrastive vision-language objective at training time, OR pretrained VLM grounding head with action decoder fine-tuned on top.
- Eval connection: this is exactly the kind of generalization failure a learned evaluator should catch — does the policy *use* its inputs, or pattern-match to the training distribution?

</details>

**Q31.** **Uncertainty estimation on a chunked DiT policy**: for a real deployment, how do you add uncertainty estimation — and how does that uncertainty feed an *evaluation* signal?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Sources of uncertainty: aleatoric (action noise from teleoperator inconsistency) vs. epistemic (out-of-distribution observation).
- Approach for a DiT chunked policy: ensemble at the chunk level — sample multiple action chunks at different diffusion seeds, measure variance. High variance signals epistemic uncertainty.
- Eval-of-eval connection: that variance becomes a *feature* for the learned evaluator — "policy was uncertain on this scenario, escalate for human review."

</details>

---

## 9. Evaluation Methodology Deep Dives

**Q32.** **Evaluation-of-evaluation**: how would you design a system to know whether your learned evaluator is correct? What's the meta-eval loop, and how do you avoid infinite regress?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Meta-eval loop: hold out a *human-labeled* gold subset that the learned evaluator never sees during development; measure evaluator precision/recall against it.
- Avoid infinite regress by: anchoring on humans at the bottom of the stack — humans evaluate the evaluator, evaluator evaluates the model.
- Operational: budget ~5% of human labeling for evaluator calibration *permanently*, not as a one-time project.

</details>

**Q33.** [CASE — 10–15 min] **Golden-set framework for AV behavior eval**: how would you bootstrap one? Who labels, what's the taxonomy, how do you handle long-tail scenarios that are release-blocking but rare in the dataset?

<details markdown="1">
<summary><b>Suggested answer (6-stage walkthrough)</b></summary>

1. **Problem framing.** Discrete behavioral events (cut-ins, hard braking, lane-change-without-signal) vs. continuous quality (comfort, lane-keeping deviation). Get the taxonomy from the team — don't assume which subtypes are release-blocking.
2. **Ground-truth source.** Stratified sampling: every rule-flagged event in fleet logs becomes a golden positive (cheap labels). Augment with human-labeled long-tail mined from fleet replay (expensive, targeted). Ultra-rare scenarios from synthetic generation.
3. **Method.** Per-scenario-type sub-golden-sets so we measure precision/recall *by behavior category*, not aggregate. Aggregate metrics hide tail failures. Each sub-set sized to give CI tight enough to detect a 2pp regression.
4. **Eval-of-eval.** Hold out ~5% of human labels permanently for evaluator calibration — never seen during evaluator training. Measure evaluator precision/recall against that frozen set, not against the model under test.
5. **Production loop.** Refresh quarterly OR on a drift trigger (fleet distribution shift, new ODD region). Track evaluator-vs-human agreement as a control chart over time.
6. **Failure modes.** (a) Labeler bias if a single team owns the golden set → rotate labelers + measure inter-rater. (b) Synthetic edge cases drifting from real-world distribution → periodic real-fleet calibration. (c) Golden-set staleness as behavior changes across model versions → tie refresh schedule to model versions, not calendar.

</details>

**Q34.** **LLM-as-a-Judge calibration drift**: same prompts, same model version — judges still drift. What's your monitoring + recalibration cadence, and at what signal do you re-train vs. re-prompt vs. replace?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Drift sources: model version updates, prompt template changes, distribution shift in inputs.
- Monitoring: rolling-window precision on a fixed calibration set, daily.
- Recalibration ladder: re-prompt (cheap) → few-shot example refresh (medium) → fine-tune judge (expensive) → replace judge architecture (rare).
- Trigger: judge precision on calibration slice drops > 3pp week-over-week.

</details>

**Q35.** [CASE — 10–15 min] **Agentic workflow for evaluating a complex driving scenario** (e.g., unprotected left turn): walk through how you'd chain VLM perception → retrieval → structured reasoning → metric. Where's the human in the loop?

<details markdown="1">
<summary><b>Suggested answer (6-stage walkthrough)</b></summary>

1. **Problem framing.** Multiple correctness criteria — yielded correctly, accepted gap was safe, acceleration was comfortable, stayed in lane, completed in reasonable time. Single-score collapses information the release manager needs; per-criterion + aggregated is right.
2. **Ground-truth source.** Human raters score per-criterion on a sampled set; that sample is the calibration set. Inter-rater agreement (Cohen's kappa or Krippendorff's alpha) measured before trusting it as ground truth.
3. **Method (the agentic chain).** (a) VLM perception extracts agents/lanes/signals from the clip. (b) Retrieval over similar past clips gives a behavioral baseline. (c) Structured CoT with per-criterion prompt templates. (d) Per-criterion score + aggregated scenario score with explicit weights.
4. **Eval-of-eval.** Per-criterion judge precision/recall vs. human labels on calibration. Watch judge-pair correlation across criteria — if all criteria correlate at >0.95, you've collapsed to a single dimension and the per-criterion structure adds no signal; simplify.
5. **Production loop.** Log the CoT trace alongside the score so debugging "why did this clip block release" is possible. Sample-based human review of release-blocking verdicts before the metric actually gates a deployment.
6. **Failure modes.** (a) VLM perception missing a subtle agent (motorcyclist in blind spot) → false-greenlight; mitigate with multi-frame, higher-res clip, redundant perception. (b) CoT criteria disagree on overall verdict → use disagreement itself as escalation. (c) Retrieval pool contaminated with the AV's own past behavior → judge becomes self-reinforcing; exclude same-model clips from retrieval.

</details>

**Q36.** **Inter-rater reliability on subjective driving labels** (e.g., "this lane change was uncomfortable"): which agreement metric, and what's the threshold to call a label trustworthy?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Categorical: Cohen's kappa (2 raters), Fleiss' kappa or Krippendorff's alpha (>2 raters).
- Ordinal (comfort 1–5): weighted kappa or ICC.
- Threshold: kappa > 0.6 substantial, > 0.8 strong. Under 0.6 means the labeling rubric needs work, not the raters.

</details>

**Q37.** **Block-or-greenlight metric design**: what statistical guarantees do you need before a learned metric can gate a software release? How do you communicate a probabilistic eval signal to release managers used to deterministic pass/fail?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Need: estimated false-positive rate (greenlights a regression) and false-negative rate (blocks a good release).
- Approach: confidence intervals on the metric using bootstrap; require lower-bound > threshold for greenlight.
- Communication: translate to release-manager language — "this metric, at this CI, says we are 95% confident regression is below X%."

</details>

**Q38.** **Video understanding for at-scale eval**: tradeoff between a frozen big VLM scoring per clip vs. a fine-tuned smaller model. What's the cost/latency/accuracy frontier?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Frozen big: zero training cost, broad capability, expensive inference, drift-resistant on prompt change.
- Fine-tuned small: cheap inference, narrow capability, requires labeled data + retraining cadence.
- Frontier: hybrid — small model for high-volume bulk; frozen big for sampling-based audit + edge-case scoring.

</details>

**Q39.** [CASE — 10–15 min] **Rule-based → learned eval transition**: how do you sunset rules without losing the safety floor they provide? What's the migration playbook?

<details markdown="1">
<summary><b>Suggested answer (6-stage walkthrough)</b></summary>

1. **Problem framing.** Rules: cheap, deterministic, known-incomplete (don't catch behaviors they weren't written for). Learned eval: expensive, probabilistic, known-broader. The migration must not lose the safety floor while extending coverage.
2. **Ground-truth source.** Rule-based eval on the historical fleet *is* the initial ground truth on rule-coverage scenarios — every rule-flagged event is a known positive, every non-flagged event in rule-coverage scope is a known negative. Human review supplies ground truth for non-rule-coverage scenarios.
3. **Method.** (a) Train learned eval to *match* rule-eval on rule-coverage scenarios — the consistency check. (b) Extend learned eval to non-rule-coverage scenarios — the value-add. (c) Validate non-rule-coverage extensions against human review on a stratified sample.
4. **Eval-of-eval.** Set thresholds: rule-coverage agreement > X% (target ~99%) before learned eval is trusted in rule-coverage cases. Non-rule-coverage agreement with humans > Y% (lower bar, ~90%, because human IRR caps the ceiling).
5. **Production loop.** Phased rollout — keep rules running in parallel as a "second opinion" for the first N months. Learned eval *only blocks releases* when both rule and learned eval agree on a regression, OR when learned eval flags a regression in non-rule-coverage scope. Disagreements escalate to human review. Sunset rules per-category as trust accrues, not all-at-once.
6. **Failure modes.** (a) Learned eval *overfits* to the rule pattern and doesn't generalize → measure on held-out non-rule-coverage scenarios, not just rule recovery. (b) Safety-critical rules sunsetted too early because aggregate agreement looked good but tail cases failed → sunset by category with explicit per-category trust thresholds. (c) Rule and learned eval drift apart over time → schedule periodic agreement audits tied to model versions.

</details>

---

## 10. Domain Curveballs

**Q40.** **AV scenario-based evaluation**: name the public-domain landmarks. What should you read?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Waymo's safety-case methodology and behavioral catalog.
- NHTSA behavioral taxonomies.
- Recent learned-planner literature: MotionLM, Wayformer, VAD.
- Public datasets: nuScenes, Waymo Open Motion Dataset.

</details>

**Q41.** **Define a "cut-in" precisely** enough that two annotators would agree. What are the edge cases?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Working definition: an adjacent-lane vehicle entering the ego lane within a TTC threshold (e.g., < 3 sec) ahead of ego, without sufficient gap.
- Edge cases: zipper merges (intentional and protocol-following — not really a cut-in?); slow-speed urban (TTC threshold needs adjustment); cyclist/scooter cut-ins (different hazard class).
- Real production taxonomies have 5+ subtypes — start from the team's existing definitions, don't invent your own.

</details>

**Q42.** **Custom agent vs. DSPy / LangChain / CrewAI** for an eval-of-eval workflow: what would you pick and why?

<details markdown="1">
<summary><b>Suggested answer</b></summary>

- Custom: full control over prompt templating, tool registry, retry/fallback logic.
- DSPy: treats the prompt as something to optimize against an eval metric — maps directly to learned-evaluation workflows. Strong fit when eval metrics drive the optimization.
- LangChain: ecosystem; con: abstraction overhead.
- CrewAI: multi-agent orchestration — overkill unless the eval workflow has multiple cooperating agents.
- For an eval-of-eval pipeline, DSPy is the natural starting point.

</details>

---

## Further Reading

References organized by section. All links verified at time of writing.

### Production inference (§1)

- [PyTorch 2.2 release blog — AOTInductor announcement](https://pytorch.org/blog/pytorch2-2/) — official launch post for AOT graph compilation; pairs with [the AOTInductor docs](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) for the API.
- [H100 Transformer Engine: FP8 explained](https://blogs.nvidia.com/blog/h100-transformer-engine/) — NVIDIA's high-level pitch on FP8; for the technical deep-dive see [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) and [Per-Tensor and Per-Block Scaling Strategies for FP8](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/).
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP'23. Foundation reading for autoscaling and KV-cache economics behind §1 Q4.

### LLM-as-a-Judge & evaluation methodology (§2, §4, §9)

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., NeurIPS'23. The canonical paper; covers position bias, verbosity bias, and self-enhancement bias — which is the "judge rubber-stamps the captioner" failure mode in Q7.
- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714) — Khattab et al., ICLR'24. The framework paper.
- [DSPy GitHub](https://github.com/stanfordnlp/dspy) and [dspy.ai docs](https://dspy.ai/) — implementation and current API.
- **DSPy applied — concrete cases:**
  - [Optimizing Databricks LLM Pipelines with DSPy (JetBlue case study)](https://www.databricks.com/blog/optimizing-databricks-llm-pipelines-dspy) — **the link to read first if §10 Q42 felt abstract.** Walks through how JetBlue replaced manual prompt tuning in a multi-stage RAG chatbot with DSPy + LLM-as-a-Judge metrics, and got 2× faster deployment than their prior LangChain pipeline.
  - [Is It Time To Treat Prompts As Code? A Multi-Use Case Study For Prompt Optimization Using DSPy](https://arxiv.org/html/2507.03620v1) — five applied use cases: guardrail enforcement, hallucination detection in code, code generation, routing agents, prompt evaluation.
  - [DSPy + MLflow for Automatically Optimizing LLM Programs](https://notebooks.databricks.com/devrel/mlflow/2024-11-27-dspy.html) — Databricks DevRel runnable notebook tying DSPy optimizers to MLflow tracking.

### Annotation pipelines & inter-rater reliability (§5)

- [The Kappa Statistic in Reliability Studies (NIH)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/) — clean Cohen's kappa primer with the Landis & Koch (1977) interpretation thresholds referenced in §9 Q36.
- [Cohen, Fleiss & Krippendorff: IAA Metrics & Implementation](https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability) — interactive tutorial with worked examples for each metric.
- [Understanding Krippendorff's Alpha (Encord)](https://encord.com/blog/interrater-reliability-krippendorffs-alpha/) — practical guide for when alpha beats kappa (ordinal labels, missing data, >2 raters).

### VLM fine-tuning & LoRA (§7)

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR'22. Reference for §7 Q19–Q20 on rank, target modules, learning rate behavior.
- [microsoft/LoRA](https://github.com/microsoft/LoRA) — original implementation.

### Sim-to-real, VLA models, and action chunking (§8)

- [Domain Randomization for Transferring DNNs from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — Tobin et al., IROS'17. The DR foundation paper for §8 Q29.
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705) — Zhao et al., RSS'23. Action Chunking Transformer; the chunking pattern referenced in §8 Q27.
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) — Chi et al., RSS'23. Receding-horizon control + diffusion action heads — the policy architecture pattern that GR00T N1's DiT chunked policy inherits.
- [Accelerate Generalist Humanoid Robot Development with NVIDIA Isaac GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/) — NVIDIA technical blog on the GR00T N1 model and synthetic-data blueprint that §8 builds on.
- [Cosmos World Foundation Models (NVIDIA)](https://blogs.nvidia.com/blog/cosmos-world-foundation-models/) and [Simplify End-to-End AV Development with Cosmos](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/) — the Cosmos Predict / Reason / Transfer stack referenced throughout §7 and §8.

### Uncertainty in policies (§8 Q31)

- [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) — Kendall & Gal, NIPS'17. The canonical aleatoric vs. epistemic decomposition for deep models. Worth reading for the framing alone — the "noise inherent in observations" vs. "uncertainty in the model that more data could explain away" distinction is exactly the split invoked in §8 Q31.
- [Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation](https://arxiv.org/html/2410.14868v1) — applied to chunked diffusion policies; shows how chunk-level diffusion variance can serve as an OOD signal, which is the operationalization §8 Q31 sketches.

### Autonomous-vehicle eval & planner literature (§10)

- [Waymo's Safety Case Blueprint (blog)](https://waymo.com/blog/2023/03/a-blueprint-for-av-safety-waymos/) — accessible entry point.
- [Building a Credible Case for Safety: Waymo's Approach](https://arxiv.org/abs/2306.01917) — the formal write-up; introduces the Case Credibility Assessment framework relevant to §9 Q37 (release-gating with statistical guarantees).
- [MotionLM: Multi-Agent Motion Forecasting as Language Modeling](https://arxiv.org/abs/2309.16534) — Seff et al., ICCV'23 (Waymo).
- [Wayformer: Motion Forecasting via Simple & Efficient Attention Networks](https://arxiv.org/abs/2207.05844) — Nayakanti et al.
- [VAD: Vectorized Scene Representation for Efficient Autonomous Driving](https://arxiv.org/abs/2303.12077) — Jiang et al., ICCV'23. End-to-end vectorized planner; pair with [VADv2](https://arxiv.org/abs/2402.13243) for the probabilistic planning extension.

---

## Closing Notes

If a question landed without your having a clean mental model for the answer, that's the signal — the fastest way to get unstuck is usually to write out your own version of the answer first, then compare. The point of the toggle isn't to give you the answer; it's to let you check your own.

The two CASE questions (Q33, Q35, Q39) are deliberately framed as 10–15 minute structured answers. If you're prepping for a senior-IC interview, practice them out loud against a timer — written answers feel different from spoken ones, and the case format rewards explicit stage-by-stage structure over free-form depth.
