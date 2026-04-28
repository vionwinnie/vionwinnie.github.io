---
title: "Video VLMs as Judges on Waymo's E2E Driving Set: A First-Principles Walkthrough"
category: engineering
excerpt: "Three off-the-shelf video VLMs as zero-shot scenario classifiers on Waymo's E2E driving val set. Two ways to measure their confidence — sample-and-count and single-pass logits — and why only one of the three is calibrated well enough to gate a human-rater queue."
tags: autonomous-driving vision-language-models evaluation uncertainty-quantification waymo
comments: true
math: true
---

**Target audience:** ML practitioners with general transformer/VLM background who want to know whether off-the-shelf video vision-language models can sit in front of a human-rater queue on autonomous-vehicle scenario data — and what predictive uncertainty actually buys you in that role.

---

## Table of Contents

1. [Why this question matters](#1-why-this-question-matters)
2. [The data: Waymo Open Dataset E2E challenge](#2-the-data-waymo-open-dataset-e2e-challenge)
3. [Stitching 8 cameras into one VLM input](#3-stitching-8-cameras-into-one-vlm-input)
4. [The judge setup — three video VLMs, one prompt](#4-the-judge-setup--three-video-vlms-one-prompt)
5. [Analysis 1: greedy accuracy and variation under sampling](#5-analysis-1-greedy-accuracy-and-variation-under-sampling)
6. [The TU / AU / EU framework — what each one measures](#6-the-tu--au--eu-framework--what-each-one-measures)
7. [Analysis 2: per-clip predictive entropy from single-pass logits](#7-analysis-2-per-clip-predictive-entropy-from-single-pass-logits)
8. [The triage funnel — VLM as a router for human raters](#8-the-triage-funnel--vlm-as-a-router-for-human-raters)
9. [What we still can't measure](#9-what-we-still-cant-measure)
10. [Key references](#10-key-references)

---

## 1. Why this question matters

Autonomous-vehicle perception teams generate driving log data faster than human annotators can label it. The standard pipeline looks like *raw clip* → *human rater* → *training set*, and the human is the bottleneck. A natural question: can a pretrained video VLM look at the clip first and decide *what kind of scenario* it shows — at minimum well enough to **route** clips to the right rater queue, or to **flag** the unusual ones for closer review?

This post walks through a small empirical study answering that question on Waymo's End-to-End driving val set. We test three off-the-shelf video VLMs as zero-shot scenario classifiers, then ask not just *are they accurate* but *do their confidence signals tell us anything useful*. The headline finding is that **only one of the three is calibrated well enough to be used as a triage signal**, and getting to that conclusion requires distinguishing several different notions of "uncertainty" — which the post unpacks from first principles.

---

## 2. The data: Waymo Open Dataset E2E challenge

Waymo's [End-to-End driving dataset](https://waymo.com/open/data/e2e/) (WOD-E2E) is the camera-only benchmark from the 2024 challenge. Each sequence is 8 seconds of synchronized 8-camera video at 10 Hz, with ego pose and a small set of derived labels.

The label we care about for this experiment is the **scenario cluster** — a sequence-level tag drawn from a 10-class taxonomy that Waymo published with the challenge:

| Cluster | What it captures |
|---|---|
| **Intersections** *(manifest typo: `Interections`)* | ego approaches / traverses an intersection — original typo preserved everywhere downstream so labels match the published manifest |
| **Foreign Object Debris** | something on the road that shouldn't be there |
| **Cyclist** | a cyclist is the safety-relevant agent |
| **Pedestrian** | a pedestrian is the safety-relevant agent |
| **Multi-Lane Maneuvers** | ego changes between two or more lanes |
| **Single-Lane Maneuvers** | within-lane behavior (slowing, stopping, smooth following) |
| **Special Vehicles** | emergency vehicle, school bus, etc. |
| **Cut_ins** | another vehicle cuts into ego's lane |
| **Construction** | construction zone present |
| **Others** | catch-all |

Two facts about these labels matter for what follows:

1. **Sequence-level, not frame-level.** A "Cyclist" label means *somewhere in the 8 seconds, the cyclist is the safety-relevant agent.* It does not mean the cyclist is visible in every frame.
2. **Sensor-suite-level, not single-camera-level.** A "Cyclist" label does not say *which camera* the cyclist is in. We verified empirically (by playing back individual sequences) that cyclist sequences often have the cyclist visible only in the rear cameras (CAM_7 and CAM_8) for most of the clip, entering the front camera briefly when ego passes them. This becomes important in the next section.

We sampled **5 clips per cluster × 10 clusters = 50 stratified clips** from the val set as our eval set.

---

## 3. Stitching 8 cameras into one VLM input

Most video VLMs accept a single video as input — not an arbitrary set of cameras. So we composite the 8 cameras into a single 2×4-tile video at 4 Hz, then hand that to the model:

![8-camera composite layout showing four forward-facing tiles in row 1 (FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_LEFT) and four mixed tiles in row 2 (SIDE_RIGHT, CAM_6, CAM_7 rear, CAM_8 rear), with a dashed cyclist trajectory traced from CAM_7 to CAM_8 to FRONT and a small ego-vehicle marker in the center.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/camera_layout.svg)

Why we did not just use the front camera: a "Cyclist" sequence is labeled as such because *across the full sensor suite, somewhere in the 8 seconds, a cyclist is the safety-relevant agent*. That cyclist often appears first in the rear cameras (overtaking from behind), then in the side cameras, and only briefly in the front camera at the moment ego passes. Front-only judging would systematically miss most of the trajectory and would never see the cyclist on the clips where ego never overtakes.

Below is one of those exact cases — a cyclist visible in CAM_7 / CAM_8 (bottom-right tiles) for most of the clip, only entering the front camera near the end:

<video src="{{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/cyclist_rear_to_front.mp4" controls width="700" preload="metadata"></video>

For comparison, here are three other clusters, all rendered as the same 8-camera composite:

**Intersections** — ego approaches an intersection:

<video src="{{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/intersection.mp4" controls width="700" preload="metadata"></video>

**Cut_ins** — another vehicle cuts into ego's lane:

<video src="{{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/cut_in.mp4" controls width="700" preload="metadata"></video>

**Foreign Object Debris** — object on the road:

<video src="{{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/foreign_object_debris.mp4" controls width="700" preload="metadata"></video>

These are the exact MP4 files passed into the VLMs in the experiments below.

---

## 4. The judge setup — three video VLMs, one prompt

We test three off-the-shelf, open-weights video VLMs:

| Judge | Backbone | Why we picked it |
|---|---|---|
| **Cosmos-Reason 2-2B** | Qwen3-VL | NVIDIA's reasoning-focused video VLM — native video input, small enough to run quickly |
| **Video-LLaVA-7B** | LanguageBind / Vicuna | Established baseline for video-language tasks |
| **Molmo2-8B** | Allen AI multimodal stack | Strong recent benchmark on video QA / grounding |

All three receive the same composite video and a prompt that lists the 10 clusters and asks for the dominant scenario. Ground truth is Waymo's published cluster label.

![Single-pass logit-extraction pipeline: composite 8-camera video flows into the VLM, the multi-choice prompt is appended, the model's logits at the answer position are restricted to the 10 letter-token IDs A-J, softmaxed into a 10-class distribution, then summarized as a per-clip predictive entropy H of p.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/judge_pipeline.svg)

### One methodology bug worth naming up front

The first version of our eval set printed the cluster name in the title bar of every frame (`Cyclist | seq 0fff5ea6 | frame 5/32`). Both Cosmos and Video-LLaVA were reading the answer off the input, producing artificially high accuracy (62% and 70% respectively). After re-rendering the eval set without the title-bar text and re-running, accuracy collapsed to **20% and 10%** — the leak was doing essentially all the work. The numbers reported below are all from the leak-free clean run; the leaked run is preserved as a forensic artifact.

This is the textbook eval-of-eval failure where the *measurement instrument contaminates the measurement*. The fix is straightforward (no signal correlated with the label appears in the input), but the right takeaway is general: when an off-the-shelf model gives suspiciously good zero-shot numbers on a domain it was not trained on, look for the leak first.

With the leak removed and a clean 50-clip eval set in hand, the next two sections walk through what each judge actually does on this data — first by looking at the answers themselves (Section 5), then by looking at how confident the model is in those answers (Sections 6 and 7).

---

## 5. Analysis 1: greedy accuracy and variation under sampling

### Greedy top-1 (one forward pass per clip, deterministic)

| Judge | Top-1 vs Waymo | Distinct cluster strings predicted | Failure mode |
|---|---|---|---|
| Cosmos-Reason 2-2B | **20%** (10/50) | 12 (case/spacing variants) | Defaults to `single_lane_maneuvers` on uncertain inputs |
| Video-LLaVA-7B | **10%** (5/50) | 1 — `Intersections` × 50 | Constant function — copies the prompt's example values |
| Molmo2-8B | **14%** (7/50) | 3 — Intersections (32), Multi-Lane (17), Cyclist (1) | Binary default; ignores 7 of 10 categories |

Random baseline for 10-class classification is 10%. Cosmos is barely above chance, Video-LLaVA *is* chance, Molmo collapses to a coarser binary than the taxonomy expects. In zero-shot terms, none of these models is usable as a labeling oracle on this dataset.

But the more interesting observation is that **all three fail differently**. They are not making the same mistake — Cosmos hallucinates scene reasoning, VL ignores the video and copies the prompt, Molmo bins everything into two classes. Three failures with no shared signal is qualitatively different from "three judges making correlated errors", and it shapes what the rest of the pipeline can do (more on this in Section 8).

### Variation across 10 runs at temperature 0.3

Greedy decoding gives one answer per clip. To see how much each judge wobbles when allowed to sample, we re-ran the same 50 clips with `temperature=0.3, do_sample=True`, **N=10 trials per clip**, and looked at how often the 10 answers agreed:

| Judge | Clips where all 10 trials agree (unanimous) | Most-common modal-vote prediction |
|---|---|---|
| Cosmos-Reason 2-2B | **31 / 50** | matches greedy answer on those 31 clips |
| Molmo2-8B | **31 / 50** | matches greedy answer on those 31 clips |
| Video-LLaVA-7B | **8 / 50** | wobbles substantially on the other 42 |

Cosmos and Molmo are deterministic-by-default even under sampling: on roughly two-thirds of clips they say the same thing 10 times in a row. Video-LLaVA is the opposite — it is the *most* variable judge under sampling, despite being the model that produced a perfectly constant `Intersections` answer under greedy decoding. Sampling exposes that VL has substantial probability mass on alternative tokens that greedy decoding hides; the constant-function behavior is an artifact of the argmax, not of the underlying distribution being narrow.

This is enough to motivate the next question. We have a notion of "the model wobbled across 10 trials," but it is sparse (binary "did it flip-flop or not" for most clips), and it does not distinguish *the model is genuinely uncertain about a hard scene* from *the model is undertrained and randomly guessing*. To separate those, we need real machinery.

---

## 6. The TU / AU / EU framework — what each one measures

When a probabilistic classifier hands us a distribution $p$ over classes, the scalar uncertainty in that distribution is its **Shannon entropy**, in bits:

$$
H(p) = -\sum_{i=1}^{K} p_i \log_2 p_i
$$

- $K$: number of classes (10 here)
- $p_i$: probability the classifier assigns to class $i$
- $H(p) \in [0, \log_2 K]$ — zero when one class has all the mass, $\log_2 K$ when the distribution is uniform

But "uncertainty" is not one thing. The standard decomposition (Houlsby et al. 2011 on BALD; Depeweg et al. 2018 on the AU/EU split) splits it into two qualitatively different sources:

$$
\underbrace{H(\bar{p})}_{\text{TU}}
\;=\;
\underbrace{\frac{1}{N}\sum_{i=1}^{N} H(p_i)}_{\text{AU}}
\;+\;
\underbrace{H(\bar{p}) \;-\; \frac{1}{N}\sum_{i=1}^{N} H(p_i)}_{\text{EU}}
$$

- $N$: number of stochastic forward passes (sampled completions, dropout masks, or ensemble members — whatever is producing the variability).
- $p_i$: the model's predictive distribution on trial $i$ (a 10-dim probability vector here).
- $\bar{p} = \frac{1}{N}\sum_{i=1}^{N} p_i$: the *mean* predictive distribution across the $N$ trials.
- $H(p_i)$: Shannon entropy of one trial's distribution, defined as in the equation above.
- **Total uncertainty (TU)** $= H(\bar{p})$. The entropy of the *averaged* distribution. It captures how much spread the predictive answers have *in aggregate*.
- **Aleatoric uncertainty (AU)** $= \frac{1}{N}\sum_i H(p_i)$. The *average* per-trial entropy. This is uncertainty that is intrinsic to the input — when even a single confident answer would have spread mass across multiple plausible classes, AU is high. AU is what you get when the *scene itself is genuinely ambiguous* (cyclist on the edge of a multi-lane maneuver — both labels are defensible).
- **Epistemic uncertainty (EU)** $= \text{TU} - \text{AU}$. By the math this is the **mutual information** between the prediction and the model's randomness. It captures *the model disagreeing with itself across trials* — different trials give different confident answers — which is the signature of a model that lacks the knowledge to commit. EU is what you get when *more training data would shrink the uncertainty*.

The decomposition is useful because the two sources demand different responses: high AU means *the labeler will probably disagree with themselves too — defer to consensus or accept that this clip is ambiguous*; high EU means *the model needs to learn more — collect more training data of this type*.

There is a crucial caveat in our setup. The decomposition only works when each $p_i$ is a *full distribution*, not a single sampled class. Section 7 explains why and what we did about it.

---

## 7. Analysis 2: per-clip predictive entropy from single-pass logits

There are two ways to get the $N$ stochastic predictions the decomposition needs. The first — *sample-and-count* — runs $N$ forward passes with `temperature > 0` and takes the sampled token from each. The second — *single-pass logits* — runs one forward pass with greedy decoding and reads the model's full softmax distribution at the answer position.

![Side-by-side comparison: sample-and-count (left, blue) shows 10 one-hot trial bars stacked vertically with their aggregate vote distribution and the formulas TU = H of p-bar, AU = expectation of H of p_i = 0 highlighted in red, EU = TU; single-pass logits (right, green) shows one continuous 10-class softmax distribution and the formula H(p) per clip.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/uq_methods_comparison.svg)

The two approaches are not interchangeable for the AU/EU split:

- In **sample-and-count**, each trial produces one *sampled* class. As a distribution that single answer is one-hot — `[0, 0, 1, 0, ..., 0]` — and the entropy of any one-hot distribution is zero. So $\mathbb{E}_i\!\left[H(p_i)\right] = 0$, $\text{AU} = 0$, and $\text{EU} = \text{TU}$ identically. We can measure TU but the decomposition collapses.
- In **single-pass logits**, we get the model's actual softmax distribution at the answer position from one forward pass. This is one full $p$ per clip — no $N$, no average. We can compute $H(p)$ directly as the per-clip predictive entropy, which is a real continuous signal.

Both methods give us *something*, but neither gives us a real AU/EU split from a single-method run. (To get that, you need $N$ forward passes *and* full per-trial distributions — for example, $N$ greedy passes with input perturbation.) For this study we ran the single-pass logit version because (a) it is roughly 16× cheaper in GPU time and (b) the per-clip $H(p)$ it produces is a denser, more usable signal than the sparse vote-distribution TU from $N=10$ sampling.

### How we extracted the logits

```python
out = model.generate(
    **inputs, max_new_tokens=1,
    output_scores=True, return_dict_in_generate=True,
    do_sample=False,                          # greedy
)
logits   = out.scores[0][0]                   # full vocab logits at answer position
class_logits = logits[letter_token_ids]       # restrict to A..J (the 10 cluster letters)
p        = torch.softmax(class_logits.float(), dim=0).cpu().numpy()
H        = -(p * np.log2(p + 1e-12)).sum()    # bits, ∈ [0, log2(10) ≈ 3.32]
```

The prompt is reformulated as multi-choice (each cluster gets a letter A-J, the model is asked to *answer with one letter*) so that the answer position is a single token. We restrict the vocab-sized logit vector to the 10 letter-token IDs and softmax to get a clean 10-class distribution.

### Headline numbers (50 clips × 3 judges, single-pass)

| Judge | Top-1 acc | Mean $H(p)$ | Mean top-class prob | $H(p)$ on correct | $H(p)$ on wrong | Escalation $\Delta$ |
|---|---|---|---|---|---|---|
| **Cosmos-Reason 2-2B** | 20% | 1.700 bits | 0.566 | 1.437 | 1.765 | **+0.329** ✓ informative |
| **Video-LLaVA-7B** | 10% | 0.374 bits | **0.949** | 0.349 | 0.377 | +0.027 ≈ noise |
| **Molmo2-8B** | 14% | 1.208 bits | 0.718 | 1.352 | 1.185 | **−0.167** ✗ anti-informative |

The **escalation signal $\Delta$** is `mean H(p) on wrong predictions − mean H(p) on correct predictions`. If positive, the model is more uncertain when it is wrong — useful as a "flag for human review" signal. If negative, the model is *more* confident when wrong — actively misleading as a confidence proxy.

### Per-cluster and distributional view

![Grouped bar chart showing mean H(p) per cluster for each of the three judges across all 10 Waymo clusters, with a dashed reference line at the maximum entropy log2 of 10 ≈ 3.32 bits. Cosmos and Molmo bars are noticeably taller across most clusters than Video-LLaVA, which sits near zero on most.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/results_per_cluster.png)

![Three-panel histogram showing H(p) split by correct (green) versus wrong (red) predictions for each judge. Cosmos shows wrong predictions clustering at higher entropy than correct ones (delta = +0.33 bits). Video-LLaVA shows almost all clips at low entropy regardless of correctness. Molmo shows correct predictions at higher entropy than wrong ones (delta = -0.17).]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/results_h_vs_correctness.png)

![Three-panel histogram of the per-judge H(p) distribution across all 50 clips. Cosmos has a broad distribution centered around 1.7 bits, Video-LLaVA is sharply concentrated near zero, Molmo is roughly uniformly spread between 0.5 and 2.0 bits.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/results_h_histogram.png)

![Reliability diagram (3 panels, one per judge) plotting top-class probability bin (x-axis) against empirical accuracy in that bin (y-axis), with a dashed perfect-calibration y = x diagonal. Cosmos sits below the diagonal across the prob range. Video-LLaVA's points are concentrated in the 0.9-1.0 confidence bin with empirical accuracy near 0.1 — extreme overconfidence. Molmo is bimodal with no monotonic relationship.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/results_calibration.png)

### Three observations worth knowing

**Cosmos's $H(p)$ is a real continuous escalation signal.** The +0.329 bit gap between wrong and correct predictions is comfortably above noise — wrong predictions sit on a noticeably wider distribution than correct ones (see middle panel above). And critically the $H(p)$ from single-pass logits is *continuous*: every clip gets a real-valued entropy, so production code can do `if H > 1.6: send to human` rather than the binary `did_it_flip_flop_under_sampling` you would get from a vote-distribution proxy on $N$ sampled trials. A Cosmos-based learned evaluator can use $H(p)$ as a graded confidence score that *actually orders clips by risk*.

**Video-LLaVA is severely overconfident.** Mean top-class probability is 0.95 with top-1 accuracy of 10%. The reliability diagram makes this concrete — VL's predictions live almost entirely in the 0.9-1.0 confidence bin, where the empirical accuracy is also ~10%. *Any* downstream pipeline that gated on top-class probability would over-trust this judge by an order of magnitude. The escalation signal $\Delta = +0.027$ is essentially noise — VL's wrong and correct predictions are equally low-entropy.

**Molmo's $H(p)$ goes the wrong way.** Mean $H(p)$ on correct clips (1.35) is *higher* than mean $H(p)$ on wrong clips (1.19). Two readings, both bad for production use: either the correct answers happen on hard clips where Molmo is correctly uncertain (and just gets lucky on the argmax), or the multi-choice letter-mapping interacts with Molmo's tokenizer in a way that distorts the softmax. Either way, Molmo's $H(p)$ cannot be used as an escalation signal — gating on `H > threshold` would systematically suppress correct answers.

The general lesson: **per-judge calibration must be measured, not assumed.** Three judges, three completely different relationships between predicted confidence and empirical accuracy. A learned-eval framework that batched these 3 models behind a generic "if confidence high, accept" rule would silently produce a bias-amplifying pipeline.

---

## 8. The triage funnel — VLM as a router for human raters

The point of running this experiment is not to replace human raters with VLMs — none of the three judges is accurate enough for that, and even if they were, the AV labeling problem is too high-stakes to hand to a 20%-accurate model. The point is to use the VLM as a **router** that splits the incoming clip stream into the right downstream queue:

![Triage funnel diagram: incoming clip stream (blue) flows into three VLM judges (orange center node), then fans out into three outcome routes — auto-bin in green for clips where all 3 judges agree and H(p) is low, human review in orange for clips where H(p) is high or judges disagree, and novel-scenario candidate in purple for clips where H(p) is high and the distribution is uniform-ish. A dashed callout below the central node notes that only Cosmos's H(p) is calibrated — per-judge gating is required.]({{ site.baseurl }}/assets/img/blog/vlm-judge-waymo/triage_funnel.svg)

The three branches map to three different downstream costs:

1. **Auto-bin** — low $H(p)$ AND all 3 judges agree → batch-accept the cluster label without human review. The cheapest route. From our 50-clip sample this would catch the easy intersections and the unambiguous cyclists, freeing rater time for the genuinely hard clips. (In our run, 0 of 50 clips had all-3-agree-and-correct, but that's a function of how bad these particular zero-shot judges are; a fine-tuned Cosmos would dramatically improve this.)
2. **Human review** — high $H(p)$ OR judges disagree → send to a rater. Standard cost. This is the case where the VLMs admit uncertainty (or contradict each other), which is exactly when human judgment is most valuable.
3. **Novel-scenario candidate** — high $H(p)$ AND none of the judges is confident AND the predicted distribution is roughly uniform across multiple non-default classes → flag as potentially out-of-taxonomy. This is the most interesting bucket. If the VLMs collectively give up — none of them lock onto a confident prediction *and* the class distribution looks like the model is groping — it is a hint that the clip might not fit any of the existing 10 categories cleanly. Routing those clips to a *taxonomy-review* queue (rather than to a regular rater) lets the dataset evolve.

**The calibration finding from Section 7 is the gating constraint on this design.** Only Cosmos's $H(p)$ is monotonic with correctness. Video-LLaVA's confidence is meaningless (95% confident at 10% accuracy), and Molmo's $H(p)$ goes the wrong way. So the practical funnel uses Cosmos's $H(p)$ as the primary uncertainty gate, with the other two judges contributing as cross-judge agreement checks rather than as confidence sources. *Routing rules cannot be uniform across judges* — they must be derived per judge from a calibration set.

---

## 9. What we still can't measure

Three known gaps in this study, in order of cost to fix:

**A real AU/EU split.** Single-pass logits give one $p$ per clip; sample-and-count with one-hot trials gives $\text{AU} = 0$. To measure both AU and EU meaningfully you need $N$ forward passes that each produce a *full* distribution. The natural way to generate that variability is input perturbation — for example $N$ different temporal samplings of the same clip, or $N$ paraphrases of the prompt — reading the full softmax from each pass. This is the natural next iteration if a downstream consumer specifically needs to distinguish "scene is genuinely ambiguous" (high AU) from "model doesn't know" (high EU). For the production triage use case in Section 8, the per-clip $H(p)$ from one greedy pass is already enough to rank clips by risk, so we did not run this round.

**Cross-judge ensemble calibration.** Section 8's design uses three judges as cross-checks, but we have not measured whether ensemble agreement (e.g. all-3-confidently-agree) is itself a calibrated signal. It probably isn't on this small sample — and given that all 3 judges fail differently, an ensemble may be no better calibrated than any single one. Worth running on a ≥500-clip set before deploying anything.

**No fine-tuning baseline.** All three judges are zero-shot. The natural comparison is *Cosmos with LoRA fine-tuning on a 200-clip Waymo-labeled subset* — based on similar literature, expect a 30-50 percentage-point lift on top-1 accuracy. The interesting question is whether fine-tuning *also* improves the calibration of $H(p)$, or whether the model just gets more confidently wrong.

---

## 10. Key references

| Year | Paper / Resource | Relevance |
|---|---|---|
| 2011 | Houlsby et al., [Bayesian Active Learning by Disagreement (BALD)](https://arxiv.org/abs/1112.5745) | The mutual-information formulation of epistemic uncertainty used in the TU = AU + EU decomposition |
| 2017 | Kendall & Gal, [What Uncertainties Do We Need in Bayesian Deep Learning?](https://arxiv.org/abs/1703.04977) | Canonical reference for the aleatoric / epistemic split in deep models |
| 2018 | Depeweg et al., [Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-Sensitive Learning](https://arxiv.org/abs/1710.07283) | The exact AU/EU decomposition we used |
| 2024 | Waymo, [End-to-End Driving Dataset / Challenge](https://waymo.com/open/data/e2e/) | The dataset and the 10-cluster taxonomy |
| 2024 | LanguageBind / Lin et al., [Video-LLaVA](https://arxiv.org/abs/2311.10122) | One of the three judges |
| 2024 | NVIDIA, [Cosmos-Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | One of the three judges (Qwen3-VL-based) |
| 2024 | Allen AI, [Molmo / Molmo2](https://huggingface.co/allenai/Molmo2-8B) | One of the three judges |
| 2017 | Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Background on reliability diagrams and the kind of post-hoc recalibration (Platt scaling, isotonic) that would be the next step for any of these three judges before production use |
