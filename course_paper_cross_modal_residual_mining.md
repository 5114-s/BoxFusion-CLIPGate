# Extending BoxFusion with Cross-Modal Residual Proposal Mining for Online Indoor Multi-View 3D Object Detection

**Author:** [Your Name]  
**Student ID:** [Your Student ID]  
**Course:** Data Mining  
**Affiliation:** [Your School and University]  
**Date:** August 2026

## Abstract

Online indoor 3D object detection is a fundamental perception capability for embodied robots, augmented-reality devices, and intelligent agents. Unlike offline methods that reconstruct a complete point cloud before inference, online systems must continuously integrate posed RGB-D observations under strict latency and memory constraints. A persistent problem is that BoxFusion-style online detectors can miss partially visible, small, or geometrically ambiguous objects, while naively appending proposals from a second detector introduces many false positives. This paper formulates the recovery of missed objects as a cross-modal residual proposal mining problem. We propose a training-free framework, Cross-Modal Residual Mining (CMRM), that first extracts 3D proposals not matched by a frozen BoxFusion-based multi-view detector, ranks them using metric-depth and free-space evidence together with multi-view DINOv3 appearance consistency, and then verifies the most promising candidates using cached promptable masks and registered RGB-D measurements. A conservative Top-5 intersection rule combines ranking and verification, and accepted boxes are appended below the minimum score of all frozen detections. We explicitly separate two evaluation layers. Under the class-agnostic protocol of the original BoxFusion paper, BoxFusion reports 37.46/31.36/13.41 AP at 3D IoU thresholds of 0.15/0.25/0.50, outperforming the reported online comparator OnlineAnySeg. Our strictly paired local extension uses a later frozen R3-active anchor containing 1,759 detections; this anchor obtains 41.4869/36.8917/23.2102 under the local frozen-anchor protocol. CMRM mines 12,549 residual candidates and accepts 170, increasing the paired result to 44.9866/40.0373/24.7644, or +3.4997/+3.1456/+1.5542 points. Values from the two protocols are reported separately and are never directly subtracted. A held-out 90-scene analysis shows that the final rule achieves candidate precision of 57.2% at IoU 0.25 and 36.6% at IoU 0.50. These results demonstrate that residual error analysis can be converted into a practical multimedia data-mining stage without retraining or altering the frozen anchor.

**Keywords:** data mining; multi-view 3D object detection; RGB-D; cross-modal fusion; residual proposal mining; foundation models; indoor scene understanding

## 1. Introduction

Indoor 3D object detection estimates the location, size, orientation, and semantic identity of objects such as chairs, tables, cabinets, displays, and doors. It is a central component of robotic navigation, manipulation, spatial question answering, and augmented reality. The problem is especially challenging for an embodied agent because observations arrive as a stream rather than as a complete scan. The same object may be strongly visible in one frame, truncated in another, and almost fully occluded in a third. Camera motion, depth holes, reflective surfaces, clutter, and pose error further distort the geometric evidence.

Traditional indoor detectors generally assume that the entire scene has already been reconstructed as a point cloud or voxel grid. VoteNet [3], FCAF3D [4], and 3DETR [5], for example, learn powerful representations from globally available 3D measurements. These methods are effective for benchmark evaluation, but complete reconstruction introduces delay and memory cost. It also conflicts with applications in which a robot needs an answer while it is still exploring. BoxFusion [1] addresses this gap with a reconstruction-free online pipeline: a single-view RGB-D detector predicts boxes in individual frames, the boxes are transformed into a common coordinate system, and multi-view association and optimization fuse repeated observations. The design is efficient and naturally supports open-vocabulary semantics through vision-language features.

Efficiency alone does not eliminate missed detections. A high-confidence threshold is usually required to prevent a long video from accumulating false positives. Consequently, objects receiving weak evidence in every individual view may never enter the global object memory. The discarded evidence is not necessarily random. Some rejected boxes correspond to real objects that are small, partly occluded, or observed at an unfavorable angle. A complementary 3D detector can recover such objects, but its unmatched output is also noisy. The key research question is therefore not simply how to generate more boxes, but how to mine a large residual pool and identify the small subset that adds true objects without damaging precision.

This question has a clear data-mining interpretation. Each unmatched proposal is an instance described by heterogeneous attributes: detector confidence, 3D location, size, depth support, free-space violation, image appearance, cross-view agreement, and mask consistency. The desired output is a binary decision indicating whether the proposal is a reliable novel object. The problem resembles classification and anomaly filtering, but labels are scarce and the evidence modalities have different failure patterns. It is also related to association-rule reasoning: a proposal is credible when several conditions co-occur, such as strong depth support, repeated appearance across views, and multiple consistent image masks.

This paper proposes Cross-Modal Residual Mining (CMRM), a training-free extension to a BoxFusion-based online multi-view 3D detection system. Original BoxFusion is the methodological foundation and the primary literature baseline. For the new residual module, however, the immediate experimental control is a later frozen R3-active prediction set. This distinction prevents an invalid subtraction between results produced by different label and scoring protocols. A complementary TR3D proposal branch supplies residual candidates that do not match the frozen anchor. CMRM ranks candidates using metric depth, free-space evidence, and DINOv3 features [7]. It then matches projected hypotheses to cached promptable segmentation masks inspired by the Segment Anything framework [8] and uses registered depth to test whether a 3D box is supported in multiple images. Only the intersection of the per-scene Top-5 ranked candidates and the mask-plus-depth verification set is appended. This conservative design is important because a permissive verifier can improve recall while simultaneously flooding the output with false positives.

The contributions are fourfold. First, we reinterpret missed-object recovery as residual proposal mining rather than as replacement of the original detector. Second, we introduce a cross-modal evidence model that joins 3D geometry, free-space constraints, self-supervised appearance, 2D masks, and temporal recurrence. Third, we use score-safe materialization: every accepted proposal is ranked below every frozen detection, preserving the original output and isolating the effect of added recall. Fourth, we report a reproducible 100-scene experiment and a held-out analysis that separates candidate precision, oracle novelty, and official detection AP. This separation makes both improvements and failure modes auditable.

Figure 1 summarizes the relationship between the original BoxFusion backbone, the frozen local R3 prediction set, and the proposed residual branch. Panels (a)--(c) provide a conceptual summary of BoxFusion. Panel (d) shows the auxiliary CMRM stage: TR3D proposals are filtered against the frozen R3 detections, ranked with cross-modal evidence, verified across views, and then merged without changing any frozen prediction.

![Figure 1. Schematic overview. Panels (a)--(c) summarize the original BoxFusion backbone; panel (d) shows CMRM as an auxiliary residual-mining stage applied to the frozen local R3 prediction set.](figures/course_paper/figure1_method_pipeline.png)

## 2. Related Work

### 2.1 Indoor 3D object detection

Point-cloud detectors learn object geometry directly from reconstructed scenes. VoteNet [3] uses deep Hough voting to move surface features toward possible object centers. FCAF3D [4] replaces hand-designed anchors with a fully convolutional sparse-voxel formulation, while 3DETR [5] uses transformer-based set prediction. TR3D [6] emphasizes a lightweight, practical, fully convolutional detector and can fuse RGB and 3D features. These methods benefit from a scene-level point cloud but normally operate after reconstruction. Their residual predictions are nevertheless useful in our setting because their geometric inductive bias differs from that of frame-based detection.

Cubify Anything [2] studies single-frame indoor 3D detection at scale and introduces a transformer that directly predicts 3D boxes from RGB or RGB-D features. Its frame-level inference is suitable for streaming systems. BoxFusion [1] uses this model as a proposal generator and fuses detections across posed views without constructing a dense scene model. Our work does not modify the released single-view predictor or the baseline association process. Instead, it investigates the residual objects missed after standard multi-view fusion.

### 2.2 Open-vocabulary and multi-view representation learning

CLIP [9] learns a shared image-text representation from large-scale natural-language supervision and enables zero-shot recognition. OpenScene [10] transfers image-language features to 3D points, and OpenMask3D [11] aggregates multi-view CLIP features for class-agnostic 3D instances. These studies show that multi-view appearance provides information beyond local geometry. In an online detector, however, semantics should not be allowed to create unsupported geometry. We therefore use appearance to rank or associate residual evidence, while mask and depth agreement determine whether a candidate can be materialized.

DINOv3 [7] provides high-quality self-supervised dense visual features. Compared with a category score, such features can express whether depth-supported regions from different viewpoints depict the same physical object even when the category is unknown. CMRM uses a frozen DINOv3-S/16+ encoder; no ScanNet validation labels are used to fine-tune it.

### 2.3 Promptable segmentation and cross-modal verification

Segment Anything [8] demonstrated that a promptable segmentation foundation model can produce high-quality object masks under zero-shot transfer. A projected 3D box supplies a natural image-space prompt. A mask alone, however, may select an adjacent object or a large background surface. Registered depth provides an independent physical test: pixels belonging to a true candidate should lie near the depth interval predicted by its 3D box. CMRM therefore treats masks and depth as complementary transactions. Repeated support across at least two views forms a high-confidence rule.

### 2.4 Residual mining as data mining

Most detector fusion methods combine all predictions and then apply non-maximum suppression. That strategy obscures which subsystem caused a change. Our formulation begins with the residual set relative to a frozen anchor. It then applies ranking, association, and rule-based classification only to unmatched candidates. This is similar to hard-example mining, except that the objective is to recover true false negatives rather than to retrain a network. It is also related to anomaly detection: candidates that strongly violate observed free space or show unstable cross-view features are treated as anomalous hypotheses. The final pipeline converts a large, weakly structured pool into a small, interpretable set of actionable patterns.

## 3. Methodology

### 3.1 Problem formulation

Let a posed RGB-D stream be defined as

\[
\mathcal{S}=\{(I_t,D_t,K_t,T_t)\}_{t=1}^{T}. \tag{1}
\]

where \(I_t\) is an RGB image, \(D_t\) is its depth map, \(K_t\) is the camera intrinsic matrix, and \(T_t\) maps camera coordinates to a world coordinate system. The frozen online detector produces an anchor set

\[
\mathcal{A}=\{(b_i,s_i,e_i)\}_{i=1}^{N}. \tag{2}
\]

where \(b_i\) is a 3D bounding box, \(s_i\) is confidence, and \(e_i\) is an optional semantic embedding. A complementary detector produces proposals \(\mathcal{P}\). We remove proposals geometrically matched to \(\mathcal{A}\), yielding the residual pool

\[
\mathcal{R}=\left\{p\in\mathcal{P}: \max_{a\in\mathcal{A}} \operatorname{IoU}_{\mathrm{AABB}}(p,a)\le 0.15\right\}. \tag{3}
\]

Equations (1)--(3) define the streaming observations, frozen detections, and unmatched residual pool, respectively. The residual test is class agnostic and uses axis-aligned 3D IoU in the common world frame. The goal is to construct a selector \(g(p,\mathcal{S})\in\{0,1\}\) that maximizes additional true-positive coverage while keeping false-positive growth small. During inference, \(g\) cannot access ground truth. Ground truth is used only for evaluation and retrospective analysis.

### 3.2 Frozen anchor and residual generation

The anchor is the existing online multi-view prediction tree. Freezing it serves two purposes. First, the main detector remains deployable even if the residual branch is disabled. Second, paired comparison becomes exact: every original row and score can be checked before and after the extension.

TR3D is used as a complementary proposal source because its sparse 3D convolutional representation differs from frame-level box regression. Its predictions are transformed to the same world frame. Equation (3) removes every proposal whose maximum class-agnostic AABB IoU with the frozen anchor exceeds 0.15. This step produced 12,549 unmatched candidates over the 100 evaluation scenes. The large ratio between residual candidates and accepted boxes illustrates the mining difficulty: simply appending all candidates would be unacceptable.

### 3.3 Stage 1: geometry and appearance ranking

For each residual proposal \(p\), the R2a observer selects up to five valid views with the largest projected box areas. At pixel stride four, every sampled depth ray is classified as box support (\(S\)), occluded (\(O\)), free-space contradiction (\(F\)), or invalid (\(I\)). Let \(n_t^k\) be the count of class \(k\) in view \(t\), \(n_t=\sum_k n_t^k\), \(v_t\) indicate a valid projection, and \(\rho_t^k=n_t^k/n_t\). Supportive and contradictory views are defined explicitly as

\[
\begin{aligned}
u_t(p)&=\mathbb{1}\!\left[v_t=1\land n_t\ge16\land \rho_t^S\ge0.10\land\rho_t^F\le0.50\right],\\
c_t(p)&=\mathbb{1}\!\left[v_t=1\land n_t\ge16\land \rho_t^F>0.50\land\rho_t^F>\rho_t^S\right].
\end{aligned} \tag{4}
\]

Equation (4) fixes the per-view decisions without learned thresholds. Let \(V=\sum_t v_t\), \(U=\sum_t u_t\), and \(C=\sum_t c_t\). The aggregate fractions \(\rho^S,\rho^F,\rho^I\) are computed after pooling the corresponding ray counts across selected views. The three depth reliability factors are

\[
\begin{aligned}
q_{con}(p)&=\frac{U+1}{V+2},\\
q_{ctr}(p)&=\begin{cases}C/V,&V>0,\\0,&V=0,\end{cases}\\
q_{dep}(p)&=\frac{\rho^S}{\rho^S+\rho^F}(1-\rho^I).
\end{aligned} \tag{5}
\]

A zero denominator in \(q_{dep}\) returns zero. This smoothed formulation rewards repeated support, penalizes observed free space, and remains well defined for sparsely observed proposals.

For appearance consistency, support rays are projected into the RGB image and mapped to unique cells of a frozen DINOv3-S/16+ dense feature map. The selected cell features are averaged and \(\ell_2\)-normalized to form \(z_t(p)\). If \(m\) valid feature views are available, CMRM uses the pairwise cosine *mean*, not the median:

\[
\begin{aligned}
\bar c(p)&=\frac{2}{m(m-1)}\sum_{t<u}z_t(p)^\top z_u(p),\\
q_{app}(p)&=\begin{cases}\operatorname{clip}((\bar c(p)+1)/2,0,1),&m\ge2,\\0.5,&m<2.\end{cases}
\end{aligned} \tag{6}
\]

Let \(s_d(p)\) be the frozen TR3D confidence. Equations (5) and (6) enter the exact deterministic ranking score

\[
\begin{aligned}
r_{depth}(p)&=s_d(p)(0.5+0.5q_{con})(1-0.5q_{ctr})(0.5+0.5q_{dep}),\\
r_{C1}(p)&=r_{depth}(p)(0.75+0.25q_{app}).
\end{aligned} \tag{7}
\]

There is no candidate-pool normalization or learned coefficient in Eq. (7). Candidates are sorted stably by \(r_{C1}\) within each scene, and the Top-10 are forwarded to Stage 2, leaving 1,000 source candidates. Temporal span and center/size stability are logged for audit but are not used by the reported route. In oracle analysis, this pool contains 209, 194, and 106 novel matches at IoU 0.15, 0.25, and 0.50.

### 3.4 Stage 2: mask and depth verification

Each Top-10 candidate is projected into scheduled RGB frames and matched class agnostically against precomputed promptable-mask proposals; the reported run does not issue a new rectangular prompt for every residual. For projected box footprint \(B_t\) and mask \(M_t\), define mask--box IoU \(J_t\), mask containment \(C_t^M\), and box coverage \(C_t^B\). A mask is eligible when

\[
\begin{aligned}
J_t&=\frac{|M_t\cap B_t|}{|M_t\cup B_t|},\quad C_t^M=\frac{|M_t\cap B_t|}{|M_t|},\quad C_t^B=\frac{|M_t\cap B_t|}{|B_t|},\\
M_t\ \text{eligible}&\Longleftrightarrow J_t\ge0.02\ \lor\ (C_t^M\ge0.10\land C_t^B\ge0.10).
\end{aligned} \tag{8}
\]

Among masks admitted by Eq. (8), the deterministic evidence score selects one mask per view:

\[
\begin{aligned}
E_t={}&0.15(s_t^M+J_t+C_t^M+C_t^B+f_t^{exp})+0.10f_t^{in}\\
&+0.05\left(f_t^{cc}+\min(n_t^D/24,1)+\min(n_t^{cc}/16,1)\right).
\end{aligned} \tag{9}
\]

The eligible mask with maximum Eq. (9) is retained. Here \(s_t^M\) is mask confidence, \(n_t^D\) is the valid mask-depth pixel count, and \(f_t^{exp}\) is the fraction of backprojected points inside a 1.25-times expanded 3D box. Points in that box are voxelized at 0.05 m with 26-neighbor connectivity. The best box-aligned component has \(n_t^{cc}\) points, original-box fraction \(f_t^{in}\), and expanded-support fraction \(f_t^{cc}\). A projection provides strong support only when

\[
h_t(p)=\mathbb{1}\!\left[A_t\ge25\land s_t^M\ge0.50\land C_t^M\ge0.10\land C_t^B\ge0.10\land n_t^D\ge24\land f_t^{exp}\ge0.15\land n_t^{cc}\ge16\land f_t^{in}\ge0.20\right]. \tag{10}
\]

In Eq. (10), \(A_t\) is projected area in pixels. Let \(H=\sum_t h_t\), \(N_{cc}=\sum_t h_tn_t^{cc}\), and let \(\bar f^{exp}\) be the mean expanded-box support over strong views. The complete mask-depth gate is

\[
\begin{aligned}
\bar f^{exp}(p)&=\begin{cases}H^{-1}\sum_t h_t f_t^{exp},&H>0,\\0,&H=0,\end{cases}\\
g_{md}(p)&=\mathbb{1}\!\left[H\ge2\land N_{cc}\ge64\land\bar f^{exp}\ge0.25\right].
\end{aligned} \tag{11}
\]

Equation (11) is stricter than a two-view count alone: it also requires 64 component points across strong views and mean expanded-box support of at least 0.25. It is an association rule across transactions: “mask agreement AND metric-depth structure in at least two views implies candidate validity.”

The initial mask-plus-depth route accepts 294 candidates over all 100 scenes. Although it is much cleaner than the 1,000-candidate pool, held-out precision remains slightly below the pre-registered targets. We therefore report it transparently and analyze a stricter rule rather than presenting it as a successful final classifier.

**Table 1. Fixed inference parameters for the reported CMRM route. Diagnostic-only gates are excluded.**

| Stage | Parameter group | Fixed value |
|---|---|---|
| Residual generation | Class-agnostic anchor matching | maximum AABB IoU ≤ 0.15 |
| Stage 1 depth | Selected views; sampling; ray margin | Top-5 by area; stride 4; 0.05 m |
| Stage 1 depth | Valid depth; samples per view | 0.10–8.00 m; at least 16 |
| Stage 1 view rules | Supportive; contradictory free space | support ≥ 0.10, free ≤ 0.50; free > 0.50 and > support |
| Stage 1 appearance | Encoder; input; missing-pair prior | DINOv3-S/16+; 960 × 960; 0.50 |
| Stage 1 output | Per-scene verification budget | Top-10 by \(r_{C1}\) |
| Stage 2 projection | Area; initial mask match | ≥ 25 px; IoU ≥ 0.02 or containment/coverage ≥ 0.10 |
| Stage 2 mask/depth | Mask score; valid depth pixels | ≥ 0.50; ≥ 24 in 0.10–8.00 m |
| Stage 2 local support | Expansion; voxel; per-view component | 1.25×; 0.05 m; \(n^{cc}\ge16\), \(f^{in}\ge0.20\) |
| Stage 2 final gate | Strong views; total points; mean support | \(H\ge2\); \(N_{cc}\ge64\); \(\bar f^{exp}\ge0.25\) |
| Materialization | Per-scene final rank budget | rank ≤ 5 and \(g_{md}=1\) |

### 3.5 Conservative intersection and score-safe materialization

The final selection intersects mask-depth verification with the five highest-ranked candidates in each scene:

\[
g(p)=g_{md}(p)\land \mathbb{1}[\operatorname{rank}_{scene}(p)\le 5]. \tag{12}
\]

The gate in Eq. (12) accepts 170 candidates. It combines two partly independent views of reliability: Stage 1 favors strong geometric and appearance evidence, while Stage 2 requires direct pixel-level confirmation.

Accepted candidates are appended to the anchor set without editing anchor coordinates, classes, or scores. Let \(s_{min}=\min_{a\in\mathcal{A}}s(a)\). New candidate scores are positive, preserve their Stage-1 order, and satisfy

\[
0<s(p)<s_{min}\quad \forall p\in\mathcal{C}. \tag{13}
\]

Therefore, by Eq. (13), all anchor detections remain before all residual detections in the global ranking. This score-safe policy prevents new hypotheses from suppressing or reordering strong original predictions. It also gives a clean interpretation of any AP change: the system gains low-score recall after exhausting the frozen anchor list.

## 4. Experimental Design

### 4.1 Dataset and protocol

ScanNetV2 contains 2.5 million RGB-D views from 1,513 indoor scenes with camera poses, reconstructed surfaces, and semantic annotations [12]. Both the BoxFusion paper and the local repository experiment use 100 validation scenes, but numerical comparability requires more than an equal scene count. The original paper reports class-agnostic AP after converting all methods to axis-aligned boxes, whereas the local experimental lineage evaluates its frozen R3-active tree with its stored predictions and evaluator configuration. Input to the online branch consists of posed RGB-D frames; the residual branch and mask-depth verifier use the same registered observations.

To reduce development leakage, ten evenly spaced scenes form a development partition and the remaining 90 scenes form a held-out analysis partition. The official headline result is reported on the fixed 100-scene protocol because it corresponds to the project’s established evaluation, while selection behavior is also shown separately on heldout90. No model is trained on validation ground truth. Ground truth is accessed only by the evaluation scripts and oracle analysis.

### 4.2 Separation of reported and paired baselines

We use two explicitly labeled layers of evidence:

1. **Reported literature baseline.** Values are transcribed from Table 1 of the BoxFusion paper. They establish that original BoxFusion is a strong reconstruction-free online method under the authors’ class-agnostic ScanNetV2 protocol. They are not used to calculate the gain of CMRM.
2. **Paired experimental baseline.** The frozen R3-active prediction tree is the exact input to the residual materialization experiment. Its 1,759 rows remain unchanged when 170 CMRM candidates are appended. Only the difference between this anchor and its CMRM output is interpreted as the effect of the proposed module.

Table 2 reproduces the relevant online comparison reported in the original paper. BoxFusion exceeds OnlineAnySeg by 6.07, 9.55, and 3.49 AP points at IoU 0.15, 0.25, and 0.50, respectively. These numbers motivate BoxFusion as the research foundation, while the later paired experiment tests whether residual mining can further improve a frozen BoxFusion-based system.

**Table 2. Results reported by the original BoxFusion paper on 100 ScanNetV2 validation scenes under its class-agnostic protocol.**

| Method | Online | Open vocabulary | AP@0.15 | AP@0.25 | AP@0.50 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| OnlineAnySeg | Yes | Yes | 31.39 | 21.81 | 9.92 | 15 |
| Original BoxFusion | Yes | Yes | **37.46** | **31.36** | **13.41** | **20** |

### 4.3 Metrics

The primary metric is mean average precision at 3D IoU thresholds 0.15, 0.25, and 0.50. Lower thresholds measure object discovery and coarse localization, while IoU 0.50 places greater emphasis on accurate box geometry. We additionally report candidate hit precision

\[
P(hit@\tau)=\frac{|\{p:\max_j IoU_{3D}(p,b_j^{gt})\ge\tau\}|}{|\mathcal{C}|}. \tag{14}
\]

Equation (14) measures independent candidate hit precision. We also report novel true positives, defined as ground-truth objects matched by a residual candidate but not by the frozen anchor under the same IoU threshold. This distinction matters because an apparently accurate candidate may duplicate an object already detected by the baseline.

### 4.4 Reproducibility and audit controls

The experiment uses a frozen scene list, frozen anchor predictions, and immutable cached evidence. Inference-side ground-truth and CLIP access are disabled in the residual confirmation branch. The active output contains the original 1,759 anchor rows followed by 170 appended rows. Loading-based and byte-level audits verify that all original rows remain unchanged and that the residual scores lie below the anchor score floor. The prediction tree hash is checked before and after observer-only experiments. The unmodified official evaluator is then applied to both trees.

These controls are more than engineering details. Data-mining systems can show illusory gains when evaluation labels leak into feature construction, when a new module silently changes baseline scores, or when unmatched predictions are evaluated with a different script. A frozen and paired protocol makes such errors detectable.

## 5. Results and Analysis

### 5.1 Protocol-aware interpretation

Figure 2 deliberately uses different chart forms to separate the literature comparison from the local paired experiment. The left panel reports class-agnostic AP under the BoxFusion paper’s protocol. The right panel uses a paired dumbbell plot for local mAP, directly connecting each frozen R3 value to its CMRM result. The higher numerical values in the right panel do not prove that R3 is better than original BoxFusion because the metrics, stored predictions, semantic handling, and evaluation lineage differ. Only within-panel comparisons are valid.

![Figure 2. Detection results under two distinct protocols. Left: literature-reported class-agnostic AP under the BoxFusion protocol. Right: paired local mAP for the frozen R3 set and CMRM. Cross-panel numerical comparison is invalid.](figures/course_paper/figure2_protocol_results.png)

### 5.2 Candidate filtering behavior

Table 3 shows how progressively stronger rules change the candidate pool. The source Top-10 route keeps 1,000 of 12,549 residuals. On all 100 scenes, only 23.9% of these candidates hit a ground-truth object at IoU 0.25 and 10.9% do so at IoU 0.50. Mask-plus-depth confirmation reduces the pool to 294 and approximately doubles precision. The final Top-5 intersection retains 170 boxes and raises precision to 60.0% at IoU 0.25 and 38.8% at IoU 0.50.

**Table 3. Residual candidate quality under different selection rules.**

| Partition and rule | Candidates | P(hit)@0.25 | P(hit)@0.50 | Novel TP@0.25 | Novel TP@0.50 |
|---|---:|---:|---:|---:|---:|
| All100, source Top-10 | 1,000 | 23.9% | 10.9% | 194 | 106 |
| All100, mask2 + depth | 294 | 46.9% | 24.8% | 113 | 70 |
| All100, Top-5 ∩ mask2 + depth | 170 | **60.0%** | **38.8%** | 97 | 65 |
| Heldout90, source Top-10 | 900 | 22.9% | 10.0% | 169 | 87 |
| Heldout90, mask2 + depth | 251 | 45.0% | 23.5% | 94 | 56 |
| Heldout90, Top-5 ∩ mask2 + depth | 145 | **57.2%** | **36.6%** | 79 | 52 |

The held-out trend closely follows the all-scene trend, suggesting that the stricter intersection is not benefiting only from the ten development scenes. At IoU 0.25, held-out precision rises by 34.3 percentage points relative to the source pool. At IoU 0.50, it rises by 26.6 points. The retained count decreases by 83.9%, from 900 to 145, yet 52 of the 87 available novel IoU-0.50 matches remain. Thus, the filter discards most noise while retaining about 59.8% of the strict-IoU recovery opportunity present in the Top-10 pool.

Figure 3 shows the complete selection funnel. The per-scene Top-10 stage retains 7.97% of the raw pool; two-view mask-and-depth verification retains 29.4% of that ranked set; and the final Top-5 intersection retains 57.8% of the verified set. Overall, only 1.35% of the 12,549 unmatched residuals are materialized, meaning that 98.65% are rejected. This steep reduction illustrates why residual recovery is fundamentally a data-mining problem rather than a simple detector union.

![Figure 3. Candidate mining funnel from 12,549 unmatched residuals to 170 verified proposals, with stage-wise retention and candidate hit precision.](figures/course_paper/figure3_candidate_funnel.png)

The mask2-plus-depth rule deserves careful interpretation. The pre-registered held-out target required precision of at least 50% at IoU 0.25 and 25% at IoU 0.50. Its measured values, 45.0% and 23.5%, fail both targets. Reporting only the all100 numbers would hide this failure. The stricter Top-5 intersection is a subsequent diagnostic rule and should be described as such. This distinction strengthens the experimental analysis because it separates confirmatory and exploratory conclusions.

### 5.3 Paired detection accuracy

Table 4 compares the frozen R3-active anchor with the score-safe CMRM output under the same local evaluator. These are the only rows from which a CMRM improvement is calculated.

**Table 4. Paired ScanNet-100 detection results under the local frozen-anchor protocol.**

| Method | mAP@0.15 | mAP@0.25 | mAP@0.50 |
|---|---:|---:|---:|
| Frozen R3-active anchor (BoxFusion-based) | 41.4869 | 36.8917 | 23.2102 |
| Frozen R3-active anchor + CMRM | **44.9866** | **40.0373** | **24.7644** |
| Absolute improvement | **+3.4997** | **+3.1456** | **+1.5542** |
| Relative improvement | +8.44% | +8.53% | +6.70% |

Within this paired protocol, CMRM improves all three thresholds. The largest absolute gain occurs at IoU 0.15, followed closely by IoU 0.25. This pattern is expected for a residual recovery module: it primarily finds objects absent from the anchor rather than refining every original box. The gain of 1.5542 at IoU 0.50 remains meaningful because it shows that a subset of recovered proposals also has sufficiently accurate geometry. This statement must not be rewritten as “CMRM improves original BoxFusion by 7.53 AP@0.15,” because 44.9866 and the paper-reported 37.46 belong to different protocols.

The relative improvement at IoU 0.25 is 8.53%, slightly larger than at the other thresholds. Indoor applications often use IoU 0.25 because partial observations make strict box fitting difficult. Under this operating point, 97 of the selected all100 candidates represent novel oracle matches, and their low-score insertion increases integrated precision-recall performance by 3.1456 points.

Because the 1,759 baseline boxes are unchanged, the improvement cannot be attributed to re-scoring the anchor, favorable non-maximum suppression, or coordinate refinement. Materialization from cached evidence takes 0.997 seconds, although this number is not end-to-end runtime: TR3D, DINOv3, and mask evidence had already been computed. It should therefore not be presented as real-time system latency.

### 5.4 Ranking analysis and oracle comparison

To determine whether better scoring could yield much larger gains, we compare three low-score ordering policies on heldout90. Fixed-low append produces 44.1668/38.8692/23.7779 AP. Ordering the same candidates by Stage-1 evidence gives 44.2210/38.9197/23.8068. An oracle ordering using ground-truth overlap gives 44.3354/39.0286/23.8584. Relative to the heldout anchor of 41.1340/36.1489/22.4782, these correspond to gains of 3.0328/2.7203/1.2997, 3.0870/2.7708/1.3286, and 3.2014/2.8797/1.3802.

The gap between Stage-1 ranking and oracle ordering is only 0.1144, 0.1089, and 0.0516 AP points at the three thresholds. Therefore, ranking within the selected set is not the principal bottleneck. More sophisticated score learning might produce a small improvement, but the larger opportunity lies in proposal geometry and coverage. Some true objects receive no usable proposal; others receive a proposal whose box is too inaccurate to pass IoU 0.50. Future research should focus on residual box refinement, view selection, and proposal generation rather than merely replacing the deterministic rank with a complex classifier.

### 5.5 Why cross-modal mining works

The results reflect complementarity among modalities. A 3D detector score captures learned objectness but may be high on clutter. Free-space evidence rejects hypotheses occupying regions that RGB-D rays have already observed as empty. Depth support confirms physical surfaces but may accept walls or tabletops. DINOv3 consistency identifies recurring visual entities but does not guarantee correct metric extent. Promptable masks delineate image objects but sometimes leak into adjacent instances. Requiring agreement among these signals reduces the chance that one failure mode dominates.

The Top-5 constraint also has a useful scene-level interpretation. A long video can generate many residuals, so independent candidate thresholds do not control the number of false positives accumulated per room. A rank budget imposes a simple prior that only a few baseline misses should be recovered from each scene. This prior may be imperfect for unusually crowded rooms, but it stabilizes precision in the present evaluation.

### 5.6 Failure cases and limitations

First, the method cannot recover an object if the complementary detector never proposes it. This explains why candidate coverage, rather than ranking, remains the main limitation. Second, projected masks can select a neighboring object when 3D calibration is imperfect or two objects overlap in the image. Multi-view confirmation reduces but does not eliminate this error. Third, reflective and transparent surfaces produce unreliable depth, making the depth rule overly conservative for mirrors, glass tables, and displays.

Fourth, the Top-5 scene budget is hand-designed. It may reject valid objects in large or cluttered environments and accept too many in nearly empty rooms. An adaptive budget based on scene size, explored volume, or calibrated false-discovery rate would be preferable. Fifth, the headline 100-scene active replay uses a rule identified after the initial mask-depth gate failed its held-out target. The result is a valid engineering measurement but not an independently pre-registered confirmation. A future study should freeze the Top-5 intersection and evaluate it on a new dataset or untouched test partition.

Finally, cached materialization time does not measure online latency. End-to-end deployment must include complementary proposal generation, DINOv3 feature extraction, mask inference, and data transfer. Efficient scheduling could run expensive verification only when the robot is stationary or when uncertainty is high.

## 6. Innovation, Practical Value, and Future Work

The principal innovation is architectural rather than a new backbone: missed detections are treated as a mineable residual database with explicit provenance. This differs from ordinary ensemble fusion, which mixes predictions from multiple detectors and makes attribution difficult. The frozen-anchor design supports rollback, paired evaluation, and safe deployment.

The second innovation is the separation of semantic ranking from geometric authorization. Foundation-model appearance features help decide which hypotheses deserve attention, but they cannot create a 3D object by themselves. Pixel masks and metric depth provide the evidence required for output. This division is particularly valuable in open-vocabulary systems, where semantic similarity can be strong even for an incorrectly localized region.

The method has realistic applications. A domestic robot can recover a small waste bin seen only briefly; an augmented-reality headset can maintain stable object anchors while a user scans a room; and a warehouse assistant can identify long-tail objects without reconstructing an entire building. Since the baseline output is preserved, the residual branch can be enabled selectively when additional compute is available.

Several extensions are promising. A calibrated classifier could replace the hard intersection after collecting training data from ScanNet training scenes rather than validation scenes. Its inputs should retain interpretable modality groups, and scene-disjoint training should prevent leakage. Box refinement could optimize center and dimensions against multi-view depth and mask boundaries. Association rules could be learned with a false-discovery-rate objective instead of fixed thresholds. Finally, active perception could direct the camera toward residual hypotheses whose uncertainty would be most reduced by another view, turning passive data mining into a closed-loop exploration strategy.

## 7. Conclusion

This paper studied residual proposal mining for online indoor multi-view 3D object detection. Original BoxFusion, which reports 37.46/31.36/13.41 class-agnostic AP on its ScanNetV2 protocol, provides the reconstruction-free methodological foundation. The proposed CMRM extension preserves a later frozen R3-active anchor and analyzes only the unmatched output of a complementary 3D proposal branch. It combines metric-depth and free-space evidence, DINOv3 cross-view appearance, cached promptable masks, and repeated RGB-D support. A conservative Top-5 intersection converts 12,549 raw residuals into 170 accepted boxes. In the separate, strictly paired frozen-anchor protocol, CMRM improves mAP from 41.4869/36.8917/23.2102 to 44.9866/40.0373/24.7644 at IoU 0.15/0.25/0.50. The reported BoxFusion values and local paired values are not directly subtracted. The held-out selection precision and oracle-ordering analysis show both the value and the limits of the approach: cross-modal verification removes substantial noise, while future gains depend mainly on better proposal coverage and geometry. More broadly, the study demonstrates that careful data-mining concepts—residual analysis, heterogeneous feature fusion, association rules, conservative classification, and auditable evaluation—can produce practical improvements in embodied 3D perception without retraining the frozen detector.

## References

[1] Y. Lan, C. Zhu, Z. Gao, J. Zhang, Y. Cao, R. Yi, Y. Wang, and K. Xu, “BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion,” *Computer Graphics Forum*, vol. 44, no. 7, e70254, 2025.

[2] J. Lazarow, D. Griffiths, G. Kohavi, F. Crespo, and A. Dehghan, “Cubify Anything: Scaling Indoor 3D Object Detection,” arXiv:2412.04458, 2024.

[3] C. R. Qi, O. Litany, K. He, and L. J. Guibas, “Deep Hough Voting for 3D Object Detection in Point Clouds,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 9277–9286.

[4] D. Rukhovich, A. Vorontsova, and A. Konushin, “FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection,” in *Proceedings of the European Conference on Computer Vision (ECCV)*, 2022.

[5] I. Misra, R. Girdhar, and A. Joulin, “An End-to-End Transformer Model for 3D Object Detection,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 2906–2917.

[6] D. Rukhovich, A. Vorontsova, and A. Konushin, “TR3D: Towards Real-Time Indoor 3D Object Detection,” arXiv:2302.02858, 2023.

[7] O. Siméoni et al., “DINOv3,” *arXiv preprint arXiv:2508.10104*, 2025.

[8] A. Kirillov et al., “Segment Anything,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 4015–4026.

[9] A. Radford et al., “Learning Transferable Visual Models from Natural Language Supervision,” in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021, pp. 8748–8763.

[10] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, “OpenScene: 3D Scene Understanding with Open Vocabularies,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.

[11] A. Takmaz, E. Fedele, R. W. Sumner, M. Pollefeys, F. Tombari, and F. Engelmann, “OpenMask3D: Open-Vocabulary 3D Instance Segmentation,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[12] A. Dai, A. X. Chang, M. Savva, M. Halber, T. Funkhouser, and M. Nießner, “ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 5828–5839.

[13] H. Wang, Y. Cong, O. Litany, Y. Gao, and L. J. Guibas, “3DIoUMatch: Leveraging IoU Prediction for Semi-Supervised 3D Object Detection,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 14615–14624.

[14] B. Cheng, L. Sheng, S. Shi, M. Yang, and D. Xu, “Back-Tracing Representative Points for Voting-Based 3D Object Detection in Point Clouds,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8963–8972.
