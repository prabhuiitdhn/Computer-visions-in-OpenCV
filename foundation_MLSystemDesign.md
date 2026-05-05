# ML System Design — Deep Conceptual Notes

Interview prep for **Staff Machine Learning Engineer @ Netradyne**
Focus: End-to-end ML system design for Computer Vision — model design, optimization, evaluation, latency, throughput, deployment, and production monitoring.

A few things to orient the study plan before we dive in:

Netradyne context — they build AI dashcam systems for commercial fleet safety. Their core product (Driveri) runs CV models on-device (edge inference) detecting:

Driver distraction / drowsiness / phone use
Tailgating / following distance
Lane departure, stop sign compliance
Harsh braking / acceleration events
This means your interview will heavily emphasize:

Real-time edge inference (latency, throughput, model compression)
CV pipeline design (detection, tracking, classification chains)
Data pipeline at scale (dashcam footage from thousands of trucks)
Model evaluation on imbalanced, real-world data
Continuous learning / model refresh in production
Suggested learning path (high → low level):

1. ML System Design Framework       ← start here
2. Problem Framing & Requirements
3. Data Pipeline Design
4. Model Architecture Selection
5. Training Pipeline
6. Evaluation & Metrics
7. Model Optimization (quantization, pruning, distillation)
8. Latency & Throughput Analysis
9. Edge Deployment
10. Production Monitoring & MLOps
---

## ML System Design: The Complete Framework

---

### What Is ML System Design?

ML System Design is the discipline of **architecting end-to-end machine learning systems** that work reliably in production — not just on a benchmark. It bridges the gap between a notebook experiment and a system that:

- Processes millions of data points per day
- Makes predictions in milliseconds on constrained hardware
- Degrades gracefully when data distribution shifts
- Can be retrained, monitored, and improved continuously

For a Staff-level interview, you are expected to **own the entire vertical** — from raw data to deployed model to production monitoring.

---

### The Golden Rule of ML System Design

> **A model is only 10% of the system. The other 90% is everything that surrounds it.**

```
What most people think ML is:
  Data → Model → Predictions

What ML in production actually is:

  ┌─────────────────────────────────────────────────────────┐
  │                  ML SYSTEM                              │
  │                                                         │
  │  Raw Data                                               │
  │     ↓                                                   │
  │  Data Collection & Ingestion Pipeline                   │
  │     ↓                                                   │
  │  Data Validation & Quality Checks                       │
  │     ↓                                                   │
  │  Feature Engineering / Preprocessing                    │
  │     ↓                                                   │
  │  ┌─────────────────────┐                               │
  │  │      MODEL          │  ← only this part in notebook  │
  │  │  Train / Evaluate   │                               │
  │  └─────────────────────┘                               │
  │     ↓                                                   │
  │  Model Versioning & Registry                            │
  │     ↓                                                   │
  │  Serving Infrastructure (edge / cloud / hybrid)        │
  │     ↓                                                   │
  │  Monitoring & Alerting                                  │
  │     ↓                                                   │
  │  Feedback Loop → back to Data Collection               │
  └─────────────────────────────────────────────────────────┘
```

---

### The 8-Phase ML System Design Framework

Every ML system design problem — regardless of domain — can be structured into 8 phases. Memorize this order. In an interview, walk through each phase systematically.

```
Phase 1:  PROBLEM FRAMING
          What are we actually solving? Business metric → ML metric.

Phase 2:  DATA STRATEGY
          What data do we need? How do we collect, label, store it?

Phase 3:  FEATURE ENGINEERING & PREPROCESSING
          How do we transform raw inputs into model-ready tensors?

Phase 4:  MODEL DESIGN
          What architecture? Why? What are the trade-offs?

Phase 5:  TRAINING PIPELINE
          How do we train at scale? Loss functions, augmentation,
          distributed training, experiment tracking.

Phase 6:  EVALUATION
          How do we know the model is good? Which metrics?
          Offline vs. online evaluation.

Phase 7:  OPTIMIZATION & DEPLOYMENT
          How do we make it fast enough for production?
          Quantization, pruning, distillation, serving.

Phase 8:  MONITORING & ITERATION
          How do we know it's still working in production?
          Data drift, model decay, retraining triggers.
```

---

### Applied to Netradyne: The Dashcam CV System

```
Business problem:
  "Detect unsafe driving events in real-time from dashcam footage
   and generate coachable alerts for fleet managers."

What this means technically:

  Input:   Video stream from forward-facing + driver-facing cameras
           Running on an embedded device in each truck

  Output:  Per-frame or per-event classification:
           - Driver distracted? (phone, looking away, drowsy)
           - Unsafe following distance?
           - Lane departure?
           - Speed violation?
           - Harsh braking / cornering?

  Constraints:
           - Must run on embedded hardware (NVIDIA Jetson class)
           - Latency: < 100ms per frame (≥ 10 FPS real-time)
           - No reliable internet connection (edge inference)
           - Battery/power limited
           - Must work: day/night, rain/fog, different truck cabs
```

---

### Phase 1: Problem Framing — The Most Critical Step

#### Step 1.1: Translate Business Metric → ML Metric

```
Business metric:              ML metric proxy:
──────────────────────────    ────────────────────────────────────
Reduce accident rate          High recall on safety-critical events
Reduce false complaints       High precision on driver alerts
Fleet manager trust           Low false positive rate per hour
Driver retention              Minimize nuisance alerts
System uptime                 Model robustness / graceful degradation
```

**The tension at Netradyne:**

```
High recall (catch every unsafe event):
  ✓ Never misses a dangerous event
  ✗ Many false alarms → driver frustration → fleet churns

High precision (only alert when certain):
  ✓ Every alert is legitimate → driver trusts system
  ✗ Misses real events → accidents not caught

Resolution: define operating point on the precision-recall curve
  based on business priorities:
  - Safety-critical (drowsy driving): high recall, accept FP
  - Nuisance-cost events (minor distraction): high precision
```

#### Step 1.2: Define the Output Space

```
Classification type:    Binary (safe/unsafe) per event
                        Multi-class (which type of event)
                        Multi-label (multiple events simultaneously)

Temporal scope:         Per-frame classification
                        Event-level (start/end of event)
                        Trip-level summary

Confidence required:    Hard decision (threshold)
                        Soft probability (downstream calibration)
```

#### Step 1.3: Define Constraints Explicitly

```
HARD CONSTRAINTS (cannot violate):
  □ Inference latency ≤ X ms per frame
  □ Model size ≤ X MB (fits in device memory)
  □ Power consumption ≤ X watts
  □ Works offline (no cloud dependency)

SOFT CONSTRAINTS (optimize for):
  □ Maximize detection accuracy
  □ Minimize annotation cost
  □ Minimize retraining frequency
  □ Maximize generalization across camera types
```

---

### Phase 2: Data Strategy — "Data is the Moat"

#### The Data Flywheel

```
More deployed devices
        ↓
More diverse real-world data collected
        ↓
Better training data → Better model
        ↓
Better model → More accurate detections
        ↓
More customer value → More deployed devices (loop)
```

#### Data Sources for Dashcam CV

```
Source 1: Production fleet (most valuable)
  + Real distribution: actual lighting, weather, driver behaviors
  + Scale: millions of hours of footage
  - Labeling is expensive (need human review)
  - Class imbalance: rare events (accidents) extremely scarce

Source 2: Simulation / Synthetic data
  + Cheap, infinite, controllable distribution
  + Can generate rare events (accidents, extreme weather)
  - Domain gap: synthetic → real requires careful tuning

Source 3: Public datasets (ADAS, driver monitoring)
  + BDD100K, DMD (Driver Monitoring Dataset), DriveFace
  + Good for pretraining, not fine-tuning
  - Different distribution from Netradyne's specific cameras

Source 4: Augmented/Programmatic data
  + Add artificial rain, night, glare to existing frames
  + Cheap way to improve robustness
  - Augmentation must match real degradation statistics
```

#### Data Labeling Strategy

```
Labeling approaches (cheapest → most expensive):

1. WEAK SUPERVISION (programmatic labels):
   Use rule-based heuristics as noisy labels
   e.g., GPS shows sudden deceleration → "harsh braking" label
   Cost: near-zero. Accuracy: ~70-80%

2. SEMI-SUPERVISED:
   Label 1% of data. Train model. Use model to pseudo-label rest.
   Iteratively improve. Cost: low.

3. ACTIVE LEARNING:
   Model identifies UNCERTAIN samples → send only those to humans
   Focuses annotation budget on highest-value examples
   Cost: medium. Very efficient for rare events.

4. HUMAN-IN-THE-LOOP:
   Full human annotation for ground truth.
   Only needed for: hard cases, safety-critical labels, evaluation set.
   Cost: high.
```

#### Class Imbalance — The Core Data Problem

```
Typical class distribution in dashcam footage:

  Normal driving:          95.0%    ← overwhelming majority
  Minor distraction:        3.5%
  Phone use:                0.8%
  Drowsiness:               0.5%
  Severe distraction:       0.15%
  Near-collision event:     0.05%

If you train naively:
  Model predicts "normal" for every frame → 95% accuracy
  But: 0% recall on the safety events you care about!
```

Solutions:
```
1. Weighted loss: upweight minority classes
   loss = Σ w_c · CrossEntropy(y_c, ŷ_c),  w_c ∝ 1/freq_c

2. Focal loss: down-weight easy negatives
   FL(p_t) = -α_t(1-p_t)^γ log(p_t)
   γ=2 makes well-classified examples contribute 100× less

3. Hard negative mining: explicitly sample hard negatives during training

4. Oversampling rare events (copy-paste augmentation for CV)

5. Two-stage: first detect "any event" (high recall),
              then classify type (high precision)
```

---

### Phase 3: Feature Engineering & Preprocessing

#### Preprocessing Pipeline for Dashcam Video

```
Raw video frame [H × W × 3, uint8]
         ↓
1. DECODING: H.264/H.265 → raw pixels (hardware-accelerated on Jetson)
         ↓
2. RESIZING: to model input size (e.g., 640×640 or 416×416)
         ↓
3. NORMALIZATION: pixel values → [0,1] or [-1,+1]
   ImageNet mean/std: μ=(0.485, 0.456, 0.406), σ=(0.229, 0.224, 0.225)
         ↓
4. COLOR SPACE: RGB vs BGR (OpenCV uses BGR — common bug!)
         ↓
5. TEMPORAL STACKING (if using temporal model):
   Stack T consecutive frames → [T × H × W × 3] tensor
         ↓
Model input tensor [B × C × H × W]
```

#### Data Augmentation Strategy

```
GEOMETRIC augmentations:
  Random crop, horizontal flip, rotation (±15°), perspective warp

PHOTOMETRIC augmentations:
  Brightness/contrast jitter, color jitter (HSV),
  Gaussian blur, Gaussian noise

DOMAIN-SPECIFIC augmentations:
  Rain overlay, lens flare, motion blur,
  IR simulation, dashboard reflection overlay, occlusion patches
```

---

### Phase 4: Model Architecture Selection

```
Decision tree for architecture selection:

Q1: What is the latency budget?
  < 10ms:   → MobileNet, EfficientNet-Lite, YOLO-Nano
  10-50ms:  → EfficientDet, YOLOv8-small, ResNet-50
  50-200ms: → YOLOv8-large, ResNet-101, Swin-Tiny

Q2: Does the task require temporal reasoning?
  No (per-frame): → CNN backbone
  Yes (drowsiness, event detection): → CNN + LSTM / Transformer

Q3: Single task or multi-task?
  Single: → task-specific head on shared backbone
  Multi:  → Multi-task learning, shared backbone, multiple heads

Q4: How much training data?
  < 10K:    → pretrained backbone, fine-tune only head
  10K-1M:   → fine-tune full model from pretrained weights
  > 1M:     → train from scratch or large pretrained model
```

#### Multi-Task Architecture for Netradyne (Example)

```
Shared backbone (EfficientNet-B2 or MobileNetV3):
  Input: [B × 3 × 384 × 384]
         ↓
  Feature extractor
         ↓
  Feature map [B × 256 × 24 × 24]
         ↙         ↓         ↘
Head 1:           Head 2:           Head 3:
Object detect     Driver state      Event classify
(vehicles,        (eyes open/       (distracted/
pedestrians)      closed, phone)    drowsy/normal)

Benefits: one forward pass, shared features, one model to deploy
```

---

### Phase 5: Training Pipeline

```
Key components:

1. EXPERIMENT TRACKING (MLflow, W&B):
   Log: hyperparameters, metrics, model artifacts, data versions

2. DISTRIBUTED TRAINING:
   DDP (DistributedDataParallel) on 8-32 GPUs

3. LEARNING RATE SCHEDULE:
   Warmup + cosine decay (standard for vision)
   OneCycleLR (good for fine-tuning)

4. MIXED PRECISION TRAINING (FP16/BF16):
   2× memory savings, 2-3× speed on modern GPUs

5. CHECKPOINTING:
   Save every N epochs + save best model (by validation metric)
   Save: weights + optimizer state + scheduler + epoch
```

---

### Phase 6: Evaluation

#### Offline vs Online Evaluation

```
OFFLINE EVALUATION (before deployment):
  Evaluate on held-out test set
  Metrics: Precision, Recall, F1, mAP, AUC-ROC, latency
  Fast, cheap, repeatable

ONLINE EVALUATION (after deployment):
  A/B test: route X% of traffic to new model
  Measure: business metrics (alert accuracy, complaint rate)
  Gold standard for measuring true real-world quality
```

#### Evaluation Slices — The Critical Concept

```
Aggregate metrics hide failures. Always evaluate on slices:

  □ Time of day: day / dusk / night
  □ Weather: clear / rain / fog / snow
  □ Camera position: different mount positions
  □ Truck type: sedan vs semi vs van
  □ Driver demographics: different skin tones for face analysis
  □ Geographic region: highway vs city vs rural

Why this matters:
  Overall F1 = 0.92 (looks great!)
  F1 on night + rain = 0.54 (catastrophic failure hidden in aggregate)
```

---

### Phase 7: Optimization & Deployment (Preview)

```
1. QUANTIZATION:       FP32 → INT8, 4× size reduction, 2-4× speedup
2. PRUNING:            Remove unimportant weights
3. KNOWLEDGE DISTIL.:  Train small model to mimic large model
4. COMPILATION:        TensorRT / TFLite — hardware-specific optimization
5. HARDWARE SELECTION: Match model to silicon (Jetson, Qualcomm DSP)
```

---

### Phase 8: Monitoring & Iteration

```
INPUT MONITORING (data drift):
  □ Input distribution shift: mean pixel value, brightness histogram
  □ Camera quality degradation: blur score, exposure levels

MODEL MONITORING (concept drift):
  □ Prediction distribution shift
  □ Alert rate spike or drop
  □ Latency: p50, p95, p99 inference time

RETRAINING TRIGGERS:
  □ Performance metric drops below threshold
  □ Distribution shift detected (KL divergence test)
  □ New camera hardware deployed
```

---

### The Interview Answer Template

```
1. Clarify requirements (2 min):
   "Before I design, let me clarify: what is the latency budget?
   What scale of data? Real-time or batch? Edge or cloud?"

2. High-level architecture (3 min):
   Draw the end-to-end pipeline. Name each component. Show data flow.

3. Deep dive on 2-3 components (10 min):
   Pick the hardest parts. Go deep on trade-offs.

4. Trade-offs and alternatives (3 min):
   "I chose A over B because... but if constraint changes to Y,
   I'd switch to B because..."

5. Failure modes and mitigations (2 min):
   "The biggest risks are X and Y. I'd mitigate by..."
```

---

**Interview one-liner:**
> ML System Design is the discipline of building reliable, production-ready ML pipelines across 8 phases: problem framing, data strategy, preprocessing, model design, training, evaluation, optimization/deployment, and monitoring. For CV systems like Netradyne's, the critical challenges are: real-time edge inference under strict latency budgets, handling severe class imbalance in rare safety events, evaluation across slices (night/weather/demographics) not just aggregate metrics, and building feedback loops that continuously improve the model from production data.

---

*End of notes — continued in next session.*
