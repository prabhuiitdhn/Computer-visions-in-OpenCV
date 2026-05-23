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

**Solutions:**

**1. Weighted loss** — upweight minority classes

$$\mathcal{L} = \sum_c w_c \cdot \text{CrossEntropy}(y_c, \hat{y}_c), \qquad w_c \propto \frac{1}{\text{freq}_c}$$

**2. Focal loss** — down-weight easy negatives

$$FL(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

$\gamma = 2$ makes well-classified examples contribute 100× less to the loss than hard misclassified ones.

**3. Hard negative mining** — explicitly sample hard negatives during training

**4. Oversampling rare events** — copy-paste augmentation for CV (paste rare event crops into normal frames)

**5. Two-stage pipeline** — first detect "any event" (high recall), then classify type (high precision)

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

## Graceful Degradation Under Distribution Shift

---

### What Is Distribution Shift?

When you train a model, it learns a function that approximates the relationship between inputs and outputs **based on the statistical distribution of your training data**. Distribution shift is when the data seen in production is statistically different from training data.

```
Training distribution:   P_train(X, Y)
Production distribution: P_prod(X, Y)

If P_train ≠ P_prod → model performance degrades
```

The question is: **how badly, how fast, and can the system recover?**

---

### The Three Types of Distribution Shift

#### Type 1: Covariate Shift — Input Distribution Changes, Relationship Doesn't

$$P_{train}(X) \neq P_{prod}(X) \quad \text{but} \quad P(Y|X) \text{ unchanged}$$

The meaning of labels hasn't changed — only what the inputs look like.

```
Example:
  Train on summer images → deploy in winter
  "Cat" still means cat. But winter cats look different:
  fur fluffier, snow background, different lighting.
  The model's learned features don't fire correctly.

Short Netradyne ref:
  Train on daytime dashcam → deploy on overnight fleet routes
```

#### Type 2: Label Shift — Class Frequencies Change

$$P_{train}(Y) \neq P_{prod}(Y) \quad \text{but} \quad P(X|Y) \text{ unchanged}$$

Each class looks the same — but how often it appears has changed.

```
Example:
  Medical diagnosis model trained on 50/50 disease/healthy dataset
  → deployed in general population where only 1% have the disease

  The appearance of the disease hasn't changed.
  But the model's decision threshold was calibrated for 50% prior
  → outputs way too many positive predictions → FP rate explodes
```

#### Type 3: Concept Drift — The Relationship Itself Changes

$$P_{train}(Y|X) \neq P_{prod}(Y|X)$$

The most dangerous. The world has changed — same input now means something different.

```
Example:
  Spam filter trained in 2018: "Congratulations you won!" → spam
  2024: same phrase appears in legitimate company raffle emails
  The text pattern is the same but the label has changed.

  Or: fraud detection model trained pre-COVID →
  post-COVID, normal transaction patterns completely changed →
  legitimate transactions flagged as fraud
```

---

### What "Graceful Degradation" Means

A system that degrades **ungracefully**:
- Fails **silently** — keeps outputting confident wrong predictions
- Team finds out weeks later from user complaints
- Damage is already done

A system that degrades **gracefully**:
- **Detects** that something has changed
- **Signals uncertainty** rather than pretending to be confident
- **Falls back** to a safer behavior
- **Alerts** the ops team automatically
- Accuracy drops **slowly and predictably**, not suddenly and catastrophically

```
Degradation spectrum:

  Catastrophic                              Graceful
  failure                                   degradation
     ↓                                           ↓
  ┌───────────────────────────────────────────────────┐
  │ Silent wrong  →  Confident  →  Flagged  →  Safe  │
  │ predictions      but wrong     uncertain   abstain│
  └───────────────────────────────────────────────────┘
                                            ↑
                                   This is the goal
```

---

### The Four Pillars of Graceful Degradation

#### Pillar 1: Out-of-Distribution (OOD) Detection

The model must know when it's seeing inputs it wasn't trained on.

```
Method 1: Confidence thresholding (naive, unreliable)
  if max(softmax) < threshold → flag as uncertain
  Problem: neural nets are famously overconfident on OOD inputs

Method 2: Calibration (temperature scaling)
  Scale logits before softmax so confidence = true accuracy
  After calibration: P(correct | confidence=0.9) ≈ 0.9
  Pre-calibration: P(correct | confidence=0.9) might be only 0.6

Method 3: Feature-space distance
  Compute distance between test input's embedding
  and the training distribution in feature space
  (Mahalanobis distance, k-NN distance)
  Large distance → likely OOD

Method 4: Ensemble disagreement
  Run N models (or MC Dropout = 1 model, N forward passes with dropout)
  If models agree → high confidence
  If models disagree → uncertain → flag as OOD
  Variance of predictions = natural uncertainty signal
```

---

#### Pillar 2: Fallback Strategies

When uncertainty is detected, have a **tiered fallback**:

```
Level 0 (normal):         Model runs normally, high confidence
Level 1 (uncertain):      Raise confidence threshold — fewer but reliable outputs
Level 2 (shift detected): Switch to simpler, more robust fallback model
                           (rule-based, heuristic, or smaller robust model)
Level 3 (severe):         Abstain entirely — output "uncertain"
                           Alert human, do NOT output a wrong prediction

Key principle: "I don't know" is always safer than a confident wrong answer
               in safety-critical systems
```

---

#### Pillar 3: Monitoring and Early Warning

```
INPUT MONITORING — detect covariate shift:
  Track statistics of input data over time:
    - Mean/std of pixel values (vision)
    - Feature distribution moments
    - Input embedding drift (cosine distance from training centroid)
  Alert when: statistics drift beyond 2–3σ of training baseline

MODEL OUTPUT MONITORING — detect label/concept shift:
  Track prediction distribution over time:
    - Histogram of class predictions per hour/day
    - Average confidence score
    - Sudden spike or drop in any class → something changed
  Alert when: output distribution differs significantly from calibration period

BUSINESS METRIC MONITORING:
  - User complaints ("wrong prediction")
  - Human review override rate
  - Confirmed event rate

Statistical drift tests:
  PSI (Population Stability Index):
    PSI = Σ (P_train - P_prod) · ln(P_train / P_prod)
    PSI < 0.1:   stable
    PSI 0.1-0.2: monitor
    PSI > 0.2:   significant shift → investigate

  KS test, MMD (Maximum Mean Discrepancy),
  Chi-squared test for categorical features
```

---

#### Pillar 4: Automated Recovery — Closing the Loop

```
Shift detected
    ↓
Sample representative data from shifted distribution
    ↓
Route to human annotation (or weak label via active learning)
    ↓
Retrain / fine-tune model on new data
    ↓
Validate on held-out set (including new shifted distribution)
    ↓
Shadow deploy (run new model alongside old, compare outputs)
    ↓
Promote new model if performance confirmed
    ↓
Monitor for next shift (loop)
```

---

### Why "Graceful" Is the Key Word

The degradation is inevitable — you **cannot prevent** distribution shift in the real world. What you **can** control is:

```
1. How fast you detect it   (monitoring latency)
2. How much harm it causes  (fallback safety net)
3. How fast you recover     (automated retraining speed)
```

```
Without graceful degradation:
  t=0    Model deployed, accuracy = 0.92
  t=30d  Accuracy = 0.71 (shift happened at t=20d, undetected)
  t=60d  User complaint triggers investigation
  t=90d  New model deployed
  → 70 days of degraded performance, discovered late

With graceful degradation:
  t=0    Model deployed, accuracy = 0.92
  t=20d  Shift detected automatically (PSI alert)
  t=21d  Fallback activated, sampling started
  t=35d  New model promoted
  → 15 days of degraded performance, detected immediately
```

---

### Summary

```
Distribution shift types:
  Covariate shift:  P(X) changes — inputs look different
  Label shift:      P(Y) changes — class frequencies change
  Concept drift:    P(Y|X) changes — the world's rules changed

Graceful degradation = 4 pillars:
  1. OOD detection:  know when you're outside training distribution
  2. Fallback tiers: have a safe behavior for each uncertainty level
  3. Monitoring:     detect shift early via statistical tests
  4. Auto-recovery:  trigger retraining automatically, close the loop
```

---

**Interview one-liner:**
> Distribution shift is inevitable in production — covariate shift changes what inputs look like, label shift changes class frequencies, and concept drift changes the input-output relationship itself. Graceful degradation means the system detects shift early via statistical monitoring (PSI, KS test, feature-space drift), signals uncertainty rather than silently producing wrong outputs, falls back through a tiered safety net, and automatically triggers retraining to close the loop — so the window of degraded performance is minimized and bounded.

---

## Data Collection & Ingestion Pipeline

---

### What Is a Data Ingestion Pipeline?

Before any model can be trained, data must travel from its **source** (camera, sensor, user action, database) to a place where it can be **processed, stored, and served** to a training job. The data ingestion pipeline is the infrastructure that makes this happen reliably, at scale, and with quality guarantees.

```
Raw world events
      ↓
  [Data Sources]        cameras, sensors, logs, APIs, databases
      ↓
  [Collection]          how data gets captured and transported
      ↓
  [Ingestion]           how data enters your ML infrastructure
      ↓
  [Validation]          is the data usable?
      ↓
  [Storage]             where does it live?
      ↓
  [Serving]             how does training/inference access it?
```

This pipeline is the **foundation of everything** — a flawed pipeline produces flawed models no matter how sophisticated your architecture is. "Garbage in, garbage out" applies to pipelines, not just data.

---

### Phase 1: Data Sources

Understanding your data sources determines every downstream design decision.

#### Source Types

```
STRUCTURED sources:
  Relational databases (PostgreSQL, MySQL)
  Data warehouses (BigQuery, Snowflake, Redshift)
  CSV / Parquet / Avro files
  Metadata stores (labels, annotations, event logs)

UNSTRUCTURED sources:
  Video streams (cameras, dashcams, surveillance)
  Image archives (S3 buckets, NAS drives)
  Audio recordings
  Document repositories

SEMI-STRUCTURED sources:
  JSON/XML logs (API responses, sensor telemetry)
  Event streams (Kafka topics, Kinesis streams)
  Time-series signals (GPS, IMU, accelerometer)

HUMAN-GENERATED sources:
  Annotation platforms (Label Studio, Scale AI, Labelbox)
  Crowdsourcing (Mechanical Turk)
  Expert review queues
```

#### Source Characteristics That Drive Design

```
Volume:     How much data per day? GB? TB? PB?
Velocity:   How fast does it arrive? Batch? Real-time stream?
Variety:    How many formats, schemas, modalities?
Veracity:   How clean/reliable is the source?
Value:      What fraction of incoming data is actually useful?

The 5 Vs of data — each one changes what pipeline architecture you need.
```

---

### Phase 2: Data Collection — Getting Data to Your Infrastructure

#### Pull vs Push Models

```
PULL (batch collection):
  Your pipeline periodically queries the source
  "Every 6 hours, pull all new video files from fleet devices"

  ✓ Simple, predictable load
  ✓ Easy to retry failures
  ✗ Latency: data is stale by the pull interval
  ✗ Misses real-time events

  Used for: offline training, periodic model refresh, archival

PUSH (streaming collection):
  Source sends data to pipeline as events happen
  "Every time a sensor reading arrives, push to Kafka topic"

  ✓ Low latency (near real-time)
  ✓ No polling overhead
  ✗ Must handle backpressure (what if pipeline is slow?)
  ✗ More complex failure handling

  Used for: online feature computation, real-time monitoring,
            live model serving, anomaly detection
```

#### Data Transport Patterns

```
Pattern 1: Direct upload (simple, small scale)
  Device → HTTP POST → Object storage (S3/GCS)
  Good for: sporadic uploads, mobile devices, small files
  Problem: doesn't scale to millions of concurrent devices

Pattern 2: Message queue (streaming, high throughput)
  Device → Kafka / Kinesis / Pub-Sub → Consumer

  Kafka architecture:
  Producers → [Topic: raw_video] → Consumer Group
                    ↓
            Partitions (parallel processing)
            Retention (store for X days for replay)

  Good for: high throughput, multiple consumers, replay
  Netradyne ref: each dashcam event → Kafka topic → parallel processors

Pattern 3: Change Data Capture (CDC)
  Monitor database for changes → stream only deltas
  Used for: keeping training data in sync with production database
  Tools: Debezium, AWS DMS

Pattern 4: Edge buffering + batch sync
  Device buffers data locally when offline
  Syncs to cloud when connectivity available
  Good for: intermittently connected devices (trucks, field sensors)
```

---

### Phase 3: Ingestion Architecture — Batch vs Stream

This is one of the most important architectural decisions.

#### Batch Ingestion

```
Source data accumulates
    ↓
Scheduled job runs (hourly / daily / weekly)
    ↓
Read large chunk → Transform → Write to data lake
    ↓
Training job reads from data lake

Tools: Apache Spark, AWS Glue, dbt, Airflow (orchestration)

┌─────────────────────────────────────────────────────┐
│              BATCH PIPELINE                         │
│                                                     │
│  Raw Store → [Spark Job] → Processed Store          │
│  (S3/GCS)    (hourly)      (Parquet/TFRecord)       │
└─────────────────────────────────────────────────────┘

Pros: simple, cheap, high throughput, easy to reprocess
Cons: latency = batch interval (hours/days of staleness)
Best for: training data preparation, periodic model retraining
```

#### Stream Ingestion (real-time)

```
Each event arrives
    ↓
Stream processor handles it immediately
    ↓
Write to serving store (feature store, database)

Tools: Apache Flink, Spark Streaming, Kafka Streams

┌─────────────────────────────────────────────────────┐
│              STREAM PIPELINE                        │
│                                                     │
│  Kafka → [Flink Job] → Feature Store / DB           │
│          (per-event)   (Redis / DynamoDB)           │
└─────────────────────────────────────────────────────┘

Pros: real-time features, immediate anomaly detection
Cons: complex, expensive, hard to backfill historical data
Best for: online serving features, real-time monitoring
```

#### Lambda Architecture (Both Combined)

```
Most production ML systems use both:

                    ┌──────────────────────────────┐
                    │         DATA SOURCES          │
                    └──────────────────────────────┘
                              ↓
              ┌───────────────┴────────────────┐
              ↓                                ↓
      BATCH LAYER                       SPEED LAYER
      (Spark, hourly jobs)              (Flink, real-time)
              ↓                                ↓
      Historical features             Real-time features
      (high accuracy, stale)          (approximate, fresh)
              ↓                                ↓
              └───────────────┬────────────────┘
                              ↓
                       SERVING LAYER
                    (merge batch + stream)
                              ↓
                         Model inference
```

---

### Phase 4: Data Validation — The Critical Gate

This is the most underbuilt component in most pipelines — and the source of the most silent failures.

#### What Can Go Wrong

```
Schema violations:
  Expected field "frame_timestamp" as int64
  Received "frame_timestamp" as string → parsing fails downstream

Missing data:
  GPS signal lost → location field = NULL
  Frame dropped → sequence has gaps → temporal model sees wrong context

Corrupt data:
  Video file partially written (device crash mid-write) → codec error
  Sensor malfunction → all readings = 0 or max value

Label errors:
  Annotator fatigue → incorrect labels on edge cases
  Label leakage → future information encoded in training features

Distribution anomalies:
  Sudden spike in brightness → camera exposed to direct sunlight
  All frames identical → camera frozen/crashed
  Zero variance in sensor readings → sensor disconnected
```

#### Validation Checks (Great Expectations style)

```
SCHEMA checks (structural):
  - Column names and types match schema
  - No unexpected NULL values in required fields
  - Values within expected ranges (e.g., pixel values 0-255)
  - Foreign key integrity (label references valid sample ID)

STATISTICAL checks (distributional):
  - Feature means within [μ_train ± 3σ_train]
  - Class label distribution matches expected ratio
  - No duplicate records (deduplication)
  - No future data leakage (timestamps are valid)

BUSINESS LOGIC checks (domain-specific):
  - Video duration > minimum threshold
  - GPS coordinates within plausible geographic bounds
  - Frame sequence has no large timestamp gaps
  - Camera calibration parameters present

REJECT / QUARANTINE policy:
  Fail schema check → reject, do not ingest, alert
  Fail statistical check → quarantine, human review
  Fail soft check → accept with warning flag, monitor
```

---

### Phase 5: Data Storage — The Right Store for the Right Data

```
COLD / RAW storage (object store):
  S3, GCS, Azure Blob Storage
  For: original raw video, unprocessed sensor data
  Properties: cheap ($/GB), durable (11 nines), slow access
  Format: native (MP4, H264) or chunked Parquet

WARM / PROCESSED storage (data lake):
  S3 with columnar format (Parquet, TFRecord, WebDataset)
  For: preprocessed training-ready tensors, extracted features
  Properties: moderate cost, fast sequential read for training
  Format: sharded TFRecord or WebDataset for streaming training

HOT / SERVING storage (feature store / database):
  Redis, DynamoDB, Cassandra, Feast
  For: low-latency real-time feature lookup at inference time
  Properties: expensive, very fast (sub-millisecond reads)
  Format: key-value or columnar with indexed access

METADATA storage (relational / document):
  PostgreSQL, MongoDB, MySQL
  For: annotation records, model versions, experiment logs,
       dataset versioning, lineage tracking
```

#### Data Versioning — Non-Negotiable

```
Problem without versioning:
  "Which training data produced model v3.2?"
  "Someone modified the labels — which model was trained on old vs new?"
  "We need to reproduce experiment from 6 months ago."
  → IMPOSSIBLE without data versioning

Solution: treat data like code
  Every dataset version gets a hash / tag
  Training job records: data_version + code_version + config → model_version

Tools:
  DVC (Data Version Control) — git for data
  Delta Lake — versioned data lake with ACID transactions
  LakeFS — git-like branching for object stores
  MLflow Datasets — lightweight versioning in experiment tracking
```

---

### Phase 6: Data Serving to Training Jobs

Training jobs need to read data **fast** — a slow data pipeline starves the GPU:

```
GPU utilization bottleneck:

  GPU waiting for data:   [GPU idle] [GPU idle] [GPU train] [GPU idle]
  → GPU utilization = 25% → expensive hardware wasted

  Optimal (data pipeline faster than GPU):
  [GPU train] [GPU train] [GPU train] [GPU train]
  → GPU utilization = 95% → fast, cheap training
```

#### Strategies for High-Throughput Data Serving

```
1. SHARDING:
   Split dataset into N shards (e.g., 1000 files of equal size)
   Each worker reads a different shard in parallel
   → N× throughput vs single file

2. PREFETCHING:
   While GPU trains on batch N, CPU loads batch N+1
   PyTorch DataLoader: num_workers=8, prefetch_factor=2
   → overlap compute and I/O completely

3. CACHING:
   Keep hot data in memory (RAM or fast SSD)
   First epoch reads from disk (slow), subsequent epochs from cache
   → 10-100× speedup for repeated training

4. COLUMNAR FORMAT:
   TFRecord / Parquet / WebDataset
   Sequential reads → avoids random seek overhead
   Compressed → less I/O bandwidth needed
   → 5-10× faster than JPEG/PNG random access

5. DISTRIBUTED DATA LOADING:
   Each training node reads its own shard from shared storage
   No single bottleneck
   Tools: Petastorm (Parquet for DL), WebDataset (tar-based)
```

---

### The Complete Pipeline in One Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                 DATA COLLECTION & INGESTION                  │
│                                                              │
│  [Sources]                                                   │
│  Cameras / Sensors / DBs / APIs                             │
│       ↓ push / pull                                          │
│  [Transport]                                                 │
│  Kafka / Kinesis / S3 Upload / CDC                          │
│       ↓                                                      │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │ BATCH LAYER  │    │ STREAM LAYER │                       │
│  │ Spark / Glue │    │ Flink/Kafka  │                       │
│  └──────┬───────┘    └──────┬───────┘                       │
│         ↓                   ↓                               │
│  [Validation Gate]  ←────────                               │
│  Schema + Stats + Logic checks                              │
│  Pass → store  |  Fail → quarantine + alert                 │
│         ↓                                                    │
│  [Storage]                                                   │
│  Raw: S3 (cold)                                             │
│  Processed: Parquet/TFRecord (warm)                         │
│  Features: Redis/Feast (hot)                                │
│  Metadata: Postgres + DVC (versioned)                       │
│         ↓                                                    │
│  [Serving to Training]                                       │
│  Sharded + Prefetched + Cached DataLoader                   │
│         ↓                                                    │
│  Training Job → Model                                        │
└──────────────────────────────────────────────────────────────┘
```

---

### Common Pipeline Failures and Mitigations

```
Failure:             Root cause:              Mitigation:
──────────────────   ──────────────────────   ──────────────────────────
Silent bad data      No validation gate       Add Great Expectations checks
GPU starvation       Slow data loading        Shard + prefetch + cache
Non-reproducible     No data versioning       DVC / Delta Lake
Training on stale    Batch interval too long  Reduce batch frequency or stream
Pipeline failure     No retry logic           Idempotent jobs + dead letter queue
Schema drift         Source changed format    Schema registry + alerting
Label leakage        Temporal split missing   Strict time-based train/val/test split
```

---

**Interview one-liner:**
> The data collection and ingestion pipeline is the infrastructure that moves raw data from sources (sensors, cameras, databases) through transport (Kafka, S3 upload), validation (schema, statistical, business logic checks), and storage (cold/warm/hot tiers) to a form that training jobs can consume efficiently. Its three critical properties are: reliability (no silent data loss or corruption), throughput (GPU should never wait for data), and reproducibility (every training run must be traceable to an exact data version).

---

## Feature Engineering / Preprocessing

---

### What Is Feature Engineering?

A model cannot learn from raw data directly — it learns from **numerical representations** of that data. Feature engineering is the process of transforming raw input into a form that makes the underlying patterns **easier for a model to discover**.

```
Raw Data                    Features                    Model
──────────                  ────────                    ─────
Pixel bytes        →        Normalized tensors    →     CNN
Timestamps         →        Hour-of-day, day-of-week →  GBM
GPS coordinates    →        Speed, heading delta  →     LSTM
Raw text           →        TF-IDF / embeddings   →     Transformer
Video frames       →        Optical flow, HOG     →     Classifier
```

The core tension:
- **Classical ML** (GBMs, SVMs, logistic regression) — heavily dependent on manual feature engineering. The model only learns from what you explicitly give it.
- **Deep learning** — learns features from raw data automatically. But good preprocessing still matters enormously.

---

### Why Preprocessing Matters Even for Deep Learning

Common misconception: "CNNs learn features automatically, so preprocessing doesn't matter."

Reality:

```
Without normalization:
  Layer 1 receives values in [0, 255]
  Gradients scale with input magnitude
  → learning rate must be tiny to avoid explosion
  → training is slow and unstable

With normalization (values in [0, 1] or [-1, 1]):
  Gradients have similar magnitude across all inputs
  → larger learning rate works
  → faster convergence, better generalization

Without augmentation:
  Model memorizes exact pixel patterns
  → brittle to rotation, lighting, scale
  → fails on anything slightly different from training

With augmentation:
  Model sees thousands of variants of each sample
  → learns invariances
  → generalizes better
```

---

### The Preprocessing Taxonomy

```
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                            │
│                                                             │
│  1. CLEANING          Remove noise, fix errors              │
│  2. TRANSFORMATION    Change scale, distribution, format    │
│  3. FEATURE CREATION  Derive new signals from raw data      │
│  4. FEATURE SELECTION Remove irrelevant / redundant features│
│  5. ENCODING          Convert categorical → numeric         │
│  6. AUGMENTATION      Artificially expand training data     │
└─────────────────────────────────────────────────────────────┘
```

---

### 1. Cleaning

Before transforming data, fix what's broken.

```
Missing values:
  Option A: Drop rows with missing values
    → simple, but loses data (bad if data is scarce)
  Option B: Imputation
    Mean/median imputation: replace with column average
    Model-based imputation: predict missing value from other features
    Forward-fill: for time series, carry last known value forward
  Option C: Indicator flag
    Add binary column "feature_X_was_missing" = 1
    → lets model learn that missingness itself is informative

Outliers:
  Option A: Remove (if sensor error / data corruption)
  Option B: Clip to [μ ± 3σ] (winsorizing)
  Option C: Log-transform to compress extreme values

  Key question: Is this outlier real (edge case to learn from)
                or corrupt (noise to ignore)?
  Never blindly remove outliers — inspect them first.

Duplicates:
  Exact duplicates → remove (inflate training statistics)
  Near-duplicates  → more subtle; use hashing or embedding similarity
  Why it matters:  if same sample is in train AND val,
                   validation accuracy is falsely inflated
```

---

### 2. Transformation — Scaling and Distribution

#### Normalization vs Standardization

$$\text{Min-Max normalization:} \quad x' = \frac{x - x_{min}}{x_{max} - x_{min}} \in [0, 1]$$

$$\text{Standardization (Z-score):} \quad x' = \frac{x - \mu}{\sigma}$$

```
When to use which:

Min-Max normalization:
  ✓ When you need values in a bounded range [0,1]
  ✓ Image pixel values (0-255 → 0.0-1.0)
  ✗ Sensitive to outliers (one extreme value compresses everything else)

Z-score standardization:
  ✓ When distribution is roughly Gaussian
  ✓ When features have very different scales (age=25, income=75000)
  ✓ Needed for distance-based models (k-NN, SVM, PCA)
  ✗ Doesn't bound the range (values can be large negatives/positives)

Neither needed for:
  Tree-based models (GBM, Random Forest) — invariant to monotonic transforms
  They split on thresholds, not distances
```

#### Log Transform — Handling Skewed Distributions

```
Raw income distribution:
  [18000, 22000, 25000, 27000, 31000, 450000]
  One billionaire skews everything → mean useless

After log transform: log(x)
  [9.8, 10.0, 10.1, 10.2, 10.3, 13.0]
  Distribution is more symmetric → model learns better

Use log transform when:
  Feature spans multiple orders of magnitude
  Distribution has a long right tail
  Examples: income, population, pixel intensities, frequency counts
```

---

### 3. Feature Creation — Domain-Driven Signal Extraction

This is where **domain knowledge** gives you an edge over pure end-to-end learning.

#### Temporal / Sequential Features

```
Raw: timestamp = 1746451200

Engineered:
  hour_of_day      = 14       (afternoon vs midnight behavior differs)
  day_of_week      = 1        (Monday — commuter traffic patterns)
  is_weekend       = 0
  days_since_event = 3        (recency signal)
  rolling_mean_7d  = 0.73     (trend smoothing)
  lag_1            = 0.81     (previous value — autocorrelation)
```

#### Spatial / Geometric Features (Vision)

```
Raw: video frame pixels

Engineered:
  Optical flow:   pixel displacement between frame t and t+1
                  → motion direction and magnitude
                  → key for action recognition, moving object detection

  HOG (Histogram of Oriented Gradients):
                  local edge orientation distributions
                  → shape descriptor, pedestrian detection

  Image pyramid:  same image at multiple scales
                  → model sees both fine detail and global context

  Depth map:      if stereo camera or LiDAR available
                  → 3D spatial reasoning

Netradyne ref: extracting optical flow from dashcam + GPS heading
               gives richer signal than raw frames alone
```

#### Interaction Features

```
Raw features: speed, road_type

Interaction: speed × road_type_highway
             = captures "speeding specifically on highways"
             Neither feature alone captures this — the combination does

General principle:
  f1 × f2  →  captures multiplicative interactions
  f1 - f2  →  captures differential/delta signals
  f1 / f2  →  captures ratios

Important: tree models discover interactions automatically
           Linear models / logistic regression need explicit interactions
```

#### Ratio and Delta Features

```
Instead of raw values, model the change:
  current_speed - speed_limit          → violation magnitude
  frame_brightness / mean_brightness   → relative exposure
  heading_change / time_delta          → angular velocity

Deltas often carry more signal than absolute values
for detecting events, anomalies, transitions
```

---

### 4. Feature Selection — Remove What Doesn't Help

More features ≠ better model. Irrelevant features add noise, slow training, and hurt generalization.

#### Methods

```
FILTER methods (fast, model-agnostic):
  Correlation with target: drop features where |corr(f, y)| ≈ 0
  Variance threshold: drop near-constant features (variance < ε)
  Mutual information: nonlinear dependency measure

  Pros: fast, model-agnostic
  Cons: ignores feature interactions

WRAPPER methods (slow, accurate):
  Forward selection: start empty, add best feature one at a time
  Backward elimination: start with all, remove worst one at a time
  Recursive Feature Elimination (RFE): model ranks features, prune weakest

  Pros: considers feature interactions
  Cons: expensive (train model N times)

EMBEDDED methods (best of both):
  L1 (Lasso) regularization: drives irrelevant feature weights → 0
  Tree feature importance: split gain per feature (Random Forest, XGBoost)
  Gradient-based: backprop gradient magnitude per input feature

  Pros: selection happens during training, efficient
  Cons: model-specific
```

#### The Curse of Dimensionality

$$\text{As dimensions } d \to \infty, \quad \text{volume of space grows as } r^d$$

```
In high dimensions, all points become equally far apart
→ distance-based models (k-NN, SVM, clustering) break down

Example:
  1D: need 10 points to cover [0,1] at resolution 0.1
  2D: need 10² = 100 points
  10D: need 10^10 = 10 billion points to maintain same density

Rule of thumb: need exponentially more data per added dimension
               → feature selection is critical for high-dimensional data
```

---

### 5. Encoding — Categorical → Numeric

Models work with numbers, not strings. Categorical features must be encoded.

```
ONE-HOT ENCODING:
  color = {red, green, blue}
  → color_red=1, color_green=0, color_blue=0

  ✓ No ordinal relationship assumed
  ✗ High cardinality explodes dimensionality
    (city with 10,000 unique values → 10,000 columns)

LABEL ENCODING (ordinal):
  size = {small=0, medium=1, large=2}
  ✓ Compact
  ✗ Implies ordering — wrong for nominal categories

TARGET ENCODING:
  Replace category with mean of target variable for that category
  city → mean(fraud_rate | city="NYC") = 0.023
  ✓ Compact, captures target relationship
  ✗ Data leakage risk — must be computed on training fold only

EMBEDDING (for high cardinality):
  Learn a dense vector representation (like word embeddings)
  city_id → embed(city_id) ∈ R^32
  ✓ Handles millions of categories
  ✓ Captures semantic similarity
  Used in: recommendation systems, NLP, entity representations
```

---

### 6. Data Augmentation — Expand the Training Distribution

Augmentation is especially critical in vision — it's how you train robust models without collecting 10× more real data.

#### Image Augmentation

```
GEOMETRIC transforms (label-preserving):
  Random crop        → invariance to position
  Horizontal flip    → invariance to left-right orientation
  Random rotation    → invariance to rotation
  Scale jitter       → invariance to object size

  ⚠ Must be label-aware:
    Flipping a "turn signal left" image → should be "turn signal right"
    Rotating a digit "6" by 180° → becomes "9"

PHOTOMETRIC transforms:
  Brightness / contrast jitter   → invariance to lighting conditions
  Gaussian noise injection        → robustness to sensor noise
  Color jitter (hue, saturation)  → invariance to color temp
  Random erase / cutout           → occlusion robustness

ADVANCED:
  Mixup:    x_new = λx_1 + (1-λ)x_2,  y_new = λy_1 + (1-λ)y_2
            Interpolates between two training samples — regularization
  CutMix:   Paste rectangular region of one image onto another
  Mosaic:   Combine 4 images into one grid (used in YOLOv4+)
  AutoAugment / RandAugment: learn optimal augmentation policy from data

Netradyne ref: simulate night / rain / glare by photometric transforms
               on daytime labeled data — saves annotation cost
```

---

### The Train/Val/Test Split — Where Preprocessing Can Leak

This is one of the most common and costly mistakes in ML engineering:

```
DATA LEAKAGE via preprocessing:

WRONG:
  1. Compute mean/std on entire dataset
  2. Normalize entire dataset
  3. Split into train/val/test
  → val/test have seen training distribution statistics
  → validation accuracy is artificially inflated

CORRECT:
  1. Split into train/val/test FIRST
  2. Fit scaler on TRAINING SET ONLY
  3. Apply scaler to val and test using training statistics

scaler = StandardScaler()
scaler.fit(X_train)          # learn μ, σ from training only
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)   # use training μ, σ
X_test  = scaler.transform(X_test)  # use training μ, σ
```

#### Time-Series: Use Temporal Split, Not Random Split

```
WRONG (random split for time series):
  Train: [t=1,3,5,7,9]  Val: [t=2,4,6,8,10]
  → model can see the future during training (leakage)

CORRECT (temporal split):
  Train: t ∈ [0, T_train]
  Val:   t ∈ [T_train, T_val]
  Test:  t ∈ [T_val, T_end]
  → strict causal ordering, no lookahead
```

---

### Feature Store — Centralized Feature Management

In production systems with many models, feature engineering logic gets duplicated across teams. The feature store solves this.

```
WITHOUT feature store:
  Team A builds "speed delta" feature → computes in Python
  Team B builds same feature → computes differently in Java
  → inconsistency: training and serving compute different values → bugs

WITH feature store:
  Feature "speed_delta_1s" computed once, stored centrally
  All teams read same feature → consistent train/serve parity
  New model can reuse existing features → faster iteration

Feature store architecture:
  OFFLINE store: Parquet/S3 — for training (batch, historical)
  ONLINE store:  Redis/DynamoDB — for inference (real-time, low latency)

  Write path: feature pipeline computes → writes to both
  Read path:  training reads offline / serving reads online

Tools: Feast, Tecton, Hopsworks, AWS SageMaker Feature Store
```

---

### Train-Serve Skew — The Silent Killer

```
Training:   features computed in Python/Spark (offline batch)
Serving:    features computed in Java/C++ (real-time)
            → any implementation difference = skew

Example:
  Training:  moving_avg = mean(last 5 speed values)
             Python: includes current frame in average
  Serving:   C++: excludes current frame (off-by-one)
             → feature value is different at inference
             → model sees a distribution it never trained on

Mitigation:
  1. Feature store (single source of truth)
  2. Share feature computation code (Python for both, or ONNX pipeline)
  3. Log online features at inference time
     → compare distribution to offline training features
     → alert on divergence
```

---

### Summary

```
Preprocessing pipeline stages:
  1. Cleaning:         fix missing values, outliers, duplicates
  2. Transformation:   normalize / standardize / log-transform
  3. Feature creation: temporal deltas, spatial signals, interactions
  4. Feature selection: remove irrelevant features (L1, RFE, importance)
  5. Encoding:         categorical → numeric (OHE, target enc, embedding)
  6. Augmentation:     expand training distribution (geometric, photometric)

Critical rules:
  → Fit all scalers/encoders on training set ONLY
  → Use temporal split for time-series data
  → Centralize features in a feature store to prevent train-serve skew
  → Augmentation must be label-aware
```

---

**Interview one-liner:**
> Feature engineering transforms raw data into representations that surface the underlying patterns more clearly for a model — covering cleaning, scaling, feature creation, selection, encoding, and augmentation. The most critical correctness rule is: all preprocessing statistics must be computed on the training set only and applied consistently to val/test, otherwise you leak information and get falsely optimistic evaluation metrics. In production, a feature store prevents train-serve skew by ensuring the same feature computation logic is used offline and online.

---

## Model Versioning & Registry

---

### Why Model Versioning Exists

In software engineering, every code change is tracked in Git — you can see who changed what, when, why, and roll back instantly. Without this, production bugs become impossible to diagnose and deployments become terrifying.

Models have the same problem, but worse:

```
A model is not just code.
A model is the product of:

  Code version       (training script, architecture)
+ Data version       (which samples, which labels, which split)
+ Config version     (hyperparameters, augmentation policy)
+ Environment        (library versions, hardware)
─────────────────────────────────────────
= Model artifact     (weights file: model.pt / model.h5 / model.onnx)

Change ANY of those → different model behavior

Without versioning:
  "Why did accuracy drop in production?"
  "Which model is running right now?"
  "Can we reproduce the model from 3 months ago?"
  → IMPOSSIBLE to answer
```

Model versioning is the practice of **tracking every model artifact alongside the exact conditions that produced it**, so the system is auditable, reproducible, and recoverable.

---

### What a Model Version Is

A model version is not just a weights file. It is a **bundle**:

```
Model Version v2.4.1
┌────────────────────────────────────────────────────────┐
│  ID:            model_v2.4.1_20260505_abc123           │
│  Artifact:      s3://models/detector/v2.4.1/model.onnx │
│  Code commit:   git SHA = f3a91c2                      │
│  Data version:  dataset_v3.2 (DVC hash = 8f2e...)      │
│  Config:        lr=1e-4, epochs=50, batch=32, aug=v3   │
│  Metrics:       mAP=0.847, F1=0.812, latency=18ms      │
│  Environment:   Python 3.10, PyTorch 2.1, CUDA 12.1    │
│  Created by:    training_job_id=train_789              │
│  Created at:    2026-05-05T14:32:00Z                   │
│  Status:        Staging                                │
│  Tags:          [night-robust, post-augv3]             │
└────────────────────────────────────────────────────────┘
```

Every field is needed. If you only store the weights file, you have no way to answer:
- "Why does this model perform worse at night?" — need data version
- "How do I reproduce this exact model?" — need code + config + env
- "Is it safe to deploy this?" — need metrics + lineage

---

### The Model Lifecycle — States a Model Passes Through

```
         Experiment
             ↓
         [Candidate]    ← training job produces a new artifact
             ↓
         [Staging]      ← passes automated quality gates
             ↓             (accuracy, latency, bias checks)
         [Production]   ← shadow deploy → promote if metrics confirmed
             ↓
         [Deprecated]   ← newer model supersedes it
             ↓
         [Archived]     ← retained for audit/compliance, not serving

Transitions are explicit and logged — not automatic overwriting
```

The key design principle: **no model should silently replace another**. Every promotion is a deliberate, tracked event.

---

### The Model Registry — Central Catalog

The model registry is a system of record for all model versions across their lifecycle. Think of it as "GitHub for models."

```
┌──────────────────────────────────────────────────────────────┐
│                      MODEL REGISTRY                          │
│                                                              │
│  Model: "vehicle_detector"                                   │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────┐  │
│  │ Version  │ Status   │ mAP      │ Latency  │ Deployed  │  │
│  ├──────────┼──────────┼──────────┼──────────┼───────────┤  │
│  │ v1.0.0   │ Archived │ 0.71     │ 24ms     │ 2025-01   │  │
│  │ v1.3.2   │ Archived │ 0.79     │ 22ms     │ 2025-06   │  │
│  │ v2.1.0   │ Deprecated│ 0.83   │ 20ms     │ 2025-11   │  │
│  │ v2.4.1   │ Production│ 0.847  │ 18ms     │ 2026-05   │  │
│  │ v2.5.0   │ Staging  │ 0.851   │ 19ms     │ —         │  │
│  └──────────┴──────────┴──────────┴──────────┴───────────┘  │
│                                                              │
│  Model: "drowsiness_classifier"                              │
│  ... (same structure)                                        │
└──────────────────────────────────────────────────────────────┘
```

The registry answers at any moment:
- What is in production right now?
- What was in production on date X?
- What models are candidates for promotion?
- Which training run produced this artifact?

---

### Semantic Versioning for Models

Borrow the convention from software: `MAJOR.MINOR.PATCH`

```
MAJOR version bump:
  Architecture change (ResNet → EfficientDet)
  Output format change (different classes, different schema)
  Breaking change — downstream consumers must update

MINOR version bump:
  Retrained on new data
  New feature added (e.g., added night-specific augmentation)
  Non-breaking improvement

PATCH version bump:
  Bug fix in preprocessing
  Minor fine-tuning
  Threshold calibration update

Example:
  v1.0.0  →  v2.0.0  :  switched from SSD to YOLO architecture
  v2.0.0  →  v2.3.0  :  retrained with 3 months of new data
  v2.3.0  →  v2.3.1  :  fixed off-by-one in NMS threshold
```

---

### Model Lineage — The Full Audit Trail

Lineage is the ability to trace a model artifact back to every input that produced it.

```
model_v2.4.1
    │
    ├── Training job: train_789
    │       ├── Code: git commit f3a91c2
    │       │       └── Diff from v2.3.0: added CBAM attention module
    │       ├── Data: dataset_v3.2
    │       │       ├── Source: s3://raw/fleet_data/2025-Q4/
    │       │       ├── Labels: annotation_batch_47 (Scale AI)
    │       │       └── Preprocessing: pipeline_v5 (DVC DAG hash)
    │       ├── Config: hyperparams_v3.yaml
    │       │       └── lr=1e-4, warmup=5ep, augment=heavy
    │       └── Environment: docker://training:cuda12.1-pt2.1
    │
    ├── Evaluation: eval_job_334
    │       ├── Test set: dataset_v3.2/test_split
    │       ├── Metrics: {mAP: 0.847, F1: 0.812, FPS: 55}
    │       └── Slice metrics: {night: 0.81, rain: 0.79, highway: 0.87}
    │
    └── Deployment: deploy_112
            ├── Target: edge_device_fleet_v4
            └── Shadow period: 2026-04-28 → 2026-05-04
```

Why lineage matters in production:
- **Bug found** → trace back to data batch that caused it
- **Regulatory audit** → prove exactly what data trained which model
- **Performance regression** → compare lineage of v2.4.1 vs v2.5.0 to find what changed

---

### Deployment Strategies — How Models Graduate to Production

#### Blue-Green Deployment

```
Current state:
  Production traffic → [Model Blue: v2.4.1]

Deploy new version:
  Production traffic → [Model Blue: v2.4.1]  (100%)
  Staging            → [Model Green: v2.5.0]  (0%)

Switch (instant, atomic):
  Production traffic → [Model Green: v2.5.0]  (100%)
  Standby            → [Model Blue: v2.4.1]   (0%) ← kept for rollback

Rollback (if issue found):
  One command: flip traffic back to Blue
  No redeployment needed — Blue is already running

Cost: 2× infrastructure during transition window
Best for: when you want zero-downtime atomic cutover
```

#### Canary Deployment

```
Phase 1: Route 5% of traffic to new model
  [v2.5.0] ← 5%
  [v2.4.1] ← 95%

Phase 2: Monitor metrics (accuracy, latency, error rate)
  If metrics OK → increase to 20%
  If metrics degrade → roll back to 0%

Phase 3: Gradual ramp
  5% → 20% → 50% → 100%

Benefit: limits blast radius of a bad model
         only 5% of users/requests affected if model is broken
Best for: high-stakes production systems, safety-critical applications
```

#### Shadow Deployment (Safest)

```
Production traffic → [v2.4.1] → returns response to user
                         ↓
                    (also) → [v2.5.0] → response discarded
                                        (never shown to user)
                                        metrics logged internally

Compare: v2.4.1 outputs vs v2.5.0 outputs on SAME real inputs
→ zero user impact
→ real production distribution
→ only promote v2.5.0 when metrics confirmed

Cost: 2× inference compute during shadow period
Best for: when you can't afford any degradation in user experience
Netradyne ref: shadow deploy new dashcam model before fleet-wide rollout
```

---

### Rollback — The Safety Net

Every promotion must have a documented rollback plan:

```
Rollback triggers (automated):
  Error rate    > threshold_error
  Latency P99   > threshold_latency
  Accuracy      < threshold_accuracy (if ground truth available)
  OOD detection rate spike

Rollback execution:
  Registry has previous production version flagged
  Deployment system re-points traffic to previous artifact
  Target: < 5 minutes from detection to rollback

Rollback test:
  Run rollback drill periodically (like fire drills)
  "Can we roll back within SLA?" → practice this, don't assume

What NOT to do:
  Delete old model artifacts (needed for rollback and audit)
  Have only one model version deployed (no fallback)
  Manual-only rollback (too slow in a production incident)
```

---

### Tools

```
Open source / cloud:
  MLflow Model Registry    — open source, widely used
                             tracks runs, metrics, artifacts, stages
  Weights & Biases         — experiment tracking + model registry
  DVC + GCP/S3             — data + model versioning via Git
  Hugging Face Hub         — model registry for NLP/vision models

Cloud-managed:
  AWS SageMaker Model Registry
  GCP Vertex AI Model Registry
  Azure ML Model Registry

Each provides:
  REST API for programmatic access
  Webhook on stage transitions (trigger CI/CD pipeline)
  Access control (who can promote to production)
  Artifact storage with deduplication
```

---

### The CI/CD Pipeline for Models (MLOps)

Model versioning plugs into a CI/CD pipeline just like code:

```
Code commit (git push)
      ↓
CI: run unit tests on training code
      ↓
Trigger training job
      ↓
Evaluate model on test set
      ↓
Automated quality gates:
  mAP > 0.82?  ✓
  Latency < 25ms? ✓
  Fairness check (slice metrics balanced)? ✓
  No regression vs current production? ✓
      ↓
Register model → status: Staging
      ↓
Shadow deploy (compare vs production)
      ↓
Human approval (optional for critical systems)
      ↓
Canary → ramp → Production
      ↓
Monitor → alert on regression → auto-rollback if triggered
```

---

### Summary

```
Model versioning = tracking every artifact + its full lineage
  Code + Data + Config + Env → reproducible, auditable model

Model registry = central catalog of all versions + lifecycle state
  Experiment → Staging → Production → Deprecated → Archived

Versioning scheme: MAJOR.MINOR.PATCH (same semantics as software)

Deployment strategies (safest to riskiest):
  Shadow      → zero user impact, compare on real traffic
  Canary      → small % of traffic, limit blast radius
  Blue-Green  → instant atomic switch, instant rollback

Rollback = must be automated, < 5min, tested regularly
```

---

**Interview one-liner:**
> Model versioning treats every trained artifact as an immutable snapshot of the code, data, config, and environment that produced it — the model registry is the central catalog that tracks each version through its lifecycle (staging → production → deprecated) with full lineage. In production, safe deployment requires a canary or shadow strategy with automated quality gates and a rollback plan that can execute in under 5 minutes, because a bad model rollout without a fast recovery path is a production incident waiting to happen.

---

## Blue-Green Deployment — Deep Dive

---

### What `Production traffic → [Model Blue: v2.4.1]` Means

```
Production traffic → [Model Blue: v2.4.1]
```

- **Production traffic** = real incoming requests from real users/devices hitting your system right now
- **→** = is being routed to
- **[Model Blue]** = a named deployment slot (just a label — "Blue" is the currently live slot)
- **v2.4.1** = the specific model version running in that slot

It means: **"Right now, all real traffic is being served by model artifact v2.4.1, which is sitting in the Blue deployment slot."**

---

### The Core Idea — Two Identical Environments

Blue-Green works by maintaining **two identical production environments** at all times:

```
┌─────────────────────────────────────────────────────────┐
│                   LOAD BALANCER / ROUTER                │
│                                                         │
│        100% traffic ──────────────────┐                 │
│                                       ↓                 │
│   ┌─────────────────┐    ┌─────────────────┐           │
│   │   BLUE SLOT     │    │   GREEN SLOT    │           │
│   │  (currently     │    │  (currently     │           │
│   │   LIVE)         │    │   IDLE/STAGING) │           │
│   │                 │    │                 │           │
│   │  Model v2.4.1   │    │  Model v2.5.0   │           │
│   │  3 replicas     │    │  3 replicas     │           │
│   │  GPU allocated  │    │  GPU allocated  │           │
│   └─────────────────┘    └─────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

Both slots are **fully running** — same hardware, same replicas, same configuration. The **only difference** is that the load balancer sends traffic to one of them.

---

### The Full Sequence Step by Step

#### Step 1 — Normal state (before deployment)

```
Router:  100% → Blue (v2.4.1)
Green:   empty / old version / idle
```

Users hit Blue. Green is free.

#### Step 2 — Deploy new version to Green

```
Router:  100% → Blue (v2.4.1)   ← users still hitting this
Green:   deploy v2.5.0, run smoke tests, warm up
```

You deploy the new model **into Green** while Blue continues serving users uninterrupted. Users notice nothing.

Smoke tests on Green:
- Can the model load?
- Does it respond to a test request?
- Is latency within bounds?
- Does output schema match expected format?

#### Step 3 — The Switch (atomic, instant)

```
BEFORE:  Router → 100% Blue (v2.4.1)
                  0%   Green (v2.5.0)

ONE COMMAND: update router config

AFTER:   Router → 0%   Blue (v2.4.1)   ← now standby
                  100% Green (v2.5.0)  ← now live
```

This is a **single atomic operation** — a config change in the load balancer. One moment everyone is on v2.4.1, the next everyone is on v2.5.0.

#### Step 4 — Rollback (if something is wrong)

```
Problem detected at t+10min:
  Error rate spiked
  Latency P99 jumped from 18ms → 95ms

ROLLBACK:  Router → 100% Blue (v2.4.1)   ← flip back

Time to rollback: seconds (just a router config change)
Blue was never torn down — it was waiting exactly for this
```

This is the key safety property: **Blue is kept alive and warm specifically so rollback is instant**. You're not redeploying — you're re-pointing a pointer.

#### Step 5 — After rollback investigation

```
Blue serving traffic again
Green (broken v2.5.0) → debug → fix → redeploy into Green
Once fixed and verified → switch again
```

---

### Why "Blue" and "Green"?

```
Just arbitrary color names for the two slots:
  Blue  = currently live (production)
  Green = standby (being prepared, or kept for rollback)

After a successful switch:
  Green becomes the new live
  Blue becomes the new standby

Some teams keep colors fixed to slots, not to roles.
Doesn't matter — the important thing is: two slots, one router.
```

---

### Comparison to Other Deployment Strategies

```
Strategy       Traffic split        Rollback speed    Risk
────────────   ─────────────────    ──────────────    ────────────────
Blue-Green     0% or 100%           Seconds           100% users hit
               (hard cut)           (router flip)     new version at once

Canary         5% → 20% → 100%     Minutes           5% users affected
               (gradual)            (ramp down)       if model is bad

Shadow         0% (mirrored only)   N/A               Zero users affected
               (never serves users) (never live)      (never serves users)
```

Blue-Green is **faster** but **higher blast radius** than canary.
Canary is safer but slower to reach full deployment.
Shadow is safest but requires double the compute and gives no real rollout.

---

### The Infrastructure Behind It

```
Load balancer options:
  NGINX / HAProxy           → config reload = instant switch
  AWS ALB (target groups)   → swap target group in one API call
  Kubernetes Ingress        → update service selector label
  Istio (service mesh)      → VirtualService weight: [blue: 0, green: 100]

Kubernetes example:
  Blue deployment:  labels: {app: model, slot: blue,  version: v2.4.1}
  Green deployment: labels: {app: model, slot: green, version: v2.5.0}
  Service selector: slot: blue   ← change this to "green" = switch done
```

---

**Interview one-liner:**
> `Production traffic → [Model Blue: v2.4.1]` means the load balancer is routing 100% of real requests to the Blue deployment slot running model v2.4.1. Blue-Green works by keeping two identical live environments — you deploy to the idle slot, then flip the router atomically. Rollback is instant because the old slot stays warm and ready, never torn down until the new version is fully confirmed stable.

---

## Shadow Period — Deep Dive

---

### What the Line Means

```
Shadow period: 2026-04-28 → 2026-05-04
```

- **Shadow period** = a window of time during which the new model (v2.5.0) was running in shadow mode — receiving real production traffic, computing predictions, but **never returning those predictions to users**
- **2026-04-28** = shadow mode started (new model began receiving mirrored traffic)
- **2026-05-04** = shadow mode ended (team reviewed collected metrics, decided to promote)
- **Duration: 7 days** — one week of real-world validation before any user was exposed

---

### What "Shadow Mode" Actually Does

```
Normal flow (no shadow):
  User request ──→ [Production Model v2.4.1] ──→ Response to user

Shadow flow:
  User request ──→ [Production Model v2.4.1] ──→ Response to user
       │
       └─── (copy of same request)
                    ↓
             [Shadow Model v2.5.0] ──→ Response DISCARDED
                                        (never sent to user)
                                        logged internally for analysis
```

The user always gets v2.4.1's response. v2.5.0 sees the same exact input, computes its own output, but that output goes **only to your internal logging/metrics system** — never to the user.

---

### Why Run a Shadow Period?

```
Canary:    5% of real users receive new model output
           → if model is wrong, 5% of users are affected
           → acceptable for most systems, not for safety-critical ones

Blue-Green: 100% of users flip instantly
           → if model is wrong, all users are affected briefly

Shadow:    0% of users ever receive new model output
           → you can run it for days/weeks with zero risk
           → you get real production distribution data
           → you compare outputs systematically before any exposure
```

Shadow gives you the **most honest evaluation possible** — not a held-out test set (which may be stale), not a synthetic benchmark, but the actual live traffic your system sees today.

---

### What Gets Measured During the Shadow Period

```
For each incoming request, you record:

  request_id:       req_8472910
  timestamp:        2026-04-29T08:14:33Z
  input:            [video frame tensor]
  v2.4.1 output:    {class: "hard_braking", confidence: 0.91}
  v2.5.0 output:    {class: "hard_braking", confidence: 0.94}
  agreement:        ✓ (both models agree)

Aggregated over 7 days:

  Agreement rate:          94.2%   ← how often do models agree?
  v2.5.0 higher conf:      71%     ← new model more certain on same inputs
  v2.5.0 lower latency:    18ms vs 20ms  ← faster too
  Disagreement cases:      5.8%    ← where do they differ? → human review
  v2.5.0 error rate:       0.003%  ← infrastructure errors (OOM, timeout)
```

---

### The 7-Day Shadow Period — Why That Duration?

The duration is chosen to cover **natural variation** in your production traffic:

```
Day 1 (Mon):  weekday morning commute traffic
Day 2 (Tue):  normal weekday
Day 3 (Wed):  normal weekday
Day 4 (Thu):  normal weekday
Day 5 (Fri):  end-of-week patterns
Day 6 (Sat):  weekend driving patterns (different from weekday)
Day 7 (Sun):  weekend driving patterns
────────────────────────────────────────────────
Full 7 days:  model has seen a complete weekly cycle of distribution
              including day/night, weekday/weekend, rush hour/off-peak
```

If you ran shadow for only 1 day, you might miss the weekend distribution, night distribution, or a rare edge case that only appears on certain days.

---

### What Happens at the End of the Shadow Period

```
Shadow period ends: 2026-05-04

Review meeting / automated gate:

  ✓ Agreement rate > 90%?          → 94.2% ✓
  ✓ No catastrophic disagreements? → reviewed 5.8% cases, none critical ✓
  ✓ Latency within budget?         → 18ms < 25ms limit ✓
  ✓ Error rate acceptable?         → 0.003% < 0.01% threshold ✓
  ✓ Slice metrics (night, rain)?   → night improved 0.79→0.83 ✓

Decision: PROMOTE v2.5.0 to production

Next step:
  Option A: Direct Blue-Green switch (v2.5.0 is already proven)
  Option B: Short canary (5%→100% over 1 day) as final safety check
```

---

### Shadow vs A/B Test — Common Confusion

```
Shadow deployment:
  New model NEVER serves users
  Comparing model outputs to each other
  Goal: "is the new model safe to release?"
  No user impact possible

A/B test:
  Both models serve real users (different user groups)
  Comparing user outcomes (clicks, complaints, conversions)
  Goal: "which model produces better user behavior?"
  Users ARE affected by whichever variant they get

Use shadow when:   you need zero risk before any exposure
Use A/B when:      you need real user outcome data to make a decision
```

---

### Cost of Shadow Deployment

```
During shadow period:
  v2.4.1: running normally (production cost)
  v2.5.0: running in parallel (same hardware cost)
  → 2× inference compute cost for the shadow duration

7 days × 2× compute = 14 device-days of extra inference cost

Trade-off:
  Cost of 7 days double compute     vs    cost of a bad model in production
  = manageable infrastructure cost  vs    user harm, incident response,
                                          emergency rollback, trust damage

For safety-critical systems: shadow period cost is always worth it
```

---

**Interview one-liner:**
> `Shadow period: 2026-04-28 → 2026-05-04` means the new model ran for 7 days receiving a mirror of all real production requests, computing predictions, but never returning them to users — all outputs were logged internally and compared against the live model. The 7-day window ensures the model is evaluated across a full weekly traffic cycle (day/night, weekday/weekend) before any user is exposed, making it the safest possible pre-promotion validation.

---

## Register Model → Status: Staging — Deep Dive

---

### What the Line Means

```
Register model → status: Staging
```

- **Register model** = the training job has finished, metrics have been computed, and the artifact (weights file + metadata) is being written into the model registry as a tracked version
- **→** = transition / assignment
- **status: Staging** = the model's lifecycle state is set to `Staging` — it has passed automated quality gates but is **not yet in production**; it is a candidate under evaluation

It is the boundary between **"experiment that ran"** and **"candidate that might ship"**.

---

### The Full Context — Where This Step Sits

```
Training job runs
      ↓
Metrics computed (mAP, F1, latency, slice metrics)
      ↓
Automated quality gates evaluated:
  mAP > threshold?        ✓
  Latency < budget?       ✓
  No regression vs prod?  ✓
  Fairness check?         ✓
      ↓
  ALL PASS → Register model → status: Staging   ← THIS LINE
      ↓
  ANY FAIL → artifact discarded / status: Failed
                 (never enters registry in a promotable state)
```

The registration itself only happens **after gates pass**. A model that fails quality gates never reaches Staging.

---

### What "Registering" Actually Does

When you call `register model`, the registry system does the following atomically:

```
1. Generate a unique version ID
     model_name:    "vehicle_detector"
     version:       v2.5.0
     run_id:        train_job_892
     artifact_uri:  s3://models/vehicle_detector/v2.5.0/model.onnx

2. Compute and store artifact checksum
     sha256: 3f8a9c2e...  ← guarantees artifact integrity
     nobody can silently swap the weights file later

3. Record full lineage
     code_commit:   git SHA a1b2c3d
     data_version:  dataset_v3.3
     config:        hyperparams_v4.yaml
     environment:   docker://train:cuda12.1-pt2.1

4. Store evaluation metrics
     mAP: 0.851
     F1:  0.829
     latency_p50: 16ms
     latency_p99: 22ms
     slice_night: 0.83
     slice_rain:  0.80

5. Set lifecycle status = Staging

6. Emit event / webhook
     → triggers downstream CI/CD pipeline
     → notifies team channel
     → potentially triggers shadow deployment automatically
```

All of this is written as a single transaction. The model is now **permanently in the registry** — immutable, versioned, traceable.

---

### What "Staging" Status Means

Staging is not a vague "almost ready" state. It has a precise meaning:

```
Staging means:
  ✓ Passed automated quality gates
  ✓ Artifact is valid and checksummed
  ✓ Metrics are within acceptable range
  ✗ NOT yet validated on live production traffic
  ✗ NOT yet approved for user-facing serving

Staging is a holding area for models that are:
  - Awaiting shadow deployment
  - Awaiting human review
  - Awaiting A/B test setup
  - Awaiting compliance/safety sign-off
```

---

### The Full Lifecycle — All Status Values

```
┌─────────────┬────────────────────────────────────────────────────┐
│  Status     │  Meaning                                           │
├─────────────┼────────────────────────────────────────────────────┤
│ Candidate   │ Training job finished, quality gates not yet run   │
├─────────────┼────────────────────────────────────────────────────┤
│ Staging     │ Passed automated gates                             │
│             │ Under evaluation (shadow, human review, A/B)       │
│             │ Not serving users                                  │
├─────────────┼────────────────────────────────────────────────────┤
│ Production  │ Actively serving real user traffic                 │
├─────────────┼────────────────────────────────────────────────────┤
│ Deprecated  │ Superseded by newer Production model               │
│             │ Kept for rollback window (e.g., 30 days)           │
├─────────────┼────────────────────────────────────────────────────┤
│ Archived    │ Past rollback window, retained for audit/compliance │
├─────────────┼────────────────────────────────────────────────────┤
│ Failed      │ Did not pass quality gates, never promotable       │
└─────────────┴────────────────────────────────────────────────────┘
```

---

### What Happens After Staging

```
Status: Staging
    │
    ├── Path A: Shadow deploy
    │     Run model on mirrored traffic for N days
    │     Compare outputs to current production model
    │     If agreement rate + metrics OK → promote
    │
    ├── Path B: Human review gate
    │     Senior ML engineer reviews slice metrics
    │     Signs off → promote
    │
    ├── Path C: Compliance / safety review
    │     Audit trail review, explainability report
    │     Legal / safety team signs off → promote
    │
    └── Path D: Automated promotion
          If ALL thresholds met AND no human gate configured
          System automatically promotes to Production
          (only for low-risk model updates)
```

---

### The Quality Gates That Determine If Staging Is Reached

```
PERFORMANCE gates:
  mAP_new > mAP_production × 0.98   ← no regression (2% tolerance)
  F1_new  > F1_production  × 0.98
  latency_p99 < latency_budget_ms

SLICE gates (critical — don't just check aggregate):
  mAP_night > threshold_night
  mAP_rain  > threshold_rain
  mAP_heavy_traffic > threshold_heavy
  → prevents a model that's great on average but terrible at night

INFRASTRUCTURE gates:
  Model loads without error
  Model responds to sample request
  Output schema matches expected format
  Memory footprint within device budget

BIAS / FAIRNESS gates:
  Performance gap between subgroups < max_gap
  False positive rate parity check

If ANY gate fails → status: Failed, not Staging
```

---

### Multiple Staging Candidates

```
At any moment, the registry might have:

  vehicle_detector v2.3.1 → Production
  vehicle_detector v2.4.0 → Staging  (from last week's training run)
  vehicle_detector v2.4.1 → Staging  (from today's training run)
  vehicle_detector v2.2.0 → Deprecated

Two Staging candidates for the same model.
Registry tracks all simultaneously.

Team decision:
  Compare v2.4.0 vs v2.4.1 in shadow
  Promote the better one
  Move the other to Deprecated or Archive
```

---

### In Code (MLflow Example)

```python
import mlflow
from mlflow.tracking import MlflowClient

# After training job completes
with mlflow.start_run() as run:
    mlflow.log_params({"lr": 1e-4, "epochs": 50})
    mlflow.log_metrics({"mAP": 0.851, "latency_p99": 22})
    mlflow.pytorch.log_model(model, "model")

# Register model → status: Staging
client = MlflowClient()
model_version = client.create_model_version(
    name="vehicle_detector",
    source=f"runs:/{run.info.run_id}/model",
    run_id=run.info.run_id
)

# Transition to Staging
client.transition_model_version_stage(
    name="vehicle_detector",
    version=model_version.version,
    stage="Staging"
)
# Model is now in registry, status = Staging
# Downstream webhook fires → triggers shadow deployment pipeline
```

---

**Interview one-liner:**
> `Register model → status: Staging` means the training job's artifact has passed all automated quality gates (accuracy, latency, slice metrics, schema checks) and has been formally entered into the model registry as a tracked, immutable, versioned entry. Staging is a holding state — the model is a validated candidate but has not yet been proven on live production traffic. It then goes through shadow deployment or human review before being promoted to Production.

---

## Canary → Ramp → Production — Deep Dive

---

### What the Line Means

```
Canary → ramp → Production
```

- **Canary** = start by sending a small slice of real traffic (5–10%) to the new model
- **ramp** = incrementally increase that percentage as confidence grows
- **Production** = 100% of traffic now flows to the new model; it is the production model

It's a **controlled, reversible promotion process** — not a single event but a sequence of checkpoints.

---

### Why "Canary"?

The name comes from coal miners who carried canary birds underground. If toxic gas was present, the canary died first — giving miners an early warning before they were harmed.

```
In deployment:
  The canary = the small fraction of traffic (5%) exposed to the new model
  The miners = the remaining 95% of users still on the old model

If the new model fails:
  The "canary" traffic shows the problem first
  95% of users are completely unaffected
  You roll back before the damage spreads
```

---

### The Full Ramp Sequence

```
Time →
────────────────────────────────────────────────────────────────────

t=0      Deploy v2.5.0 to canary slot
         Route: 5% → v2.5.0
                95% → v2.4.1 (production)

         Monitor for 1–2 hours:
           Error rate? Latency? Output distribution?
           ✓ All OK → proceed

t=2h     Ramp to 20%
         Route: 20% → v2.5.0
                80% → v2.4.1

         Monitor for 2–4 hours:
           ✓ All OK → proceed

t=6h     Ramp to 50%
         Route: 50% → v2.5.0
                50% → v2.4.1

         Monitor for 4–8 hours:
           ✓ All OK → proceed

t=14h    Ramp to 100%
         Route: 100% → v2.5.0
                0%   → v2.4.1

         v2.5.0 is now Production
         v2.4.1 is now Deprecated (kept warm for rollback window)
```

---

### How Traffic Is Split — The Mechanics

#### Method 1: Request-Level Splitting (stateless)

```
Each incoming request is independently assigned:
  Random number r ∈ [0, 1]
  if r < 0.05 → send to v2.5.0
  else        → send to v2.4.1

Result: statistically ~5% of all requests go to new model
Pros: simple, no state needed
Cons: same user can get v2.4.1 on one request, v2.5.0 on next
      → inconsistent experience within a session
```

#### Method 2: User/Device-Level Splitting (sticky)

```
Hash the user_id or device_id:
  bucket = hash(device_id) % 100
  if bucket < 5 → always route to v2.5.0
  else          → always route to v2.4.1

Result: a specific device ALWAYS sees the same model version
Pros: consistent experience per device
Cons: the "canary" cohort is fixed — must verify it's representative
```

For ML inference (stateless requests), request-level splitting is usually fine.
For user-facing products, sticky routing is preferred.

---

### What Gets Monitored at Each Ramp Stage

```
LATENCY:
  p50, p95, p99 response time
  Alert threshold: p99 > 1.5× baseline
  Why: new model might be heavier, slower on edge cases

ERROR RATE:
  HTTP 5xx errors, inference exceptions, OOM crashes
  Alert threshold: error_rate > 0.1%
  Why: model might crash on certain input shapes

OUTPUT DISTRIBUTION:
  Class prediction histogram
  Mean confidence score
  Alert if: distribution shifts significantly from v2.4.1
  Why: model might be overconfident or predicting wrong classes

BUSINESS METRICS (if ground truth available fast enough):
  False positive rate (human review override rate)
  Precision on high-stakes events

RESOURCE METRICS:
  GPU memory usage, CPU utilization
  Alert if: resource usage higher than expected
  Why: new model might not fit within device budget at scale
```

---

### The Decision Logic at Each Checkpoint

```
At each ramp stage, three possible outcomes:

  1. PROCEED (all metrics within thresholds)
       → increase traffic percentage
       → continue monitoring

  2. HOLD (metrics borderline, not clearly failing)
       → stay at current percentage
       → extend monitoring window
       → investigate specific failing cases

  3. ROLLBACK (metric breaches threshold)
       → immediately revert traffic to 0% on new model
       → 100% back to v2.4.1
       → blast radius = only the % that was on canary at that moment
```

---

### Blast Radius — Why Canary Limits Damage

```
Scenario: v2.5.0 has a critical bug on night-time frames

Without canary (Blue-Green hard cut):
  t=0: 100% of traffic switches to v2.5.0
  t=5min: bug discovered
  100% of users experienced the bug for 5 minutes
  Blast radius = 100%

With canary:
  t=0:    5% traffic on v2.5.0
  t=3min: automated alert fires (error rate threshold breached)
  Rollback executed
  Blast radius = 5% for 3 minutes ≈ very small
```

---

### Ramp Speed — How Fast Should You Increase?

```
Aggressive ramp (when confident):
  5% → 25% → 100%
  Over: 2 hours total
  When: model is minor update (fine-tune only), well-understood changes

Conservative ramp (safety-critical):
  1% → 5% → 20% → 50% → 100%
  Over: 3–7 days
  When: architecture change, new data distribution, safety-critical system

Factors that should slow your ramp:
  High-stakes outcomes (medical, safety, financial)
  Large architecture change
  New geographic region / new data distribution
  Limited observability

Factors that allow faster ramp:
  Extensive shadow period completed
  Minor hyperparameter / fine-tune change
  Strong automated monitoring with fast alerting
```

---

### Canary in Kubernetes / Istio (Concrete Infrastructure)

```yaml
# Istio VirtualService — traffic splitting
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-inference
spec:
  http:
  - route:
    - destination:
        host: model-v241     # old production
      weight: 95             # 95% of traffic
    - destination:
        host: model-v250     # canary
      weight: 5              # 5% of traffic
```

To ramp: change `weight: 5` → `weight: 20` → `weight: 50` → `weight: 100`.
Each change is a single config update — no redeployment needed.

---

### The Three Strategies Are a Sequence, Not Alternatives

```
Step 1: Shadow deploy (zero risk, real traffic comparison)
  Run for 7 days, collect agreement metrics
  Gate: agreement > 90%, no catastrophic disagreements

Step 2: Canary (controlled real exposure)
  5% → 20% → 50% → 100% over hours/days
  Gate: latency, error rate, output distribution at each stage

Step 3: Full Production (Blue-Green atomic switch at 100%)
  Old model retained as Deprecated for rollback window

Shadow gives pre-release confidence.
Canary limits blast radius during promotion.
Blue-Green enables instant rollback after full ramp.
```

---

### Summary

```
Canary → ramp → Production:

  Canary:     send 5% of real traffic to new model
              monitor → proceed / hold / rollback

  Ramp:       incrementally increase %
              5% → 20% → 50% → 100%
              gate at each step: latency, error rate, output distribution

  Production: 100% traffic on new model
              old model retained as Deprecated for rollback window

Key safety property: blast radius at any step = current traffic %
                     problem at 5% canary → only 5% of users affected
```

---

**Interview one-liner:**
> Canary → ramp → Production is a gradual traffic migration strategy where a new model starts receiving a small fraction (5%) of real requests, is monitored for latency, error rate, and output distribution at each step, and has its traffic percentage incrementally increased only if all metrics remain within thresholds — so if a failure is detected at any stage, the rollback only affects the fraction of traffic already on the new model, not all users. The ramp speed is calibrated to the risk level: hours for minor updates, days for architecture changes or safety-critical systems.

---

## Model Registry = Central Catalog of All Versions + Lifecycle State

---

### The Core Analogy

Think of a model registry the same way you think of a **library catalog**.

```
Library catalog:
  Every book has:   title, author, edition, location, availability status
  You can search:   "find all editions of book X"
  You can ask:      "which copy is currently checked out?"
  You can track:    when it was added, who borrowed it, where it is now

Model registry:
  Every model has:  name, version, artifact location, metrics, status
  You can search:   "find all versions of vehicle_detector"
  You can ask:      "which version is currently in production?"
  You can track:    who trained it, what data, what metrics, what stage
```

Without a catalog, a library is just a room full of books with no way to find anything. Without a registry, a model zoo is just a folder of `.onnx` files with no provenance, no state, no history.

---

### What "Central Catalog" Means

**Central** = single source of truth. Every team, every system, every pipeline reads from and writes to the same registry.

```
WITHOUT central catalog:

  Team A:  "our latest model is in s3://models/team-a/latest.pt"
  Team B:  "we have model_v3_final_FINAL2.pt on the NAS drive"
  Infra:   "production is running... we think it's the one from March?"
  Audit:   "which model was running on 2026-01-15?" → nobody knows

  Result:
  - No one agrees on what's deployed
  - Can't reproduce past results
  - Incident response is chaotic
  - Compliance/audit is impossible

WITH central catalog (registry):

  Everyone queries the same system:
  GET /registry/models/vehicle_detector?status=Production
  → returns exactly one answer: v2.4.1, deployed 2026-05-05, metrics: {...}

  Single source of truth eliminates ambiguity entirely
```

---

### What "All Versions" Means

The registry retains **every version ever registered** — not just the current one. It is append-only by design:

```
vehicle_detector version history:

  v1.0.0  →  Archived    trained: 2025-01-10  mAP: 0.71
  v1.1.0  →  Archived    trained: 2025-02-14  mAP: 0.74
  v1.3.2  →  Archived    trained: 2025-06-20  mAP: 0.79
  v2.0.0  →  Archived    trained: 2025-08-03  mAP: 0.82  (arch change: SSD→YOLO)
  v2.1.0  →  Deprecated  trained: 2025-11-15  mAP: 0.83
  v2.4.1  →  Production  trained: 2026-05-05  mAP: 0.847
  v2.5.0  →  Staging     trained: 2026-05-06  mAP: 0.851
```

This history enables:

```
1. Rollback:
   Production model v2.4.1 has a bug
   → roll back to v2.1.0 immediately (artifact still exists, still runnable)

2. Regression analysis:
   "Performance dropped between v2.0.0 and v2.1.0"
   → diff their lineage: what changed in data? code? config?

3. Reproducibility:
   "Reproduce the model from the compliance report filed in August 2025"
   → load v2.0.0 exactly: same weights, same code commit, same data hash

4. Audit trail:
   Regulator asks: "What model was making decisions on 2025-09-01?"
   → query registry by date → v2.0.0, with full lineage
```

---

### What "Lifecycle State" Means

Every model version carries a **state** — not just metadata, but a machine-readable status that controls what systems can do with it.

```
┌──────────────────────────────────────────────────────────────────┐
│                      LIFECYCLE STATE MACHINE                     │
│                                                                  │
│   [Training]                                                     │
│       ↓                                                          │
│   [Candidate] ──── fails quality gates ────→ [Failed]           │
│       ↓ passes quality gates                                     │
│   [Staging]                                                      │
│       ↓ shadow/canary validates it                               │
│   [Production] ──── superseded by newer version ──→ [Deprecated]│
│                                                        ↓        │
│                                                   [Archived]    │
└──────────────────────────────────────────────────────────────────┘
```

Each state has **enforced rules**:

```
State: Candidate
  Can be served to users?    NO
  Can be promoted?           NO (not yet evaluated)

State: Staging
  Can be served to users?    NO (shadow/testing only)
  Can be promoted to Prod?   YES (after shadow + human gate)

State: Production
  Can be served to users?    YES (this is the live model)
  How many can exist?        ONE per model name per endpoint

State: Deprecated
  Can be served to users?    NO (not actively serving)
  Can be used for rollback?  YES (artifact still available)
  Retention window:          typically 30–90 days

State: Archived
  Can be used for rollback?  NO (too old, for compliance only)
  Retention window:          years (depends on regulation)

State: Failed
  Can ever be promoted?      NO
  Purpose:                   record of what didn't work + why
```

---

### The Registry's Data Model — What It Actually Stores

```
MODEL (top-level entity):
  model_name:       "vehicle_detector"
  description:      "Detects vehicles, pedestrians, cyclists from dashcam"
  owner:            "perception-team"
  tags:             ["safety-critical", "edge-deployed"]

MODEL VERSION (one per training run):
  version_id:       "v2.4.1"
  model_name:       "vehicle_detector"
  status:           "Production"
  artifact_uri:     "s3://models/vehicle_detector/v2.4.1/model.onnx"
  artifact_hash:    sha256: "3f8a9c2e..."
  run_id:           "train_job_789"
  code_commit:      "git:f3a91c2"
  data_version:     "dataset_v3.2"
  config:           {lr: 1e-4, epochs: 50, batch: 32}
  environment:      "docker://train:cuda12.1-pt2.1"
  created_by:       "ci_pipeline"
  promoted_by:      "alice@company.com"

METRICS (linked to version):
  mAP: 0.847, F1: 0.812
  latency_p50: 14ms, latency_p99: 18ms
  slice_night: 0.81, slice_rain: 0.79

LIFECYCLE TRANSITIONS (audit log):
  timestamp         from_status   to_status    actor
  2026-05-05 14:32  —             Candidate    ci_pipeline
  2026-05-05 14:35  Candidate     Staging      auto_gate
  2026-05-05 14:36  Staging       ShadowDeploy deploy_system
  2026-05-05 22:14  ShadowDeploy  Production   alice@company.com
```

---

### How the Registry Connects to Everything Else

```
┌────────────────────────────────────────────────────────────────┐
│                       MODEL REGISTRY                           │
│                    (central catalog)                           │
└────────┬──────────────┬──────────────┬──────────────┬─────────┘
         │              │              │              │
         ↓              ↓              ↓              ↓
  [Training        [Deployment    [Monitoring    [Experiment
   Pipeline]        System]        System]        Tracker]

  "Did this        "What is       "Which model   "Compare
   training run     the current    version is     v2.4.1 vs
   produce a        Production     generating     v2.5.0 metrics
   better model     artifact to    this alert?"   side by side"
   than current      serve?"
   production?"

Each system queries the registry via API:
  GET /models/vehicle_detector/production     → returns v2.4.1
  GET /models/vehicle_detector/staging        → returns v2.5.0
  GET /models/vehicle_detector/versions       → returns all versions
  POST /models/vehicle_detector/v2.5.0/stage → transition to Production
```

---

### Enforcement — Why State Matters in Code

```python
# Deployment system reads from registry
def get_production_model(model_name: str) -> ModelArtifact:
    versions = registry.list_versions(model_name, status="Production")
    assert len(versions) == 1, "Exactly one Production model must exist"
    return versions[0].artifact

# Training pipeline checks before promoting
def promote_to_production(model_name: str, version: str):
    v = registry.get_version(model_name, version)
    assert v.status == "Staging", "Can only promote from Staging"
    assert v.metrics["mAP"] > get_production_metrics(model_name)["mAP"] * 0.98
    registry.transition(model_name, version, to_status="Production")
    registry.transition(model_name, current_production, to_status="Deprecated")
```

The registry **enforces invariants**:
- Only one version can be `Production` at a time per model
- Can only promote `Staging` → `Production` (not `Failed` → `Production`)
- Transitions are logged and reversible within the retention window

---

### Comparison: With vs Without Registry

```
WITHOUT registry:

  "Which model is in prod?"       → check with 3 people, get 3 answers
  "Roll back to last week's"      → find the file, hope it's the right one
  "Why did this prediction fail?" → no way to trace to training conditions
  "Reproduce 6-month-old model"   → impossible, data/code/config lost
  "Who approved this model?"      → nobody knows

WITH registry:

  "Which model is in prod?"       → GET /models/vehicle_detector/production
  "Roll back to last week's"      → registry.transition(v2.1.0, "Production")
  "Why did this prediction fail?" → trace run_id → code commit → data batch
  "Reproduce 6-month-old model"   → run_id + code_commit + data_version = exact replica
  "Who approved this model?"      → lifecycle_transitions log: alice@company.com, 22:14
```

---

### Summary

```
Model registry = central catalog of all versions + lifecycle state

Central catalog:
  Single source of truth — no ambiguity about what's deployed
  All teams read/write the same system via API
  Append-only history — every version retained

All versions:
  Every training run that passes gates gets a registry entry
  Full lineage stored: code + data + config + env + metrics
  Enables rollback, reproducibility, regression analysis, audit

Lifecycle state:
  Candidate → Staging → Production → Deprecated → Archived
  Each state has enforced rules (who can serve it, promote it, delete it)
  Every transition is logged: who, when, why
  State is machine-readable → other systems act on it automatically
```

---

**Interview one-liner:**
> The model registry is the single source of truth for all trained model artifacts — it stores every version with full lineage (code, data, config, metrics) and tracks each version through a lifecycle state machine (Candidate → Staging → Production → Deprecated → Archived) with enforced rules and an immutable audit log. Without it, you cannot answer "what's in production right now, who approved it, and can we roll back?" — which are exactly the questions asked during every production incident and every compliance audit.

---

## Latency Percentiles: p50, p95, p99

---

### The Core Problem With "Average Latency"

The first instinct is to measure average (mean) response time. This is almost always the wrong metric.

```
10 requests with response times (ms):
  [12, 11, 13, 12, 14, 11, 12, 13, 250, 280]

Mean = (12+11+13+12+14+11+12+13+250+280) / 10 = 62.8ms

Reality:
  8 out of 10 users got ~12ms (fast, great experience)
  2 out of 10 users got ~265ms (terrible experience)

The mean (62.8ms) describes nobody's actual experience.
It hides the slow tail completely.
```

**Percentiles** solve this. They describe the distribution honestly.

---

### What a Percentile Is — First Principles

$$p_k = \text{the value below which } k\% \text{ of observations fall}$$

To compute a percentile:
1. Sort all response times in ascending order
2. Find the value at position $k\%$ of the way through the sorted list

```
Example: 1000 requests, sorted response times (ms):
  position 1    → 8ms   (fastest request)
  position 500  → 14ms  ← p50: 50% of requests were faster than this
  position 950  → 38ms  ← p95: 95% of requests were faster than this
  position 990  → 142ms ← p99: 99% of requests were faster than this
  position 1000 → 580ms (slowest request)
```

---

### p50 — The Median (Typical User Experience)

```
p50 = 14ms

Meaning: 50% of requests completed in under 14ms
         50% of requests took 14ms or longer

p50 is the median — not the mean.
It tells you: "what does a typical user experience?"

Why use median over mean?
  Mean is pulled upward by a few very slow requests
  Median is resistant to outliers
  p50 = 14ms means half your users are getting ≤14ms
  → a concrete, honest description of the typical case

When p50 is good but p95/p99 are bad:
  Most users are happy, but a visible minority are suffering
  → there's a tail problem, not a general problem
```

---

### p95 — The "Almost Everyone" Threshold

```
p95 = 38ms

Meaning: 95% of requests completed in under 38ms
         5% of requests took 38ms or longer

Out of 1,000,000 requests: 50,000 requests exceeded 38ms
                           → 50,000 slow experiences per million

p95 is the standard SLA threshold for most web services.
"Our API responds in under 50ms p95" is a common SLA contract.

Why 95%?
  It's the boundary between "normal variation" and "tail behavior"
  Below p95: system is behaving predictably
  Above p95: you're in the slow tail — worth investigating
```

---

### p99 — The Tail (Where Problems Hide)

```
p99 = 142ms

Meaning: 99% of requests completed in under 142ms
         1% of requests took 142ms or longer

Out of 1,000,000 requests: 10,000 requests exceeded 142ms

p99 is critical for:
  High-volume systems: 1% of 10M requests/day = 100,000 slow requests
  Safety-critical systems: even 1% slow responses can cause missed events
  Real-time inference: model output arrives too late → downstream fails

p99 catches problems that p50 and p95 miss completely:
  GC pauses, cold-start delays, memory pressure, lock contention,
  occasional large input (high-res frame), GPU thermal throttling
```

---

### Visualizing the Full Distribution

```
Response time distribution:

Requests
  ↑
  │  ████████████████████████████████████████████
  │  ████████████████████████████████████████████████
  │  ██████████████████████████████████████████████████████
  │                                              ████████
  │                                                       ████
  └────────────────────────────────────────────────────────────→
  0ms    10ms   20ms   30ms   40ms  ...  140ms  ...  500ms

         ↑              ↑                  ↑
        p50            p95                p99
       (14ms)         (38ms)            (142ms)
```

The long right tail is called **"tail latency"** — the small fraction of requests that take much longer than the typical case. This tail is where real-world problems live.

---

### Why the Tail Exists — Root Causes

```
GC pauses (garbage collection):
  JVM / Python GC pauses the process for 10–200ms
  During GC: all requests queue up → slow response
  Happens unpredictably → shows up in p99, not p50

Cold start:
  First request after deployment: model not yet in GPU memory
  Model loading time (100ms–2s) added to first few requests

Memory pressure:
  When RAM/VRAM is near capacity
  OS starts swapping → latency explodes for some requests

Large input variation:
  Most frames: 640×480, fast inference
  Occasionally: high-res frame or dense scene → 3–5× longer inference
  → shows up in p99

Lock contention / batching wait:
  GPU batch inference: wait for batch to fill before processing
  If batch fills slowly: some requests wait longer

Network jitter:
  Occasional packet retransmission, routing change
  → shows up in p99, not p50
```

---

### The Percentile Hierarchy

```
Metric     Tells you                              Typical SLA
────────   ──────────────────────────────         ────────────────────
p50        Typical user experience                < 20ms (real-time CV)
p95        Near-worst-case for 95%                < 50ms
p99        Tail behavior                          < 100ms
p99.9      Extreme tail (1 in 1000)               < 500ms
max        Single worst request (not useful for SLAs — too noisy)
```

Never use `max` for SLAs — one network hiccup or GC pause makes max meaningless.

---

### Fan-Out Amplification — Why p99 Matters More at Scale

```
Simple service: 1 model call per user request
  1% of requests are slow → 1% of users affected

Complex pipeline: 1 user request → calls 5 microservices
  Each service has p99 = 1% slow
  P(at least one slow) = 1 - (0.99)^5 = ~5% of requests are slow

With 10 services:
  P(at least one slow) = 1 - (0.99)^10 = ~10%

Conclusion:
  In a microservices architecture:
  system-level slow% ≈ 1 - (1 - p99_per_service)^N

  What looks like "only 1% slow" per service becomes
  10–20% of end-to-end requests being slow.
  This is why tail latency optimization is critical at scale.
```

---

### How to Measure and Report Correctly

```
WRONG: compute average response time       → hides tail
WRONG: compute p99 over 24-hour window     → good hours average out bad hours

CORRECT: rolling p99 over short windows (1 min, 5 min)
         → catches bursts and spikes immediately

CORRECT: segment by:
  - Time of day (night traffic patterns differ)
  - Request type (video frame vs metadata query)
  - Device type (edge device vs cloud)
  - Model version (compare p99 before/after deployment)

Tools:
  Prometheus + Grafana:  histogram_quantile(0.99, ...)
  Datadog:               p99 as first-class metric
  AWS CloudWatch:        percentile statistics on custom metrics
  OpenTelemetry:         distributed tracing with latency histograms
```

---

### Latency Budget — Using Percentiles in System Design

```
Real-time dashcam inference pipeline:
  Total latency budget: 100ms (real-time constraint)

  Breakdown (p99):
  Camera capture → encode:     5ms
  Network (edge → processor):  10ms
  Preprocessing (resize/norm): 8ms
  Model inference (GPU):       25ms
  Postprocessing (NMS, etc):   5ms
  Result logging:              3ms
  ─────────────────────────────────
  Total p99:                   56ms  ← well within 100ms

If model inference p99 spikes to 80ms:
  Total = 111ms → over budget → real-time constraint violated
  → alert fires → investigate model or hardware
```

---

### p50 vs p99 in Practice — A Real Scenario

```
Before optimization:
  p50 = 18ms   ← looks fine
  p95 = 45ms   ← acceptable
  p99 = 320ms  ← PROBLEM: 1% of frames take 320ms

  Root cause: high-density frames (crowded intersections)
              take 18× longer due to NMS over many boxes

After fix (NMS optimization + max_det cap):
  p50 = 16ms   ← minor improvement
  p95 = 22ms   ← much better
  p99 = 38ms   ← FIXED: tail reduced by 8.4×

  You would have completely missed this problem
  if you only monitored p50 or mean.
```

---

### Summary

```
p50 (median):  50% of requests faster than this
               = typical user experience
               Use for: "is the system generally fast?"

p95:           95% of requests faster than this
               = standard SLA threshold
               Use for: "is the system reliably fast for almost everyone?"

p99:           99% of requests faster than this
               = tail latency threshold
               Use for: "are there pathological slow cases?"

Key insight:
  Mean hides the tail.
  p50 describes typical experience.
  p99 describes the worst experience your system regularly delivers.
  In safety-critical real-time systems, p99 is the number that matters.
```

---

**Interview one-liner:**
> p50, p95, p99 are latency percentiles — p50 is the median (typical user experience), p95 is the threshold below which 95% of requests complete (standard SLA metric), and p99 captures the slow tail that affects 1% of requests. The mean is useless for latency because it hides the tail — p99 is what matters for production SLAs because at 1M requests/day, 1% slow = 10,000 bad experiences, and in fan-out pipelines with N services, system-level p99 degrades as approximately $1-(1-p99_{service})^N$.

---

## Serving Infrastructure: Edge / Cloud / Hybrid

---

### What Is Serving Infrastructure?

Training a model is a one-time (or periodic) job. **Serving** is the continuous, production system that takes a trained model artifact and makes it available to answer real requests in real time.

```
Training job:
  Runs once (or periodically)
  Uses lots of compute, hours/days
  Produces: model artifact (weights file)

Serving infrastructure:
  Runs forever (24/7/365)
  Must respond in milliseconds
  Consumes: model artifact
  Produces: predictions on demand
```

The fundamental question is: **where does inference run?**

```
Option A: EDGE   — inference runs on the device that collects the data
Option B: CLOUD  — inference runs on centralized servers
Option C: HYBRID — some inference on edge, some on cloud, coordinated
```

---

### Option A: Edge Inference

The model runs **directly on the device** — a camera, robot, car, phone, embedded board — where the data is generated.

```
┌─────────────────────────────────────────────────────────┐
│                    EDGE DEVICE                          │
│                                                         │
│  Camera sensor                                          │
│      ↓ raw frames                                       │
│  [Preprocessing]  (resize, normalize)                   │
│      ↓ tensor                                           │
│  [Model Inference]  (TensorRT / ONNX / TFLite)         │
│      ↓ prediction                                       │
│  [Postprocessing]  (NMS, thresholding)                  │
│      ↓ result                                           │
│  [Action / Log]   (trigger alert / store event)         │
│                                                         │
│  Network: NOT REQUIRED for inference                    │
└─────────────────────────────────────────────────────────┘
```

#### Why Edge?

```
1. LATENCY:
   Cloud round-trip: capture → encode → transmit → infer → transmit back
   = 50–500ms depending on network
   Edge inference: capture → infer locally = 5–30ms

   For real-time safety systems: latency = correctness
   A collision warning 200ms late is useless

2. BANDWIDTH:
   Raw 1080p @ 30fps ≈ 3–8 Gbps uncompressed
   1000 edge devices × 2 Mbps = 2 Gbps continuous upload → expensive
   1000 edge devices × 10 KB/s events = 10 MB/s → trivial
   Processing on device → only send metadata/events

3. PRIVACY / COMPLIANCE:
   Raw video never leaves the device
   Only processed metadata transmitted
   Critical for: HIPAA, GDPR, defense

4. RESILIENCE:
   Device works even when network is down / intermittent
   Trucking on rural roads, underground parking, tunnels
```

#### Edge Constraints

```
COMPUTE:
  Edge SoCs: 10–100 TOPS (vs cloud GPU: 1000+ TOPS)
  → smaller models required (MobileNet, EfficientDet-Lite, YOLOv8n)

MEMORY:
  Edge: 4–16 GB RAM, 4–8 GB VRAM
  Cloud GPU: 40–80 GB VRAM
  → model must fit in constrained memory budget

POWER:
  Edge device: 10–30W power envelope
  Cloud GPU: 300–700W
  → power efficiency (TOPS/W) is a first-class metric on edge

MODEL OPTIMIZATION for edge:
  Quantization:        FP32 → INT8 (4× smaller, 2–4× faster)
  Pruning:             remove low-importance weights (30–80% sparsity)
  Knowledge distill:   train small "student" to mimic large "teacher"
  TensorRT:            fuse layers, optimize kernels for target GPU
  TFLite / ONNX:       cross-platform deployment runtimes
```

---

### Option B: Cloud Inference

The model runs on **centralized servers** (data center GPUs). Devices send data to the cloud, receive predictions back.

```
┌──────────────┐         Network          ┌─────────────────────────┐
│ EDGE DEVICE  │  ──── request ────→      │    CLOUD INFERENCE      │
│              │                          │                         │
│ Camera       │  ←─── response ────      │  Load Balancer          │
│ Sensor       │                          │      ↓                  │
└──────────────┘                          │  Inference Servers      │
                                          │  (A100 / H100 GPUs)    │
                                          │  Large batch inference  │
                                          └─────────────────────────┘
```

#### Why Cloud?

```
1. MODEL CAPABILITY:
   Full-size models: ResNet-152, ViT-Large, GPT-4-vision
   No memory/compute constraint

2. EASY UPDATES:
   New model version → deploy to cloud fleet
   vs edge: must push OTA update to thousands of devices (risky, slow)

3. CENTRALIZED MONITORING:
   All requests go through known infrastructure
   Full observability: every request logged, every latency measured

4. BATCH PROCESSING:
   Collect data all day → run large batch inference overnight
   Maximize GPU utilization, cost-efficient for non-real-time workloads
```

#### Cloud Constraints

```
LATENCY:
  Network RTT: 10–200ms — not suitable for safety-critical real-time decisions

NETWORK DEPENDENCY:
  No network = no inference
  Unacceptable for autonomous systems in connectivity-challenged environments

PRIVACY:
  Raw data leaves the device → compliance implications

COST AT SCALE:
  Cloud GPU inference: $2–6/GPU-hour
  1000 devices × continuous inference = significant monthly bill
```

---

### Option C: Hybrid Inference

Run **different parts of the ML pipeline on different tiers** — the most common architecture in production systems at scale.

```
┌──────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                           │
│                                                                  │
│  EDGE TIER (real-time, safety-critical)                         │
│  ┌──────────────────────────────────┐                           │
│  │  Small fast model (YOLOv8n)      │                           │
│  │  Latency: 15ms                   │                           │
│  │  Task: detect events, filter     │                           │
│  │  Output: "possible hard braking" │                           │
│  └──────────────┬───────────────────┘                           │
│                 │ if interesting event detected                  │
│                 ↓ (not every frame — maybe 1% of frames)        │
│  CLOUD TIER (high accuracy, complex reasoning)                  │
│  ┌──────────────────────────────────┐                           │
│  │  Large model (ViT-Large)         │                           │
│  │  Latency: 80ms (acceptable here) │                           │
│  │  Task: classify severity, context│                           │
│  │  Output: "confirmed hard braking,│                           │
│  │           severity: 8.2/10"      │                           │
│  └──────────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

#### The Two-Stage Design Pattern

```
Stage 1 (edge) — DETECTOR / FILTER:
  Runs on every frame, continuously
  Fast, lightweight model
  Purpose: "is anything interesting happening?"
  If NO → discard frame (saves 99% of bandwidth/compute)
  If YES → send relevant clip/frame to cloud

Stage 2 (cloud) — ANALYZER / CLASSIFIER:
  Runs only on flagged events (small fraction of total frames)
  Large, accurate model
  Purpose: "what exactly is happening, how severe?"
  Output: detailed classification, confidence, metadata

This pattern:
  Reduces cloud compute cost by 50–100× (only process flagged events)
  Maintains real-time safety response on edge
  Uses cloud capacity for deep analysis where latency is acceptable
```

#### Hybrid Coordination

```
Every frame (edge only):
  Raw frame → edge model → prediction → action (alert/log)
  Nothing sent to cloud

Flagged event (edge → cloud):
  Trigger: edge model confidence in [0.4, 0.7] (uncertain)
           OR edge model detects high-severity event
  Payload: compressed video clip (2–5 seconds)
           edge model output, device metadata (GPS, speed, timestamp)

Periodic model sync (cloud → edge):
  New model version → push OTA to edge devices during idle period
  Edge device validates checksum → activates new model
  Old model kept for fallback
```

---

### Hardware Options at Each Tier

#### Edge Hardware

```
Embedded / industrial:
  NVIDIA Jetson Orin  — 275 TOPS, 15–60W, $200–500
  NVIDIA Jetson Nano  — 21 TOPS, 5–10W, $100
  Google Edge TPU     — ultra-low power, INT8 only

Automotive / dashcam:
  Ambarella CV SoC    — purpose-built dashcam processor
  TI TDA4             — automotive ADAS
  Mobileye EyeQ       — dedicated ADAS chip, safety-certified
```

#### Cloud Hardware

```
Training AND inference:
  NVIDIA A100 (80GB)  — current workhorse
  NVIDIA H100 (80GB)  — latest generation, 3× A100

Inference-optimized:
  NVIDIA T4           — cost-efficient inference, 16GB
  AWS Inferentia      — purpose-built inference chip (cheaper than GPU)
  Google TPU v4       — high-throughput batched inference
```

---

### Serving Frameworks

```
TRITON INFERENCE SERVER (NVIDIA):
  Supports: TensorRT, ONNX, PyTorch, TF
  Features: dynamic batching, model ensemble, concurrent model execution
  Use case: production cloud inference, high throughput

TORCHSERVE:
  Native PyTorch model serving, REST + gRPC, model versioning

ONNX RUNTIME:
  Universal runtime: runs PyTorch, TF, sklearn via ONNX export
  Cross-platform: edge and cloud

TENSORRT:
  NVIDIA inference optimizer: fuses layers, selects optimal kernels
  Quantization: FP32 → FP16 → INT8
  Up to 6× speedup vs raw PyTorch on same GPU

TFLITE:
  TensorFlow Lite: optimized for ARM CPUs and mobile GPUs
  Use case: Android/iOS, Raspberry Pi, microcontrollers
```

---

### Batching — Critical for Cloud Throughput

```
Single request at a time:
  GPU utilization: 5–15%
  Throughput: 50 req/sec

Batched (batch_size=32):
  GPU utilization: 85–95%
  Throughput: 1,200 req/sec   (24× throughput, same hardware cost)

Dynamic batching (Triton):
  Collect requests arriving within a 5ms window
  Bundle into one batch → run inference once → return to each requester

  Trade-off:
    Adds 0–5ms latency (batching wait)
    Multiplies throughput by 10–30×
    Worth it when throughput >> latency priority
```

---

### Decision Matrix

```
Requirement              Edge    Cloud   Hybrid
───────────────────────  ──────  ──────  ──────
Latency < 20ms           ✓       ✗       ✓ (edge stage)
Works offline            ✓       ✗       ✓ (edge stage)
Large model (>1B params) ✗       ✓       ✓ (cloud stage)
Privacy (data on-device) ✓       ✗       ✓ (raw stays on edge)
Easy model updates       ✗       ✓       partial
Low bandwidth cost       ✓       ✗       ✓ (filter on edge)
High accuracy complex    ✗       ✓       ✓ (cloud stage)
Full observability       ✗       ✓       ✓ (cloud stage)
```

---

### Summary

```
Edge inference:
  Model runs ON the device
  Pros: low latency (5–30ms), works offline, privacy, low bandwidth
  Cons: constrained compute/memory, hard to update, smaller models only
  Tools: TensorRT, TFLite, ONNX Runtime, Jetson, Coral

Cloud inference:
  Model runs on centralized GPU servers
  Pros: large models, easy updates, full observability, batching
  Cons: high latency (50–500ms), network dependent, bandwidth cost
  Tools: Triton, TorchServe, A100/H100, dynamic batching

Hybrid:
  Two-stage pipeline: fast model on edge filters,
  complex model on cloud analyzes flagged events
  Pros: combines advantages of both
  Pattern: edge detector → cloud analyzer (1% of frames go to cloud)
```

---

**Interview one-liner:**
> Serving infrastructure determines where model inference runs — edge puts the model directly on the device for sub-20ms latency, offline operation, and privacy, but is constrained to smaller quantized models; cloud puts inference on centralized GPUs for large model capability and easy updates, but adds network latency and bandwidth cost; hybrid is the most common production pattern where a lightweight edge model filters events in real time and a powerful cloud model analyzes only the flagged fraction, achieving both real-time safety response and high-accuracy deep analysis at a fraction of the bandwidth and compute cost of cloud-only inference.

---

*End of notes — continued in next session.*
