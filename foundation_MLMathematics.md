# ML / AI Mathematical Foundations
## Interview Preparation — Numerical Calculation & Code

---

## Table of Contents
1. [Simple PyTorch Training Code](#simple-pytorch-training-code)
2. [CNN Math — Output Size & Parameters](#block-a--cnn-math)
3. [SGD Update — Numerical](#block-b--sgd-update-numerical)
4. [ViT Dimensions — Q, K, V](#block-c--vit-dimensions-q-k-v)
5. [Classification Metrics — Numerical](#block-d--classification-metrics-numerical)
6. [Extra Prep — Optimizer, BatchNorm, IoU, Cross-Entropy](#block-e--extra-prep-questions)

---

## Simple PyTorch Training Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# ── 1. CONFIG ──────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 5
BATCH_SIZE = 64
LR         = 1e-3

# ── 2. DATA ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_ds = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
val_ds   = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── 3. MODEL ───────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28×28 → 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14×14 → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(DEVICE)

# ── 4. LOSS & OPTIMIZER ────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ── 5. TRAIN / EVAL FUNCTIONS ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs     = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ── 6. TRAINING LOOP ───────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vl_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
          f"Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

# ── 7. SAVE ────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "simple_cnn.pth")
print("Model saved.")
```

---

## BLOCK A — CNN Math

### Q1. Output Size Formula

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

| Scenario | After Layer 1 | After Layer 2 |
|---|---|---|
| Input=32, K=3, P=1, S=1 | $\frac{32-3+2}{1}+1 = 32$ | 32 |
| Input=32, K=3, P=0, S=1 | $\frac{32-3}{1}+1 = 30$ | 28 |
| Input=32, K=3, P=0, S=2 | $\frac{32-3}{2}+1 = 15$ | $\frac{15-3}{2}+1 = 7$ |

**Key insight:** Same padding ($P=1$ with $K=3$) preserves spatial size. Stride downsamples.

---

### Q2. Parameter Count — 3 Channels → 10 Channels, Kernel=3×3

$$\text{Params} = K_h \times K_w \times C_{in} \times C_{out} + C_{out}\ (\text{bias})$$

$$= 3 \times 3 \times 3 \times 10 + 10 = 270 + 10 = \boxed{280}$$

**General rule:**
- No bias: $K^2 \times C_{in} \times C_{out}$
- With bias: $K^2 \times C_{in} \times C_{out} + C_{out}$

**Extended examples:**

| $C_{in}$ | $C_{out}$ | K | Params (with bias) |
|---|---|---|---|
| 3 | 10 | 3 | 280 |
| 32 | 64 | 3 | 18,496 |
| 64 | 128 | 3 | 73,856 |
| 1 | 32 | 5 | 832 |

---

## BLOCK B — SGD Update (Numerical)

### Setup
$$L = \frac{1}{2}(w \cdot x - y)^2, \quad w=2,\ x=3,\ y=10,\ \eta=0.01$$

### Step-by-step

**Forward pass:**
$$\hat{y} = w \cdot x = 2 \times 3 = 6$$

**Loss before update:**
$$L = \frac{1}{2}(6-10)^2 = \frac{1}{2}(16) = 8$$

**Gradient:**
$$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x = (6-10)(3) = -12$$

**Weight update:**
$$w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w} = 2 - 0.01 \times (-12) = \boxed{2.12}$$

**Loss after update:**
$$L_{\text{new}} = \frac{1}{2}(2.12 \times 3 - 10)^2 = \frac{1}{2}(6.36-10)^2 = \frac{1}{2}(13.25) \approx \boxed{6.63}$$

**Loss reduced from 8 → 6.63 after one step.**

---

## BLOCK C — ViT Dimensions (Q, K, V)

### Setup
Image: 224×224, Patch size: 16×16, $d_{model}=768$, single head

### Calculations

$$N_{\text{patches}} = \frac{224}{16} \times \frac{224}{16} = 14 \times 14 = 196$$

$$\text{Sequence length} = 196 + 1\ (\text{CLS token}) = 197$$

### Dimension Table (Single Head)

| Matrix | Shape | Note |
|---|---|---|
| Input $X$ | $197 \times 768$ | tokens × model dim |
| $W_Q,\ W_K,\ W_V$ | $768 \times 768$ | projection weight matrices |
| $Q = X W_Q$ | $197 \times 768$ | query matrix |
| $K = X W_K$ | $197 \times 768$ | key matrix |
| $V = X W_V$ | $197 \times 768$ | value matrix |
| $QK^T$ | $197 \times 197$ | attention score map |
| $\text{softmax}(QK^T/\sqrt{d_k})$ | $197 \times 197$ | attention weights |
| Output | $197 \times 768$ | after weighted sum of V |

### Multi-head ($h=12$)
$$d_{\text{head}} = \frac{768}{12} = 64$$

Each head: $Q, K, V$ shape $= 197 \times 64$, $W_Q, W_K, W_V$ shape $= 768 \times 64$

**Scaling factor:** $\sqrt{d_k} = \sqrt{64} = 8$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## BLOCK D — Classification Metrics (Numerical)

### Given
**TP = 6, FN = 2, TN = 10, FP = 2, Total = 20**

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP = 6 | FN = 2 |
| **Actual Negative** | FP = 2 | TN = 10 |

### All Metrics

| Metric | Formula | Calculation | Value |
|---|---|---|---|
| **Accuracy** | $\frac{TP+TN}{Total}$ | $\frac{6+10}{20}$ | **0.80** |
| **Precision** | $\frac{TP}{TP+FP}$ | $\frac{6}{6+2}$ | **0.75** |
| **Recall / Sensitivity / TPR** | $\frac{TP}{TP+FN}$ | $\frac{6}{6+2}$ | **0.75** |
| **Specificity / TNR** | $\frac{TN}{TN+FP}$ | $\frac{10}{10+2}$ | **0.833** |
| **FPR (Fall-out)** | $\frac{FP}{FP+TN}$ | $\frac{2}{2+10}$ | **0.167** |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P+R}$ | $\frac{2(0.75)(0.75)}{0.75+0.75}$ | **0.75** |
| **ROC point** | (FPR, TPR) | — | **(0.167, 0.75)** |

### Dataset Insight
- **Slight imbalance:** 8 positives vs 12 negatives
- Precision = Recall = F1 = 0.75 → symmetric errors (equal FP and FN)
- **Best metric to report:** F1 Score (handles imbalance better than accuracy)
- Accuracy of 0.80 is slightly inflated by the majority negative class

---

## BLOCK E — Extra Prep Questions

### E1. Momentum SGD (Numerical)

**Setup:** $w=1$, $v=0$, gradient $g=0.5$, $\mu=0.9$, $\eta=0.1$

$$v_{\text{new}} = \mu v + g = 0.9(0) + 0.5 = 0.5$$
$$w_{\text{new}} = w - \eta v_{\text{new}} = 1 - 0.1(0.5) = \boxed{0.95}$$

---

### E2. Adam Update — Step 1 (Numerical)

**Setup:** $g=0.5$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=0.001$, $t=1$

$$m_1 = (1-\beta_1) \cdot g = 0.1 \times 0.5 = 0.05$$
$$v_1 = (1-\beta_2) \cdot g^2 = 0.001 \times 0.25 = 2.5 \times 10^{-4}$$

**Bias correction:**
$$\hat{m} = \frac{m_1}{1-\beta_1^t} = \frac{0.05}{0.1} = 0.5$$
$$\hat{v} = \frac{v_1}{1-\beta_2^t} = \frac{2.5 \times 10^{-4}}{0.001} = 0.25$$

**Update:**
$$w_{\text{new}} = w - \frac{\eta \cdot \hat{m}}{\sqrt{\hat{v}} + \epsilon} = w - \frac{0.001 \times 0.5}{\sqrt{0.25}} = w - \frac{0.0005}{0.5} = w - \boxed{0.001}$$

---

### E3. Receptive Field After 2 Conv Layers

**Formula:** $RF_L = RF_{L-1} + (K-1) \times \prod_{i=1}^{L-1} S_i$

For K=3, S=1 throughout:
$$RF_1 = K = 3$$
$$RF_2 = 3 + (3-1) \times 1 = \boxed{5}$$
$$RF_3 = 5 + (3-1) \times 1 = 7$$

**General rule:** Each conv layer with K=3, S=1 adds 2 to the receptive field.

With stride S=2 at layer 1:
$$RF_2 = 3 + (3-1) \times 2 = \boxed{7}$$

---

### E4. Batch Normalization — Normalize Manually

**Batch:** $x = [2, 4, 6, 8]$, $\gamma=1$, $\beta=0$

$$\mu = \frac{2+4+6+8}{4} = 5$$
$$\sigma^2 = \frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4} = \frac{9+1+1+9}{4} = 5$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \approx \frac{x_i - 5}{\sqrt{5}}$$

| $x_i$ | $\hat{x}_i$ |
|---|---|
| 2 | $\frac{-3}{2.236} = -1.342$ |
| 4 | $\frac{-1}{2.236} = -0.447$ |
| 6 | $\frac{+1}{2.236} = +0.447$ |
| 8 | $\frac{+3}{2.236} = +1.342$ |

With $\gamma=1, \beta=0$: output $= \hat{x}$ (no scale/shift)

---

### E5. Cross-Entropy Loss (Numerical)

**Softmax output:** $[0.7,\ 0.2,\ 0.1]$, **True label:** class 0

$$L = -\sum_i y_i \log(\hat{p}_i) = -1 \cdot \log(0.7) = -(-0.357) = \boxed{0.357}$$

**If true label were class 1:**
$$L = -\log(0.2) = \boxed{1.609}$$

**Key insight:** Loss decreases as predicted probability for correct class increases toward 1.

---

### E6. IoU Calculation (Numerical)

**Given:** Predicted box area = 50, GT box area = 60, Intersection = 30

$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}} = \frac{\text{Intersection}}{\text{Area}_A + \text{Area}_B - \text{Intersection}}$$

$$\text{IoU} = \frac{30}{50 + 60 - 30} = \frac{30}{80} = \boxed{0.375}$$

**Threshold rules:**
- IoU ≥ 0.5 → True Positive (PASCAL VOC standard)
- IoU ≥ 0.75 → stricter (COCO standard)
- IoU < threshold → False Positive

---

### E7. Softmax vs Sigmoid — When to Use

| Scenario | Function | Why |
|---|---|---|
| Multi-class (1 of N) | Softmax | Probabilities sum to 1 |
| Multi-label (any subset) | Sigmoid per class | Each class independent |
| Binary classification | Sigmoid or Softmax(2 classes) | Equivalent |

**Softmax:**
$$\hat{p}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

For $z = [2, 1, 0]$:
$$\hat{p} = \left[\frac{e^2}{e^2+e^1+e^0},\ \frac{e^1}{e^2+e^1+e^0},\ \frac{e^0}{e^2+e^1+e^0}\right] = [0.665,\ 0.245,\ 0.090]$$

---

### E8. Vanishing Gradient — Why It Happens (Numerical Intuition)

Sigmoid gradient: $\sigma'(x) = \sigma(x)(1-\sigma(x))$, maximum = 0.25 at $x=0$

For a 5-layer network, gradient at layer 1:
$$\frac{\partial L}{\partial W^{(1)}} = \prod_{l=2}^{5} \sigma'(x^{(l)}) \leq (0.25)^4 = \boxed{0.0039}$$

After 10 layers: $(0.25)^{10} \approx 10^{-6}$ → gradient essentially zero.

**ReLU fix:** gradient = 1 for positive activations → product stays 1 → no vanishing.

---

### E9. Weight Initialization — Xavier vs He

**Xavier (for Sigmoid/Tanh):**
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\ +\sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$

For $n_{in}=256$, $n_{out}=128$: range $= \pm\sqrt{\frac{6}{384}} = \pm 0.125$

**He (for ReLU):**
$$W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{in}}}\right)$$

For $n_{in}=256$: $\sigma = \sqrt{\frac{2}{256}} = \sqrt{0.0078} = 0.088$

---

### E10. Learning Rate Scaling Rule (Large Batch)

**Linear scaling rule:** when batch size increases by factor $k$, scale LR by $k$

| Batch Size | LR |
|---|---|
| 32 | 0.01 |
| 64 | 0.02 |
| 256 | 0.08 |
| 1024 | 0.32 |

**Square root rule (alternative):** $\text{LR} \propto \sqrt{\text{Batch Size}}$

For BS=32 → BS=256 (8× increase): $\text{LR}_{\text{new}} = 0.01 \times \sqrt{8} = 0.028$

---

## Quick Reference — Formulas Cheat Sheet

| Formula | Expression |
|---|---|
| Conv output size | $\lfloor(I - K + 2P)/S\rfloor + 1$ |
| Conv params | $K^2 \times C_{in} \times C_{out} + C_{out}$ |
| SGD update | $w \leftarrow w - \eta \nabla_w L$ |
| Momentum SGD | $v \leftarrow \mu v + g;\ w \leftarrow w - \eta v$ |
| Adam | $w \leftarrow w - \eta \hat{m}/(\sqrt{\hat{v}}+\epsilon)$ |
| Cross-entropy | $L = -\sum y_i \log \hat{p}_i$ |
| Softmax | $\hat{p}_i = e^{z_i}/\sum_j e^{z_j}$ |
| Precision | $TP/(TP+FP)$ |
| Recall | $TP/(TP+FN)$ |
| F1 | $2PR/(P+R)$ |
| Specificity | $TN/(TN+FP)$ |
| FPR | $FP/(FP+TN)$ |
| IoU | Intersection / Union |
| Receptive field | $RF_L = RF_{L-1} + (K-1) \times S^{L-1}$ |
| BatchNorm | $\hat{x} = (x-\mu)/\sqrt{\sigma^2+\epsilon}$ |
| ViT patches | $(H/P) \times (W/P)$ |
| Attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ |
