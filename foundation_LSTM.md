# LSTM — Long Short-Term Memory

---

## The Problem LSTM Solves

Standard RNNs process sequences by passing a hidden state $h_t$ forward:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**Vanishing gradient problem**: during backpropagation through time (BPTT), gradients are multiplied by $W_h$ at every timestep. If $|W_h| < 1$, gradients shrink exponentially → early timesteps get near-zero gradient → RNN can't learn long-range dependencies.

```
Sequence:  "The cat, which sat on the mat for hours, was hungry"
               ↑                                        ↑
           "cat" (subject)                        "was" (verb)

RNN forgets "cat" by the time it needs to predict "was" (singular)
LSTM remembers it.
```

---

## The Core Idea — A Separate Memory Cell

LSTM adds a **cell state** $C_t$ — a conveyor belt running through the sequence with only minor linear interactions (no squashing). Information can flow unchanged across many timesteps.

```
C_{t-1} ──────────────────────────────────────────→ C_t
              ↑ forget          ↑ add new info
              │                 │
h_{t-1}, x_t → [gates] ────────┘
```

Three **gates** control what flows through — each is a sigmoid layer outputting values in $[0, 1]$ (0 = block completely, 1 = pass completely).

---

## The 4 Equations — Step by Step

Given input $x_t$ and previous hidden state $h_{t-1}$:

**1. Forget Gate** — what to erase from cell state:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- $f_t \approx 0$ → forget this dimension of $C_{t-1}$
- $f_t \approx 1$ → keep this dimension

**2. Input Gate** — what new info to write:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$ decides *how much* to write
- $\tilde{C}_t$ is the *candidate* values to write (range $[-1, +1]$)

**3. Cell State Update** — combine forget + write:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$\odot$ = element-wise multiply. This is the key: **additive update** — no matrix multiply on $C$, so gradients flow back without vanishing.

**4. Output Gate** — what to expose as hidden state:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

---

## Visual Architecture

```
        x_t  h_{t-1}
         │      │
    ┌────┴──────┴────┐
    │                │
    │  f_t = σ(·)    │──→ C_{t-1} ──×(f_t)──┐
    │  i_t = σ(·)    │                        ├──→ C_t ──→ tanh ──×(o_t)──→ h_t
    │  C̃_t = tanh(·) │──→ i_t × C̃_t ─────────┘
    │  o_t = σ(·)    │──────────────────────────────────────────────────────↑
    └────────────────┘
```

---

## Why the Cell State Solves Vanishing Gradients

The gradient of the loss w.r.t. $C_{t-k}$ (k steps back):

$$\frac{\partial \mathcal{L}}{\partial C_{t-k}} = \frac{\partial \mathcal{L}}{\partial C_t} \prod_{j=t-k}^{t} f_j$$

It's a **product of forget gates** — not a product of weight matrices. If the network learns $f_j \approx 1$ for a memory it needs to preserve, the gradient flows back unchanged. The network can **learn** to preserve gradients by learning to keep the forget gate open.

Compare to vanilla RNN: $\prod W_h^k$ — fixed weight matrix raised to the $k$th power → inevitably vanishes or explodes.

---

## Each Gate's Role with a Language Example

Sentence: *"The dog, despite being tired, **barked**"*

| Gate | Action | Example |
|------|--------|---------|
| **Forget** | Erase stale info | Forget "tired" (adjective) after the comma — irrelevant to verb |
| **Input** | Write new info | Write "dog" = singular subject to cell state |
| **Output** | Expose relevant info | When predicting "barked", expose subject number from cell state |

---

## Parameters Count

For a single LSTM layer with input size $d$, hidden size $h$:

```
4 gates × (W_h: h×h  +  W_x: h×d  +  b: h)
= 4 × (h² + hd + h)
= 4h(h + d + 1)
```

Example: $d=512$, $h=512$ → $4 \times 512 \times (512 + 512 + 1) \approx 4.2M$ parameters per layer.

---

## LSTM Variants

**Peephole connections** (Gers & Schmidhuber, 2000):
Gates also look at the cell state directly:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t, C_{t-1}] + b_f)$$

**GRU (Gated Recurrent Unit)** — simplified LSTM:
- Merges forget + input → single **update gate** $z_t$
- Merges cell + hidden state → just $h_t$
- 2 gates instead of 3 → fewer parameters, faster, comparable performance on most tasks

$$z_t = \sigma(W_z[h_{t-1}, x_t])$$

$$r_t = \sigma(W_r[h_{t-1}, x_t])$$

$$h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tanh(W[r_t \odot h_{t-1}, x_t])$$

**Bidirectional LSTM**:
```
→ LSTM: x₁ → x₂ → x₃ → x₄ → x₅    (forward context)
← LSTM: x₅ → x₄ → x₃ → x₂ → x₁    (backward context)
Output at each step: [h_forward ; h_backward]   (concatenated)
```

---

## LSTM in Face Analysis Context

| Use Case | How LSTM Is Used |
|---------|-----------------|
| **Video face recognition** | Sequence of frame embeddings → LSTM → aggregated identity embedding |
| **Facial expression over time** | Frame-level AU (Action Unit) detections → LSTM → temporal expression label |
| **Lip reading** | Frame sequence of mouth crops → CNN features → LSTM → phoneme/word |
| **Age progression** | Temporal modeling of aging trajectory |
| **Liveness detection** | Sequence of frames → LSTM → detect blink/motion patterns (anti-spoofing) |

---

## LSTM vs Transformer

| | LSTM | Transformer |
|--|------|-------------|
| Long-range dependency | Good, not perfect (gate saturation) | Perfect (direct attention between any two positions) |
| Parallelization | Sequential — $O(T)$ serial steps | Fully parallel over sequence |
| Memory complexity | $O(h)$ per step | $O(T^2)$ attention matrix |
| Training speed | Slow for long sequences | Fast (GPU-friendly) |
| Best use today | Short sequences, on-device inference | Long sequences, large-scale training |

---

## Punchline for Interviews

> LSTM solves the vanishing gradient problem by introducing a cell state $C_t$ updated additively ($C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$) — gradients flow back as a product of forget gates rather than weight matrices, so the network can learn to preserve long-range memory by keeping $f_t \approx 1$. Three sigmoid gates (forget, input, output) each in $[0,1]$ control what to erase, write, and expose at each step. In face analysis, LSTMs model temporal sequences — video frame embeddings, expression dynamics, lip movements — but have been largely replaced by Transformers for tasks where full-sequence parallelism matters.
