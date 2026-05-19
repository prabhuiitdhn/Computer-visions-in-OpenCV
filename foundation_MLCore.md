# ML Core Foundations

---

## Optimizers & Gradient Descent

### **Optimizers in Machine Learning**
Optimizers are algorithms or methods used to adjust the weights and biases of a machine learning model to minimize the loss function. The loss function quantifies the error between the predicted output and the actual target. Optimizers play a critical role in training neural networks and other machine learning models by guiding the model toward the optimal set of parameters.

---

### **Gradient Descent: The Core of Optimization**
Gradient Descent is the most widely used optimization algorithm in machine learning. It is an iterative optimization algorithm used to minimize a function by moving in the direction of the steepest descent as defined by the negative of the gradient.

#### **Mathematical Formulation**
Given a loss function $L(\theta)$, where $\theta$ represents the model parameters (weights and biases), the goal is to find $\theta^*$ that minimizes $L(\theta)$.

The update rule for Gradient Descent is:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

Where:
- $\theta_t$: Parameters at iteration $t$
- $\eta$: Learning rate (step size)
- $\nabla_\theta L(\theta_t)$: Gradient of the loss function with respect to $\theta$

---

### **Types of Gradient Descent**
1. **Batch Gradient Descent**
   - Computes the gradient using the entire dataset.
   - **Advantages**: Converges to the global minimum for convex functions.
   - **Disadvantages**: Computationally expensive for large datasets.

2. **Stochastic Gradient Descent (SGD)**
   - Computes the gradient using a single data point at each iteration.
   - **Advantages**: Faster updates, can escape local minima due to noise.
   - **Disadvantages**: High variance in updates, leading to oscillations.

3. **Mini-Batch Gradient Descent**
   - Computes the gradient using a small batch of data points.
   - **Advantages**: Combines the benefits of Batch and SGD, efficient and stable.
   - **Disadvantages**: Requires tuning the batch size.

---

### **Challenges in Gradient Descent**
1. **Choosing the Learning Rate ($\eta$)**
   - A small $\eta$ leads to slow convergence.
   - A large $\eta$ may cause divergence or overshooting.

2. **Local Minima and Saddle Points**
   - Non-convex loss functions may have multiple local minima or saddle points.

3. **Vanishing/Exploding Gradients**
   - Common in deep networks, where gradients become too small or too large.

---

### **Advanced Optimizers**
To address the challenges of Gradient Descent, advanced optimizers have been developed. These optimizers adapt the learning rate or use momentum to accelerate convergence.

1. **Momentum**
   - Adds a fraction of the previous update to the current update:

     $$v_t = \gamma v_{t-1} + \eta \nabla_\theta L(\theta_t)$$

     $$\theta_{t+1} = \theta_t - v_t$$

   - **Advantages**: Accelerates convergence, reduces oscillations.

2. **Adagrad (Adaptive Gradient Algorithm)**
   - Adapts the learning rate for each parameter based on the historical gradients:

     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \nabla_\theta L(\theta_t)$$

   - **Advantages**: Handles sparse data well.
   - **Disadvantages**: Learning rate diminishes over time.

3. **RMSProp (Root Mean Square Propagation)**
   - Modifies Adagrad by using an exponentially decaying average of squared gradients:

     $$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$$

     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \nabla_\theta L(\theta_t)$$

   - **Advantages**: Solves Adagrad's diminishing learning rate problem.

4. **Adam (Adaptive Moment Estimation)**
   - Combines Momentum and RMSProp:

     $$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

     $$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

     $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

     $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

   - **Advantages**: Works well in practice, robust to noisy gradients.

---

### **Practical Considerations**
1. **Learning Rate Scheduling**
   - Use techniques like step decay, exponential decay, or cyclic learning rates to adjust $\eta$ during training.

2. **Gradient Clipping**
   - Clip gradients to prevent exploding gradients in deep networks.

3. **Weight Initialization**
   - Proper initialization (e.g., Xavier, He initialization) reduces the risk of vanishing/exploding gradients.

4. **Regularization**
   - Techniques like L1/L2 regularization, dropout, and batch normalization improve generalization and stability.

---

### **Conclusion**
Optimizers and Gradient Descent are the backbone of training machine learning models. While Gradient Descent provides the foundation, advanced optimizers like Adam and RMSProp address its limitations, enabling efficient and robust training of complex models. For a senior researcher, understanding the nuances of these algorithms and their practical implications is crucial for designing and optimizing state-of-the-art machine learning systems.

---

## Steepest Descent

### **Core Idea**
The method of Steepest Descent involves finding the direction in which the function decreases most rapidly at a given point. This direction is given by the **negative gradient** of the function at that point.

#### **Mathematical Formulation**
Given a function $f(x)$, the goal is to find $x^*$ such that:

$$f(x^*) = \min f(x)$$

The update rule for Steepest Descent is:

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

Where:
- $x_k$: Current point in the parameter space.
- $\nabla f(x_k)$: Gradient of $f(x)$ at $x_k$, representing the direction of steepest ascent.
- $-\nabla f(x_k)$: Direction of steepest descent.
- $\alpha_k$: Step size (learning rate) at iteration $k$.

---

### **Key Steps in Steepest Descent**
1. **Initialization**: Start with an initial guess $x_0$.
2. **Compute Gradient**: Calculate $\nabla f(x_k)$ at the current point $x_k$.
3. **Determine Step Size**: Choose $\alpha_k$, which controls how far to move in the direction of $-\nabla f(x_k)$.
4. **Update**: Move to the next point:
   $$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$
5. **Repeat**: Iterate until convergence, i.e., until $\|\nabla f(x_k)\|$ is sufficiently small or the change in $f(x)$ is negligible.

---

### **Geometric Intuition**
- The gradient $\nabla f(x)$ points in the direction of the steepest increase of the function.
- The negative gradient $-\nabla f(x)$ points in the direction of the steepest decrease.
- Steepest Descent moves along $-\nabla f(x)$ to reduce the function value as quickly as possible.

Imagine standing on a hill and trying to descend to the bottom. The steepest descent direction at any point is the direction in which the slope is steepest downward.

---

### **Step Size ($\alpha_k$)**
The choice of $\alpha_k$ is critical for the performance of Steepest Descent:
1. **Fixed Step Size**:
   - A constant $\alpha_k$ is used for all iterations.
   - **Advantages**: Simple to implement.
   - **Disadvantages**: May lead to slow convergence or overshooting.

2. **Line Search**:
   - $\alpha_k$ is chosen at each iteration to minimize $f(x_k - \alpha_k \nabla f(x_k))$.
   - **Advantages**: Ensures optimal progress at each step.
   - **Disadvantages**: Computationally expensive.

3. **Adaptive Step Size**:
   - Adjust $\alpha_k$ dynamically based on the progress of the algorithm.
   - Common in modern optimization methods like Adam or RMSProp.

---

### **Convergence**
- Steepest Descent converges to a local minimum if $f(x)$ is differentiable and convex.
- The rate of convergence depends on the condition number of the Hessian matrix $H$ (second derivative of $f(x)$):
  - **Well-conditioned** ($H$ has similar eigenvalues): Faster convergence.
  - **Ill-conditioned** ($H$ has widely varying eigenvalues): Slower convergence, zig-zagging behavior.

---

### **Advantages**
1. **Simplicity**: Easy to understand and implement.
2. **General Applicability**: Works for a wide range of differentiable functions.
3. **Foundation for Modern Methods**: Forms the basis for Gradient Descent and its variants.

---

### **Disadvantages**
1. **Slow Convergence**: Especially for ill-conditioned problems.
2. **Sensitive to Step Size**: Poor choice of $\alpha_k$ can lead to divergence or slow progress.
3. **Local Minima**: May converge to a local minimum for non-convex functions.

---

### **Comparison with Gradient Descent**
Steepest Descent is often used interchangeably with Gradient Descent, but there are subtle differences:
- **Steepest Descent**: Focuses on minimizing $f(x)$ along the exact direction of steepest decrease, often using line search to determine $\alpha_k$.
- **Gradient Descent**: Typically uses a fixed or adaptive step size without exact line search.

---

### **Practical Considerations**
1. **Preconditioning**:
   - Transform the problem to improve the condition number of $H$.
   - Example: Scale the variables or use second-order methods like Newton's method.

2. **Momentum**:
   - Add a fraction of the previous update to the current update to accelerate convergence and reduce oscillations.

3. **Modern Optimizers**:
   - Algorithms like Adam, RMSProp, and Adagrad build on the principles of Steepest Descent but adapt the step size and direction dynamically.

---

### **Conclusion**
Steepest Descent is a fundamental optimization method that provides the foundation for many modern algorithms. While it has limitations in terms of convergence speed and sensitivity to step size, its simplicity and effectiveness make it a cornerstone of optimization theory and practice.
