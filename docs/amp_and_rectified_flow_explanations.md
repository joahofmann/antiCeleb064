# GPU Mixed Precision (AMP) & Rectified Flow (RF) Explanations

This document summarizes the technical discussions and explanations regarding Automatic Mixed Precision (AMP), training visual trajectories, and the sampling speedups of Rectified Flow (RF) on the GPU.

---

## 1. GPU Mixed Precision (AMP)

### What is AMP?
AMP stands for **Automatic Mixed Precision** (implemented via `torch.cuda.amp` or `torch.amp` in PyTorch). It is a method of accelerating deep learning training on modern NVIDIA GPUs (utilizing Tensor Cores) by mixing different floating-point precisions:

* **FP32 (Single Precision - 32-bit):** Standard float format. High precision and range, but slower to compute and uses more memory (4 bytes per value).
* **FP16 (Half Precision - 16-bit):** Low precision float format. Fast to compute and uses 50% less memory (2 bytes per value), but prone to numerical underflow/overflow.
* **BF16 (Brain Floating Point - 16-bit):** Alternative 16-bit format with the same dynamic range as FP32, making training more stable than FP16 on supported GPUs (Ampere architecture and newer).

### How AMP works in our code
In the notebook, AMP is used in the forward and backward passes of both Flow Matching (Phase 1) and Rectified Flow (Phase 2) training loops via two main components:
1. **`torch.cuda.amp.autocast()`:** Automatically casts compatible mathematical operations (like matrix multiplications and convolutions) to 16-bit float formats, while keeping precision-critical calculations (like loss functions) in FP32.
2. **`torch.cuda.amp.GradScaler()`:** Scales the loss before backpropagation to prevent extremely small gradients from underflowing (becoming zero) in FP16, then automatically unscales them before updating the optimizer weights.

```python
scaler = torch.cuda.amp.GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 2. Visual Convergence (FM vs. RF) after Short Training

When inspecting the generated output image grids after a short training run (e.g., 5 epochs), the grids for **Flow Matching (100 steps)** and **Rectified Flow (1 step)** look very similar. This is explained by the following code and training behaviors:

### A. Fresh Weights Initialization
In **Cell 9** (Phase 2), the Rectified Flow model is initialized from scratch:
```python
Modell_RF = SimpleUNet().to(Device)
```
It does not load or copy the trained weights of Phase 1 (`Modell`). This means both Phase 1 and Phase 2 train a completely fresh, randomly-initialized U-Net for the exact same number of epochs (5 epochs) on the exact same 10,000 cached images. Given they share identical architectures and parameters, their feature capacity at 5 epochs remains highly similar.

### B. Similar Mathematical Targets
Both models minimize the Mean Squared Error (MSE) between their predicted vector field ($v$) and a linear target velocity vector:
* **Flow Matching:** Trains to predict the vector field pointing from noise ($x_0$) to real images ($x_1$).
* **Rectified Flow:** Trains to predict the vector field pointing from noise ($x_0$) to FM-generated images ($x_{1\_rectified}$).

Because both tasks involve learning vector field mapping using the same U-Net capacity, their visual convergence curves are structurally parallel.

---

## 3. Why Rectified Flow is 100x Faster

The "100x speedup" refers entirely to **inference (image generation/sampling) speed**, not to training speed.

### Curved vs. Straight Trajectories
* **Flow Matching / Diffusion (Curved Paths):** The learned mathematical trajectories from random noise ($x_0$) to a final image ($x_1$) are curved in vector space. To generate an image accurately, an ODE solver must take many tiny steps (typically **50 to 100 steps**) to follow the curve, calling the U-Net at every step:
  $$x_0 \rightarrow x_{0.01} \rightarrow x_{0.02} \rightarrow \dots \rightarrow x_{1.0} \quad \text{(100 U-Net calls)}$$
* **Rectified Flow (Straight Paths):** Retraining the model on generated pairs $(x_0, x_{1\_rectified})$ effectively "straightens" the trajectories into straight lines in vector space. Because the paths are straight, the solver can predict the trajectory direction directly at $t=0$ and jump to the target image in **exactly 1 step**:
  $$x_1 = x_0 + v(x_0, 0) \quad \text{(1 U-Net call)}$$

Since calling the U-Net is the main computational bottleneck, generating high-quality images in 1 step instead of 100 steps results in a **100x speedup in sampling time**.

### Particle Connection vs. Average Velocity Field Analogy
* **In Flow Matching (FM):** We only learn a global velocity field representing the **mean (average) velocity** at each point $(x, t)$. Because different trajectories cross over each other, the model has to average-out these crossing paths. The particles must navigate this curved, global average field, necessitating step-by-step path calculation (ODE integration) during inference.
* **In Rectified Flow (RF):** We establish a direct, straight-line "bridge" connecting a specific particle of noise ($x_0$) to a specific particle of the generated image ($x_1$). Because they are uniquely linked, the paths do not cross, and the U-Net learns a straight, constant velocity vector field, eliminating the need for step-by-step path integration during inference.
