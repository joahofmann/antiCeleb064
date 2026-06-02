# Notebook Parameters and Optimal ZBook Settings

This document provides a technical definition of the running parameters used in the notebook, explains their mathematical and physical interdependences, and presents the optimal settings to complete the entire training run on a ZBook (RTX A5000 Laptop GPU) in under **one hour**.

---

## 1. Parameters Definition & Consequences

### **`EPOCHS`**
* **Definition:** The number of complete passes the training algorithm makes through the entire dataset.
* **Technical/Mathematical Consequence:**
  $$\text{Total Gradient Updates} = \text{EPOCHS} \times \frac{\text{Dataset Size}}{\text{BATCH_SIZE}}$$
  Increasing epochs increases training accuracy and model quality (convergence), but training time scales **linearly** with epochs. If you double the epochs, you double the training duration. Excessively high epochs can lead to **overfitting** (where the model memorizes the training pictures instead of learning general features).

### **`TEST_IMAGE_LIMIT`**
* **Definition:** The maximum number of images allowed to load into the active dataset (slicing the dataset size).
* **Technical/Mathematical Consequence:** Directly determines the "Dataset Size" variable. Slicing the dataset is the **most significant lever** for controlling training time. Slicing it to 20,000 images means the dataset is 10% of the original CelebA size, resulting in a **10x reduction** in training time.

### **`BATCH_SIZE`**
* **Definition:** The number of training samples processed together in a single forward and backward pass.
* **Technical/Mathematical Consequence:**
  * **Gradient noise:** Larger batches compute a cleaner, more accurate average gradient step (less stochastic noise).
  * **VRAM consumption:** Higher batch size consumes more GPU memory (VRAM). On an RTX A5000 (16 GB), you can easily run high batch sizes (like `64`) for $64\times 64$ images without hitting an Out-of-Memory (OOM) error. Larger batches maximize GPU parallelization via **Tensor Cores**, speeding up processing per image.

### **`LEARNING_RATE`**
* **Definition:** The step size the optimizer (Adam) takes in the direction of the gradient to update the network weights.
* **Technical/Mathematical Consequence:** In gradient updates: $\theta_{t+1} = \theta_t - \eta \cdot \nabla \mathcal{L}$ (where $\eta$ is the learning rate). If the learning rate is too high, the model diverges (loss explodes to `NaN`). If too low, training will be extremely slow or get stuck.

### **`IMAGE_SIZE`**
* **Definition:** The spatial resolution (width and height in pixels) of the square images.
* **Technical/Mathematical Consequence:** Determines the inputs ($3 \times H \times W$) to the U-Net. Computational complexity of convolutional layers scales **quadratically** with image size ($O(H^2 \cdot W^2)$). Doubling the resolution (e.g. from 64 to 128) results in **4x VRAM consumption** and **4x longer processing time**.

### **`Reflow ODE Steps`**
* **Definition:** The number of discretization steps used by the Euler solver to integrate the velocity field from time $t=0$ (noise) to $t=1$ (clean image).
* **Technical/Mathematical Consequence:** 
  * Higher steps (e.g., 50) ensure high-quality straight paths for Reflow, reducing blurring.
  * Lower steps (e.g., 10) are faster but cause integration errors (curved trajectories) which degrade generation quality.

---

## 2. Interdependences

1. **VRAM constraints (Image Size vs. Batch Size):**
   Memory usage is proportional to $\text{BATCH_SIZE} \times \text{IMAGE_SIZE}^2$. If you double the image size, you must divide the batch size by 4 to avoid OOM crashes.
2. **Gradient Stability (Batch Size vs. Learning Rate):**
   * *Linear Scaling Rule:* Larger batches yield cleaner gradients. If you increase the `BATCH_SIZE`, you can (and should) slightly increase the `LEARNING_RATE` to optimize faster.
3. **Phase 2 Startup Overhead (Dataset Limit vs. ODE Steps):**
   To initialize Phase 2, the model must compute the rectified targets for every image in the dataset. This requires:
   $$\text{Total Forward Passes} = \text{Reflow ODE Steps} \times \frac{\text{Dataset Size}}{\text{BATCH_SIZE}}$$
   If your dataset limit is huge and ODE steps are high, this single calculation step can take up to an hour before Phase 2 training even begins.

---

## 3. Optimal Settings for ZBook (< 1 Hour Run)

Your ZBook is equipped with an **NVIDIA RTX A5000 Laptop GPU (16 GB VRAM)**. This is a workstation-grade GPU with massive memory capacity. By increasing `BATCH_SIZE` to 64, we saturate the Tensor Cores (making training faster per image) and can train on **20,000 images** for **150 epochs** in about **40 minutes**, achieving high-quality face generation:

### **Configure these parameters in Cell 2:**

| Parameter | Recommended Setting | Explanation |
| :--- | :---: | :--- |
| **`TEST_MODE`** | **`True`** | *Use the test notebook parameters to customize.* |
| **`TEST_IMAGE_LIMIT`** | **`20000`** | Cuts dataset down to 10% of CelebA. Plenty of variety for learning faces. |
| **`BATCH_SIZE`** | **`64`** | Fits comfortably in your 16GB VRAM. Drastically speeds up step times. |
| **`LEARNING_RATE`** | **`0.0003`** (3e-4) | Slightly raised to match the larger batch size (64). |
| **`EPOCHS`** | **`150`** | Ensures deep convergence and sharp details in under an hour. |
| **`IMAGE_SIZE`** | **`64`** | Keep at 64x64 to preserve fast processing speeds. |
| **`Reflow ODE Steps`** | **`50`** | Keeps trajectories perfectly straight for 1-step generation. |

### **Mathematical Time Estimation on RTX A5000:**
* **Steps per Epoch:** $\frac{20000}{64} \approx 312$ steps.
* **GPU Speed (Batch Size 64):** $\approx 25$ ms per step.
* **Time per Epoch:** $312 \times 25\text{ ms} = 7.8$ seconds.
* **Phase 1 Training (150 epochs):** $150 \times 7.8\text{ s} \approx \mathbf{19.5\text{ minutes}}$.
* **Reflow Target Generation (Cell 18):** $50 \text{ steps} \times 312 \text{ batches} \times 8 \text{ ms (eval only)} \approx \mathbf{2\text{ minutes}}$.
* **Phase 2 Training (150 epochs):** $150 \times 7.8\text{ s} \approx \mathbf{19.5\text{ minutes}}$.
* **Total Run Time:** $19.5\text{ min} + 2\text{ min} + 19.5\text{ min} = \mathbf{41\text{ minutes}}$.
