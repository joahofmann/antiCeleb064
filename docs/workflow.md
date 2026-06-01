# Flow Matching and Rectified Flow Training Workflow

This document outlines the end-to-end machine learning pipeline implemented in `FM_RM_Colab_ZBook01.ipynb` for training a generative U-Net model using Flow Matching (FM) and Rectified Flow (RF).

---

## 1. Environment Setup & Configuration
* **Automatic Environment Switch:** The notebook detects if it is running in Google Colab or on a local machine. If running in Colab, it automatically mounts Google Drive for persistent checkpoints.
* **Hyperparameters Configuration:** Defines critical parameters such as:
  * `IMAGE_SIZE = 64` (the resolution of the square training images)
  * `EPOCHS` (total iterations over the dataset)
  * `BATCH_SIZE` (number of samples processed at once)
  * `Device` selection (`cuda` if an NVIDIA GPU is present, else `cpu`)

---

## 2. Dataset Loading & Preprocessing
* **Custom Dataset (`BildDatensatz`):** Recursively traverses a folder to find image files (`.png`, `.jpg`, `.jpeg`, `.bmp`).
* **Transformations:**
  * Rescales images to `64x64`.
  * Converts them to PyTorch Tensors.
  * Normalizes the pixels to the range `[-1, 1]` to optimize stable training.
* **DataLoader:** Batches the preprocessed images for sequential training input.

---

## 3. U-Net Architecture
* **Sinusoidal Position Embeddings:** Encodes the continuous time step $t \in [0, 1]$ into a high-dimensional vector. This allows the model to differentiate flow states at various stages.
* **Velocity Net (`SimpleUNet`):** Serves as the velocity vector field $v(x_t, t)$. It takes a noisy/partially-generated image $x_t$ along with the time embedding $t$, and outputs a predicted velocity vector.

---

## 4. Phase 1: Flow Matching (1-RF)
Trains the U-Net to map random noise to target images:
1. **Checkpoint Recovery:** Automatically loads existing weights from Drive if `RESUME_FROM_EPOCH` is set.
2. **Linear Path Interpolation:** Given real training images $x_1$ and random noise $x_0 \sim \mathcal{N}(0, I)$:
   * Draw a random time step $t \sim \text{Uniform}(0, 1)$.
   * Interpolate linearly to get the intermediate point:
     $$x_t = t \cdot x_1 + (1 - t) \cdot x_0$$
   * The target velocity vector is:
     $$v_{\text{target}} = x_1 - x_0$$
3. **Loss Optimization:** The U-Net predicts the velocity $v_{\text{pred}} = \text{Modell}(x_t, t)$. The network is optimized using Mean Squared Error (MSE):
   $$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} [\|v_{\text{pred}} - (x_1 - x_0)\|^2]$$

---

## 5. Flow Matching Inference (ODE Solving)
To generate new images using the trained Phase 1 model:
* Start with pure Gaussian noise $x_0$ at $t=0$.
* Step-by-step, integrate the predicted velocity field forward to $t=1$ using a **forward Euler ODE solver**:
  $$x_{t+dt} = x_t + v(x_t, t) \cdot dt$$
  > [!NOTE]
  > Since Flow Matching paths can cross, the velocity field is curved. This requires a relatively high number of integration steps (typically 50 steps) to produce high-quality samples.
* Scale pixels back to `[0, 1]` for visualization.

---

## 6. Phase 2: Reflow (Rectified Flow / 2-RF)
To speed up generation, the paths are straightened through a process called Reflow:
1. **Re-pairing Trajectories:** The trained Phase 1 Flow Matching model takes the original noise inputs $x_0$ and solves the ODE to generate the corresponding clean outputs $x_1^{\text{rectified}}$.
2. **Straightened Dataset:** Creates a new matched dataset of $(x_0, x_1^{\text{rectified}})$ that maps noise directly to images without path intersections.
3. **Phase 2 Training:** A secondary model (`Modell_RF`) is trained on these straight trajectories.

---

## 7. 1-Step Inference Comparison
* Demonstrates the primary benefit of **Rectified Flow**: since trajectories are straightened, the model does not require 50 ODE steps to generate images.
* It can compute images in **just a single step** ($dt = 1$) starting from noise $x_0$:
  $$x_1 = x_0 + v_{\text{RF}}(x_0, 0)$$
* The notebook executes a visual comparison between 50-step generation and ultra-fast 1-step generation.
