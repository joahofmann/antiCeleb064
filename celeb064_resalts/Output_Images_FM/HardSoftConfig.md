Here is the complete summary of all the hardware, software, and hyperparameter configuration parameters used for the completed **100-epoch comparative run**:

### 1. Hardware & Environment Configuration
* **Active Compute Device:** `cuda` (NVIDIA RTX A5000 Laptop GPU with 16GB VRAM).
* **Conda Environment:** `zbook_gpu`
* **CPU Fallback (`FORCE_CPU`):** `False` (GPU acceleration enabled).
* **Data Preloading/Caching:** RAM preloading/caching is enabled. It walked the directory `C:/Data/celeb064` and loaded all **10,000 CelebA images** directly into system RAM before training started, bypassing slow SSD/HDD disk I/O loops.

### 2. Software & Package Stack
* **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.optim`).
* **Automatic Mixed Precision (AMP):** Enabled (`torch.cuda.amp.autocast` + `GradScaler`) to accelerate U-Net computations via 16-bit float tensor operations on GPU Tensor Cores.
* **Jupyter/Notebook execution kernel:** `ipykernel`/`jupyter` via `nbconvert` headless execution.
* **Evaluation Libraries:** NumPy, Matplotlib (for visualization grid/plot exports), and `scipy.linalg` (for 2D Wasserstein-2 Fréchet Distance calculations).

### 3. Hyperparameter Configuration (Cell 2)
* **Test Mode (`TEST_MODE`):** `True`
* **Target Epochs (`EPOCHS`):** `100` (for FM, RF, and DDPM models alike).
* **Batch Size (`BATCH_SIZE`):** `64` (optimal memory footprint for 16GB VRAM U-Net training).
* **Learning Rate (`LEARNING_RATE`):** `3e-4` (tuned Adam optimizer learning rate).
* **Image Size (`IMAGE_SIZE`):** `64` (resolves to 64x64x3 RGB image tensors).
* **Dataset Limit (`TEST_IMAGE_LIMIT`):** `40,000` (naturally capped at the physical `10,000` images found in the folder).
* **Automatic Recovery (`AUTO_RESUME`):** `True` (scans checkpoints and automatically resumes training from the latest checkpoint if interrupted).

### 4. U-Net Architecture Parameters (Cell 6)
* **Type:** Time-conditional `SimpleUNet` with U-Net skips.
* **Input Channels (`in_channels`):** `3` (RGB).
* **Output Dimension (`out_dim`):** `3` (predicts 3D noise vector for DDPM, or 3D velocity vector for FM/RF).
* **Down-path Channels (`down_channels`):** `(64, 128, 256, 512)`
* **Up-path Channels (`up_channels`):** `(512, 256, 128, 64)`
* **Time Embedding Dimension (`time_emb_dim`):** `32` (converts continuous time step $t \in [0, 1]$ into a 32-dimensional Sinusoidal Position Embedding vector).

### 5. Diffusion & Flow Schedules
* **Inference/ODE Steps:** `50` steps (used for Euler integration in Flow Matching/Rectified Flow, and reverse sampling in DDPM).
* **Flow Matching Step Size (`dt`):** `0.02` ($1.0 / 50$ steps).
* **DDPM Steps (`DDPM_STEPS`):** `50` steps.
* **DDPM Noise Schedule Bounds:** `BETA_START = 1e-4` to `BETA_END = 0.02`.
* **DDPM Schedule Type:** Linear schedule.