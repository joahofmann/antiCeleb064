# Comparative Evaluation Walkthrough (DDPM vs. Flow Matching vs. Rectified Flow)

We have successfully integrated a complete **Denoising Diffusion Probabilistic Model (DDPM)** training pipeline and comparative evaluation framework into the new notebook [FM_RM_Colab_ZBook02_test.ipynb](file:///C:/Users/joach/antiG/antiCeleb064/FM_RM_Colab_ZBook02_test.ipynb) and executed a clean 5-epoch test run on the GPU.

---

## 1. Accomplished Tasks

- **Notebook Setup:**
  - Created [FM_RM_Colab_ZBook02_test.ipynb](file:///C:/Users/joach/antiG/antiCeleb064/FM_RM_Colab_ZBook02_test.ipynb) to host the comparative framework while preserving the original `ZBook01` notebook.
- **DDPM Pipeline Integration:**
  - **Schedules**: Configured linear beta schedules and precomputed cumulative alphas for forward and reverse diffusion.
  - **U-Net Instance**: Instantiated a third `SimpleUNet` (`Modell_DDPM`) to isolate DDPM training.
  - **Phase 3 Training Loop**: Added the DDPM training cell minimizing noise prediction MSE loss with GradScaler AMP support.
  - **Reverse Sampler**: Implemented the stochastic reverse diffusion sampler `Generiere_Bilder_DDPM()`.
- **Clean Run Execution**:
  - Cleared output folders and ran the new notebook end-to-end on GPU under the `zbook_gpu` environment.
  - The 5-epoch run of all three phases (Flow Matching, Rectified Flow, and DDPM) completed successfully in **~19 minutes**.
- **GitHub Sync**:
  - Committed and pushed the executed notebook to the main remote branch.

---

## 2. Side-by-Side Quantitative Results

After 5 epochs of training, the comparative metrics table was populated:

```
===============================================================================================
                                EVALUATION RESULTS TABLE
===============================================================================================
Method                         | NFE   | Time (ms/img)   | Chamfer Dist    | FID Equiv    | Straightness
-----------------------------------------------------------------------------------------------
DDPM (50 steps)                | 50    | 56.15           | 0.8774          | 8.6860       | N/A         
Flow Matching (50 steps)       | 50    | 55.94           | 1.0026          | 3.2204       | 0.9819      
Rectified Flow (50 steps)      | 50    | 55.95           | 0.9474          | 6.1907       | 0.9952      
Rectified Flow (1 step)        | 1     | 1.01            | 0.8939          | 5.8597       | 1.0000      
===============================================================================================
```

### Analysis of the Metrics:
1. **Computation & Speed (NFE & Time):**
   - DDPM, Flow Matching, and Rectified Flow (50 steps) all take **~56 ms/image** (identical U-Net evaluation cost).
   - Rectified Flow (1 step) executes in **1.01 ms/image**, yielding a **55x speedup** with precise GPU timing.
2. **FID-Equivalent (Distribution Alignment Proxy):**
   - Flow Matching reaches the best alignment value (`3.2204`) at 5 epochs.
   - DDPM lags behind at `8.6860`, demonstrating that Flow Matching/Rectified Flow exhibit faster and cleaner quality convergence early on.
3. **Trajectory Straightness:**
   - Flow Matching has a straightness score of `0.9819` (curved path).
   - Rectified Flow (50 steps) successfully straightens this to `0.9952`.
   - Rectified Flow (1 step) is mathematically straight (`1.0000`).
   - DDPM is stochastic and does not follow a continuous ODE vector field, hence straightness is `N/A`.
4. **Chamfer Distance (Edge Alignment):**
   - DDPM (50 steps) yields a strong edge alignment score (`0.8774`), followed closely by 1-step Rectified Flow (`0.8939`).

---

## 3. Generated Outputs & Validation

All checkpoints and image outputs have been successfully written to their target folders under [Alex_RM](file:///C:/Users/joach/antiG/antiCeleb064/Alex_RM):

### Training Loss Comparison Plot
The step-wise training losses (smoothed) for all three phases across 780 training batches/steps:

![Training Loss Comparison Plot](file:///C:/Users/joach/.gemini/antigravity/brain/cbe776a8-c256-4cb7-b611-fba8eb42479c/loss_comparison.png)

### Checkpoints Written
- **DDPM Checkpoints:** Saved to [Checkpoints_DDPM](file:///C:/Users/joach/antiG/antiCeleb064/Alex_RM/Checkpoints_DDPM).
  - `ddpm_checkpoint_epoch_1.pth` to `ddpm_checkpoint_epoch_5.pth`
  - `ddpm_final.pth` & `ddpm_final_checkpoint.pth`
- **Flow Matching Checkpoints:** Saved to [Checkpoints_FM](file:///C:/Users/joach/antiG/antiCeleb064/Alex_RM/Checkpoints_FM).
  - `flow_matching_checkpoint_epoch_1.pth` to `flow_matching_checkpoint_epoch_5.pth`
  - `flow_matching_final_checkpoint.pth`
- **Rectified Flow Checkpoints:** Saved to [Checkpoints_RF](file:///C:/Users/joach/antiG/antiCeleb064/Alex_RM/Checkpoints_RF).
  - `rectified_flow_checkpoint_epoch_1.pth` to `rectified_flow_checkpoint_epoch_5.pth`
  - `rectified_flow_final.pth` & `rectified_flow_final_checkpoint.pth`

### Image Grids Generated

````carousel
![DDPM Grid Output](file:///C:/Users/joach/.gemini/antigravity/brain/cbe776a8-c256-4cb7-b611-fba8eb42479c/ddpm_output_grid.png)
<!-- slide -->
![Flow Matching Grid Output](file:///C:/Users/joach/.gemini/antigravity/brain/cbe776a8-c256-4cb7-b611-fba8eb42479c/flow_matching_output_grid.png)
<!-- slide -->
![Rectified Flow 1-Step Grid Output](file:///C:/Users/joach/.gemini/antigravity/brain/cbe776a8-c256-4cb7-b611-fba8eb42479c/rectified_flow_1step_output.png)
````
