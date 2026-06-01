# Resuming Training from Checkpoints

This document outlines how the model pipeline in `FM_RM_Colab_ZBook01_test.ipynb` manages loading checkpoints from the local path (`C:\Users\joach\antiG\antiCeleb064\Alex_RM\Checkpoints_FM`) and how the training loop length is corrected dynamically.

---

## 1. Checkpoint Directory Path Mapping

The notebook handles path resolution dynamically depending on the execution environment:
* **Google Colab:** Set up via Google Drive directories.
* **Local Machine (ZBook):** Resolves paths relative to the workspace:
  * `DRIVE_BASE = "./Alex_RM"`
  * `DRIVE_CHECKPOINTS = DRIVE_BASE + "/Checkpoints_FM"` (maps to `C:\Users\joach\antiG\antiCeleb064\Alex_RM\Checkpoints_FM`)

---

## 2. Checkpoint Loading Flow (Cell 10)

When resuming training from a specific epoch (e.g., `RESUME_FROM_EPOCH = 25`), the following operations occur:

### Step 1: Check File Existence
Constructs the file path to the checkpoint and verifies its presence on disk:
```python
checkpoint_load_path = DRIVE_CHECKPOINTS + f"/flow_matching_checkpoint_epoch_{RESUME_FROM_EPOCH}.pth"
if os.path.exists(checkpoint_load_path):
```

### Step 2: Load the State Dictionary
Loads the checkpoint's state (metadata, model weights, optimizer variables) from storage:
```python
checkpoint = torch.load(checkpoint_load_path, map_location=Device)
```

### Step 3: Restore Model Weights
Loads the saved parameters into the U-Net architecture:
```python
Modell.load_state_dict(checkpoint['model_state_dict'])
```

### Step 4: Restore Optimizer State
Restores the optimizer's momentum, learning rate schedulers, and weight updates:
```python
Optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Step 5: Adjust the Starting Epoch
Aligns the starting epoch of the training loop to continue from the next consecutive epoch:
```python
START_EPOCH = checkpoint['epoch'] + 1
```

---

## 3. Dynamic Epoch Correction

### The Problem
If the model loaded a checkpoint from epoch `25` (setting `START_EPOCH = 26`) but the program was set to test mode with `EPOCHS = 5`, the training loop:
```python
for Epoche in range(START_EPOCH, EPOCHS + 1):
```
would translate to `range(26, 6)`. Because the starting index is higher than the ending index, **the loop would run 0 times and skip training entirely.**

### The Solution
We added dynamic corrections to the loop bounds directly inside Cell 10, adjusting the value of `EPOCHS` dynamically to match the loaded data:

```python
    START_EPOCH = checkpoint['epoch'] + 1
    
    # Dynamically adjust the total number of epochs (EPOCHS) based on the loaded file:
    if TEST_MODE:
        EPOCHS = START_EPOCH + 4  # Trains for exactly 5 epochs (e.g., epochs 26 to 30)
    else:
        if EPOCHS < START_EPOCH:
            EPOCHS = START_EPOCH + (200 - checkpoint['epoch'])  # Standardizes full training to 200 epochs total
```

This guarantees that the training loop runs for the intended remaining duration (either 5 epochs for testing or the correct total remaining epochs for production) without manual manual editing of parameters.
