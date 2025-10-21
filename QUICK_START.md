# VQ-VAE Quick Start Guide

## 🚀 5-Minute Start

### 1. Install (2 min)

```bash
cd /Users/fumingyang/Downloads/WaferTools_V3
pip install tensorflow pillow numpy matplotlib tqdm
```

### 2. Train Mode 1 (2 min setup)

```bash
python train_mode1_toponly.py \
    --data_dir /path/to/your/em/images \
    --output_dir ./my_first_run \
    --epochs 10 \
    --batch_size 2
```

### 3. Evaluate (1 min)

```bash
python evaluate_vqvae.py \
    --model_dir ./my_first_run \
    --image /path/to/test_image.png \
    --output ./results
```

**Done!** Check `./results/` for reconstructed image and metrics.

---

## 📋 Command Cheatsheet

### Training Commands

```bash
# Mode 1: 1024× compression (fastest)
python train_mode1_toponly.py --data_dir ./data --output_dir ./runs/m1

# Mode 2: 204× compression (better quality)
python train_mode2_twolevel.py --data_dir ./data --output_dir ./runs/m2 --epochs 150

# Custom hyperparameters
python train_mode1_toponly.py \
    --data_dir ./data \
    --output_dir ./runs/custom \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 1e-4
```

### Evaluation Commands

```bash
# Single image
python evaluate_vqvae.py \
    --model_dir ./runs/m1 \
    --image test.png \
    --output ./eval_single

# Batch of images
python evaluate_vqvae.py \
    --model_dir ./runs/m1 \
    --image_dir ./test_images \
    --output ./eval_batch \
    --batch \
    --max_images 30
```

### Monitoring

```bash
# TensorBoard (view training progress)
tensorboard --logdir ./runs/m1/tensorboard --port 6006

# Then open: http://localhost:6006
```

---

## 📊 Quick Mode Comparison

| Feature | Mode 1 | Mode 2 | Mode 3* |
|---------|--------|--------|---------|
| Compression | 1024× | 204× | 1024× |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Training Time | Fast | Moderate | Slow |
| Use Case | Segmentation | Texture | High compression + detail |

*Mode 3 requires additional implementation

---

## 🔍 File Structure After Training

```
./runs/m1/
├── config.json              ← Model configuration
├── model.weights.h5         ← Final weights
├── best_loss.weights.h5     ← Best checkpoint
├── training.csv             ← Metrics log
├── history.json             ← Full history
├── samples/                 ← Validation samples
└── tensorboard/             ← TensorBoard logs
```

---

## 💡 Common Recipes

### Recipe 1: Quick Test Run

```bash
# 10 epochs, small dataset
python train_mode1_toponly.py \
    --data_dir ./small_dataset \
    --output_dir ./test_run \
    --epochs 10 \
    --batch_size 1
```

### Recipe 2: Production Training

```bash
# Mode 2 with careful settings
python train_mode2_twolevel.py \
    --data_dir ./full_dataset \
    --output_dir ./production_run \
    --epochs 200 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### Recipe 3: Resume from Checkpoint

Currently the scripts train from scratch. To resume, manually load weights:

```python
from vqvae_modules import VQVAETopOnly
model = VQVAETopOnly(...)
model.load_weights('./runs/m1/best_loss.weights.h5')
# Continue training...
```

### Recipe 4: Batch Reconstruction

```bash
# Reconstruct all test images
python evaluate_vqvae.py \
    --model_dir ./runs/m1 \
    --image_dir ./test_set \
    --output ./reconstructions \
    --batch
```

---

## 🎯 Expected Results

### Mode 1 (Top-Only)
- **PSNR:** 28-32 dB (typical)
- **SSIM:** 0.85-0.92
- **Compression:** ~1024×
- **Training:** 50-100 epochs sufficient
- **Use:** Good for segmentation tasks

### Mode 2 (Two-Level)
- **PSNR:** 30-35 dB (typical)
- **SSIM:** 0.90-0.95
- **Compression:** ~204×
- **Training:** 100-200 epochs recommended
- **Use:** Best for texture preservation

---

## ⚡ Performance Tips

### Faster Training
```bash
# Reduce batch processing
--batch_size 1

# Fewer epochs for testing
--epochs 20

# Use smaller images (modify config.py)
```

### Better Quality
```bash
# More epochs
--epochs 200

# Lower learning rate (more stable)
--learning_rate 1e-4

# Use Mode 2 instead of Mode 1
python train_mode2_twolevel.py ...
```

### Less Memory
```bash
# Smaller batch size
--batch_size 1

# Gradient accumulation (implement if needed)
```

---

## 🐛 Quick Fixes

### "Out of memory"
```bash
python train_mode1_toponly.py --batch_size 1 ...
```

### "No images found"
Check your `--data_dir` has `.png`, `.jpg`, or `.tif` files

### "Model not found"
Verify `--model_dir` contains `config.json` and `.weights.h5` files

### Low PSNR (<25 dB)
- Train longer (more epochs)
- Check if images are properly normalized
- Try Mode 2 instead

---

## 📚 Learn More

- **Full guide:** `VQVAE_README.md`
- **Installation:** `INSTALLATION_GUIDE.md`
- **Conversion details:** `CONVERSION_SUMMARY.md`

---

## 🎓 Example Session

```bash
# Step 1: Prepare data
mkdir -p ./my_em_data
# ... copy your EM images here ...

# Step 2: Train Mode 1 (fast test)
python train_mode1_toponly.py \
    --data_dir ./my_em_data \
    --output_dir ./exp1 \
    --epochs 50

# Step 3: Monitor (in another terminal)
tensorboard --logdir ./exp1/tensorboard

# Step 4: Evaluate
python evaluate_vqvae.py \
    --model_dir ./exp1 \
    --image ./my_em_data/test_image.png \
    --output ./exp1_eval

# Step 5: Check results
open ./exp1_eval/panel.png  # macOS
# or
xdg-open ./exp1_eval/panel.png  # Linux

# Step 6: If satisfied, train Mode 2 for better quality
python train_mode2_twolevel.py \
    --data_dir ./my_em_data \
    --output_dir ./exp2 \
    --epochs 150
```

---

**That's it!** You're ready to compress EM images. 🎉

For detailed documentation, see `VQVAE_README.md`.

