# VQ-VAE EM Image Compression

This directory contains modular Python scripts for training and evaluating VQ-VAE models for electron microscopy (EM) image compression. The code is organized into reusable modules and three training modes with different compression ratios.

## 📁 Project Structure

```
WaferTools_V3/
├── vqvae_modules/           # Core modules
│   ├── __init__.py
│   ├── layers.py            # VQ-VAE layers (residual blocks, VectorQuantizerEMA)
│   ├── models.py            # Model architectures (VQVAETopOnly, VQVAETwoLevel)
│   ├── data_utils.py        # Data loading, preprocessing, tiling
│   ├── eval_utils.py        # Evaluation metrics, visualization, export
│   └── config.py            # Configuration presets for each mode
│
├── train_mode1_toponly.py   # Mode 1: Top-Only VQ-VAE (1024× compression)
├── train_mode2_twolevel.py  # Mode 2: Two-Level VQ-VAE (204× compression)
├── train_mode3_prior.py     # Mode 3: Template for Transformer prior
├── evaluate_vqvae.py        # Evaluation script
└── VQVAE_README.md          # This file
```

## 🎯 Compression Modes

### Mode 1: Top-Only VQ-VAE (1024× compression)

**Best for:** Whole-cell segmentation without requiring a trained prior

- **Compression ratio:** ~1024×
- **Key features:**
  - Extreme compression using only top-level quantizer
  - Membrane predictions remain stable
  - Suitable for segmentation tasks
- **Known artifacts:** Occasional 2D membrane breaks (requires 3D repair)
- **Training time:** Shortest

**Usage:**
```bash
python train_mode1_toponly.py \
    --data_dir /path/to/em/images \
    --output_dir ./runs/mode1 \
    --batch_size 2 \
    --epochs 100
```

### Mode 2: Two-Level VQ-VAE (204× compression)

**Best for:** Preserving overall cellular texture with moderate compression

- **Compression ratio:** ~204×
- **Key features:**
  - Uses both top (coarse) and bottom (fine) quantizers
  - Good global appearance retention
  - Better texture preservation than Mode 1
- **Known artifacts:** Possible vesicle shape deformation
- **Training time:** Moderate

**Usage:**
```bash
python train_mode2_twolevel.py \
    --data_dir /path/to/em/images \
    --output_dir ./runs/mode2 \
    --batch_size 2 \
    --epochs 150
```

### Mode 3: Two-Level + Transformer Prior (1024× compression)

**Best for:** High compression with more detail restored

- **Compression ratio:** ~1024×
- **Key features:**
  - Top-VQ tokens condition prediction of Bottom-VQ tokens
  - Bottom level fills in details
  - Higher quality than Mode 1 at same compression
- **Requirements:** 
  - Pre-trained Mode 2 model
  - Transformer prior implementation (template provided)
- **Training time:** Longest (Mode 2 + Transformer)

**Note:** Mode 3 requires additional implementation of the Transformer prior network. See `train_mode3_prior.py` for structure.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow>=2.10.0 numpy pillow matplotlib tqdm
```

### 2. Prepare Your Data

Organize your EM images in a directory:
```
/path/to/em/images/
├── image_001.png
├── image_002.png
├── image_003.tif
└── ...
```

Supported formats: PNG, JPG, TIFF (grayscale or RGB, will be converted to grayscale)

### 3. Train a Model

**Start with Mode 1 for fastest results:**

```bash
python train_mode1_toponly.py \
    --data_dir /path/to/em/images \
    --output_dir ./runs/mode1_experiment \
    --epochs 100
```

**For better quality, use Mode 2:**

```bash
python train_mode2_twolevel.py \
    --data_dir /path/to/em/images \
    --output_dir ./runs/mode2_experiment \
    --epochs 150
```

### 4. Evaluate the Model

**Single image evaluation:**

```bash
python evaluate_vqvae.py \
    --model_dir ./runs/mode1_experiment \
    --image /path/to/test_image.png \
    --output ./eval_results
```

**Batch evaluation:**

```bash
python evaluate_vqvae.py \
    --model_dir ./runs/mode1_experiment \
    --image_dir /path/to/test/images \
    --output ./eval_results \
    --batch \
    --max_images 30
```

## 📊 Output Structure

After training, the output directory contains:

```
runs/mode1_experiment/
├── config.json              # Model configuration
├── model.weights.h5         # Final model weights
├── best_loss.weights.h5     # Best weights by validation loss
├── best_ssim.weights.h5     # Best weights by SSIM
├── training.csv             # Training metrics log
├── history.json             # Full training history
├── vq_top_embeddings.npy    # Codebook embeddings
├── vq_top_ema_*.npy         # EMA statistics
├── tensorboard/             # TensorBoard logs
└── samples/                 # Validation samples
    ├── sample_000_input.png
    ├── sample_000_recon.png
    └── ...
```

Evaluation results:

```
eval_results/
├── results.json             # Metrics and compression stats
├── results.csv              # (batch mode) Per-image results
├── input.png                # Original image
├── recon.png                # Reconstructed image
└── panel.png                # Side-by-side comparison
```

## 🔧 Configuration

### Custom Hyperparameters

Edit `vqvae_modules/config.py` or pass arguments to training scripts:

```bash
python train_mode1_toponly.py \
    --data_dir ./data \
    --output_dir ./runs/custom \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 1e-4
```

### Key Hyperparameters

**Mode 1 (Top-Only):**
- `image_size`: 1024 (tile size)
- `top_grid`: 32 (latent resolution)
- `num_embeddings_top`: 256 (codebook size)
- `commitment_cost_top`: 1.0 (encoder commitment)

**Mode 2 (Two-Level):**
- `top_grid`: 32 (coarse resolution)
- `bottom_grid`: 64 (fine resolution)
- `num_embeddings_top`: 256
- `num_embeddings_bottom`: 256
- `commitment_cost_bottom`: 0.15 (lower for fine details)

## 📈 Evaluation Metrics

The scripts report:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better (typical: 25-35 dB)
- **SSIM** (Structural Similarity): Higher is better (0-1 scale, typical: 0.85-0.95)
- **Compression ratio**: e.g., 1024× means original is 1024× larger
- **Bits per pixel (bpp)**: Lower is better (typical: 0.004-0.04)

## 🛠️ Module Usage

You can also import and use the modules in your own code:

```python
from vqvae_modules import VQVAETopOnly, create_tiled_dataset, evaluate_single_image
from vqvae_modules.config import get_config

# Load config
config = get_config('top_only')

# Build model
model = VQVAETopOnly(
    image_size=config['image_size'],
    top_grid=config['top_grid'],
    num_embeddings_top=config['num_embeddings_top'],
    # ... other params
)

# Create dataset
train_ds = create_tiled_dataset(
    file_paths=['image1.png', 'image2.png'],
    tile_size=1024,
    batch_size=2,
    shuffle=True
)

# Train
model.compile(optimizer='adam', loss='mae')
model.fit(train_ds, epochs=10)
```

## 🔬 Technical Details

### Vector Quantization

The VQ-VAE uses **Exponential Moving Average (EMA)** for codebook updates instead of straight gradient descent:

- More stable training
- No codebook collapse
- Better convergence

### Data Handling

Large images (e.g., 2048×4096) are automatically:

1. **Tiled** into 1024×1024 patches
2. **Processed** in batches
3. **Stitched** back together during inference

This allows training on arbitrarily large images with limited GPU memory.

### Architecture

**Encoder:** Progressive downsampling with stride-2 convolutions
- 1024 → 512 → 256 → 128 → 64 → 32 (for top-only)
- Residual refinement at bottleneck

**Decoder:** Progressive upsampling with transposed convolutions
- Mirror of encoder
- 32 → 64 → 128 → 256 → 512 → 1024

**Quantizer:** EMA-based vector quantization
- L2 distance for code lookup
- Straight-through gradient estimator
- Perplexity tracking

## 💡 Tips & Best Practices

1. **Start with Mode 1** to verify data pipeline and training setup
2. **Monitor perplexity** in TensorBoard - should be 50-200 for good codebook usage
3. **Use ReduceLROnPlateau** - automatically reduces learning rate when validation plateaus
4. **Save validation samples** - visually inspect reconstructions during training
5. **3D consistency** - For Mode 1, apply 3D morphological operations to repair membrane breaks
6. **Data augmentation** - Consider adding random flips/rotations for better generalization

## 🐛 Troubleshooting

**Low perplexity (<10):** Codebook collapse - try:
- Increasing `commitment_cost`
- Reducing learning rate
- Increasing codebook size

**High validation loss:** Overfitting - try:
- Increasing dataset size
- Using dropout/regularization
- Early stopping

**Out of memory:** Try:
- Reducing `batch_size`
- Using smaller `image_size` (e.g., 512)
- Enabling mixed precision training

**Poor reconstructions:** Try:
- Training longer
- Using Mode 2 instead of Mode 1
- Adjusting commitment costs

## 📚 References

- VQ-VAE: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
- VQ-VAE-2: Razavi et al., "Generating Diverse High-Fidelity Images with VQ-VAE-2", NeurIPS 2019

## 📝 Original Notebook

This modular structure was extracted from `vqvae2_em_compression.ipynb` for:
- Better code organization
- Easier experimentation
- Version control
- Deployment to servers

The notebook contained all three modes in a single file with inline configuration.

## 🤝 Contributing

To add new features:

1. Add model components to `vqvae_modules/models.py`
2. Add data processing to `vqvae_modules/data_utils.py`
3. Add evaluation metrics to `vqvae_modules/eval_utils.py`
4. Create a new training script following the existing patterns

---

**Questions?** Check the inline documentation in each module for detailed API information.

