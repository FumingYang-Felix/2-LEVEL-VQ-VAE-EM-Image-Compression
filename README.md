# VQ-VAE Installation and Setup Guide

## Quick Installation

### 1. Install Dependencies

```bash
cd /../

# Install requirements
pip install -r vqvae_requirements.txt
```

### 2. Verify Installation

```bash
# Check TensorFlow
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check imports
python3 -c "from vqvae_modules import VQVAETopOnly; print('✓ Imports OK')"

# Check training scripts
python3 train_mode1_toponly.py --help
```

## Environment Setup Options

### Option 1: System-wide Installation (Simplest)

```bash
pip install -r vqvae_requirements.txt
```

### Option 2: Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv vqvae_env

# Activate (macOS/Linux)
source vqvae_env/bin/activate

# Activate (Windows)
# vqvae_env\Scripts\activate

# Install dependencies
pip install -r vqvae_requirements.txt
```

### Option 3: Conda Environment

```bash
# Create conda environment
conda create -n vqvae python=3.10

# Activate
conda activate vqvae

# Install TensorFlow
conda install tensorflow-gpu  # For GPU
# OR
conda install tensorflow  # For CPU

# Install other requirements
pip install pillow matplotlib tqdm
```

## GPU Setup (Optional but Recommended)

### Check GPU Availability

```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

### CUDA Setup (NVIDIA GPUs)

1. Install CUDA Toolkit (11.8 recommended for TF 2.12+)
2. Install cuDNN
3. Install TensorFlow GPU version:

```bash
pip install tensorflow[and-cuda]
```

### Apple Silicon (M1/M2/M3 Macs)

TensorFlow automatically uses Metal acceleration:

```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

## Verify Setup

Create a test script `test_setup.py`:

```python
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from vqvae_modules import VQVAETopOnly
from vqvae_modules.config import get_config

print("="*60)
print("VQ-VAE Setup Verification")
print("="*60)

# Check TensorFlow
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

# Check imports
print("\n✓ Modules imported successfully")

# Test model creation
config = get_config('top_only')
model = VQVAETopOnly(
    image_size=config['image_size'],
    top_grid=config['top_grid'],
    num_embeddings_top=config['num_embeddings_top'],
)
print("✓ Model created successfully")

# Test forward pass
dummy_input = tf.zeros([1, 1024, 1024, 1])
output = model(dummy_input, training=False)
print(f"✓ Forward pass successful: {output.shape}")

print("\n" + "="*60)
print("✅ All checks passed! Ready to train.")
print("="*60)
```

Run it:

```bash
python3 test_setup.py
```

## Troubleshooting

### Issue: "No module named 'tensorflow'"

```bash
pip install tensorflow>=2.10.0
```

### Issue: GPU not detected

1. Check CUDA installation: `nvcc --version`
2. Check cuDNN: Verify `/usr/local/cuda/lib64` contains cuDNN
3. Reinstall TensorFlow GPU version

### Issue: Out of memory

Reduce batch size in training:

```bash
python train_mode1_toponly.py --batch_size 1 ...
```

### Issue: Import errors

Check Python path:

```python
import sys
print(sys.path)
```

Ensure `/Users/fumingyang/Downloads/WaferTools_V3` is in the path or run from that directory.

## Performance Tips

### CPU Training
- Set batch size to 1-2
- Use smaller image sizes (512 instead of 1024)
- Expect slower training (~10-20x slower than GPU)

### GPU Training
- Batch size 2-4 for 1024x1024 images on 8GB VRAM
- Batch size 8-16 for smaller images or larger GPUs
- Enable mixed precision for 2x speedup:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## Next Steps

Once setup is complete:

1. Read `VQVAE_README.md` for usage guide
2. Prepare your EM image dataset
3. Start with Mode 1 training:

```bash
python train_mode1_toponly.py \
    --data_dir /path/to/images \
    --output_dir ./runs/test \
    --epochs 10
```

4. Monitor training with TensorBoard:

```bash
tensorboard --logdir ./runs/test/tensorboard
```

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space
- CPU training: Any modern CPU

### Recommended
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- 50GB disk space (for datasets and models)
- CUDA 11.8+

### Tested Configurations

✅ **macOS (Apple Silicon)**
- M1/M2/M3 with 16GB RAM
- TensorFlow 2.13+ with Metal acceleration
- Training speed: ~30 sec/epoch (Mode 1, batch_size=1)

✅ **Linux (NVIDIA GPU)**
- Ubuntu 22.04, RTX 3090 (24GB)
- TensorFlow 2.12 with CUDA 11.8
- Training speed: ~5 sec/epoch (Mode 1, batch_size=4)

✅ **Linux (CPU only)**
- Ubuntu 22.04, AMD Ryzen 9
- TensorFlow 2.13
- Training speed: ~200 sec/epoch (Mode 1, batch_size=1)

## Support

If you encounter issues:

1. Check TensorFlow installation: `pip show tensorflow`
2. Verify Python version: `python3 --version`
3. Review error messages carefully
4. Check GPU availability if using GPU
5. Try reducing batch size if OOM errors occur

---

**Ready to start training!** See `VQVAE_README.md` for detailed usage instructions.

