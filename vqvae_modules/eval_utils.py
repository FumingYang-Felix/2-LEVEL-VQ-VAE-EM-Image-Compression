"""
Evaluation and visualization utilities for VQ-VAE models.
"""

import os
import json
import math
import time
import csv
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from .data_utils import reconstruct_full_image


def to_uint8(x):
    """Convert [0, 1] float array to uint8 image."""
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


def save_png(img, path):
    """Save numpy array as PNG image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    Image.fromarray(to_uint8(img)).save(path, 'PNG')


def compute_metrics(original, reconstructed):
    """
    Compute PSNR and SSIM metrics.
    
    Args:
        original: Original image [H, W] or [B, H, W, C] in [0, 1]
        reconstructed: Reconstructed image [H, W] or [B, H, W, C] in [0, 1]
        
    Returns:
        Dictionary with 'psnr' and 'ssim' keys
    """
    # Ensure 4D tensors
    if len(original.shape) == 2:
        original = original[None, ..., None]
    elif len(original.shape) == 3:
        original = original[None, ...]
    
    if len(reconstructed.shape) == 2:
        reconstructed = reconstructed[None, ..., None]
    elif len(reconstructed.shape) == 3:
        reconstructed = reconstructed[None, ...]
    
    orig_t = tf.convert_to_tensor(original, dtype=tf.float32)
    recon_t = tf.convert_to_tensor(reconstructed, dtype=tf.float32)
    
    psnr = float(tf.reduce_mean(tf.image.psnr(orig_t, recon_t, max_val=1.0)).numpy())
    ssim = float(tf.reduce_mean(tf.image.ssim(orig_t, recon_t, max_val=1.0)).numpy())
    
    return {'psnr': psnr, 'ssim': ssim}


def compute_compression_stats(image_size, top_grid, num_embeddings_top,
                               bottom_grid=None, num_embeddings_bottom=None):
    """
    Compute compression statistics.
    
    Args:
        image_size: Original image size (e.g., 1024)
        top_grid: Top latent grid size (e.g., 32)
        num_embeddings_top: Number of top codebook entries
        bottom_grid: Bottom latent grid size (optional, for two-level)
        num_embeddings_bottom: Number of bottom codebook entries (optional)
        
    Returns:
        Dictionary with compression statistics
    """
    bits_per_top = int(math.ceil(math.log2(num_embeddings_top)))
    tokens_top = top_grid * top_grid
    latent_bits_top = tokens_top * bits_per_top
    
    if bottom_grid is not None and num_embeddings_bottom is not None:
        bits_per_bottom = int(math.ceil(math.log2(num_embeddings_bottom)))
        tokens_bottom = bottom_grid * bottom_grid
        latent_bits_bottom = tokens_bottom * bits_per_bottom
        latent_bits_total = latent_bits_top + latent_bits_bottom
    else:
        bits_per_bottom = None
        tokens_bottom = None
        latent_bits_total = latent_bits_top
    
    orig_bits = image_size * image_size * 8  # 8-bit grayscale
    ratio = orig_bits / latent_bits_total
    bpp = latent_bits_total / (image_size * image_size)
    
    stats = {
        'image_size': image_size,
        'tokens_top': tokens_top,
        'bits_per_token_top': bits_per_top,
        'latent_bits_top': latent_bits_top,
        'orig_bits': orig_bits,
        'compression_ratio': round(ratio, 2),
        'bpp': round(bpp, 6)
    }
    
    if bottom_grid is not None:
        stats.update({
            'tokens_bottom': tokens_bottom,
            'bits_per_token_bottom': bits_per_bottom,
            'latent_bits_bottom': latent_bits_bottom,
            'latent_bits_total': latent_bits_total
        })
    
    return stats


def evaluate_single_image(model, image_path, config, output_dir, save_panel=True):
    """
    Evaluate model on a single image and save results.
    
    Args:
        model: Trained VQ-VAE model
        image_path: Path to test image
        config: Model configuration dictionary
        output_dir: Directory to save results
        save_panel: Whether to save visualization panel
        
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconstruct image
    orig, recon = reconstruct_full_image(
        model, image_path,
        tile_size=config['image_size'],
        batch_size=config.get('batch_size', 2)
    )
    
    # Compute metrics
    metrics = compute_metrics(orig, recon)
    
    # Compute compression stats
    comp_stats = compute_compression_stats(
        config['image_size'],
        config['top_grid'],
        config['num_embeddings_top'],
        config.get('bottom_grid'),
        config.get('num_embeddings_bottom')
    )
    
    # Save images
    save_png(orig, os.path.join(output_dir, 'input.png'))
    save_png(recon, os.path.join(output_dir, 'recon.png'))
    
    # Save panel
    if save_panel:
        fig = plt.figure(figsize=(10, 5), dpi=150)
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(orig, cmap='gray')
        ax1.set_title(f"Input")
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(recon, cmap='gray')
        ax2.set_title(f"Recon (PSNR: {metrics['psnr']:.2f} dB)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'panel.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Combine results
    results = {
        'image_path': image_path,
        **metrics,
        **comp_stats
    }
    
    # Save JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[Evaluation] {os.path.basename(image_path)}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    print(f"  Compression: {comp_stats['compression_ratio']:.1f}x (~{comp_stats['bpp']:.5f} bpp)")
    
    return results


def evaluate_batch_images(model, image_paths, config, output_dir, batch_size=2):
    """
    Evaluate model on multiple images and save aggregate statistics.
    
    Args:
        model: Trained VQ-VAE model
        image_paths: List of image paths to evaluate
        config: Model configuration dictionary
        output_dir: Directory to save results
        batch_size: Batch size for processing tiles
        
    Returns:
        Dictionary with aggregate statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for idx, path in enumerate(image_paths):
        print(f"\n[{idx+1}/{len(image_paths)}] Evaluating {os.path.basename(path)}...")
        
        # Reconstruct
        orig, recon = reconstruct_full_image(
            model, path,
            tile_size=config['image_size'],
            batch_size=batch_size
        )
        
        # Compute metrics
        metrics = compute_metrics(orig, recon)
        
        # Save images
        stem = f"{idx:04d}"
        save_png(orig, os.path.join(output_dir, f'{stem}_input.png'))
        save_png(recon, os.path.join(output_dir, f'{stem}_recon.png'))
        
        results.append({
            'index': idx,
            'image_path': path,
            **metrics
        })
    
    # Compute aggregate statistics
    avg_psnr = float(np.mean([r['psnr'] for r in results]))
    avg_ssim = float(np.mean([r['ssim'] for r in results]))
    std_psnr = float(np.std([r['psnr'] for r in results]))
    std_ssim = float(np.std([r['ssim'] for r in results]))
    
    # Compression stats
    comp_stats = compute_compression_stats(
        config['image_size'],
        config['top_grid'],
        config['num_embeddings_top'],
        config.get('bottom_grid'),
        config.get('num_embeddings_bottom')
    )
    
    summary = {
        'count': len(image_paths),
        'avg_psnr': round(avg_psnr, 4),
        'avg_ssim': round(avg_ssim, 6),
        'std_psnr': round(std_psnr, 4),
        'std_ssim': round(std_ssim, 6),
        **comp_stats
    }
    
    # Save JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    
    # Save CSV
    with open(os.path.join(output_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'image_path', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"[BATCH EVALUATION COMPLETE]")
    print(f"  N = {len(image_paths)}")
    print(f"  Avg PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  Avg SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  Compression: {comp_stats['compression_ratio']:.1f}x (~{comp_stats['bpp']:.5f} bpp)")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return summary


def export_model_bundle(model, config, output_dir, history=None, 
                        val_dataset=None, num_samples=1):
    """
    Export trained model bundle with weights, codebooks, and samples.
    
    Args:
        model: Trained VQ-VAE model
        config: Model configuration dictionary
        output_dir: Directory to save bundle
        history: Training history (optional)
        val_dataset: Validation dataset for samples (optional)
        num_samples: Number of validation samples to save
        
    Returns:
        Path to bundle directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model once to initialize weights
    dummy_input = tf.zeros([1, config['image_size'], config['image_size'], 1])
    _ = model(dummy_input, training=False)
    
    # Save weights
    weights_path = os.path.join(output_dir, 'model.weights.h5')
    model.save_weights(weights_path)
    print(f"✅ Saved weights: {weights_path}")
    
    # Save codebooks
    if hasattr(model, 'vq_top'):
        np.save(os.path.join(output_dir, 'vq_top_embeddings.npy'), 
                model.vq_top.embeddings.numpy())
        np.save(os.path.join(output_dir, 'vq_top_ema_cluster_size.npy'),
                model.vq_top.ema_cluster_size.numpy())
        np.save(os.path.join(output_dir, 'vq_top_ema_dw.npy'),
                model.vq_top.ema_dw.numpy())
        print("✅ Saved top codebook")
    
    if hasattr(model, 'vq_bottom'):
        np.save(os.path.join(output_dir, 'vq_bottom_embeddings.npy'),
                model.vq_bottom.embeddings.numpy())
        np.save(os.path.join(output_dir, 'vq_bottom_ema_cluster_size.npy'),
                model.vq_bottom.ema_cluster_size.numpy())
        np.save(os.path.join(output_dir, 'vq_bottom_ema_dw.npy'),
                model.vq_bottom.ema_dw.numpy())
        print("✅ Saved bottom codebook")
    
    # Save config with timestamp
    config_with_meta = {
        **config,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': model.__class__.__name__
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_with_meta, f, indent=2)
    print("✅ Saved config")
    
    # Save training history
    if history is not None and hasattr(history, 'history'):
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(history.history, f, indent=2)
        print("✅ Saved training history")
    
    # Save validation samples
    if val_dataset is not None and num_samples > 0:
        samples_dir = os.path.join(output_dir, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        taken = 0
        for xb, _ in val_dataset.take(num_samples):
            yb = model(xb, training=False)
            
            for i in range(min(xb.shape[0], num_samples - taken)):
                orig = (xb.numpy()[i, ..., 0] + 0.5)
                recon = (yb.numpy()[i, ..., 0] + 0.5)
                
                save_png(orig, os.path.join(samples_dir, f'sample_{taken:03d}_input.png'))
                save_png(recon, os.path.join(samples_dir, f'sample_{taken:03d}_recon.png'))
                taken += 1
                
                if taken >= num_samples:
                    break
            
            if taken >= num_samples:
                break
        
        print(f"✅ Saved {taken} validation samples")
    
    print(f"\n{'='*60}")
    print(f"Model bundle saved to: {output_dir}")
    print(f"{'='*60}")
    
    return output_dir

