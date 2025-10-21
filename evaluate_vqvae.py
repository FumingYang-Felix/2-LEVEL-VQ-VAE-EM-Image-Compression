#!/usr/bin/env python3
"""
Evaluate a trained VQ-VAE model on test images.

Usage:
    # Evaluate single image
    python evaluate_vqvae.py --model_dir ./runs/mode1 --image /path/to/test.png --output ./eval_results
    
    # Evaluate batch of images
    python evaluate_vqvae.py --model_dir ./runs/mode1 --image_dir /path/to/test_images --output ./eval_results --batch
"""

import os
import argparse
import json
import glob
import tensorflow as tf

from vqvae_modules.models import VQVAETopOnly, VQVAETwoLevel
from vqvae_modules.eval_utils import evaluate_single_image, evaluate_batch_images
from vqvae_modules.data_utils import natural_sort_key


def load_model_from_bundle(model_dir):
    """
    Load a trained VQ-VAE model from a bundle directory.
    
    Args:
        model_dir: Path to model bundle directory containing config.json and weights
        
    Returns:
        Tuple of (model, config)
    """
    # Load config
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading model: {config.get('model_type', 'Unknown')}")
    print(f"Mode: {config.get('mode', 'Unknown')}")
    
    # Build model based on mode
    mode = config.get('mode', 'top_only')
    
    if mode == 'top_only':
        model = VQVAETopOnly(
            image_size=config['image_size'],
            top_grid=config['top_grid'],
            num_hiddens_top=config['num_hiddens_top'],
            embedding_dim_top=config['embedding_dim_top'],
            num_embeddings_top=config['num_embeddings_top'],
            commitment_cost_top=config['commitment_cost_top']
        )
    elif mode in ['two_level', 'two_level_with_prior']:
        model = VQVAETwoLevel(
            image_size=config['image_size'],
            top_grid=config['top_grid'],
            bottom_grid=config['bottom_grid'],
            num_hiddens_top=config['num_hiddens_top'],
            embedding_dim_top=config['embedding_dim_top'],
            num_embeddings_top=config['num_embeddings_top'],
            num_hiddens_bottom=config['num_hiddens_bottom'],
            embedding_dim_bottom=config['embedding_dim_bottom'],
            num_embeddings_bottom=config['num_embeddings_bottom'],
            commitment_cost_top=config['commitment_cost_top'],
            commitment_cost_bottom=config['commitment_cost_bottom']
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Build model by running a dummy forward pass
    dummy_input = tf.zeros([1, config['image_size'], config['image_size'], 1])
    _ = model(dummy_input, training=False)
    
    # Load weights
    weights_path = os.path.join(model_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        # Try alternative names
        alt_weights = [
            os.path.join(model_dir, 'best_loss.weights.h5'),
            os.path.join(model_dir, 'best_ssim.weights.h5'),
        ]
        for alt in alt_weights:
            if os.path.exists(alt):
                weights_path = alt
                break
        else:
            raise FileNotFoundError(f"Weights file not found in {model_dir}")
    
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    print("Model loaded successfully")
    
    return model, config


def main(args):
    print("\n" + "="*60)
    print("VQ-VAE Model Evaluation")
    print("="*60 + "\n")
    
    # Load model
    model, config = load_model_from_bundle(args.model_dir)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.batch:
        # Batch evaluation mode
        if not args.image_dir:
            raise ValueError("--image_dir is required for batch evaluation")
        
        print(f"\nCollecting images from: {args.image_dir}")
        
        # Collect images
        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        image_paths = [
            p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
            if os.path.splitext(p)[1].lower() in extensions
        ]
        image_paths = sorted(image_paths, key=natural_sort_key)
        
        if args.max_images:
            image_paths = image_paths[:args.max_images]
        
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("No images found!")
            return
        
        # Evaluate batch
        results = evaluate_batch_images(
            model,
            image_paths,
            config,
            args.output,
            batch_size=config.get('batch_size', 2)
        )
        
    else:
        # Single image evaluation mode
        if not args.image:
            raise ValueError("--image is required for single image evaluation")
        
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")
        
        print(f"\nEvaluating: {args.image}")
        
        # Evaluate single image
        results = evaluate_single_image(
            model,
            args.image,
            config,
            args.output,
            save_panel=True
        )
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Results saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained VQ-VAE model'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Directory containing trained model bundle'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for evaluation'
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Directory containing test images (for batch evaluation)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./eval_results',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch evaluation mode'
    )
    
    parser.add_argument(
        '--max_images',
        type=int,
        help='Maximum number of images to evaluate (for batch mode)'
    )
    
    args = parser.parse_args()
    main(args)

