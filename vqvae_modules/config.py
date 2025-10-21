"""
Configuration presets for VQ-VAE training.
"""


# Mode 1: Top-Only VQ-VAE (1024× compression)
MODE1_CONFIG = {
    'mode': 'top_only',
    'image_size': 1024,
    'tile_size': 1024,
    'top_grid': 32,
    'num_hiddens_top': 128,
    'embedding_dim_top': 96,
    'num_embeddings_top': 256,
    'commitment_cost_top': 1.0,
    
    'batch_size': 2,
    'epochs': 100,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    
    'train_split': 0.8,  # First 80% for training, rest for validation
    'shuffle_buffer': 2048,
}


# Mode 2: Two-Level VQ-VAE (204× compression)
MODE2_CONFIG = {
    'mode': 'two_level',
    'image_size': 1024,
    'tile_size': 1024,
    'top_grid': 32,
    'bottom_grid': 64,
    
    'num_hiddens_top': 128,
    'embedding_dim_top': 96,
    'num_embeddings_top': 256,
    'commitment_cost_top': 0.25,
    
    'num_hiddens_bottom': 128,
    'embedding_dim_bottom': 64,
    'num_embeddings_bottom': 256,
    'commitment_cost_bottom': 0.15,
    
    'batch_size': 2,
    'epochs': 150,
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    
    'train_split': 0.8,
    'shuffle_buffer': 2048,
}


# Mode 3: Two-Level + Transformer Prior (1024× compression)
# Note: Mode 3 requires pre-trained Mode 2 model + additional transformer training
MODE3_CONFIG = {
    'mode': 'two_level_with_prior',
    'image_size': 1024,
    'tile_size': 1024,
    'top_grid': 32,
    'bottom_grid': 64,
    
    # Same architecture as Mode 2
    'num_hiddens_top': 128,
    'embedding_dim_top': 96,
    'num_embeddings_top': 256,
    'commitment_cost_top': 0.25,
    
    'num_hiddens_bottom': 128,
    'embedding_dim_bottom': 64,
    'num_embeddings_bottom': 256,
    'commitment_cost_bottom': 0.15,
    
    # Transformer prior hyperparams
    'transformer_layers': 8,
    'transformer_heads': 8,
    'transformer_dim': 512,
    'transformer_dropout': 0.1,
    
    'batch_size': 2,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    'train_split': 0.8,
    'shuffle_buffer': 2048,
    
    # Path to pre-trained Mode 2 model
    'pretrained_vqvae_path': None,  # Set this before training
}


def get_config(mode='top_only'):
    """
    Get configuration for a specific mode.
    
    Args:
        mode: One of 'top_only', 'two_level', or 'two_level_with_prior'
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'top_only': MODE1_CONFIG,
        'two_level': MODE2_CONFIG,
        'two_level_with_prior': MODE3_CONFIG,
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(configs.keys())}")
    
    return configs[mode].copy()


def update_config(config, **kwargs):
    """
    Update configuration with custom values.
    
    Args:
        config: Base configuration dictionary
        **kwargs: Key-value pairs to update
        
    Returns:
        Updated configuration dictionary
    """
    updated = config.copy()
    updated.update(kwargs)
    return updated

