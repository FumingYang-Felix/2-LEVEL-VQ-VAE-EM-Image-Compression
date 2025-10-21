"""
Shared neural network layers for VQ-VAE models.
Includes residual blocks and EMA-based vector quantizer.
"""

import tensorflow as tf


def residual_block(x, filters):
    """Single residual block with bottleneck structure."""
    skip = x
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters//2, 3, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 1, padding='same')(x)
    return tf.keras.layers.Add()([skip, x])


def residual_stack(x, filters, num_blocks=2):
    """Stack of residual blocks followed by ReLU activation."""
    for _ in range(num_blocks):
        x = residual_block(x, filters)
    return tf.keras.layers.ReLU()(x)


class VectorQuantizerEMA(tf.keras.layers.Layer):
    """
    Vector Quantizer with Exponential Moving Average (EMA) codebook updates.
    
    Args:
        num_embeddings: Number of discrete codes in the codebook
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss (encoder commitment to codes)
        decay: EMA decay rate for codebook updates
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, num_embeddings, embedding_dim,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self._perplexity = tf.keras.metrics.Mean(name=f'{self.name}_perplexity')

    @property
    def metrics(self):
        return [self._perplexity]

    def build(self, input_shape):
        # Codebook embeddings (transposed: [embedding_dim, num_embeddings])
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=tf.keras.initializers.RandomUniform(
                -1.0/self.num_embeddings, 1.0/self.num_embeddings
            ),
            trainable=False
        )
        
        # EMA tracking variables
        self.ema_cluster_size = self.add_weight(
            name='ema_cluster_size',
            shape=(self.num_embeddings,),
            initializer='zeros',
            trainable=False
        )
        self.ema_dw = self.add_weight(
            name='ema_dw',
            shape=(self.embedding_dim, self.num_embeddings),
            initializer='zeros',
            trainable=False
        )

    def call(self, inputs, training=None):
        """
        Forward pass: quantize continuous inputs to discrete codes.
        
        Args:
            inputs: [B, H, W, C] tensor of continuous features
            training: Whether in training mode (for EMA updates)
            
        Returns:
            Quantized tensor [B, H, W, C] with straight-through gradient
        """
        shp = tf.shape(inputs)  # [B, H, W, C]
        flat = tf.reshape(inputs, [-1, self.embedding_dim])  # [N, C]

        # Compute L2 distances to all codebook entries
        dists = (
            tf.reduce_sum(flat**2, axis=1, keepdims=True)
            - 2.0 * tf.matmul(flat, self.embeddings)
            + tf.reduce_sum(self.embeddings**2, axis=0, keepdims=True)
        )  # [N, K]
        
        # Find nearest code
        idx = tf.argmax(-dists, axis=1)  # [N]
        one_hot = tf.one_hot(idx, self.num_embeddings, dtype=flat.dtype)  # [N, K]
        
        # Lookup quantized values
        quant = tf.matmul(one_hot, tf.transpose(self.embeddings))  # [N, C]
        quant = tf.reshape(quant, shp)  # [B, H, W, C]

        # EMA updates (only during training)
        def _ema_update():
            cluster_size = tf.reduce_sum(one_hot, axis=0)
            dw = tf.matmul(tf.transpose(flat), one_hot)
            
            # Update EMA statistics
            ema_cs = self.ema_cluster_size * self.decay + cluster_size * (1.0 - self.decay)
            ema_dw = self.ema_dw * self.decay + dw * (1.0 - self.decay)
            
            # Laplace smoothing
            n = tf.reduce_sum(ema_cs)
            smoothed_cs = (
                (ema_cs + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            )
            
            # Update codebook
            new_embed = ema_dw / tf.expand_dims(smoothed_cs, 0)
            
            self.ema_cluster_size.assign(ema_cs)
            self.ema_dw.assign(ema_dw)
            self.embeddings.assign(new_embed)
            return 0.0

        if training is None:
            training = tf.keras.backend.learning_phase()
        tf.cond(tf.cast(training, tf.bool), _ema_update, lambda: 0.0)

        # Commitment loss (encoder commits to chosen codes)
        e_loss = tf.reduce_mean((tf.stop_gradient(quant) - inputs)**2)
        self.add_loss(self.commitment_cost * e_loss)

        # Compute perplexity (measure of codebook usage)
        avg_probs = tf.reduce_mean(one_hot, axis=0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        self._perplexity.update_state(perplexity)

        # Straight-through estimator (copy gradients from output to input)
        quant = inputs + tf.stop_gradient(quant - inputs)
        return quant

