"""
AE_WGAN.py
===========
AE-WGAN (Denoising Autoencoder + Wasserstein GAN) for Adversarial Attack 
Generation in Anomaly-Based Network Intrusion Detection.

Based on: Arafah et al. (2025), Applied Soft Computing, 168, 112455.

Architecture (NSL-KDD — 24 features after ANOVA):
    Autoencoder:
        Encoder: 24 → 20 → 16  (ReLU)
        Decoder: 16 → 20 → 24  (ReLU)
    
    WGAN:
        Generator: 32 (noise) → 64 → 24  (ReLU)
        Critic:    24 → 16 → 1           (Linear output)

Training Configuration:
    Optimizer: Adam (lr=1e-4)
    Epochs: 100
    Batch size: 32
    Critic updates per generator update: 5
    Gradient penalty coefficient (lambda): 10

Usage:
    from AE_WGAN import AEWGAN
    model = AEWGAN(input_dim=24)
    model.train(X_attack, epochs=100)
    synthetic = model.generate(n_samples=1000)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ============================================================================
# AUTOENCODER (Denoising AE)
# ============================================================================

class DenoisingAutoencoder:
    """
    Denoising Autoencoder component of AE-WGAN.
    
    Architecture (for NSL-KDD, input_dim=24):
        Encoder: input_dim → 20 → latent_dim (16)   [ReLU]
        Decoder: latent_dim (16) → 20 → input_dim    [ReLU]
    
    The AE reduces noise in the data before passing to the WGAN,
    improving the quality of generated attacks.
    """
    
    def __init__(self, input_dim, latent_dim=16, intermediate_dim=20):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input features (24 for NSL-KDD, 10 for CICIDS).
        latent_dim : int
            Dimensionality of the latent (encoded) representation.
        intermediate_dim : int
            Dimensionality of intermediate layers.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.autoencoder = self._build_autoencoder()
    
    def _build_encoder(self):
        """Encoder: input_dim → intermediate → latent_dim"""
        inputs = layers.Input(shape=(self.input_dim,), name='encoder_input')
        x = layers.Dense(self.intermediate_dim, activation='relu', name='enc_dense1')(inputs)
        encoded = layers.Dense(self.latent_dim, activation='relu', name='enc_dense2')(x)
        return Model(inputs, encoded, name='encoder')
    
    def _build_decoder(self):
        """Decoder: latent_dim → intermediate → input_dim"""
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(self.intermediate_dim, activation='relu', name='dec_dense1')(latent_inputs)
        decoded = layers.Dense(self.input_dim, activation='relu', name='dec_dense2')(x)
        return Model(latent_inputs, decoded, name='decoder')
    
    def _build_autoencoder(self):
        """Full AE: input → encoder → decoder → reconstruction"""
        inputs = layers.Input(shape=(self.input_dim,), name='ae_input')
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        autoencoder = Model(inputs, decoded, name='autoencoder')
        autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
        return autoencoder
    
    def train(self, X, epochs=100, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the autoencoder to learn a denoised representation.
        
        Adds Gaussian noise to inputs for denoising capability.
        """
        # Add noise for denoising
        noise_factor = 0.1
        X_noisy = X + noise_factor * np.random.normal(size=X.shape)
        X_noisy = np.clip(X_noisy, 0, 1)  # keep in [0,1] range
        
        history = self.autoencoder.fit(
            X_noisy, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history
    
    def encode(self, X):
        """Encode data to latent representation."""
        return self.encoder.predict(X, verbose=0)
    
    def decode(self, Z):
        """Decode latent representation back to original space."""
        return self.decoder.predict(Z, verbose=0)
    
    def reconstruct(self, X):
        """Full reconstruction: encode then decode."""
        return self.autoencoder.predict(X, verbose=0)


# ============================================================================
# WGAN with Gradient Penalty (WGAN-GP)
# ============================================================================

class WGANGenerator(Model):
    """
    Generator network for WGAN.
    
    Architecture: noise_dim (32) → 64 → input_dim (24)
    All layers use ReLU activation.
    """
    
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu', name='gen_dense1')
        self.dense2 = layers.Dense(output_dim, activation='relu', name='gen_dense2')
    
    def call(self, z):
        x = self.dense1(z)
        return self.dense2(x)


class WGANCritic(Model):
    """
    Critic (Discriminator) network for WGAN.
    
    Architecture: input_dim (24) → 16 → 1 (linear output)
    Uses linear output for Wasserstein distance estimation.
    """
    
    def __init__(self, input_dim):
        super().__init__()
        self.dense1 = layers.Dense(16, activation='relu', name='crit_dense1')
        self.dense2 = layers.Dense(1, name='crit_dense2')  # Linear output
    
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)


# ============================================================================
# AE-WGAN COMBINED MODEL
# ============================================================================

class AEWGAN:
    """
    AE-WGAN: Denoising Autoencoder + Wasserstein GAN.
    
    The full pipeline:
        1. Train AE on real attack data → learn denoised representation
        2. Encode real attacks through AE encoder → get clean features  
        3. Train WGAN: Generator learns to produce realistic attacks,
           Critic learns to distinguish real (AE-encoded) from generated
        4. Generate synthetic attacks by sampling noise → Generator → output
    
    Parameters
    ----------
    input_dim : int
        Feature dimensionality (24 for NSL-KDD, 10 for CICIDS-2017).
    noise_dim : int
        Dimensionality of random noise input to Generator.
    latent_dim : int
        AE latent space dimensionality.
    n_critic : int
        Number of critic updates per generator update.
    gp_weight : float
        Gradient penalty coefficient (lambda).
    lr : float
        Learning rate for Adam optimizer.
    """
    
    def __init__(self, input_dim, noise_dim=32, latent_dim=16,
                 n_critic=5, gp_weight=10.0, lr=1e-4):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.lr = lr
        
        # Build components
        self.ae = DenoisingAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            intermediate_dim=max(latent_dim + 4, 20)
        )
        self.generator = WGANGenerator(noise_dim, input_dim)
        self.critic = WGANCritic(input_dim)
        
        # Optimizers
        self.gen_optimizer = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        
        # History tracking
        self.history = {
            'ae_loss': [],
            'critic_loss': [],
            'gen_loss': [],
            'wasserstein_distance': []
        }
    
    def gradient_penalty(self, real_data, fake_data, batch_size):
        """
        Compute gradient penalty for WGAN-GP.
        
        Enforces the Lipschitz constraint by penalizing the gradient
        norm of the critic's output w.r.t. interpolated samples.
        """
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated)
        
        gradients = tape.gradient(pred, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1) + 1e-8)
        penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
        
        return penalty
    
    @tf.function
    def train_critic_step(self, real_data, batch_size):
        """Single critic training step."""
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            
            real_output = self.critic(real_data, training=True)
            fake_output = self.critic(fake_data, training=True)
            
            # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
            # Critic minimizes: E[D(fake)] - E[D(real)] + gp
            critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
            gp = self.gradient_penalty(real_data, fake_data, batch_size)
            critic_loss += self.gp_weight * gp
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )
        
        wasserstein_dist = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        
        return critic_loss, wasserstein_dist
    
    @tf.function
    def train_generator_step(self, batch_size):
        """Single generator training step."""
        noise = tf.random.normal([batch_size, self.noise_dim])
        
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_output = self.critic(fake_data, training=True)
            
            # Generator minimizes: -E[D(G(z))]
            gen_loss = -tf.reduce_mean(fake_output)
        
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        return gen_loss
    
    def train(self, X_real, epochs=100, batch_size=32, ae_epochs=50, verbose=1):
        """
        Full AE-WGAN training pipeline.
        
        Step 1: Train Autoencoder on real data
        Step 2: Encode real data through trained AE
        Step 3: Train WGAN (Generator + Critic) adversarially
        
        Parameters
        ----------
        X_real : np.ndarray
            Real attack samples (N, input_dim).
        epochs : int
            Number of WGAN training epochs.
        batch_size : int
            Training batch size.
        ae_epochs : int
            Number of epochs for AE pre-training.
        verbose : int
            Verbosity level (0=silent, 1=progress, 2=detailed).
        
        Returns
        -------
        history : dict
            Training history with losses.
        """
        n_samples = X_real.shape[0]
        
        if n_samples < batch_size:
            batch_size = max(n_samples // 2, 4)
            if verbose:
                print(f"  [Warning] Small dataset ({n_samples} samples), "
                      f"reduced batch_size to {batch_size}")
        
        # ===== STEP 1: Train Autoencoder =====
        if verbose:
            print(f"\n  [AE] Training Autoencoder ({ae_epochs} epochs)...")
        
        ae_verbose = 0 if verbose < 2 else 1
        ae_history = self.ae.train(
            X_real, epochs=ae_epochs, batch_size=batch_size, verbose=ae_verbose
        )
        self.history['ae_loss'] = ae_history.history['loss']
        
        if verbose:
            final_ae_loss = ae_history.history['loss'][-1]
            print(f"  [AE] Final reconstruction loss: {final_ae_loss:.6f}")
        
        # ===== STEP 2: Encode real data =====
        X_encoded = self.ae.encode(X_real)
        X_decoded = self.ae.reconstruct(X_real)
        
        # Use reconstructed (denoised) data as "real" for WGAN training
        X_real_denoised = X_decoded.astype(np.float32)
        
        # ===== STEP 3: Train WGAN =====
        if verbose:
            print(f"\n  [WGAN] Training WGAN ({epochs} epochs, "
                  f"n_critic={self.n_critic})...")
        
        dataset = tf.data.Dataset.from_tensor_slices(X_real_denoised)
        dataset = dataset.shuffle(n_samples).batch(batch_size, drop_remainder=True)
        
        for epoch in range(epochs):
            epoch_critic_losses = []
            epoch_gen_losses = []
            epoch_w_dists = []
            
            for real_batch in dataset:
                current_batch_size = tf.shape(real_batch)[0]
                
                # Train critic n_critic times
                for _ in range(self.n_critic):
                    c_loss, w_dist = self.train_critic_step(
                        real_batch, current_batch_size
                    )
                    epoch_critic_losses.append(c_loss.numpy())
                    epoch_w_dists.append(w_dist.numpy())
                
                # Train generator once
                g_loss = self.train_generator_step(current_batch_size)
                epoch_gen_losses.append(g_loss.numpy())
            
            # Record history
            avg_c = np.mean(epoch_critic_losses)
            avg_g = np.mean(epoch_gen_losses)
            avg_w = np.mean(epoch_w_dists)
            self.history['critic_loss'].append(avg_c)
            self.history['gen_loss'].append(avg_g)
            self.history['wasserstein_distance'].append(avg_w)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"    Epoch {epoch+1}/{epochs} — "
                      f"C_loss: {avg_c:.4f}, G_loss: {avg_g:.4f}, "
                      f"W_dist: {avg_w:.4f}")
        
        if verbose:
            print(f"  [WGAN] Training complete.")
        
        return self.history
    
    def generate(self, n_samples):
        """
        Generate synthetic attack samples.
        
        Parameters
        ----------
        n_samples : int
            Number of synthetic samples to generate.
        
        Returns
        -------
        synthetic_data : np.ndarray
            Generated samples of shape (n_samples, input_dim).
        """
        noise = tf.random.normal([n_samples, self.noise_dim])
        synthetic = self.generator(noise, training=False).numpy()
        
        # Clip to [0, 1] to match min-max scaled range
        synthetic = np.clip(synthetic, 0, 1)
        
        return synthetic
    
    def plot_training_history(self, save_path=None):
        """Plot training loss curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # AE Loss
        axes[0].plot(self.history['ae_loss'], color='blue')
        axes[0].set_title('Autoencoder Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].grid(True, alpha=0.3)
        
        # WGAN Losses
        axes[1].plot(self.history['critic_loss'], label='Critic', color='red')
        axes[1].plot(self.history['gen_loss'], label='Generator', color='green')
        axes[1].set_title('WGAN Losses')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Wasserstein Distance
        axes[2].plot(self.history['wasserstein_distance'], color='purple')
        axes[2].set_title('Wasserstein Distance')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Distance')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [Plot] Saved training history to {save_path}")
        
        plt.close()
        return fig


# ============================================================================
# OVERSAMPLING ENGINE
# ============================================================================

def oversample_attacks(X_train, y_train_detail, input_dim,
                       target_increment=20000,
                       epochs=100, ae_epochs=50, batch_size=32,
                       min_samples=20, verbose=1):
    """
    Generate synthetic attack samples for each attack type using AE-WGAN.
    
    Paper reference: "The AE-WGAN architecture was trained to produce 
    adversarial attacks based on the features selected by ANOVA. Since 
    AE-WGAN is executed separately on each dataset, the adversarial 
    attacks produced were added to the original set."
    
    For each attack type with >= min_samples:
        - Train an AE-WGAN on that attack type's samples
        - Generate target_increment synthetic samples
        - Add them to the training set
    
    Parameters
    ----------
    X_train : np.ndarray
        Original training features.
    y_train_detail : pd.Series
        Detailed attack labels for each sample.
    input_dim : int
        Feature dimensionality.
    target_increment : int
        Number of synthetic samples to generate per attack type.
    epochs : int
        WGAN training epochs per attack type.
    ae_epochs : int
        AE pre-training epochs.
    batch_size : int
        Training batch size.
    min_samples : int
        Minimum samples required to train AE-WGAN on an attack type.
    verbose : int
        Verbosity level.
    
    Returns
    -------
    X_augmented : np.ndarray
        Augmented training features (original + synthetic).
    y_augmented : pd.Series
        Augmented labels.
    generation_report : dict
        Per-attack generation statistics.
    models : dict
        Trained AE-WGAN models keyed by attack type.
    """
    import pandas as pd
    
    attack_types = y_train_detail[y_train_detail != 'normal'].unique()
    
    if verbose:
        print("\n" + "=" * 60)
        print("AE-WGAN OVERSAMPLING")
        print("=" * 60)
        print(f"Attack types to process: {len(attack_types)}")
        print(f"Target synthetic samples per type: {target_increment}")
        print(f"Min samples required: {min_samples}")
    
    all_synthetic_X = []
    all_synthetic_y = []
    models = {}
    generation_report = {}
    
    for attack in sorted(attack_types):
        mask = y_train_detail == attack
        X_attack = X_train[mask]
        n_original = X_attack.shape[0]
        
        if n_original < min_samples:
            if verbose:
                print(f"\n  [{attack}] Skipped — only {n_original} samples "
                      f"(< {min_samples} minimum)")
            generation_report[attack] = {
                'original': n_original,
                'generated': 0,
                'status': 'skipped'
            }
            continue
        
        if verbose:
            print(f"\n  [{attack}] Training AE-WGAN on {n_original} samples...")
        
        # Build and train AE-WGAN for this attack type
        model = AEWGAN(
            input_dim=input_dim,
            noise_dim=32,
            latent_dim=min(16, input_dim - 2),
            n_critic=5,
            gp_weight=10.0,
            lr=1e-4
        )
        
        try:
            model.train(
                X_attack,
                epochs=epochs,
                batch_size=min(batch_size, n_original // 2),
                ae_epochs=ae_epochs,
                verbose=max(0, verbose - 1)
            )
            
            # Generate synthetic samples
            n_to_generate = target_increment
            synthetic = model.generate(n_to_generate)
            
            all_synthetic_X.append(synthetic)
            all_synthetic_y.extend([attack] * n_to_generate)
            models[attack] = model
            
            generation_report[attack] = {
                'original': n_original,
                'generated': n_to_generate,
                'status': 'success'
            }
            
            if verbose:
                print(f"  [{attack}] Generated {n_to_generate} synthetic samples.")
        
        except Exception as e:
            if verbose:
                print(f"  [{attack}] Failed: {str(e)}")
            generation_report[attack] = {
                'original': n_original,
                'generated': 0,
                'status': f'failed: {str(e)}'
            }
    
    # Combine original + synthetic
    if all_synthetic_X:
        X_synthetic = np.vstack(all_synthetic_X)
        y_synthetic = pd.Series(all_synthetic_y)
        
        X_augmented = np.vstack([X_train, X_synthetic])
        y_augmented = pd.concat([y_train_detail, y_synthetic], ignore_index=True)
    else:
        X_augmented = X_train
        y_augmented = y_train_detail
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"OVERSAMPLING COMPLETE")
        print(f"  Original samples:  {X_train.shape[0]}")
        print(f"  Synthetic samples: {X_augmented.shape[0] - X_train.shape[0]}")
        print(f"  Total samples:     {X_augmented.shape[0]}")
        print(f"{'=' * 60}")
    
    return X_augmented, y_augmented, generation_report, models


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("AE-WGAN module loaded successfully.")
    print("\nUsage:")
    print("  from AE_WGAN import AEWGAN, oversample_attacks")
    print()
    print("  # Single attack type:")
    print("  model = AEWGAN(input_dim=24)")
    print("  model.train(X_attack, epochs=100)")
    print("  synthetic = model.generate(n_samples=1000)")
    print()
    print("  # Full oversampling pipeline:")
    print("  X_aug, y_aug, report, models = oversample_attacks(")
    print("      X_train, y_train_detail, input_dim=24)")
