import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import sys

from build_model import build_denoiser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print("Step 7: Training the Residual U-Net Denoiser...")

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true)
    y = y_pred - tf.reduce_mean(y_pred)
    r_num = tf.reduce_sum(x * y)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(x)) * tf.reduce_sum(tf.square(y)))
    r = r_num / (r_den + 1e-12)
    return 1 - r

def hybrid_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    corr = correlation_coefficient_loss(y_true, y_pred)
    return mae + 0.5 * corr

try:
    print("Loading preprocessed data...")
    X_train = np.load('../../data/processed/X_train.npy')
    y_train = np.load('../../data/processed/y_train.npy')
    X_val = np.load('../../data/processed/X_val.npy')
    y_val = np.load('../../data/processed/y_val.npy')

    INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])
    print(f"Data loaded successfully. Input shape: {INPUT_SHAPE}")

    print("Building and compiling model...")
    model = build_denoiser(INPUT_SHAPE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=hybrid_loss,
        metrics=['mae']
    )
    print("Model compiled successfully.")

    EPOCHS = 100
    BATCH_SIZE = 32

    callbacks = [
        ModelCheckpoint('../../data/models/best_denoiser.keras', save_best_only=True, monitor='val_loss', mode='min'),
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]

    print(f"Starting training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    print("Training complete!")

    MODEL_SAVE_PATH = '../../data/models/denoiser_model.keras'
    print(f"Saving trained model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Residual U-Net Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Hybrid)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../../results/training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Training loss plot saved to '../../results/training_loss.png'")

    print("\n🎉 Step 7 Complete — Model trained successfully with hybrid loss.")

except FileNotFoundError:
    print("Error: Missing preprocessed data files (e.g., X_train.npy). Please run Step 5 first.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error during training: {e}")
    sys.exit(1)
