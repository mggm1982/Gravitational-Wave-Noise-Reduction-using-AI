import tensorflow as tf
from tensorflow.keras import layers, Model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def residual_block(x, filters, kernel_size=7, dropout_rate=0.1):

    shortcut = layers.Conv1D(filters, 1, padding="same")(x)

    out = layers.Conv1D(filters, kernel_size, padding="same")(x)
    out = layers.LeakyReLU(alpha=0.1)(out)
    out = layers.BatchNormalization()(out)

    out = layers.Conv1D(filters, kernel_size, padding="same")(out)
    out = layers.LeakyReLU(alpha=0.1)(out)
    out = layers.BatchNormalization()(out)

    out = layers.Dropout(dropout_rate)(out)

    return layers.Add()([shortcut, out])

def build_denoiser(input_shape):

    inputs = layers.Input(shape=input_shape)

    e1 = residual_block(inputs, 16)
    p1 = layers.MaxPooling1D(2)(e1)

    e2 = residual_block(p1, 32)
    p2 = layers.MaxPooling1D(2)(e2)

    e3 = residual_block(p2, 64)
    p3 = layers.MaxPooling1D(2)(e3)

    b = residual_block(p3, 128)

    u3 = layers.UpSampling1D(2)(b)
    d3 = layers.concatenate([u3, e3])
    d3 = residual_block(d3, 64)

    u2 = layers.UpSampling1D(2)(d3)
    d2 = layers.concatenate([u2, e2])
    d2 = residual_block(d2, 32)

    u1 = layers.UpSampling1D(2)(d2)
    d1 = layers.concatenate([u1, e1])
    d1 = residual_block(d1, 16)

    clean_pred = layers.Conv1D(1, 7, padding='same', activation='tanh')(d1)

    model = Model(inputs, clean_pred)
    return model

if __name__ == "__main__":
    print("Building Residual U-Net Denoiser with Hybrid Loss...")
    model = build_denoiser((4096, 1))
    model.summary()
