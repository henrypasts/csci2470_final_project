import tensorflow as tf
import numpy as np


class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, category_size, d_model=128, num_heads=5, dff=256, rate=0.3):
        super().__init__()
        self.category_size = category_size
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(input_dim=category_size, output_dim=d_model)
        self.pos_encoding = self.positional_encoding(1000, d_model)

        self.attention_layers = [tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model) for _ in range(3)]
        self.feed_forward_layers = [tf.keras.layers.Dense(dff, activation=tf.keras.layers.LeakyReLU()) for _ in range(3)]
        self.dropout_layers = [tf.keras.layers.Dropout(rate) for _ in range(3)]
        self.final_layer = tf.keras.layers.Dense(units=category_size, activation="linear")

    def call(self, x, training):
        attention_mask = self.create_look_ahead_mask(tf.shape(x)[1])
        x = self.embedding(x) + self.pos_encoding[:, :tf.shape(x)[1], :]

        for i in range(3):
            x = self.attention_layers[i](x, x, x, training=training, attention_mask=attention_mask)
            x = self.feed_forward_layers[i](x)
            x = self.dropout_layers[i](x, training=training)

        return self.final_layer(x)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        pos_encoding = np.concatenate([np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])], axis=-1)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
