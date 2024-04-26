import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    # Correct the calculation of positional encoding
    positions = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, depth, 2) * -(np.log(10000.0) / depth))
    angle_rads = positions * div_term
    # Initialize the positional encoding with zeros and fill in with sine and cosine functions
    pos_encoding = np.zeros((length, depth))
    pos_encoding[:, 0::2] = np.sin(angle_rads)
    pos_encoding[:, 1::2] = np.cos(angle_rads)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(100, d_model)[tf.newaxis, ...]

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        x += pos_encoding
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def call(self, query, value, key, training, mask=None):
        attn_output = self.mha(query, value, key, attention_mask=mask)
        attn_output = self.dropout(attn_output, training=training)
        out = self.layernorm(query + attn_output)
        return out


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation=tf.keras.layers.LeakyReLU())
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training):
        output = self.dense1(x)
        output = self.dropout(output, training=training)
        output = self.dense2(output)
        output = self.dropout(output, training=training)
        return self.layernorm(x + output)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()
        self.mha = BaseAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)
        self.ffn = FeedForward(d_model, dff, rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, training, mask)
        ffn_output = self.ffn(attn_output, training)
        return ffn_output


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate):
        super(DecoderLayer, self).__init__()
        self.mha1 = BaseAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)  # Masked self-attention
        self.mha2 = BaseAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)  # Encoder-Decoder attention
        self.ffn = FeedForward(d_model, dff, rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1 = self.mha1(x, x, x, training, look_ahead_mask)
        # Encoder-Decoder attention
        attn2 = self.mha2(attn1, enc_output, enc_output, training, padding_mask)
        # Final feed forward network
        ffn_output = self.ffn(attn2, training)
        return ffn_output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(vocab_size, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 rate=0.1, softmax=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        self.softmax = softmax
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation="softmax")

    def call(self, inputs, training=True):
        inp, tar = inputs
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output = self.decoder(tar, enc_output, training, combined_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def generate_next_probs(self, inps):
        enc_padding_mask = self.create_padding_mask(inps)
        enc_output = self.encoder(inps, training=False, mask=enc_padding_mask)

        # Initialize the target sequences with the last token of each input sequence
        target_seqs = inps[:, -1:]

        dec_padding_mask = self.create_padding_mask(target_seqs)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(target_seqs)[1])
        combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)

        dec_output = self.decoder(target_seqs, enc_output, training=False, look_ahead_mask=combined_mask,
                                  padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)

        # Get the predicted probabilities for the next time step for each sequence
        next_probs = final_output[:, -1, :]

        return next_probs

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model.numpy(),  # convert d_model to its original type if necessary
            'warmup_steps': self.warmup_steps
        }
