import tensorflow as tf
from sentiment.transformer import SentimentTransformerBlock


class SentimentModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(SentimentModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.transformer_layer = SentimentTransformerBlock(d_model=512, num_heads=8, ff_dim=512, dropout=0.1)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.transformer_layer(x)
        x = self.output_layer(x)
        return x
    
model = SentimentModel(input_shape=(10,), num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()