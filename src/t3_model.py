import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    output = tf.matmul(attention_weights, v)
    
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, name='MultiHeadAttention', d_model=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=True)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=True)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class InputEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, name='InputEncodingLayer', d_model=512, **kwargs):
        super(InputEncodingLayer, self).__init__(trainable=True, name=name, **kwargs)
        self.d_model = d_model


    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.d_features = input_shape[-1]
        
        # d-dimensional projection
        # input_shape = (batch_size, features (m), sequence length (w))
        # U = XW + B : ∈ (w x d)
        #   W ∈ (m x d)
        #   X ∈ (w x m)
        #   B ∈ (w x d)
        self.wp = self.add_weight(name='wp', shape=(self.d_features, self.d_model), initializer='uniform', trainable=True)
        self.bp = self.add_weight(name='bb', shape=(self.seq_len, self.d_model), initializer='uniform', trainable=True)
        
        # positional encodings
        self.we = self.add_weight(name='we', shape=(self.seq_len, self.d_model), initializer='uniform', trainable=True)
        
        super(InputEncodingLayer, self).build(input_shape)


    def call(self, inputs):

        # Apply d-dimensional projection (U)
        U = tf.matmul(inputs, self.wp) + self.bp    # (batch_size, inp_seq_len, d_model)
        
        # Apply positional encodings (U')
        Up = U + self.we
        
        return Up


class FeedForwardNetworkLayer(tf.keras.layers.Layer):
    def __init__(self, name='FeedForwardNetworkLayer', d_model=512, dff=2048, **kwargs):
        super(FeedForwardNetworkLayer, self).__init__(name=name, **kwargs)

        self.dense1  = tf.keras.layers.Dense(dff, activation='relu') # (batch_size, seq_len, dff)
        self.out     = tf.keras.layers.Dense(d_model)                # (batch_size, seq_len, d_model)

    
    def call(self, input, training=False):
        x = self.dense1(input, training=training)
        return self.out(x, training=training)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, name='EncoderLayer', num_heads=8, head_size=128, dff=2048, dropout=0, d_model=512, **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        
        self.mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        
        self.ffn = FeedForwardNetworkLayer(d_model=d_model, dff=dff)

        self.layernorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.BatchNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)


    def call(self, x, training=False):
#         attn_output = self.mha([x, x], training=training)
        attn_output, _ = self.mha(x, x, x, None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class T3(tf.keras.Model):
    def __init__(self, name='T3', num_heads=8, head_size=128, dff=2048, d_model=512, num_layers=1, dropout=0, final_activation='linear', num_outputs=1, **kwargs):
        super(T3, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.input_encoding = InputEncodingLayer(d_model=d_model)
        self.dropout = dropout
        self.outlen = num_outputs
        self.attention_layers = [
            EncoderLayer(
                num_heads=num_heads, 
                head_size=head_size, 
                dff=dff, 
                d_model=d_model, 
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ]
        self.flatten = tf.keras.layers.Flatten()
        self.linear = tf.keras.layers.Dense(self.outlen, activation='linear', use_bias=True)

        
    def call(self, inputs, training=False):

        x = self.input_encoding(inputs)
        
        for i in range(self.num_layers):
            x = self.attention_layers[i](x, training=training)

        x = self.flatten(x)

        x = self.linear(x, training=training)

        return x


    def get_config(self):
        return super().get_config()