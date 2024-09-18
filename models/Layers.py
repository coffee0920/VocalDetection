from tensorflow.keras import layers as klayers
from utils.initial_generator import InitialGenerator
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

class STSA(klayers.Layer):

    def __init__(self, **kwargs):
        super(STSA, self).__init__(name='STSA', **kwargs)

        sin_amplitudes = InitialGenerator.initial_wave(cycle_nums=1024, duration=2048, sin_wave=True, exp=False, window='hann')
        sin_amplitudes = np.reshape(sin_amplitudes, [2048, 1, 1024])

        cos_amplitudes = InitialGenerator.initial_wave(cycle_nums=1024, duration=2048, sin_wave=False, exp=False, window='hann')
        cos_amplitudes = np.reshape(cos_amplitudes, [2048, 1, 1024])

        self.sin_conv = klayers.Conv1D(
            1024, 2048, strides=[512], padding='same', name='sin_conv1d', use_bias=False, trainable=False, weights=[sin_amplitudes]
        )
        self.sin_square = klayers.Lambda(K.square, name='sin_square')

        self.cos_conv = klayers.Conv1D(
            1024, 2048, strides=[512], padding='same', name='cos_conv1d', use_bias=False, trainable=False, weights=[cos_amplitudes]
        )
        self.cos_square = klayers.Lambda(K.square, name='cos_square')
        self.sqrt_lambda = klayers.Lambda(K.sqrt, name='sqrt')
        self.uscl_reshape = klayers.Reshape(target_shape=[63, 1024, 1], name='uscl_reshape')

    def call(self, x, **kwargs):
        x = klayers.add((self.sin_square(self.sin_conv(x)), self.cos_square(self.cos_conv(x))))
        x = self.sqrt_lambda(x)
        return self.uscl_reshape(x)

class USCLLayer(klayers.Layer):

    def __init__(self, filters, kernel_size, stride=1, pooling=False, padding='same', index=0, activation='relu', **kwargs):
        super(USCLLayer, self).__init__(**kwargs)
        self.pooling = pooling

        if self.pooling:
            self.pool = klayers.MaxPool2D(kernel_size)

        self.conv_1 = klayers.Conv2D(filters, kernel_size, strides=stride, name=f'uscl_{filters}_{index}_1', padding=padding)
        self.bn_1 = klayers.BatchNormalization(name=f'uscl_{filters}_{index}_1_bn')
        self.relu_1 = klayers.Activation(activation, name=f'uscl_{filters}_{index}_1_{activation}')

    def call(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        if self.pooling:
            x = self.pool(x)

        return x

class USCLLayer2(klayers.Layer):

    def __init__(self, filters, kernel_size, stride=1, pooling=False, padding='same', index=0, activation='relu', **kwargs):
        super(USCLLayer2, self).__init__(**kwargs)
        self.pooling = pooling

        if self.pooling:
            self.pool = klayers.MaxPool2D(kernel_size)

        # self.conv_1 = klayers.Conv2D(filters, kernel_size, strides=stride, name=f'uscl_{filters}_{index}_1', padding=padding)
        self.bn_1 = klayers.BatchNormalization(name=f'uscl_{filters}_{index}_1_bn')
        self.relu_1 = klayers.Activation(activation, name=f'uscl_{filters}_{index}_1_{activation}')

    def call(self, x):
        # x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        if self.pooling:
            x = self.pool(x)

        return x
    
class SelfAttentionCNN(tf.keras.Model):
    def __init__(self, batch_size, height, width, input_channels, num_heads, **kwargs):
        super(SelfAttentionCNN, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.num_heads = num_heads

        self.f = klayers.Conv2D(filters=self.num_heads, kernel_size=(1, 1), padding='valid', use_bias=False)
        self.g = klayers.Conv2D(filters=self.num_heads, kernel_size=(1, 1), padding='valid', use_bias=False)
        self.h = klayers.Conv2D(filters=self.num_heads, kernel_size=(1, 1), padding='valid', use_bias=False)

        # Define Softmax and Multiply layers
        self.softmax = klayers.Softmax(axis=-1)
        self.multiply = klayers.Multiply()

        # Define trainable weights Wv
        self.Wv_layers = [self.add_weight(name=f'Wv_Z_{i}', shape=(1,), initializer='ones', trainable=True) for i in range(self.num_heads)]

        # Define Gamma
        self.Gamma = K.variable(1.0)

    def call(self, x):
        # Calculate F, G, H
        F = self.f(x)
        G = self.g(x)
        H = self.h(x)

        # Split from each filter
        F = tf.split(F,num_or_size_splits=self.num_heads, axis=-1)
        G = tf.split(G,num_or_size_splits=self.num_heads, axis=-1)
        H = tf.split(H,num_or_size_splits=self.num_heads, axis=-1)

        # Calculate S
        S = []
        for i in range(self.num_heads):
            s = []
            for j in range(self.num_heads):
                product = tf.multiply(F[i], G[j]) # Perform element-wise multiplication
                sum_product = tf.reduce_sum(product)  # Calculate the sum of each element in the feature map
                s.append(sum_product)
                del product
                del sum_product
            S.append(s)
        
        S = tf.transpose(S)

        Beta = tf.map_fn(self.process_column, S, dtype=tf.float32)
        
        Z_i = []
        for i in range(self.num_heads):
            Z_temp = []
            for j in range(self.num_heads):
                temp = Beta[i][j]
                H_mul_beta = tf.multiply(temp, H[j])  # Z_i = Beta[i][j] * H[j]
                Z_temp.append(H_mul_beta)
            Z_i.append(tf.add_n(Z_temp))

        del Beta

        O = []
        for _ in range(self.input_channels):
            Wv_Z = []
            for i in range(self.num_heads):
                Wv_Zi = tf.multiply(self.Wv_layers[i],Z_i[i])  # Wv_ij * Z_i
                Wv_Z.append(Wv_Zi)
            O.append(tf.add_n(Wv_Z))    # O = sum(Wv_ij * Z_i)

        gamma_O = self.Gamma * O  
   
        reshape_O = tf.reshape(gamma_O,(-1,self.height,self.width,self.input_channels))
        Y = reshape_O + x     # Y = gamma * O + x

        return Y
    # use tf.map_fn instead of recurrent 
    def process_column(self, column):
        column_softmax = self.softmax(column)
        return column_softmax