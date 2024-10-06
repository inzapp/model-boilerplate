"""
Authors : inzapp

Github url : https://github.com/inzapp/model-boilerplate

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import tensorflow as tf


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.infos = [[16, 1], [32, 1], [64, 1], [128, 1], [256, 1], [512, 1]]

    def build(self, unet_depth=3, fcn=True, bn=False, activation='leaky'):
        if fcn:
            return self.build_fcn_model(unet_depth=unet_depth, bn=bn, activation=activation)
        else:
            return self.build_scaling_model(unet_depth=unet_depth, bn=bn, activation=activation)

    def build_fcn_model(self, unet_depth, bn, activation):
        input_layer = tf.keras.layers.Input(shape=(self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels))
        x = input_layer
        xs = []
        channels, n_convs = self.infos[0]
        for _ in range(n_convs):
            x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
        xs.append(x)
        for i in range(unet_depth):
            channels, n_convs = self.infos[i+1]
            x = self.conv2d(x, channels, 3, 2, bn=bn, activation=activation)
            for _ in range(n_convs):
                x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
            if i < unet_depth - 1:
                xs.append(x)
        for i in range(unet_depth, 0, -1):
            channels, n_convs = self.infos[i-1]
            x = self.conv2d(x, channels, 1, 1, bn=bn, activation=activation)
            x = self.conv2dtranspose(x, channels, 4, 2, bn=bn, activation=activation)
            x = self.add([x, xs.pop()])
            for _ in range(n_convs):
                x = self.conv2dtranspose(x, channels, 4, 1, bn=bn, activation=activation)
        output_layer = self.output_layer(x, input_layer, name='output')
        return tf.keras.models.Model(input_layer, output_layer)

    def build_scaling_model(self, unet_depth, bn, activation):
        input_layer = tf.keras.layers.Input(shape=(self.cfg.input_rows, self.cfg.input_cols, self.cfg.input_channels))
        x = input_layer
        xs = []
        channels, n_convs = self.infos[0]
        for _ in range(n_convs):
            x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
        for i in range(unet_depth):
            xs.append(x)
            x = self.maxpooling2d(x)
            channels, n_convs = self.infos[i+1]
            for _ in range(n_convs):
                x = self.conv2d(x, channels, 3, 1, bn=bn, activation=activation)
        for i in range(unet_depth, 0, -1):
            channels, n_convs = self.infos[i-1]
            x = self.conv2d(x, channels, 1, 1, bn=bn, activation=activation)
            x = self.upsampling2d(x)
            x = self.add([x, xs.pop()])
            for _ in range(n_convs):
                x = self.conv2dtranspose(x, channels, 4, 1, bn=bn, activation=activation)
        output_layer = self.output_layer(x, input_layer, name='output')
        return tf.keras.models.Model(input_layer, output_layer)

    def output_layer(self, x, input_layer, additive=True, name='output'):
        if additive:
            x = self.conv2d(x, self.cfg.input_channels, 1, 1, activation='linear')
            x = self.add([x, input_layer], name=name)
        else:
            x = self.conv2d(x, self.cfg.input_channels, 1, 1, activation='linear')
        return x

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2dtranspose(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_initializer=self.kernel_initializer(),
            kernel_regularizer=self.kernel_regularizer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def maxpooling2d(self, x):
        return tf.keras.layers.MaxPooling2D()(x)

    def upsampling2d(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, x, name=None):
        return tf.keras.layers.Add(name=name)(x)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def kernel_regularizer(self, l2=0.0005):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation, name=None):
        if activation == 'linear':
            return x
        elif activation == 'leaky':
            return tf.keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        else:
            return tf.keras.layers.Activation(activation=activation, name=name)(x) if activation != 'linear' else x

