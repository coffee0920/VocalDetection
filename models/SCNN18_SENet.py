from tensorflow.keras import Model
from tensorflow.keras import layers as klayers
from models.Layers import STSA, USCLLayer

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    squeeze = klayers.GlobalAveragePooling2D()(input_x)

    excitation = klayers.Dense(out_dim // ratio, activation='relu', name=layer_name + '_fully_connected1')(squeeze)
    excitation = klayers.Dense(out_dim, activation='sigmoid', name=layer_name + '_fully_connected2')(excitation)
    excitation = klayers.Reshape((1, 1, out_dim))(excitation)

    scale = klayers.Multiply()([input_x, excitation])

    return scale

class SCNN18_SENet(Model):
    def __init__(self, input_shape, nb_classes, **kwargs):
        super(SCNN18_SENet, self).__init__(**kwargs)

        self.input_layer = klayers.Input(input_shape, name='input')

        self.STSA = STSA()

        self.uscl_conv1 = USCLLayer(64, (3, 2), (3, 2), False, 'valid', 1, name='uscl_conv1')
        self.uscl_conv2 = USCLLayer(64, (1, 2), (1, 1), False, 'same', 2, name='uscl_conv2')
        self.uscl_conv3 = USCLLayer(64, (1, 2), (1, 1), False, 'same', 3, name='uscl_conv3')
        self.uscl_conv4 = USCLLayer(64, (1, 2), (1, 1), True, 'same', 4, name='uscl_conv4')
        self.uscl_conv5 = USCLLayer(64, (1, 2), (1, 1), True, 'same', 5, name='uscl_conv5')
        self.uscl_conv6 = USCLLayer(64, (1, 2), (1, 1), False, 'same', 6, name='uscl_conv6')
        self.uscl_conv7 = USCLLayer(64, (1, 2), (1, 1), True, 'same', 7, name='uscl_conv7')
        self.uscl_conv8 = USCLLayer(128, (1, 2), (1, 1), False, 'same', 8, name='uscl_conv8')
        self.uscl_conv9 = USCLLayer(128, (1, 2), (1, 1), False, 'same', 9, name='uscl_conv9')
        self.uscl_conv10 = USCLLayer(128, (1, 2), (1, 1), True, 'same', 10, name='uscl_conv10')
        self.uscl_conv11 = USCLLayer(128, (1, 2), (1, 1), True, 'same', 11, name='uscl_conv11')
        self.uscl_conv12 = USCLLayer(128, (1, 2), (1, 1), False, 'same', 12, name='uscl_conv12')
        self.uscl_conv13 = USCLLayer(128, (1, 2), (1, 1), True, 'same', 13, name='uscl_conv13')
        self.uscl_conv14 = USCLLayer(256, (1, 2), (1, 1), True, 'same', 14, name='uscl_conv14')
        self.uscl_conv15 = USCLLayer(256, (1, 2), (1, 1), True, 'same', 15, name='uscl_conv15')
        self.uscl_conv16 = USCLLayer(256, (1, 2), (1, 1), False, 'same', 16, name='uscl_conv16')

        self.final_uscl = klayers.Conv2D(256, (3, 2), (1, 1), 'same', name='final_uscl')
        self.final_uscl_relu = klayers.Activation('relu')
        self.final_uscl_bn = klayers.BatchNormalization(name='final_uscl_bn')
        self.final_pool = klayers.MaxPool2D((3, 2))

        self.final_conv = klayers.Conv2D(256, (1, 1), padding='same', name='final_conv')
        self.final_relu = klayers.Activation('relu')
        self.final_bn = klayers.BatchNormalization(name='final_bn')

        self.dropout = klayers.Dropout(0.5, name='uscl_dropout')
        self.flatten = klayers.Flatten(name='flatten')
        self.out_dense = klayers.Dense(nb_classes, name="Dense_2nb", activation='softmax')

    def model(self):
        return Model(inputs=[self.input_layer], outputs=self.call(self.input_layer))

    def call(self, x, **kwargs):
        x = self.STSA(x)
        x = self.uscl_conv1(x)
        # x = Squeeze_excitation_layer(x, 64, 8, 'seblock1')
        x = self.uscl_conv2(x)
        # x = Squeeze_excitation_layer(x, 64, 2, 'seblock2')
        x = self.uscl_conv3(x)
        # x = Squeeze_excitation_layer(x, 64, 8, 'seblock3')
        x = self.uscl_conv4(x)
        # x = Squeeze_excitation_layer(x, 64, 2, 'seblock4')
        x = self.uscl_conv5(x)
        # x = Squeeze_excitation_layer(x, 64, 8, 'seblock5')
        x = self.uscl_conv6(x)
        # x = Squeeze_excitation_layer(x, 64, 8, 'seblock6')
        x = self.uscl_conv7(x)
        # x = Squeeze_excitation_layer(x, 64, 2, 'seblock7')
        x = self.uscl_conv8(x)
        # x = Squeeze_excitation_layer(x, 128, 8, 'seblock8')
        x = self.uscl_conv9(x)
        # x = Squeeze_excitation_layer(x, 128, 2, 'seblock9')
        x = self.uscl_conv10(x)
        # x = Squeeze_excitation_layer(x, 128, 8, 'seblock10')
        x = self.uscl_conv11(x)
        # x = Squeeze_excitation_layer(x, 128, 2, 'seblock11')
        x = self.uscl_conv12(x)
        # x = Squeeze_excitation_layer(x, 128, 8, 'seblock12')
        x = self.uscl_conv13(x)
        # x = Squeeze_excitation_layer(x, 128, 2, 'seblock13')
        x = self.uscl_conv14(x)
        # x = Squeeze_excitation_layer(x, 256, 8, 'seblock14')
        x = self.uscl_conv15(x)
        # x = Squeeze_excitation_layer(x, 256, 8, 'seblock15')
        x = self.uscl_conv16(x)
        # x = Squeeze_excitation_layer(x, 256, 2, 'seblock16')
        x = self.final_uscl(x)
        x = self.final_uscl_relu(x)
        x = self.final_uscl_bn(x)
        # x = Squeeze_excitation_layer(x, 256, 8, 'seblock17')
        # if add SE Net after Layer 17, remove maxpooling since SE Net hava already done it
        # x = self.final_pool(x)
        x = self.final_conv(x)
        x = self.final_relu(x)
        x = self.final_bn(x)
        x = Squeeze_excitation_layer(x, 256, 2, 'seblock18')
        x = self.dropout(x)
        x = self.flatten(x)

        return self.out_dense(x)