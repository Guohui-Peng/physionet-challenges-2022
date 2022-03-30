#!/usr/bin/env python

# The model used by our team
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K


# Modified ResNet for challenge
class RESNET_C(keras.Model):

    def __init__(self, include_top=False, nb_classes=1, feature_maps = 64):
        super(RESNET_C, self).__init__()

        self.include_top = include_top

        self.conv_11 = keras.layers.Conv2D(filters=feature_maps, kernel_size=8, padding='same')
        self.BN_11 = keras.layers.BatchNormalization()
        self.relu_11 = keras.layers.Activation("relu")

        self.conv_12 = keras.layers.Conv2D(filters=feature_maps, kernel_size=5, padding='same')
        self.BN_12 = keras.layers.BatchNormalization()
        self.relu_12 = keras.layers.Activation("relu")

        self.conv_13 = keras.layers.Conv2D(filters=feature_maps, kernel_size=3, padding='same')
        self.BN_13 = keras.layers.BatchNormalization()
        self.relu_13 = keras.layers.Activation("relu")

        self.conv_1e = keras.layers.Conv2D(filters=feature_maps, kernel_size=1, padding='same')
        self.BN_1e = keras.layers.BatchNormalization()
        self.add_1 = keras.layers.add

        self.conv_21 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=8, padding='same')
        self.BN_21 = keras.layers.BatchNormalization()
        self.relu_21 = keras.layers.Activation("relu")

        self.conv_22 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=5, padding='same')
        self.BN_22 = keras.layers.BatchNormalization()
        self.relu_22 = keras.layers.Activation("relu")

        self.conv_23 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=3, padding='same')
        self.BN_23 = keras.layers.BatchNormalization()
        self.relu_23 = keras.layers.Activation("relu")

        self.conv_2e = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=1, padding='same')
        self.BN_2e = keras.layers.BatchNormalization()
        self.add_2 = keras.layers.add

        self.conv_31 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=8, padding='same')
        self.BN_31 = keras.layers.BatchNormalization()
        self.relu_31 = keras.layers.Activation("relu")

        self.conv_32 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=5, padding='same')
        self.BN_32 = keras.layers.BatchNormalization()
        self.relu_32 = keras.layers.Activation("relu")

        self.conv_33 = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=3, padding='same')
        self.BN_33 = keras.layers.BatchNormalization()
        self.relu_33 = keras.layers.Activation("relu")
        self.add_3 = keras.layers.add
 
        # self.conv_3e = keras.layers.Conv2D(filters=feature_maps * 2, kernel_size=1, padding='same')
        self.BN_3e = keras.layers.BatchNormalization()

        self.global_pooling = keras.layers.GlobalAveragePooling2D(name='global_pooling')

        self.classifier = keras.layers.Dense(nb_classes, activation='sigmoid', name='FC')


    def call(self, inputs):
        # Block 1        
        x11 = self.conv_11(inputs)
        x11 = self.BN_11(x11)
        x11 = self.relu_11(x11)

        x12 = self.conv_12(x11)
        x12 = self.BN_12(x12)
        x12 = self.relu_12(x12)

        x13 = self.conv_13(x12)
        x13 = self.BN_13(x13)

        e1 = self.conv_1e(inputs)
        e1 = self.BN_1e(e1)

        b1 = self.add_1([e1, x13])
        b1 = self.relu_13(b1)

        # Block 2
        x21 = self.conv_21(b1)
        x21 = self.BN_21(x21)
        x21 = self.relu_21(x21)

        x22 = self.conv_22(x21)
        x22 = self.BN_22(x22)
        x22 = self.relu_22(x22)

        x23 = self.conv_23(x22)
        x23 = self.BN_23(x23)

        e2 = self.conv_2e(b1)
        e2 = self.BN_2e(e2)

        b2 = self.add_2([e2, x23])
        b2 = self.relu_23(b2)

        # Block 3
        x31 = self.conv_31(b2)
        x31 = self.BN_31(x31)
        x31 = self.relu_31(x31)

        x32 = self.conv_32(x31)
        x32 = self.BN_32(x32)
        x32 = self.relu_32(x32)

        x33 = self.conv_33(x32)
        x33 = self.BN_33(x33)

        # e3 = self.conv_3e(b2)
        e3 = self.BN_3e(b2)

        b3 = self.add_3([e3, x33])
        b3 = self.relu_33(b3)

        out_layer = self.global_pooling(b3)
        if self.include_top == True:
            out_layer = self.classifier(out_layer)

        return out_layer


# ResNet with MLP
class RESNET_MLP(keras.Model):
    def __init__(self, nb_classes):
        super(RESNET_MLP, self).__init__()

        # Layers
        self.resnet = RESNET_C()

        self.dense_1 = keras.layers.Dense(54, activation='relu')
        self.dense_2 = keras.layers.Dense(4, activation='sigmoid')

        self.concatenate = keras.layers.concatenate

        self.classifier = keras.layers.Dense(nb_classes, activation='sigmoid', name='FC')

    def call(self, inputs):
        input_a, input_b = inputs

        resnet = self.resnet(input_a)

        mlp = self.dense_1(input_b)
        mlp = self.dense_2(mlp)

        combined = self.concatenate([resnet, mlp])
        return self.classifier(combined)


@tf.function
def F1_Score(y_true, y_pred):
    """
    Calculate F1 score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

