
import tensorflow as tf
from tensorflow.keras import layers, models
class ModelSelector:
    def __init__(self, image_size, color_space,model):
        self.image_size = image_size
        self.color_space = color_space
        self.model = model
    # Cargar MobileNetV2 con pesos preentrenados de ImageNet, sin la parte superior
    def load_mobilenetv2(self):
        base_model = tf.keras.applications.MobileNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')

        # Congelar las capas del modelo base para que no se entrenen
        base_model.trainable = False
        return base_model
    def load_vgg16(self):
        base_model = tf.keras.applications.VGG16(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')
    def load_vgg19(self):
        base_model = tf.keras.applications.VGG19(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')
        # Congelar las capas del modelo base para que no se entrenen
        base_model.trainable = False
        return base_model
    def load_resnet50(self):
        base_model = tf.keras.applications.ResNet50(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')

        # Congelar las capas del modelo base para que no se entrenen
        base_model.trainable = False
        return base_model
    def load_inceptionv3(self):
        base_model = tf.keras.applications.InceptionV3(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')
    def load_inception_resnet_v2(self):
            base_model = tf.keras.applications.InceptionResNetV2(input_shape=(self.image_size, self.image_size, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        # Congelar las capas del modelo base para que no se entrenen
            base_model.trainable = False
            return base_model
    def load_efficientnet(self):
        base_model = tf.keras.applications.EfficientNetV2B0(input_shape=(self.image_size, self.image_size, 3),
                                                include_top=False,
                                                weights='imagenet')

        # Congelar las capas del modelo base para que no se entrenen
        base_model.trainable = False
        return base_model
    def load_model(self):
        if self.model == 'mobilenetv2':
            return self.load_mobilenetv2()
        elif self.model == 'vgg16':
            return self.load_vgg16()
        elif self.model == 'resnet50':
            return self.load_resnet50()
        elif self.model == 'inceptionv3':
            return self.load_inceptionv3()
        elif self.model == 'efficientnet':
            return self.load_efficientnet()
        elif self.model == 'inception_resnet_v2':
            return self.load_inception_resnet_v2()
        elif self.model == 'vgg19':
            return self.load_vgg19()
        else:
            return None