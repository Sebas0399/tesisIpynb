
import tensorflow as tf
import keras_cv
class ModelSelector:
    def __init__(self, image_size, color_space,model):
        self.image_size = image_size
        self.color_space = color_space
        self.model = model
    def load_resnet50(self):
        base_model = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_resnet50_pretained(self):
        base_model = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_resnet152(self):
        base_model=keras_cv.models.ResNetV2Backbone.from_preset("resnet152_v2",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_cspdarknet(self):
        if(self.image_size in [96, 192, 384, 768]):
            base_model=keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_m",input_shape=(self.image_size, self.image_size, 3))
        else:
            raise ValueError("Image size must be one of [96, 192, 384, 768]")
        return base_model
    def load_cspdarknet_pretained(self):
        if(self.image_size in [48, 96, 192, 384]):
            base_model=keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_tiny_imagenet",input_shape=(self.image_size, self.image_size, 3))
        else:
            raise ValueError("Image size must be one of [48, 96, 192, 384]")
        return base_model
    def load_densenet169(self):
        base_model=keras_cv.models.DenseNetBackbone.from_preset("densenet169",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_densenet169_pretrained(self):
        base_model=keras_cv.models.DenseNetBackbone.from_preset("densenet169_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_efficientnet(self):
        base_model=keras_cv.models.EfficientNetLiteBackbone.from_preset("efficientnetlite_b4",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_efficientnet_V1(self):
        base_model=keras_cv.models.EfficientNetV1Backbone.from_preset("efficientnetv1_b7",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_efficientnet_V2_pretained(self):
        base_model=keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_s_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_efficientnet_V2(self):
        base_model=keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b3",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_mobilenet_V3_pretrained(self):
        base_model=keras_cv.models.MobileNetV3Backbone.from_preset("mobilenet_v3_large_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_mobilenet_V3(self):
        base_model=keras_cv.models.MobileNetV3Backbone.from_preset("mobilenet_v3_large",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_yolo_v8(self):
        base_model=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_yolo_v8_pretained(self):
        base_model=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone_coco",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    def load_model(self):
        if(self.model=="resnet50"):
            return self.load_resnet50()
        elif(self.model=="resnet50_pretained"):
            return self.load_resnet50_pretained()
        elif(self.model=="resnet152"):
            return self.load_resnet152()
        elif(self.model=="cspdarknet"):
            return self.load_cspdarknet()
        elif(self.model=="cspdarknet_pretained"):
            return self.load_cspdarknet_pretained()
        elif(self.model=="densenet169"):
            return self.load_densenet169()
        elif(self.model=="densenet169_pretrained"):
            return self.load_densenet169_pretrained()
        elif(self.model=="efficientnet"):
            return self.load_efficientnet()
        elif(self.model=="efficientnet_V1"):
            return self.load_efficientnet_V1()
        elif(self.model=="efficientnet_V2_pretained"):
            return self.load_efficientnet_V2_pretained()
        elif(self.model=="efficientnet_V2"):
            return self.load_efficientnet_V2()
        elif(self.model=="mobilenet_V3_pretrained"):
            return self.load_mobilenet_V3_pretrained()
        elif(self.model=="mobilenet_V3"):
            return self.load_mobilenet_V3()
        elif(self.model=="yolo_v8"):
            return self.load_yolo_v8()
        elif(self.model=="yolo_v8_pretained"):
            return self.load_yolo_v8_pretained()
        else:
            return None