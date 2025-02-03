
import tensorflow as tf
import keras_cv
class ModelSelector:
    def __init__(self, image_size, color_space,model):
        self.image_size = image_size
        self.color_space = color_space
        self.model = model
  
    def load_resnet50_pretained(self):
        base_model = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
  
    
    def load_efficientnet_V2_pretained(self):
        base_model=keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_s_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    
    def load_mobilenet_V3_pretrained(self):
        base_model=keras_cv.models.MobileNetV3Backbone.from_preset("mobilenet_v3_large_imagenet",input_shape=(self.image_size, self.image_size, 3))
        return base_model
    
    def load_model(self):
       
        if(self.model=="resnet50_pretained"):
            return self.load_resnet50_pretained()
        
        elif(self.model=="efficientnet_V2_pretained"):
            return self.load_efficientnet_V2_pretained()

        elif(self.model=="mobilenet_V3_pretrained"):
            return self.load_mobilenet_V3_pretrained()
        
        else:
            return None