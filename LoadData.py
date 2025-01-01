import os
from keras.preprocessing import image_dataset_from_directory

class LoadData:
    def __init__(self, image_size, color_space, batch_size):
        self.image_size = image_size
        self.color_space = color_space
        self.batch_size = batch_size

    def load_data(self, dir, subset):
        # Determinar la ruta del conjunto de datos
        subset_dir = os.path.join(dir, subset)
        # Crear el dataset infiriendo automáticamente los nombres de las clases
        dataset = image_dataset_from_directory(
            directory=subset_dir,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            label_mode='categorical',
            color_mode=self.color_space
        )
        return dataset

    def load_train_data(self, dir):
        return self.load_data(dir, 'train')

    def load_test_data(self, dir):
        return self.load_data(dir, 'test')

    def load_validation_data(self, dir):
        return self.load_data(dir, 'val')
