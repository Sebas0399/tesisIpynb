import os
from sklearn.model_selection import train_test_split
import shutil
from configparser import ConfigParser
class SplitData:
    def __init__(self):
        configfile_name = "config.ini" 
        config= ConfigParser()
        config.read(configfile_name)
        # Set test al val size
        self.test_size =float( config.get("dataset","test_size"))
        self.val_size=float(config.get("dataset","val_size"))
        self.image_size = config.get("dataset","val_size")
        self.color_space = config.get("dataset","color_mode")
        self.dataset_dir=config.get("dataset","base_dir")
    def split(self):
        if not os.path.exists('Nuevo'):
            base_dir="Nuevo"
            os.makedirs(base_dir)
            os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'val'), exist_ok=True)
            os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

            # Listar clases (carpetas)
            classes = os.listdir(self.dataset_dir)

            for class_name in classes:
                class_dir = os.path.join(self.dataset_dir, class_name)
                images = os.listdir(class_dir)
                
                # Dividir en train y temp (que luego se dividirá en val y test)
                print(self.test_size*2)
                train_images, temp_images = train_test_split(images, test_size=self.test_size*2, random_state=42)
                val_images, test_images = train_test_split(temp_images, test_size=(self.val_size*2)+0.1, random_state=42)  # 20% para val y 20% para test
                
                # Crear carpetas de clase en train, val y test
                os.makedirs(os.path.join(base_dir, 'train', class_name), exist_ok=True)
                os.makedirs(os.path.join(base_dir, 'val', class_name), exist_ok=True)
                os.makedirs(os.path.join(base_dir, 'test', class_name), exist_ok=True)
                
                # Mover las imágenes a sus respectivas carpetas
                for img in train_images:
                    shutil.move(os.path.join(class_dir, img), os.path.join(base_dir, 'train', class_name, img))
                    
                for img in val_images:
                    shutil.move(os.path.join(class_dir, img), os.path.join(base_dir, 'val', class_name, img))
                    
                for img in test_images:
                    shutil.move(os.path.join(class_dir, img), os.path.join(base_dir, 'test', class_name, img))
        else:
            os.rmdir('Nuevo')
SplitData().split()