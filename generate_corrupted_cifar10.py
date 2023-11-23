from PIL import Image
import os
import glob
from PIL import Image
from imagecorruptions import corrupt
import numpy as np
item_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for type_cor in ['brightness','contrast','defocus_blur','gaussian_noise']:
    for _class_ in item_list:
        path_orginal = './data/cifar10/test/' + _class_
        path = './data/cifar10_5_'+ type_cor +'/test/' + _class_
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print("The new directory is created!")
        image_names = glob.glob(path_orginal + '/*.png')
        for image_name in image_names:
            path_to_image = image_name
            print(path_to_image)
            image = Image.open(path_to_image)
            image = np.array(image)
            corrupted = corrupt(image, corruption_name=type_cor, severity=5)
            im = Image.fromarray(corrupted)
            im.save(path_to_image.replace('cifar10', 'cifar10_5_'+ type_cor))

