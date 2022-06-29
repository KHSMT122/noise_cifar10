import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class CustomImageDataGenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """    
    def random_transform(self, src, seed):
        np.random.seed(seed)
        random = np.random.randint(0, 2)
        if random == 0:
            img = self.addGaussianNoise(src, seed=seed)
        if random == 1:
            img = self.addSaltPepperNoise(src, seed=seed)
        return img
    """

    # gaussianblurと何が違う？(つまりいらなくね？)
    def addGaussianNoise(self, src, sigma):
        row, col, ch = src.shape
        mean = 0
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = src + gauss
        return noisy

    def addSaltPepperNoise(self, src, amount):
        row, col, ch = src.shape
        s_vs_p = 0.5
        sp_img = src.copy()

        # salt
        num_salt = np.ceil(amount * src.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in src.shape]
        sp_img[coords[:-1]] = (1, 1, 1)

        # pepper
        num_pepper = np.ceil(amount * src.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in src.shape]
        sp_img[coords[:-1]] = (0, 0, 0)

        return sp_img
