import numpy as np 
import cv2


class ImageProcessor:
    """
    this class processes the input image to fit the model
    """
    def __init__(self):
        pass

    def process_image(self, img):
        img = cv2.resize(np.float32(img), (28, 28))
        img = cv2.blur(img, (2,2))
        _, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
        img = (img-127.5)/127.5
        img = img.reshape((1,1,28,28))
        return img