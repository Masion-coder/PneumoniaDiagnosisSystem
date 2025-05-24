from matplotlib import pyplot as plt
import torch
import matplotlib.image as mpimg  # mpimg 用于读取图片
from modules import MyTransforms

def CLAHE(fname: str) -> None:
    img = mpimg.imread(fname)

    clahe = MyTransforms.CLAHE()

    plt.subplot(1, 2, 1).imshow(img, cmap='gray')

    plt.title('Original')

    plt.subplot(1, 2, 2).imshow(clahe.__call__(img), cmap='gray')

    plt.title('CLAHE')

    plt.show()

if __name__ == '__main__':
    CLAHE('./datasets/chest_xray/train/NORMAL/NORMAL2-IM-1422-0001.jpeg')