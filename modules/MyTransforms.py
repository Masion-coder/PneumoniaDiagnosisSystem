import cv2
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transforms

class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        # 创建 CLAHE 对象
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, img):
        # 将 PIL 图像转换为 NumPy 数组
        img_np = np.array(img)
        
        # 创建 CLAHE 对象（推迟到 __call__ 方法中创建）
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        
        # 应用 CLAHE
        img_clahe = clahe.apply(img_np)
        
        # 将 NumPy 数组转换回 PIL 图像
        img_clahe_pil = Image.fromarray(img_clahe)
        
        return img_clahe_pil
    
def TrainCompose() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(), # 将图片转换为单通道灰度图
            transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转图片
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转图片
            transforms.RandomRotation(degrees=(-20, 20)), # 随机对图片进行旋转
            CLAHE(), # 对图片进行clahe处理
            transforms.ToTensor(), # 将图片转换为Tensor，同时把图片的数据从[0, 255]转换到[0, 1]
            transforms.Normalize(mean=(0.5,), std=(0.5,)), # 归一化到[-1,1]，均值和标准差都是0.5(可调)
            transforms.Resize(size=(956, 956), antialias=True), # 将图片缩放
        ]
    )

def TestCompose() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(), # 将图片转换为单通道灰度图
            CLAHE(), # 对图片进行clahe处理
            transforms.ToTensor(), # 将图片转换为Tensor，同时把图片的数据从[0, 255]转换到[0, 1]
            transforms.Normalize(mean=(0.5,), std=(0.5,)), # 归一化到[-1,1]，均值和标准差都是0.5(可调)
            transforms.Resize(size=(956, 956), antialias=True), # 将图片缩放
        ]
    )