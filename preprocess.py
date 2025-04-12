import cv2
import numpy as np
from numpy.typing import NDArray
# import torch
# import torchvision.transforms as transforms
# import kornia


class Preprocess:
    @staticmethod
    def center(image):
        """将眼底图像居中并填充为正方形"""
        # 1. 找边框
        # 转为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用固定二值法
        otsuThresh, binaryImage = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        # 创建一个椭圆核函数
        EllipseKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # 闭合
        binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, EllipseKernel)
        # 寻找最外层轮廓（RETR_EXTERNAL）
        # 使用简化坐标表示（CHAIN_APPROX_SIMPLE，只保留轮廓拐点坐标）
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return image
        # 假设最大轮廓为眼底
        maxContour = max(contours, key=cv2.contourArea)
        # 找中心
        M = cv2.moments(maxContour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # 获取最大轮廓的边界框
        x, y, w, h = cv2.boundingRect(maxContour)

        r = max(cx - x, x + w - cx, cy - y, y + h - cy)
        r = int(r * 1.05)

        imageH = image.shape[0]
        imageW = image.shape[1]

        topPad = max(abs(r - cy), 0)
        bottomPad = max(abs(cy + r - imageH), 0)
        leftPad = max(abs(r - cx), 0)
        rightPad = max(abs(cx + r - imageW), 0)

        # 2. 调整图片大小
        # Padding First
        centeredImage = cv2.copyMakeBorder(
            image,
            top=topPad,
            bottom=bottomPad,
            left=leftPad,
            right=rightPad,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        # Crop remaining
        cx = cx + leftPad
        cy = cy + topPad

        leftBound = cx - r
        rightBound = cx + r
        topBound = cy - r
        bottomBound = cy + r
        centeredImage = centeredImage[topBound:bottomBound, leftBound:rightBound]

        return centeredImage

    @staticmethod
    def reflective_pad(image):
        """反射式填充"""
        # 二值化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, binaryImage = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)

        a = image.shape[0]
        O = a // 2

        # 计算内径
        # 计算最近的背景(0)
        distTransform = cv2.distanceTransform(binaryImage, cv2.DIST_L2, maskSize=5)
        innerR = distTransform[O, O]
        if innerR == 0:
            return image

        yCoords, xCoords = np.indices((a, a))

        distSquared = (xCoords - O) ** 2 + (yCoords - O) ** 2

        paddedMask = (distSquared > innerR ** 2) & np.all(image == (0, 0, 0), axis=2)

        padded = image.copy()

        dy = yCoords[paddedMask] - O
        dx = xCoords[paddedMask] - O
        dist = np.sqrt(dx ** 2 + dy ** 2)
        k = (innerR - 2) / (dist + 1e-6)  # 减2像素缓冲，避免边界残留
        ReflectionY = (O + k * dy).astype(int)
        ReflectionX = (O + k * dx).astype(int)

        # 镜像点坐标（带双线性插值）
        mirroredY = np.clip(ReflectionY + ReflectionY - yCoords[paddedMask], 0, a - 2)
        mirroredX = np.clip(ReflectionX + ReflectionX - xCoords[paddedMask], 0, a - 2)

        # 双线性插值取色（避免锯齿）
        x1 = mirroredX.astype(int)
        y1 = mirroredY.astype(int)
        x2 = np.minimum(x1 + 1, a - 1)
        y2 = np.minimum(y1 + 1, a - 1)

        alpha = mirroredX - x1
        beta = mirroredY - y1
        # 扩展为三维
        alpha = alpha[:, np.newaxis]
        beta = beta[:, np.newaxis]

        padded[paddedMask] = (
                (1 - alpha) * (1 - beta) * image[y1, x1] +
                alpha * (1 - beta) * image[y1, x2] +
                (1 - alpha) * beta * image[y2, x1] +
                alpha * beta * image[y2, x2]
        ).astype(np.uint8)

        return padded

    @staticmethod
    def concat(image1, image2) -> NDArray[np.uint8]:
        image1 = cv2.resize(image1, (224, 224), interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, (224, 224), interpolation=cv2.INTER_CUBIC)
        horizontal_concat = np.concatenate((image1, image2), axis=1)

        return horizontal_concat

    @staticmethod
    def clahe(image: NDArray[np.uint8]):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe_ = cv2.createCLAHE(
            clipLimit=2.0,  # 最大对比度
            tileGridSize=(8, 8)  # 直方图的网格大小
        )
        lab[..., 0] = clahe_.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def random_rotate_flip(image: NDArray[np.uint8], seed:int = 0):
        if seed % 5 == 0:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif seed % 5 == 1:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif seed % 5 == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif seed % 5 == 3:
            image = cv2.flip(image, 0)
        else:
            image = cv2.flip(image, 1)
        return image

    @staticmethod
    def resize(image: NDArray[np.uint8]):
        return cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)


# class TorchPreprocessor:
#     def __init__(self, device='cuda'):
#         self.device = device
#
#         # 使用JIT编译加速核心操作
#         self.clahe_layer = torch.jit.script(kornia.enhance.CLAHE(
#             clip_limit=2.0,
#             grid_size=(8, 8)
#         )).to(device)
#
#         # 预定义常用变换
#         self.resize = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
#         self.augmentations = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(90)
#         ])
#
#     def center_crop(self, image_tensor):
#         """GPU加速的居中裁剪"""
#         # 创建二值掩码
#         gray = kornia.color.rgb_to_grayscale(image_tensor)
#         _, binary = torch.where(gray > 1e-6, 1, 0)
#
#         # 查找非零区域边界
#         non_zero = torch.nonzero(binary)
#         min_y, max_y = non_zero[:, 1].min(), non_zero[:, 1].max()
#         min_x, max_x = non_zero[:, 2].min(), non_zero[:, 2].max()
#
#         # 计算裁剪参数
#         center = torch.tensor([(min_x + max_x) // 2, (min_y + max_y) // 2], device=self.device)
#         size = torch.max(max_x - min_x, max_y - min_y) * 1.05
#
#         # 执行裁剪
#         return kornia.geometry.transform.crop_by_boxes(
#             image_tensor,
#             torch.tensor([[
#                 center[0] - size / 2,  # x_min
#                 center[1] - size / 2,  # y_min
#                 center[0] + size / 2,  # x_max
#                 center[1] + size / 2  # y_max
#             ]], device=self.device),
#             mode='bilinear'
#         )
#
#     def reflective_pad(self, img_tensor):
#         """GPU反射填充"""
#         return kornia.augmentation.PadTo((img_tensor.shape[-2] * 2, img_tensor.shape[-1] * 2),
#                                     pad_mode='reflect')(img_tensor)
#
#     def process(self, image_path):
#         # 读取图像并转为Tensor
#         cv_img = cv2.imread(image_path)
#         tensor_img = kornia.utils.image_to_tensor(cv_img, keepdim=False).float() / 255.
#         tensor_img = tensor_img.to(self.device)
#
#         # 执行处理流水线
#         tensor_img = self.center_crop(tensor_img)
#         tensor_img = self.reflective_pad(tensor_img)
#         tensor_img = self.clahe_layer(tensor_img)
#         tensor_img = self.augmentations(tensor_img)
#
#         return tensor_img

# if __name__ == '__main__':
#     import random
#     preprocess = Preprocess()
#     TP = TorchPreprocessor()
#     # image = cv2.imread(r"P:\Datasets\ysg\glaucoma\advanced_glaucoma\1.png")
#     # image = preprocess.center(image)
#     # image = preprocess.reflective_pad(image)
#     # image = preprocess.clahe(image)
#     # cv2.imshow('image', image)
#     t_image = TP.process(r"P:\Datasets\ysg\glaucoma\advanced_glaucoma\1.png")
#
#     # 转换步骤
#     numpy_img = t_image.detach().cpu()  # 1. 移回CPU并脱离计算图
#     numpy_img = numpy_img.permute(0, 2, 3, 1)  # 2. 调整维度顺序从NCHW->NHWC
#     numpy_img = numpy_img.squeeze()  # 3. 去除批量维度（假设batch_size=1）
#     numpy_img = (numpy_img * 255).numpy().astype(np.uint8)  # 4. 反归一化+类型转换
#
#     # 颜色空间转换（如果预处理包含RGB操作需要添加）
#     numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
#
#     # 显示图像
#     cv2.imshow('Processed Image', numpy_img)
#
#     cv2.waitKey(0)