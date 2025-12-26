import time
import os
import numpy as np
from scipy import stats
import scipy.ndimage as ndimage
from typing import Tuple, List, Union

class ImageKNN:
    """基于K-近邻的图像分类器"""
    
    def __init__(self, n_neighbors: int = 3, image_size: Tuple[int, int] = (28, 28)):
        """
        初始化分类器
        
        参数:
            n_neighbors: int, 近邻数量
            image_size: Tuple[int, int], 输入图像的期望大小 (高度, 宽度)
        """
        self.n_neighbors = n_neighbors
        self.image_size = image_size
        self._X_train = None
        self._y_train = None
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：调整大小、归一化
        
        参数:
            image: np.ndarray, 输入图像
            
        返回:
            np.ndarray: 预处理后的图像特征向量
        """
        # 确保图像是2D的
        if image.ndim > 2:
            # 如果是彩色图像，转换为灰度
            image = np.mean(image, axis=2)
            
        # 调整图像大小
        image = ndimage.zoom(image, 
                           (self.image_size[0] / image.shape[0],
                            self.image_size[1] / image.shape[1]))
        
        # 归一化到 [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        # 展平为1D特征向量
        return image.flatten()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ImageKNN':
        """
        训练分类器
        
        参数:
            X_train: np.ndarray, 形状为 (n_samples, height, width) 的训练图像
            y_train: np.ndarray, 训练标签
            
        返回:
            self: ImageKNN
        """
        # 预处理所有训练图像
        self._X_train = np.array([self._preprocess_image(img) for img in X_train])
        self._y_train = np.array(y_train)
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测测试图像的类别
        
        参数:
            X_test: np.ndarray, 形状为 (n_samples, height, width) 的测试图像
            
        返回:
            np.ndarray: 预测的类别标签
        """
        if self._X_train is None or self._y_train is None:
            raise ValueError("请先调用fit方法训练分类器")
            
        # 预处理所有测试图像
        X_test_processed = np.array([self._preprocess_image(img) for img in X_test])
        
        # 计算欧氏距离
        predictions = []
        for test_img in X_test_processed:
            # 计算与所有训练样本的距离
            distances = np.linalg.norm(self._X_train - test_img, axis=1)
            
            # 获取最近的k个邻居的索引
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # 获取这些邻居的标签
            k_nearest_labels = self._y_train[k_nearest_indices]
            
            # 通过投票选择最终类别（取众数）
            predicted_label = stats.mode(k_nearest_labels, keepdims=False)[0]
            predictions.append(predicted_label)
            
        return np.array(predictions)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        预测类别概率
        
        参数:
            X_test: np.ndarray, 形状为 (n_samples, height, width) 的测试图像
            
        返回:
            np.ndarray: 每个类别的概率估计
        """
        if self._X_train is None or self._y_train is None:
            raise ValueError("请先调用fit方法训练分类器")
            
        X_test_processed = np.array([self._preprocess_image(img) for img in X_test])
        unique_classes = np.unique(self._y_train)
        probabilities = []
        
        for test_img in X_test_processed:
            distances = np.linalg.norm(self._X_train - test_img, axis=1)
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self._y_train[k_nearest_indices]
            
            # 计算每个类别的概率
            class_probs = []
            for class_label in unique_classes:
                prob = np.mean(k_nearest_labels == class_label)
                class_probs.append(prob)
            
            probabilities.append(class_probs)
            
        return np.array(probabilities)