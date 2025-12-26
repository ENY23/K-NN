import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from image_knn import ImageKNN
import time

def test_image_knn():
    # 1. 加载数据
    print("正在加载数据...")
    digits = load_digits()
    X, y = digits.images, digits.target
    
    # 2. 划分训练集和测试集
    print("划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. 创建并训练分类器
    print("训练分类器...")
    classifier = ImageKNN(n_neighbors=3, image_size=(8, 8))  # MNIST数字图像是8x8的
    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 4. 预测
    print("进行预测...")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 5. 评估性能
    accuracy = accuracy_score(y_test, y_pred)
    
    # 6. 打印结果
    print("\n=== 性能评估 ===")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"预测时间: {prediction_time:.2f} 秒")
    print(f"准确率: {accuracy:.4f}")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=[f"数字{i}" for i in range(10)]))

    # 7. 可视化预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(X_test):
            ax.imshow(X_test[i], cmap='gray')
            ax.set_title(f'预测:{y_pred[i]}\n实际:{y_test[i]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    print("\n预测结果可视化已保存为 'prediction_examples.png'")

if __name__ == "__main__":
    test_image_knn()