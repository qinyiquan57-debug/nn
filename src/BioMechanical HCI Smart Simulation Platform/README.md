简介
本项目基于深度学习框架（如 TensorFlow、PyTorch）实现图像风格迁移，利用预训练的卷积神经网络（如 VGG19）提取图像的内容特征和风格特征，通过优化目标函数使生成图像同时匹配内容特征和风格特征。
文件结构
style_transfer-exercise.py：包含图像风格迁移核心实现的代码文件
utils.py：辅助函数模块（图像加载、预处理、后处理等）
环境要求
Python 3.x
TensorFlow 2.x 或 PyTorch 1.8+
NumPy
OpenCV 或 PIL（图像处理）
Matplotlib（结果可视化）
安装依赖
可以使用以下命令安装所需的 Python 库：
bash
# 若使用TensorFlow
pip install tensorflow numpy opencv-python matplotlib

# 若使用PyTorch
pip install torch torchvision numpy opencv-python matplotlib
核心原理
特征提取：使用预训练的 VGG 网络，选择浅层网络提取内容特征（保留图像主体结构），选择多层网络提取风格特征（通过 Gram 矩阵捕获风格信息）。
损失函数：
内容损失：衡量生成图像与内容图像在内容特征上的差异
风格损失：衡量生成图像与风格图像在风格特征上的差异
总损失：内容损失与风格损失的加权和
优化过程：以生成图像为优化变量，通过梯度下降最小化总损失，迭代更新生成图像。
使用说明
1. 准备图像
将内容图像和风格图像放入images目录（需自行创建），支持常见格式（.jpg、.png 等）。
2. 配置参数
在代码中设置关键参数：
内容图像路径、风格图像路径
内容损失权重、风格损失权重
迭代次数、学习率等
3. 运行程序
python
运行
# 示例（TensorFlow版本）
if __name__ == "__main__":
    content_path = "images/content.jpg"
    style_path = "images/style.jpg"
    generate_image = style_transfer(content_path, style_path, iterations=1000)
    save_image(generate_image, "output/generated.jpg")
测试
代码中包含可视化模块，可实时显示迭代过程中生成图像的变化，最终输出生成图像与内容图像、风格图像的对比结果，验证风格迁移效果。
报告要求
提交时请一并提交代码和报告，报告至少包含：
所选内容图像和风格图像说明
模型结构（使用的预训练网络、特征层选择）
损失函数设计（权重设置及理由）
实验结果分析（不同参数对生成效果的影响）
生成图像与原图的对比展示