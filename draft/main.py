import numpy as np
import matplotlib.pyplot as plt
import tifffile

# 1. 读取文件
file_path = './results/holden_PCA.tif'  # 请确保文件路径正确

try:
    # 使用 tifffile 读取
    # 注意：tifffile.imread 会直接将数据读取为 numpy 数组
    img = tifffile.imread(file_path)

    print("✅ 文件读取成功！")
    print("-" * 30)

    # 2. 展示基本信息
    print(f"📊 基本信息:")
    print(f"   形状 (Shape): {img.shape}")  # 应该是 (400, 400, 8) 或类似
    print(f"   数据类型 (Dtype): {img.dtype}")
    print(f"   总体最小值: {np.min(img):.4f}")
    print(f"   总体最大值: {np.max(img):.4f}")

    # 3. 处理并展示前3个波段
    # 检查维度：确保我们处理的是 (H, W, C) 格式
    if img.ndim == 3:
        if img.shape[2] >= 3:
            # 提取前3个波段
            # 注意：如果是浮点型数据 (float32)，imshow 通常期望范围是 [0, 1]
            # 如果是整型数据 (uint8/uint16)，imshow 期望 [0, 255] 或 [0, 65535]
            rgb = img[:, :, :3]

            # 数据归一化 (如果数据范围过大，强制压缩到 0-1 供显示)
            # 这一步是为了防止 float 数据过大导致 imshow 显示全白
            def normalize(array):
                array_min = array.min()
                array_max = array.max()
                return (array - array_min) / (array_max - array_min + 1e-8)  # 加极小值防止除以0

            # 对每个波段分别归一化 (或者也可以对整个 rgb 一起归一化)
            rgb_normalized = np.zeros_like(rgb, dtype=np.float32)
            for i in range(3):
                rgb_normalized[:, :, i] = normalize(rgb[:, :, i])

            # 4. 绘图显示
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_normalized)
            plt.title("前 3 个波段合成图 (RGB)")
            plt.axis('off')  # 关闭坐标轴
            plt.show()

        else:
            print("❌ 图像波段数少于3个，无法合成RGB图像。")
    else:
        print("❌ 图像不是多波段格式，无法提取前3个波段。")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 '{file_path}'。请检查路径是否正确。")
except Exception as e:
    print(f"❌ 读取文件时发生错误: {e}")
