import numpy as np
import cv2
import scipy.io as sio
import os
from methods.PCA import PCA
from methods.SFIM import SFIM
from methods.GS import GS
from methods.MTF_GLP import MTF_GLP
from methods.MTF_GLP_HPM import MTF_GLP_HPM
from methods.GSA import GSA
from methods.CNMF import CNMF
from methods.GFPCA import GFPCA

# loading data
data_path = '../HyMars_data/holden.mat'
data_dict = sio.loadmat(data_path)
print(f"key in {data_path}:", [k for k in data_dict.keys() if not k.startswith('_')])
# 排除掉以 '__' 开头的系统变量，取第一个有效变量
data_key = [k for k in data_dict.keys() if not k.startswith('_')][0]

raw_hsi = data_dict[data_key].astype(np.float32)
print(f"成功加载数据，变量名: {data_key}, 形状: {raw_hsi.shape}")

# preprocess
# 确保 shape 是 ratio 的整数倍
ratio = 4
H, W, B = raw_hsi.shape
new_H = (H // ratio) * ratio
new_W = (W // ratio) * ratio
# crop
raw_hsi = raw_hsi[:new_H, :new_W, :]
print(f"shape after cropping: {raw_hsi.shape}")

# 波段抽取

pan_idx = 250
ms_indices = [50, 100, 150, 200, 280, 310, 350, 400]
pan_data = raw_hsi[:, :, pan_idx]
ms_data_list = raw_hsi[:, :, ms_indices]

# 生成输入2：低分辨率多光谱图
ratio = 4
h, w, c = ms_data_list.shape
# 使用高斯模糊后下采样
ms_low = cv2.resize(ms_data_list, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


input_pan = normalize(pan_data)
input_pan = np.expand_dims(input_pan, -1)  # 变为 (H, W, 1)
input_ms = normalize(ms_low)  # 变为 (H/4, W/4, 8)
print(f"输入1 (PAN) 形状: {input_pan.shape}")
print(f"输入2 (Low-res MS) 形状: {input_ms.shape}")

fused_image = CNMF(input_pan, input_ms)
print(f"融合后图像形状: {fused_image.shape}")

save_dir = './results_mars/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 选前三个融合波段合成 RGB 保存以便查看效果
view_img = fused_image[:, :, [0, 1, 2]]
view_img = (normalize(view_img) * 255).astype(np.uint8)
cv2.imwrite(os.path.join(save_dir, 'Mars_CNMF_Result.jpg'), view_img)
cv2.imwrite(os.path.join(save_dir, 'Mars_CNMF_Result.tiff'), fused_image)

# 同时保存一份低分辨率的对比图
low_view = input_ms[:, :, [0, 1, 2]]
# low_view = cv2.resize(input_ms[:, :, [0, 1, 2]], (w, h), interpolation=cv2.INTER_NEAREST)
low_view = (normalize(low_view) * 255).astype(np.uint8)
# cv2.imwrite(os.path.join(save_dir, 'Mars_LowRes_Contrast_SFIM.jpg'), low_view)
