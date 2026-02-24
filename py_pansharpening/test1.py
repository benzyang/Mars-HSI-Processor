import numpy as np
import cv2
import scipy.io as sio
import os

# from methods.PCA import PCA


def norm(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

if __name__ == '__main__':
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
    ms_data_list = [raw_hsi[:, :, i] for i in ms_indices]

    # visualize
    save_dir = './mars_bands_vis/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(os.path.join(save_dir, f'PAN_Band_{pan_idx}.jpg'), norm(pan_data))

    for i, idx in enumerate(ms_indices):
        band_img = norm(ms_data_list[i])
        cv2.imwrite(os.path.join(save_dir, f'MS_Band_{i+1}_idx{idx}.jpg'), band_img)

    print(f"已成功提取并保存波段。")
    print(f"PAN 基准波段: {pan_idx}")
    print(f"MS 组成波段索引: {ms_indices}")
