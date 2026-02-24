import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio

import numpy as np
from skimage.measure import shannon_entropy


def select_best_bands(data, num_bands=8):
    """
    自动选择信息熵最大的波段
    data 维度: (H, W, Bands)
    """
    entropies = []
    for i in range(data.shape[2]):
        entropies.append(shannon_entropy(data[:, :, i]))

    # 获取熵值最大的索引
    best_indices = np.argsort(entropies)[-num_bands:]
    return sorted(best_indices)


data_path = './HyMars_data/holden.mat'
data_dict = sio.loadmat(data_path)
print(f"key in {data_path}:", [k for k in data_dict.keys() if not k.startswith('_')])
# 排除掉以 '__' 开头的系统变量，取第一个有效变量
data_key = [k for k in data_dict.keys() if not k.startswith('_')][0]

raw_hsi = data_dict[data_key].astype(np.float32)
# 示例调用
selected_indices = select_best_bands(raw_hsi)
print(f"推荐全色锐化的波段索引: {selected_indices}")
