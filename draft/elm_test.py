import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import sys
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# loading data
data_path = './HyMars_data/holden.mat'
gt_path = './HyMars_data/holden_gt.mat'
data_mat = sio.loadmat(data_path)
gt_mat = sio.loadmat(gt_path)
print(f'loading {data_path}')

data_key = [k for k in data_mat.keys() if not k.startswith('_')][0]
gt_key = [k for k in gt_mat.keys() if not k.startswith('_')][0]
data = data_mat[data_key].astype(np.float32)
gt = gt_mat[gt_key]
print(f"shape of data: {data.shape}")
print(f"shape of gt: {gt.shape}")

H, W, B = data.shape
X = data.reshape(-1, B)
y = gt.flatten()

# 归一化
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)

# PCA
print("start PCA")
t0 = time.time()
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X)
print(f"PCA completed, dimension reduced from {B} to {X_pca.shape[1]}, cost {time.time() - t0:.2f}s")

# 过滤背景
mask = y > 0
X_train_full = X_pca[mask]
y_train_full = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X_train_full, y_train_full, test_size=0.9, random_state=42, stratify=y_train_full
)


'''gemini2'''
from scipy.linalg import pinv


class CustomELM:
    def __init__(self, n_hidden=512, alpha=1e-7, random_state=42):
        self.n_hidden = n_hidden
        self.alpha = alpha  # 正则化系数 (岭回归)
        self.random_state = random_state
        self.input_weights = None
        self.bias = None
        self.output_weights = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 1. 随机初始化输入层权重和偏置
        self.input_weights = np.random.normal(size=(n_features, self.n_hidden))
        self.bias = np.random.normal(size=(1, self.n_hidden))

        # 2. 计算隐藏层输出矩阵 H
        H = self._sigmoid(np.dot(X, self.input_weights) + self.bias)

        # 3. 对标签进行 One-Hot 编码
        y_oh = np.eye(n_classes)[y]

        # 4. 计算输出权重 Beta (使用岭回归公式: Beta = (H.T * H + alpha*I)^-1 * H.T * Y)
        # 这种方式比直接求伪逆更稳定，尤其是在高光谱这种特征相关的场景下
        I = np.eye(self.n_hidden)
        self.output_weights = np.dot(np.dot(np.linalg.inv(np.dot(H.T, H) + self.alpha * I), H.T), y_oh)

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.input_weights) + self.bias)
        out = np.dot(H, self.output_weights)
        return np.argmax(out, axis=1)


# 标签必须从 0 开始
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)

# 实例化并训练
# 建议尝试增加 n_hidden 至 1024 或更高，alpha 设小一点
elm = CustomELM(n_hidden=2000, alpha=1e-8, random_state=42)

print(f"Training Custom ELM with {elm.n_hidden} hidden nodes...")
t_start = time.time()
elm.fit(X_train, y_train_le)
print(f"Training cost: {time.time() - t_start:.4f}s")

# 预测与评估
y_pred = elm.predict(X_test)

# 转回原始标签进行报告
y_pred_orig = le.inverse_transform(y_pred)

print(f"Custom ELM Accuracy: {accuracy_score(y_test, y_pred_orig):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_orig))
sys.exit(0)
'''gemini'''


class TraditionalELM:
    def __init__(self, n_hidden=1000, alpha=0.1):
        self.n_hidden = n_hidden
        self.alpha = alpha  # 正则化参数，防止准确率崩塌
        self.lb = LabelBinarizer()

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        # 1. 强力归一化 (ELM 的命根子)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # 2. 标签转 One-hot
        T = self.lb.fit_transform(y)
        n_samples, n_features = X.shape

        # 3. 随机生成输入权重和偏置 (固定种子保证可重复)
        np.random.seed(42)
        self.W = np.random.normal(size=(n_features, self.n_hidden))
        self.b = np.random.normal(size=(1, self.n_hidden))

        # 4. 计算隐藏层输出 H
        H = self._sigmoid(np.dot(X, self.W) + self.b)

        # 5. 使用岭回归(Ridge)公式计算输出权重 Beta
        # 相比直接求逆，这样更稳定: Beta = (H'H + alpha*I)^-1 * H'T
        I = np.eye(self.n_hidden)
        self.beta = np.linalg.solve(H.T @ H + self.alpha * I, H.T @ T)
        print(f"原生 ELM 训练完成，节点数: {self.n_hidden}")

    def predict(self, X):
        X = self.scaler.transform(X)
        H = self._sigmoid(np.dot(X, self.W) + self.b)
        T_pred = np.dot(H, self.beta)
        return self.lb.inverse_transform(T_pred)


# --- 使用建议 ---
# 1. 强制将 n_hidden 设为 1000 以上
# 2. 保持 test_size=0.9 (0.1训练) 看看结果
elm = TraditionalELM(n_hidden=1500, alpha=1.0)
elm.fit(X_train, y_train)
y_pred = elm.predict(X_test)

print(f"原生 ELM 准确率: {accuracy_score(y_test, y_pred):.4f}")


'''pyoselm'''
# --- 修正 1: 标签预处理 ---
# 将标签映射为 0, 1, 2, 3, 4, 5 (OS-ELM 必须从 0 开始且连续)
# le = LabelEncoder()
# y_train_full_le = le.fit_transform(y_train_full)

# # 重新划分数据集
# X_train, X_test, y_train, y_test = train_test_split(
#     X_train_full, y_train_full_le, test_size=0.9, random_state=42, stratify=y_train_full_le
# )

# n_hidden = 2000
# print('n_hidden:', n_hidden, ', activation_func: sigmoid')
# model = OSELMClassifier(n_hidden=n_hidden, activation_func='sigmoid', random_state=42)

# # 我们取前 n_initial 个样本，由于之前做了 stratify，这里大概率包含了所有类
# n_initial = 2200  # 稍微加大一点，确保安全性 (需 > n_hidden)
# X_init = X_train[:n_initial]
# y_init = y_train[:n_initial]

# # 检查初始块是否涵盖了所有类别
# if len(np.unique(y_init)) < len(np.unique(y_train)):
#     print("警告：初始块类别不全！正在重新随机抽取...")
#     # 如果不全，可以考虑手动抽样或增加 n_initial

# # 训练
# model.fit(X_init, y_init)
# model.fit(X_train[n_initial:], y_train[n_initial:])

# # 预测
# y_pred = model.predict(X_test)

# # --- 修正 3: 评估时转回原始标签 (可选) ---
# y_pred_orig = le.inverse_transform(y_pred)
# y_test_orig = le.inverse_transform(y_test)

# print(f"修正后的准确率: {accuracy_score(y_test, y_pred):.4f}")

# print(f"Accuracy: {accuracy_score(y_test_orig, y_pred_orig):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test_orig, y_pred_orig))

# 预测全图（包含 mask=0 的背景）
# all_pred = model.predict(X_pca)
# # 变回 2D 形状
# prediction_map = all_pred.reshape(H, W)

# # 遮盖掉背景 (gt == 0 的部分)
# prediction_map[gt == 0] = 0

# # 绘图
# plt.figure(figsize=(8, 6))
# plt.imshow(prediction_map, cmap='jet')
# plt.title('OSELM Classification Result')
# plt.colorbar()
# plt.show()
