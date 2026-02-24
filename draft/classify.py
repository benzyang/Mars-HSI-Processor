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


def start_train(method='svm', params=None, gridsearch=False, heatmap=False):
    if method.lower() == 'svm':
        if gridsearch:
            # 网格搜索
            print("start SVM GridSearch")
            t0 = time.time()
            from sklearn.model_selection import GridSearchCV

            # 定义测试范围
            param_grid = {
                'C': [0.1, 1, 10, 50, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01, 1, 10, 100],
            }
            grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            print(f"best params: {grid.best_params_}, cost{time.time() - t0:.2f}s")

            if heatmap:
                # 画热力图
                import pandas as pd
                import seaborn as sns

                # 将搜索结果转为表格
                results = pd.DataFrame(grid.cv_results_)
                # 提取参数和得分
                viz_data = results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
                sns.heatmap(viz_data, annot=True, cmap='YlGnBu', fmt='.3g')
                plt.title('Grid Search Accuracy Heatmap')
                plt.show()

            return grid
        else:
            if params is None:
                params = {}
            C = params.get('C', 100)
            gamma = params.get('gamma', 'scale')

            # 训练 SVM 模型
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            print(f"Running SVM with C={C}, gamma={gamma}...")
            clf.fit(X_train, y_train)
            return clf
    elif method.lower() == 'randomforest' or method.lower() == 'rf':
        if gridsearch:
            print("start RandomForest GridSearch")
            t0 = time.time()
            from sklearn.model_selection import GridSearchCV

            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 20, 50],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy'],
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
            grid.fit(X_train, y_train)
            print(f"best params: {grid.best_params_}, cost {time.time() - t0:.2f}s")

            if heatmap:
                import pandas as pd
                import seaborn as sns

                results_df = pd.DataFrame(grid.cv_results_)
                viz_df = results_df[['param_n_estimators', 'param_max_depth', 'mean_test_score']]
                viz_df['param_max_depth'] = viz_df['param_max_depth'].fillna('None')

                pivot_table = viz_df.pivot_table(
                    index='param_max_depth', columns='param_n_estimators', values='mean_test_score'
                )

                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".4f")
                plt.title('Random Forest Grid Search: Accuracy Heatmap')
                plt.xlabel('n_estimators (Number of Trees)')
                plt.ylabel('max_depth (Tree Depth)')
                plt.show()

                best_model = grid.best_estimator_
                importances = best_model.feature_importances_
                indices = np.arange(len(importances))
                plt.bar(indices, importances)
                plt.title("Feature (Band) Importance")
                plt.xlabel("PCA Components")
                plt.ylabel("Importance Score")
                plt.show()

            return grid
        else:
            if params is None:
                params = {}
            n_estimators = params.get('n_estimators', 500)
            max_depth = params.get('max_depth', None)
            min_samples_split = params.get('min_samples_split', 2)
            criterion = params.get('criterion', 'entropy')

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
            )
            print(
                f"Running RandomForest with n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}..."
            )
            clf.fit(X_train, y_train)
            return clf


clf = start_train(method='rf')

# 在测试集上评估
y_test_pred = clf.predict(X_test)
print(f"SVM test set accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\nclassification report:")
print(classification_report(y_test, y_test_pred))

# 预测全图
y_pred_all = clf.predict(X_pca)
class_map = y_pred_all.reshape(H, W)
class_map[gt == 0] = 0  # 掩盖背景

plt.figure(figsize=(6, 6))
plt.imshow(class_map, cmap='nipy_spectral')
plt.title("SVM Classification Map")
plt.axis('off')
plt.show()


# 7. 结果展示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Ground Truth")
plt.imshow(gt, cmap='nipy_spectral')
plt.subplot(1, 2, 2)
plt.title("Classification Result (RF + PCA)")
plt.imshow(class_map, cmap='nipy_spectral')
# plt.colorbar()
plt.show()
