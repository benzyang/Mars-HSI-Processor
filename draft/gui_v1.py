import sys
import os
import numpy as np
import scipy.io as sio
import cv2
import time
import matplotlib.pyplot as plt
import traceback
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QThread, Signal, QUrl, QEvent, QObject
from PySide6.QtGui import QImage, QPixmap, QDesktopServices, QPalette, QBrush
import tifffile as tiff


class MarsApp(QObject):
    def __init__(self):
        super().__init__()
        # 1. 加载 UI
        loader = QUiLoader()
        ui_file = QFile("mars.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"无法打开 UI 文件: {ui_file.errorString()}")
            return

        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            print("UI 加载失败")
            return

        self.set_window_background()

        # 2. 变量初始化
        self.data_path = None
        self.gt_path = None
        self.raw_data = None
        self.gt_data = None
        self.fused_path = None
        self.lowres_path = None
        self.fused_img = None
        self.classify_path = None
        self.gt_img_path = None
        self.ui.tabWidget.setCurrentIndex(0)

        # 3. 绑定信号与槽
        self.setup_connections()

        # 4. 显示窗口
        self.ui.show()
        self.log_info("火星高光谱处理软件已启动。")

    def setup_connections(self):
        self.ui.action_clear.triggered.connect(self.clear_all_contents)
        self.ui.btn_load_data.clicked.connect(self.load_raw_data)
        self.ui.btn_load_gt.clicked.connect(self.load_gt_data)
        self.ui.btn_run_fusion.clicked.connect(self.run_fusion)
        self.ui.btn_run_classify.clicked.connect(self.run_classify)

        self.ui.label_fused.installEventFilter(self)
        self.ui.label_lowres.installEventFilter(self)
        self.ui.label_class.installEventFilter(self)
        self.ui.label_gt.installEventFilter(self)
        self.ui.label_fused.setMouseTracking(True)
        self.ui.label_lowres.setMouseTracking(True)
        self.ui.label_class.setMouseTracking(True)
        self.ui.label_gt.setMouseTracking(True)

    def set_window_background(self):
        self.ui.setStyleSheet(
            """
            /* 1. 底层背景图 */
            QMainWindow {
                border-image: url(background_blurred1.jpg) 0 0 0 0 stretch stretch;
            }

            /* 2. 关键：通过 centralwidget 的背景色来“冲淡”背景图 */
            /* rgba 最后一位 180 表示不透明度（0-255）。数值越高，背景图越淡，界面越白 */
            #centralwidget {
                background-color: rgba(255, 255, 255, 70); 
            }

            /* 3. 让 sidebar 保持一定的区分度 */
            QGroupBox#sidebar {
                background-color: rgba(255, 255, 255, 100);
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 12px;
            }

            QTabWidget::pane {
                background-color: rgba(255, 255, 255, 80); /* 80 的透明度，很有现代感 */
                border-radius: 10px;
            }

            /* 2. 设置 Tab 页内部的 Widget 透明 */
            /* 注意：你在 Designer 里每个 Tab 下的那个直接子 Widget (如 tab_fusion, tab_classify) */
            QTabWidget QWidget {
                background: transparent;
            }

            /* 3. 设置页签栏（TabBar）透明 */
            QTabBar::tab {
                background: rgba(255, 255, 255, 100); /* 页签半透明 */
                border: 1px solid #ccc;
                padding: 5px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }

            /* 4. 设置选中的页签更亮一点，方便区分 */
            QTabBar::tab:selected {
                background: rgba(255, 255, 255, 200);
                border-bottom-color: none;
            }
        """
        )
        # bg_img = QImage("./background_blurred1.jpg")

        # # 调整图片大小以适应当前窗口（平滑缩放）
        # scaled_pixmap = bg_img.scaled(self.ui.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # # 使用 Palette 设置背景
        # palette = QPalette()
        # palette.setBrush(QPalette.Window, QBrush(scaled_pixmap))
        # self.ui.setPalette(palette)

        # # 必须设置此属性，背景图才会生效
        # self.ui.setAutoFillBackground(True)

    def clear_all_contents(self):
        self.ui.lineEdit_loaddata.clear()
        self.ui.lineEdit_loadgt.clear()
        self.ui.text_results.clear()

        self.ui.combo_fusion.setCurrentIndex(0)
        self.ui.combo_classify.setCurrentIndex(0)
        self.ui.checkBox_gridsearch.setChecked(False)

        self.ui.btn_run_fusion.setEnabled(True)
        self.ui.btn_run_classify.setEnabled(True)

        labels = [self.ui.label_ori, self.ui.label_fused, self.ui.label_lowres, self.ui.label_class, self.ui.label_gt]
        for lbl in labels:
            lbl.clear()

        self.data_path = None
        self.gt_path = None
        self.raw_data = None
        self.gt_data = None
        self.fused_path = None
        self.lowres_path = None
        self.fused_img = None
        self.classify_path = None
        self.gt_img_path = None
        self.ui.tabWidget.setCurrentIndex(0)

        self.log_info("已清空")

    def load_raw_data(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "选择数据", "", "Mat Files (*.mat)")
        if path:
            self.ui.lineEdit_loaddata.setText(path)
            self.data_path = path

    def load_gt_data(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "选择标签", "", "Mat Files (*.mat)")
        if path:
            self.ui.lineEdit_loadgt.setText(path)
            self.gt_path = path
            # data = sio.loadmat(path)
            # key = [k for k in data.keys() if not k.startswith('_')][0]
            # self.gt_data = data[key]
            # self.log_info(f"加载标签: {key}")
            # self.display_image(self.gt_data, self.ui.label_gt, "gt")

    def on_fused_label_clicked(self, event=None):
        if self.fused_path:
            self.open_img(self.fused_path)

    def on_lowres_label_clicked(self, event=None):
        if self.lowres_path:
            self.open_img(self.lowres_path)

    def on_classify_label_clicked(self, event=None):
        if self.classify_path:
            self.open_img(self.classify_path)

    def on_gt_label_clicked(self, event=None):
        if self.gt_img_path:
            self.open_img(self.gt_img_path)

    def eventFilter(self, watched, event):
        # 检查事件类型是否为鼠标按下
        if event.type() == QEvent.Type.MouseButtonPress:
            # 判断是哪一个 Label 被点击了
            if watched == self.ui.label_fused:
                self.on_fused_label_clicked()
                return True  # 表示事件已处理，不再向下传递
            elif watched == self.ui.label_lowres:
                self.on_lowres_label_clicked()
                return True
            elif watched == self.ui.label_class:
                self.on_classify_label_clicked()
                return True
            elif watched == self.ui.label_gt:
                self.on_gt_label_clicked()
                return True

        # 其他事件交给父类处理
        return super().eventFilter(watched, event)

    def open_img(self, img_path):
        if not img_path:
            return

        local_url = QUrl.fromLocalFile(os.path.abspath(img_path))
        if os.path.exists(img_path):
            QDesktopServices.openUrl(local_url)
        else:
            self.log_info(f"错误：文件不存在 {img_path}")

    def run_fusion(self):
        if not self.data_path:
            self.log_info("请先加载高光谱数据！")
            return
        self.ui.label_fused.clear()
        self.ui.label_lowres.clear()

        method = self.ui.combo_fusion.currentText()
        self.log_info(f"开始执行全色锐化融合: {method}...")

        # 禁用按钮
        self.ui.btn_run_fusion.setEnabled(False)

        # 创建并启动线程
        self.fusion_thread = FusionThread(self.data_path, method)

        # 连接信号
        self.fusion_thread.log_signal.connect(self.log_info)
        self.fusion_thread.raw_preview_signal.connect(self.display_raw_data)
        self.fusion_thread.finished_signal.connect(self.on_fusion_finished)
        self.fusion_thread.start()

    def on_fusion_finished(self, fused_img, view_img, low_view, img_names):
        self.display_image(view_img, self.ui.label_fused)
        self.display_image(low_view, self.ui.label_lowres)
        self.fused_img = fused_img
        self.fused_path = img_names[0]
        self.lowres_path = img_names[1]
        self.ui.label_fused.setCursor(Qt.PointingHandCursor)
        self.ui.label_lowres.setCursor(Qt.PointingHandCursor)

        self.ui.btn_run_fusion.setEnabled(True)
        self.ui.tabWidget.setCurrentIndex(1)
        self.log_info(f"全色锐化完成，结果保存至 {img_names[2]}")

    def run_classify(self):
        if self.gt_path is None:
            self.log_info("请先加载标签数据！")
            return
        if self.fused_img is None:
            if not self.data_path:
                self.log_info("请先加载高光谱数据或进行全色锐化！")
                return
            else:
                input_img = self.data_path
                self.ui.tabWidget.setCurrentIndex(0)
        else:
            input_img = self.fused_img
            self.ui.tabWidget.setCurrentIndex(1)

        self.ui.label_class.clear()
        self.ui.label_gt.clear()

        method = self.ui.combo_classify.currentText()
        gridsearch = self.ui.checkBox_gridsearch.isChecked()
        self.log_info(f"正在启动分类任务: {method}...")
        self.ui.btn_run_classify.setEnabled(False)
        self.classify_thread = ClassifyThread(input_img, self.gt_path, method, gridsearch)

        # 连接信号
        self.classify_thread.log_signal.connect(self.log_info)
        self.classify_thread.raw_preview_signal.connect(self.display_raw_data)
        self.classify_thread.gt_preview_signal.connect(
            lambda img, path: self.display_image(img, self.ui.label_gt, path)
        )
        self.classify_thread.finished_signal.connect(self.on_classify_finished)
        self.classify_thread.start()

    def on_classify_finished(self, class_img, save_path):
        self.display_image(class_img, self.ui.label_class, save_path)
        self.classify_path = save_path
        self.ui.btn_run_classify.setEnabled(True)
        self.ui.tabWidget.setCurrentIndex(2)

    def log_info(self, text):
        self.ui.text_results.append(f"[LOG] {text}")

    def display_image(self, img, label_widget, save_path=None):
        try:
            if len(img.shape) == 2:
                img = self.process_gt_to_rgb(img, save_path=save_path)
                self.ui.tabWidget.setCurrentIndex(2)
                if label_widget.objectName() == 'label_gt':
                    self.gt_img_path = save_path

            h, w, c = img.shape
            img_contiguous = np.ascontiguousarray(img)
            q_img = QImage(img_contiguous.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            if not pixmap.isNull():
                label_w = label_widget.width() if label_widget.width() > 0 else 400
                scaled_pixmap = pixmap.scaledToWidth(label_w, Qt.SmoothTransformation)
                label_widget.setPixmap(scaled_pixmap)
                label_widget.setCursor(Qt.PointingHandCursor)
        except Exception as e:
            self.log_info(f"图像显示失败: {e}")

    def display_raw_data(self, rgb):
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max - rgb_min > 0:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min) * 255
        rgb = rgb.astype(np.uint8)

        rgb_contiguous = rgb.copy()
        h, w, c = rgb.shape
        q_img = QImage(rgb_contiguous.data, w, h, c * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 允许 Label 缩放内容
        self.ui.label_ori.setScaledContents(True)
        self.ui.label_ori.setCursor(Qt.PointingHandCursor)
        fixed_width = self.ui.label_ori.width()
        if not pixmap.isNull():
            self.ui.label_ori.setPixmap(pixmap.scaledToWidth(fixed_width, Qt.SmoothTransformation))
        # self.ui.verticalLayout_loaddata.setSpacing(10)
        self.ui.tabWidget.setCurrentIndex(0)

    def process_gt_to_rgb(self, gt, cmap_name='nipy_spectral', save_path=None):
        norm_data = normalize(gt)
        cmap = plt.get_cmap(cmap_name)
        rgba_image = cmap(norm_data)

        # 转换为 8-bit RGB (H x W x 3)
        rgb_image = (rgba_image[:, :, :3] * 255).astype(np.uint8)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            self.log_info(f"GT已保存至: {save_path}")

        return rgb_image


class ClassifyThread(QThread):
    log_signal = Signal(str)
    raw_preview_signal = Signal(np.ndarray)
    gt_preview_signal = Signal(np.ndarray, str)
    finished_signal = Signal(np.ndarray, str)

    def __init__(self, fused_img, gt_path, method, gridsearch):
        super().__init__()
        self.gt_path = gt_path
        self.fused_img = fused_img
        self.method = method
        self.gridsearch = gridsearch

    def run(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score

            if isinstance(self.fused_img, str):
                from sklearn.decomposition import PCA

                data_mat = sio.loadmat(self.fused_img)
                data_key = [k for k in data_mat.keys() if not k.startswith('_')][0]
                data = data_mat[data_key].astype(np.float32)
                self.log_signal.emit(f"加载数据: {data_key}, 形状: {data.shape}")
                self.raw_preview_signal.emit(data[:, :, [10, 20, 30]])

                H, W, B = data.shape
                X = data.reshape(-1, B)
                X = normalize(X)

                # print("start PCA")
                t0 = time.time()
                pca = PCA(n_components=30)
                X = pca.fit_transform(X)
                self.log_signal.emit(f"PCA完成, 维度从{B}降至{X.shape[1]}")
                print(f"PCA completed, cost {time.time() - t0:.2f}s")
            else:
                H, W, B = self.fused_img.shape
                X = self.fused_img.reshape(-1, B)
                # X = normalize(X)

            gt_mat = sio.loadmat(self.gt_path)
            gt_key = [k for k in gt_mat.keys() if not k.startswith('_')][0]
            gt = gt_mat[gt_key]
            gt = gt[:H, :W]
            self.log_signal.emit(f"加载标签: {gt_key}, 形状: {gt.shape}")

            save_dir = './results/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gt_save_name = os.path.join(save_dir, f'{gt_key}.png')

            self.gt_preview_signal.emit(gt, gt_save_name)
            y = gt.flatten()

            # 过滤背景
            mask = y > 0
            X_train_full = X[mask]
            y_train_full = y[mask]

            X_train, X_test, y_train, y_test = train_test_split(
                X_train_full, y_train_full, test_size=0.9, random_state=42, stratify=y_train_full
            )

            clf = self.start_train(
                X_train, y_train, method=self.method, params=None, gridsearch=self.gridsearch, heatmap=False
            )

            # 在测试集上评估
            y_test_pred = clf.predict(X_test)
            self.log_signal.emit(f"分类准确率: {accuracy_score(y_test, y_test_pred):.4f}")
            print("classification report:")
            print(classification_report(y_test, y_test_pred))

            # 预测全图
            y_pred_all = clf.predict(X)
            class_map = y_pred_all.reshape(H, W)
            class_map[gt == 0] = 0  # 掩盖背景
            result_save_name = os.path.join(save_dir, f'{gt_key}_classification_result.png')
            self.finished_signal.emit(class_map, result_save_name)

        except Exception as e:
            self.log_signal.emit(f"分类过程中发生错误: {str(e)}")
            full_error = traceback.format_exc()
            print(full_error)

    def start_train(self, X_train, y_train, method, params=None, gridsearch=False, heatmap=False):
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        import pandas as pd
        import seaborn as sns

        if method.lower() == 'svm':
            if gridsearch:
                # 网格搜索
                print("start SVM GridSearch")
                t0 = time.time()

                # 定义测试范围
                param_grid = {
                    'C': [0.1, 1, 10, 50, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01, 1, 10, 100],
                }
                grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
                grid.fit(X_train, y_train)
                print(f"best params: {grid.best_params_}, cost{time.time() - t0:.2f}s")
                self.log_signal.emit("GridSearch 完成")

                # 画热力图
                if heatmap:
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
            from sklearn.ensemble import RandomForestClassifier

            if gridsearch:
                print("start RandomForest GridSearch")
                t0 = time.time()

                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 20, 50],
                    'min_samples_split': [2, 5],
                    'criterion': ['gini', 'entropy'],
                }
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
                grid.fit(X_train, y_train)
                print(f"best params: {grid.best_params_}, cost {time.time() - t0:.2f}s")
                self.log_signal.emit("GridSearch 完成")

                if heatmap:
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


class FusionThread(QThread):
    log_signal = Signal(str)
    raw_preview_signal = Signal(np.ndarray)
    finished_signal = Signal(np.ndarray, np.ndarray, np.ndarray, list)

    def __init__(self, data_path, method):
        super().__init__()
        self.data_path = data_path
        self.method = method

    def run(self):
        try:
            # load data
            data = sio.loadmat(self.data_path)
            data_key = [k for k in data.keys() if not k.startswith('_')][0]
            raw_hsi = data[data_key].astype(np.float32)
            self.log_signal.emit(f"加载数据: {data_key}, 形状: {raw_hsi.shape}")

            self.raw_preview_signal.emit(raw_hsi[:, :, [30, 20, 10]].copy())

            # process
            ratio = 4
            H, W, B = raw_hsi.shape
            new_H, new_W = (H // ratio) * ratio, (W // ratio) * ratio
            raw_hsi = raw_hsi[:new_H, :new_W, :]

            # 波段抽取
            pan_idx = 250
            ms_indices = [50, 100, 150, 200, 280, 310, 350, 400]
            pan_data = raw_hsi[:, :, pan_idx]
            ms_data_list = raw_hsi[:, :, ms_indices]

            h, w, _ = ms_data_list.shape
            ms_low = cv2.resize(ms_data_list, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
            input_pan = np.expand_dims(normalize(pan_data), -1)
            input_ms = normalize(ms_low)
            print(f"输入1 (PAN) 形状: {input_pan.shape}")
            print(f"输入2 (Low-res MS) 形状: {input_ms.shape}")

            self.log_signal.emit("正在执行核心算法...")
            if self.method.upper() == 'PCA':
                from methods.PCA import PCA

                fused_image = PCA(input_pan, input_ms)
            elif self.method.upper() == 'GSA':
                from methods.GSA import GSA

                fused_image = GSA(input_pan, input_ms)
            elif self.method.upper() == 'MTF_GLP_HPM':
                from methods.MTF_GLP_HPM import MTF_GLP_HPM

                fused_image = MTF_GLP_HPM(input_pan, input_ms)
            elif self.method.upper() == 'CNMF':
                from methods.CNMF import CNMF

                fused_image = CNMF(input_pan, input_ms)

            # 输出图像
            save_dir = './results/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_name = os.path.splitext(os.path.basename(self.data_path))[0]
            view_img_name = os.path.join(save_dir, f'{img_name}_{self.method}.jpg')
            low_view_name = os.path.join(save_dir, f'{img_name}_{self.method}_lowres.jpg')
            tif_name = os.path.join(save_dir, f'{img_name}_{self.method}_result.tif')
            view_img = (normalize(fused_image[:, :, [0, 1, 2]]) * 255).astype(np.uint8)
            low_view = (
                normalize(cv2.resize(input_ms[:, :, [0, 1, 2]], (w, h), interpolation=cv2.INTER_NEAREST)) * 255
            ).astype(np.uint8)

            # 任务完成，发送结果
            self.finished_signal.emit(fused_image, view_img, low_view, [view_img_name, low_view_name, tif_name])
            cv2.imwrite(view_img_name, view_img)
            cv2.imwrite(low_view_name, low_view)
            tiff.imwrite(tif_name, fused_image.astype('float32'), planarconfig='separate')

        except Exception as e:
            self.log_signal.emit(f"融合过程中发生错误: {str(e)}")
            full_error = traceback.format_exc()
            print(full_error)


def normalize(img):
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    denominator = img_max - img_min

    if denominator < 1e-10:
        return np.zeros_like(img)

    return (img - img_min) / denominator


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(current_dir, "py_pansharpening")
    if project_path not in sys.path:
        sys.path.append(project_path)

    app = QApplication(sys.argv)
    window_manager = MarsApp()
    sys.exit(app.exec())
