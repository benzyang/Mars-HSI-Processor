import sys
import os
import time
import traceback
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff

import matplotlib

matplotlib.use('Agg')  # 设置后端为非交互模式，专门用于后台绘图保存

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QThread, Signal, QUrl, QEvent, QObject, QCoreApplication
from PySide6.QtGui import QImage, QPixmap, QDesktopServices, QCursor, QGuiApplication


def normalize(img):
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    denominator = img_max - img_min
    if denominator < 1e-10:
        return np.zeros_like(img)
    return (img - img_min) / denominator


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MarsApp(QObject):
    def __init__(self):
        super().__init__()

        # 1. 动态加载 UI
        loader = QUiLoader()
        ui_file = QFile("./mars.ui")
        if not ui_file.open(QFile.ReadOnly):
            print(f"无法打开 UI 文件: {ui_file.errorString()}")
            sys.exit(-1)

        self.ui = loader.load(ui_file)
        ui_file.close()

        if not self.ui:
            print("UI 加载失败")
            sys.exit(-1)

        # 2. 核心设置
        self.set_window_style()
        self.init_variables()
        self.setup_connections()

        # 3. 拦截窗口关闭事件（防止线程残留）
        self.ui.closeEvent = self.handle_close_event

        self.ui.show()
        self.log_info("火星高光谱处理软件已启动 (V1.0 Optimized)")

    def init_variables(self):
        self.data_path = None
        self.gt_path = None
        self.fused_img = None  # 存储内存中的融合数据
        self.paths = {'fused': None, 'lowres': None, 'classify': None, 'gt': None, 'heatmap': None}

        self.fusion_thread = None
        self.classify_thread = None
        self.ui.checkBox_gridsearch.setChecked(False)
        self.ui.tabWidget.setCurrentIndex(0)

    def set_window_style(self):
        self.ui.setStyleSheet(
            """
            QMainWindow {
                border-image: url(background_blurred.jpg) 0 0 0 0 stretch stretch;
            }
            #centralwidget {
                background-color: rgba(255, 255, 255, 60); 
            }
            
            QGroupBox {
                background-color: rgba(255, 255, 255, 70);
                border: 1px solid #ccc;
                border-radius: 8px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 2px;
                color: #333;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 4px;
                
            }
            QPushButton:hover {
                background-color: #0063b1;
            }
            QPushButton:pressed {
                background-color: #004e8c;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: rgba(255, 255, 255, 70);
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 150);
                border: 1px solid #ccc;
                padding: 4px 10px;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: rgba(255, 255, 255, 210);
                font-weight: bold;
            }
        """
        )

    def setup_connections(self):
        # 按钮动作
        self.ui.action_clear.triggered.connect(self.clear_all_contents)
        self.ui.btn_load_data.clicked.connect(self.load_raw_data)
        self.ui.btn_load_gt.clicked.connect(self.load_gt_data)
        self.ui.btn_run_fusion.clicked.connect(self.run_fusion)
        self.ui.btn_run_classify.clicked.connect(self.run_classify)

        # 事件过滤器 (用于点击图片)
        labels = [self.ui.label_fused, self.ui.label_lowres, self.ui.label_class, self.ui.label_gt]
        for lbl in labels:
            lbl.installEventFilter(self)
            lbl.setMouseTracking(True)  # 允许鼠标追踪

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if watched == self.ui.label_fused:
                self.open_img(self.paths['fused'])
            elif watched == self.ui.label_lowres:
                self.open_img(self.paths['lowres'])
            elif watched == self.ui.label_class:
                self.open_img(self.paths['classify'])
            elif watched == self.ui.label_gt:
                self.open_img(self.paths['gt'])
            return True
        return super().eventFilter(watched, event)

    def handle_close_event(self, event):
        """窗口关闭时的清理工作"""
        if self.fusion_thread and self.fusion_thread.isRunning():
            self.fusion_thread.terminate()
            self.fusion_thread.wait()
        if self.classify_thread and self.classify_thread.isRunning():
            self.classify_thread.terminate()
            self.classify_thread.wait()
        event.accept()

    def clear_all_contents(self):
        self.ui.lineEdit_loaddata.clear()
        self.ui.lineEdit_loadgt.clear()
        self.ui.text_results.clear()

        # 停止正在运行的线程
        if self.fusion_thread and self.fusion_thread.isRunning():
            self.fusion_thread.terminate()
        if self.classify_thread and self.classify_thread.isRunning():
            self.classify_thread.terminate()

        self.ui.btn_run_fusion.setEnabled(True)
        self.ui.btn_run_classify.setEnabled(True)
        QApplication.restoreOverrideCursor()  # 确保光标恢复

        labels = [self.ui.label_ori, self.ui.label_fused, self.ui.label_lowres, self.ui.label_class, self.ui.label_gt]
        for lbl in labels:
            lbl.clear()

        self.init_variables()
        self.log_info("系统状态已重置")

    def load_raw_data(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "选择高光谱数据", "", "Mat Files (*.mat)")
        if path:
            self.ui.lineEdit_loaddata.setText(path)
            self.data_path = path
            self.log_info(f"已选择数据: {os.path.basename(path)}")

    def load_gt_data(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "选择标签数据", "", "Mat Files (*.mat)")
        if path:
            self.ui.lineEdit_loadgt.setText(path)
            self.gt_path = path
            self.log_info(f"已选择标签: {os.path.basename(path)}")

    def run_fusion(self):
        if not self.data_path:
            QMessageBox.warning(self.ui, "提示", "请先加载高光谱数据 (.mat)！")
            return

        self.ui.label_fused.clear()
        self.ui.label_lowres.clear()
        method = self.ui.combo_fusion.currentText()

        self.log_info(f"=== 启动全色锐化: {method} ===")
        self.set_busy(True)

        self.fusion_thread = FusionThread(self.data_path, method)
        self.fusion_thread.log_signal.connect(self.log_info)
        self.fusion_thread.raw_preview_signal.connect(self.display_raw_data)
        self.fusion_thread.finished_signal.connect(self.on_fusion_finished)
        self.fusion_thread.start()

    def on_fusion_finished(self, fused_data, view_path, lowres_path, tif_path):
        self.set_busy(False)
        self.fused_img = fused_data
        self.paths['fused'] = view_path
        self.paths['lowres'] = lowres_path

        self.display_image_from_file(view_path, self.ui.label_fused)
        self.display_image_from_file(lowres_path, self.ui.label_lowres)

        self.ui.tabWidget.setCurrentIndex(1)
        self.log_info(f"✅ 融合完成，结果已保存至: {os.path.dirname(view_path)}")

    def run_classify(self):
        if not self.gt_path:
            QMessageBox.warning(self.ui, "提示", "请先加载标签数据 (GT)！")
            return

        # 判断输入源
        if self.fused_img is not None:
            input_source = self.fused_img  # 优先使用融合数据
            self.log_info("使用当前融合结果进行分类")
        elif self.data_path:
            input_source = self.data_path  # 否则使用原始文件路径
            self.log_info("使用原始 .mat 文件进行分类")
        else:
            QMessageBox.warning(self.ui, "提示", "没有可用的输入数据！")
            return

        self.ui.label_class.clear()
        self.ui.label_gt.clear()

        method = self.ui.combo_classify.currentText()
        gridsearch = self.ui.checkBox_gridsearch.isChecked()

        self.log_info(f"=== 启动分类任务: {method} (自动优化: {gridsearch}) ===")
        self.set_busy(True)

        self.classify_thread = ClassifyThread(input_source, self.gt_path, method, gridsearch)
        self.classify_thread.log_signal.connect(self.log_info)
        self.classify_thread.raw_preview_signal.connect(self.display_raw_data)
        self.classify_thread.gt_preview_signal.connect(self.update_gt_preview)
        self.classify_thread.finished_signal.connect(self.on_classify_finished)
        self.classify_thread.heatmap_signal.connect(self.on_heatmap_generated)
        self.classify_thread.start()

    def on_classify_finished(self, class_map, save_path):
        self.set_busy(False)
        self.paths['classify'] = save_path

        self.display_image(class_map, self.ui.label_class, save_path, is_gt=True)
        self.ui.tabWidget.setCurrentIndex(2)
        self.log_info("✅ 分类任务完成")

    def on_heatmap_generated(self, heatmap_path):
        self.paths['heatmap'] = heatmap_path
        self.log_info(f"参数优化热力图已生成: {heatmap_path}")
        # self.open_img(heatmap_path)

    def update_gt_preview(self, gt_data, save_path):
        self.paths['gt'] = save_path
        self.display_image(gt_data, self.ui.label_gt, save_path, is_gt=True)

    # --- 辅助 UI 函数 ---

    def set_busy(self, is_busy):
        """设置界面忙碌状态"""
        if is_busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.ui.btn_run_fusion.setEnabled(False)
            self.ui.btn_run_classify.setEnabled(False)
        else:
            QApplication.restoreOverrideCursor()
            self.ui.btn_run_fusion.setEnabled(True)
            self.ui.btn_run_classify.setEnabled(True)

    def log_info(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.ui.text_results.append(f"<span style='color:#555'>[{timestamp}]</span> {text}")
        # 自动滚动到底部
        self.ui.text_results.verticalScrollBar().setValue(self.ui.text_results.verticalScrollBar().maximum())

    def open_img(self, img_path):
        if not img_path or not os.path.exists(img_path):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(img_path)))

    def display_raw_data(self, rgb_data):
        rgb_norm = normalize(rgb_data)
        rgb_8bit = (rgb_norm * 255).astype(np.uint8)
        self.display_image_array(rgb_8bit, self.ui.label_ori)
        self.ui.tabWidget.setCurrentIndex(0)

    def display_image(self, img, label_widget, save_path=None, is_gt=False):
        if is_gt:
            # 彩色映射
            img_norm = normalize(img)
            cmap = plt.get_cmap('nipy_spectral')
            rgba_img = cmap(img_norm)
            rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

            if save_path:
                ensure_dir(os.path.dirname(save_path))
                cv2.imwrite(save_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

            self.display_image_array(rgb_img, label_widget)
        else:
            self.display_image_array(img, label_widget)

    def display_image_from_file(self, path, label_widget):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.update_label_pixmap(label_widget, pixmap)

    def display_image_array(self, img_array, label_widget):
        """将 numpy 数组转为 QPixmap 显示"""
        h, w, c = img_array.shape
        # 必须确保内存连续，否则 QImage 显示会扭曲
        img_contiguous = np.ascontiguousarray(img_array)
        q_img = QImage(img_contiguous.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.update_label_pixmap(label_widget, pixmap)

    def update_label_pixmap(self, label, pixmap):
        if pixmap.isNull():
            return

        target_width = label.width() if label.width() > 0 else 400
        original_ratio = pixmap.height() / pixmap.width()
        target_height = int(target_width * original_ratio)
        scaled = pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # label.setScaledContents(False)
        label.setPixmap(scaled)
        label.setCursor(Qt.PointingHandCursor)


class ClassifyThread(QThread):
    log_signal = Signal(str)
    raw_preview_signal = Signal(np.ndarray)
    gt_preview_signal = Signal(np.ndarray, str)
    finished_signal = Signal(np.ndarray, str)
    heatmap_signal = Signal(str)

    def __init__(self, input_source, gt_path, method, gridsearch):
        super().__init__()
        self.input_source = input_source
        self.gt_path = gt_path
        self.method = method
        self.gridsearch = gridsearch
        self.save_dir = './results/'
        ensure_dir(self.save_dir)

    def run(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, accuracy_score

            # 1. 准备数据 X
            if isinstance(self.input_source, str):
                # 如果是路径，需要读取 + PCA
                from sklearn.decomposition import PCA

                self.log_signal.emit(f"加载原始数据: {os.path.basename(self.input_source)}")
                data_mat = sio.loadmat(self.input_source)
                key = [k for k in data_mat.keys() if not k.startswith('_')][0]
                data = data_mat[key].astype(np.float32)

                # 发送预览
                if data.shape[2] >= 30:
                    self.raw_preview_signal.emit(data[:, :, [25, 35, 55]])

                H, W, B = data.shape
                X = data.reshape(-1, B)
                X = normalize(X)

                t0 = time.time()
                pca = PCA(n_components=30)
                X = pca.fit_transform(X)
                self.log_signal.emit(f"PCA 降维完成 ({time.time() - t0:.2f}s): {B} -> 30")
            else:
                # 已经是内存中的 numpy 数组 (通常是融合后的)
                data = self.input_source
                H, W, B = data.shape
                X = data.reshape(-1, B)

            # 2. 准备标签 Y
            gt_mat = sio.loadmat(self.gt_path)
            gt_key = [k for k in gt_mat.keys() if not k.startswith('_')][0]
            gt = gt_mat[gt_key][:H, :W]  # 确保尺寸匹配

            # 保存并预览 GT
            gt_save_path = os.path.join(self.save_dir, f'{gt_key}_gt.png')
            self.gt_preview_signal.emit(gt, gt_save_path)

            y = gt.flatten()
            mask = y > 0
            X_train = X[mask]
            y_train = y[mask]

            # 3. 划分数据集
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train, y_train, test_size=0.9, random_state=42, stratify=y_train
            )

            # 4. 训练
            clf = self.train_model(X_tr, y_tr)

            # 5. 评估
            y_pred_te = clf.predict(X_te)
            acc = accuracy_score(y_te, y_pred_te)
            self.log_signal.emit(f"测试集准确率: {acc:.4f}")
            print(classification_report(y_te, y_pred_te))

            # 6. 全图预测
            self.log_signal.emit("正在生成全图分类结果...")
            y_all = clf.predict(X)
            class_map = y_all.reshape(H, W)
            class_map[gt == 0] = 0

            result_name = os.path.join(self.save_dir, f'Classification_{self.method}.png')
            self.finished_signal.emit(class_map, result_name)

        except Exception as e:
            self.log_signal.emit(f"❌ 错误: {str(e)}")
            print(traceback.format_exc())

    def train_model(self, X, y):
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        import pandas as pd
        import seaborn as sns

        model = None
        params = {}

        if self.method.lower() == 'svm':
            if self.gridsearch:
                self.log_signal.emit("正在进行 SVM 网格搜索...")
                param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 1, 0.1, 0.01]}
                grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
                grid.fit(X, y)
                self.log_signal.emit(f"最佳参数: {grid.best_params_}")
                self.generate_heatmap(grid, "SVM_Heatmap.png")
                return grid
            else:
                self.log_signal.emit("使用默认参数训练 SVM...")
                model = SVC(kernel='rbf', C=100, gamma='scale')
        elif 'random' in self.method.lower():
            if self.gridsearch:
                self.log_signal.emit("正在进行 RF 网格搜索...")
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [None, 20, 50],
                    # 'min_samples_split': [2, 5],
                    # 'criterion': ['gini', 'entropy'],
                }
                grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
                grid.fit(X, y)
                self.log_signal.emit(f"最佳参数: {grid.best_params_}")
                self.generate_heatmap(grid, "RF_Heatmap.png")
                return grid
            else:
                self.log_signal.emit("使用默认参数训练 Random Forest...")
                model = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1, random_state=42)

        model.fit(X, y)
        return model

    def generate_heatmap(self, grid_result, filename):
        try:
            import pandas as pd
            import seaborn as sns

            results = pd.DataFrame(grid_result.cv_results_)
            # 只取前两个参数画图
            params = [k for k in grid_result.param_grid.keys()]
            if len(params) >= 2:
                p1, p2 = params[0], params[1]
                # 这里的 values 需要对应 param_grid 的 key
                viz_data = results.pivot_table(index=f'param_{p1}', columns=f'param_{p2}', values='mean_test_score')

                plt.figure(figsize=(8, 6))
                sns.heatmap(viz_data, annot=True, cmap='viridis', fmt='.3f')
                plt.title('Grid Search Accuracy')
                save_path = os.path.join(self.save_dir, filename)
                plt.savefig(save_path)
                plt.close()

                self.heatmap_signal.emit(save_path)
        except Exception as e:
            print(f"绘图失败: {e}")


class FusionThread(QThread):
    log_signal = Signal(str)
    raw_preview_signal = Signal(np.ndarray)
    finished_signal = Signal(np.ndarray, str, str, str)  # data, view_path, low_path, tif_path

    def __init__(self, data_path, method):
        super().__init__()
        self.data_path = data_path
        self.method = method
        self.save_dir = './results/'
        ensure_dir(self.save_dir)

    def run(self):
        try:
            data = sio.loadmat(self.data_path)
            key = [k for k in data.keys() if not k.startswith('_')][0]
            raw_hsi = data[key].astype(np.float32)

            if raw_hsi.shape[2] > 30:
                self.raw_preview_signal.emit(raw_hsi[:, :, [25, 35, 55]].copy())

            # 模拟全色锐化输入
            ratio = 4
            H, W, B = raw_hsi.shape
            # 裁剪边缘以适应比例
            H_crop, W_crop = (H // ratio) * ratio, (W // ratio) * ratio
            raw_hsi = raw_hsi[:H_crop, :W_crop, :]

            # 提取波段
            pan = raw_hsi[:, :, 45]  # 200
            ms_indices = np.linspace(50, B - 50, 8, dtype=int)
            ms = raw_hsi[:, :, ms_indices]

            # 制造低分辨率 MS
            h_lr, w_lr = H_crop // ratio, W_crop // ratio
            ms_lr = cv2.resize(ms, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)

            # 归一化输入
            input_pan = np.expand_dims(normalize(pan), -1)
            input_ms = normalize(ms_lr)

            self.log_signal.emit(f"正在执行算法: {self.method}...")

            # 调用算法
            fused_image = None
            try:
                if self.method == 'PCA':
                    from methods.PCA import PCA

                    fused_image = PCA(input_pan, input_ms)
                elif self.method == 'GSA':
                    from methods.GSA import GSA

                    fused_image = GSA(input_pan, input_ms)
                elif self.method.upper() == 'MTF_GLP_HPM':
                    from methods.MTF_GLP_HPM import MTF_GLP_HPM

                    fused_image = MTF_GLP_HPM(input_pan, input_ms)
                elif self.method.upper() == 'CNMF':
                    from methods.CNMF import CNMF

                    fused_image = CNMF(input_pan, input_ms)
                else:
                    # 默认回退到 PCA 防止空指针
                    from methods.PCA import PCA

                    fused_image = PCA(input_pan, input_ms)
            except ImportError as ie:
                self.log_signal.emit(f"⚠️ 找不到算法模块: {ie}, 请检查 methods 文件夹")
                return

            # 保存结果
            base_name = os.path.splitext(os.path.basename(self.data_path))[0]
            view_img = (normalize(fused_image[:, :, :3]) * 255).astype(np.uint8)
            view_path = os.path.join(self.save_dir, f'{base_name}_{self.method}.png')
            cv2.imwrite(view_path, cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR))

            # 低分对比图
            low_view = (
                normalize(cv2.resize(input_ms[:, :, :3], (W_crop, H_crop), interpolation=cv2.INTER_NEAREST)) * 255
            ).astype(np.uint8)
            low_path = os.path.join(self.save_dir, f'{base_name}_LowRes.jpg')
            cv2.imwrite(low_path, cv2.cvtColor(low_view, cv2.COLOR_RGB2BGR))

            tif_path = os.path.join(self.save_dir, f'{base_name}_{self.method}.tif')
            tiff.imwrite(tif_path, fused_image.astype(np.float32))

            self.finished_signal.emit(fused_image, view_path, low_path, tif_path)

        except Exception as e:
            self.log_signal.emit(f"❌ 融合失败: {str(e)}")
            print(traceback.format_exc())


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(current_dir, "py_pansharpening")
    if project_path not in sys.path:
        sys.path.append(project_path)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    app = QApplication(sys.argv)
    window = MarsApp()
    sys.exit(app.exec())
