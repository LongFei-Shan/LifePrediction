# Author:Administrator
# Name:LSTMLossPicture
# Time:2023/4/23  9:15
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from PyQt5 import QtCore
import pyqtgraph.examples
import pyqtgraph.exporters


class LSTMLossPicture(QWidget):
    def __init__(self):
        super().__init__()
        # 设置窗口大小
        self.resize(900, 600)
        self.setWindowTitle("趋势预测")
        # 设置画图
        pg.setConfigOption("background", "k")  # 设置绘图区域背景颜色为黑色
        pg.setConfigOption("foreground", "w")  # 设置绘图区域前景色（字体，坐标轴等）颜色为黑色
        self.plot = pg.PlotWidget(self, antialias=False)
        # 加载画图函数
        self.plot.addLegend()
        # 添加网格线
        self.plot.showGrid(x=True, y=True, alpha=0.7)
        # 设置笔刷
        self.pen1 = pg.mkPen(color=(255, 0, 0), width=3)
        # 设置曲线
        self.plot1 = self.plot.plot(x=[], y=[], pen=self.pen1)
        # 添加定时器
        self.timer = QtCore.QTimer(self)
        # 将pg添加到widgets中
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def drawPlot(self):
        if not self.queue.empty():
            time, loss = self.queue.get()
            self.plot1.setData(x=time,y=loss)
            # 存储图片
            self.saveImage()

    def saveImage(self):
        ex = pyqtgraph.exporters.ImageExporter(self.plot.scene())
        ex.export(fileName="./loss.png")

    def run(self):
        self.timer.timeout.connect(self.drawPlot)
        self.timer.start(100)
