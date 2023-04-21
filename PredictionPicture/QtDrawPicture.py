from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from PyQt5 import QtCore
import pyqtgraph.examples
import pyqtgraph.exporters

# pyqtgraph.examples.run()
loop = 0
class MyQtPicture(QWidget):
    def __init__(self, Queue):
        super().__init__()
        # 设置窗口大小
        self.resize(900, 600)
        self.setWindowTitle("趋势预测")
        # 队列传递数据
        self.queue = Queue
        # 设置画图
        pg.setConfigOption("background", "k")  # 设置绘图区域背景颜色为黑色
        pg.setConfigOption("foreground", "w")  # 设置绘图区域前景色（字体，坐标轴等）颜色为黑色
        self.plot = pg.PlotWidget(self, antialias=False)
        # 加载画图函数
        self.plot.addLegend()
        # 添加网格线
        self.plot.showGrid(x=True, y=True, alpha=0.7)
        # 设置笔刷
        self.pen1 = pg.mkPen(color=(255, 0, 0), width=3, style=QtCore.Qt.DotLine)
        self.pen2 = pg.mkPen(color=(0, 255, 0), width=3)
        self.pen3 = pg.mkPen(color=(0, 255, 255), width=2)
        self.pen4 = pg.mkPen(color=(255, 255, 0), width=2, style=QtCore.Qt.DotLine)
        # 设置曲线
        self.plot1 = self.plot.plot(x=[], y=[], pen=self.pen1, name="预测")
        self.plot1.setOpacity(0.7)  # 设置曲线透明度
        self.plot2 = self.plot.plot(x=[], y=[], pen=self.pen2, name="真实")
        self.plot2.setOpacity(0.7)  # 设置曲线透明度
        # 显示±15%的阈值线
        self.plotThresholdUp = self.plot.plot(x=[], y=[], pen=self.pen4, name="±15%阈值")
        self.plotThresholdUp.setOpacity(0.7)
        self.plotThresholdDown = self.plot.plot(x=[], y=[], pen=self.pen4)
        self.plotThresholdDown.setOpacity(0.5)
        # 添加垂直线
        self.plot3 = pg.InfiniteLine(pos=0, pen=self.pen3, movable=False, name="预测起始点")
        self.plot.addItem(self.plot3)
        # 添加定时器
        self.timer = QtCore.QTimer(self)
        # 将pg添加到widgets中
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def drawPlot(self):
        global loop
        if not self.queue.empty():
            x_fore, y_fore, x_real, y_real, pos = self.queue.get()
            self.plot1.setData(x=x_fore,y=y_fore)
            self.plot2.setData(x=x_real,y=y_real)
            self.plot3.setPos(pos=pos)
            self.plot.setXRange(x_fore[0] - 100, x_fore[-1] + 10)
            # 显示±15%的阈值线
            thresholdUp = y_real*0.85
            thresholdDown = y_real*1.15
            self.plotThresholdUp.setData(x=x_real, y=thresholdUp)
            self.plotThresholdDown.setData(x=x_real, y=thresholdDown)
            self.saveImage()
            loop += 1

    def saveImage(self):
        global loop
        ex = pyqtgraph.exporters.ImageExporter(self.plot.scene())
        ex.export(fileName=f"./结果/LSTM/{loop}.png")

    def run(self):
        self.timer.timeout.connect(self.drawPlot)
        self.timer.start(100)
