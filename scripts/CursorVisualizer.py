"""
Demonstrates common image analysis tools.

Many of the features demonstrated here are already provided by the ImageView
widget, but here we present a lower-level approach that provides finer control
over the user interface.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from arrconv import cp2np

pg.setConfigOptions(imageAxisOrder="row-major")


class CursorVisualizer:
    def __init__(self, data):
        self.data = cp2np(data)
        self.app = pg.mkQApp("Data Slicing Viewer")
        self.win = pg.GraphicsLayoutWidget()

        self.initialize_window()

    def initialize_window(self):
        self.p = self.win.addPlot()
        self.img = pg.ImageItem()
        self.p.addItem(self.img)

        self.roi = pg.ROI([-8, 14], [6, 5])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addRotateHandle([0, 1], [0, 0])
        self.p.addItem(self.roi)
        self.roi.setZValue(10)  # make sure ROI is drawn above image

        hist = pg.HistogramLUTItem()
        hist.setImageItem(self.img)
        self.win.addItem(hist)

        self.win.nextRow()
        self.p2 = self.win.addPlot(colspan=2)
        self.p2.setMaximumHeight(250)
        self.win.resize(800, 800)
        self.win.show()

        self.img.setImage(self.data)
        hist.setLevels(self.data.min(), self.data.max())

        # zoom to fit imageo
        self.p.autoRange()

        self.roi.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()

        self.img.hoverEvent = self.imageHoverEvent

    # Callbacks for handling user interaction
    def updatePlot(self):
        selected = self.roi.getArrayRegion(self.data, self.img)
        self.p2.plot(selected.mean(axis=0), clear=True)

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor."""
        if event.isExit():
            self.p.setTitle("")
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        i = int(np.clip(i, 0, self.data.shape[0] - 1))
        j = int(np.clip(j, 0, self.data.shape[1] - 1))
        val = self.data[i, j]
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        self.p.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g" % (x, y, i, j, val))

    def run(self):
        self.app.exec()


if __name__ == "__main__":
    data = np.random.normal(size=(200, 100))
    cursor_visualizer = CursorVisualizer(data)
    cursor_visualizer.run()
