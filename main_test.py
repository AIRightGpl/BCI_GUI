
from GUI.mainwindow import Ui_MainWindow

if __name__ == '__main__':
    # run test
    import sys
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QApplication, QMainWindow
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())