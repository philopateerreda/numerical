import sys
import os
os.environ['QT_API'] = 'pyqt6'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTableWidget, QTableWidgetItem, QComboBox, 
                            QMessageBox, QFrame, QGridLayout, QGroupBox,
                            QTabWidget, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from moreDetails import LagrangeDetails, NewtonDetails, DetailsDialog

class InterpolationMethods:
    @staticmethod
    def lagrange_interpolation(x_points, y_points, x):
        n = len(x_points)
        result = 0.0
        
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            result += term
            
        return result
    
    @staticmethod
    def newton_divided_difference(x_points, y_points, x):
        n = len(x_points)
        coef = np.zeros([n, n])
        
        # First column is y values
        coef[:,0] = y_points
        
        # Calculate divided differences
        for j in range(1, n):
            for i in range(n-j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_points[i+j] - x_points[i])
        
        # Interpolate
        result = coef[0][0]
        mult = 1
        for j in range(1, n):
            mult *= (x - x_points[j-1])
            result += coef[0][j] * mult
            
        return result, coef

class DarkPalette(QPalette):
    def __init__(self):
        super().__init__()
        self.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)

class InterpolationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolation Methods")
        self.setMinimumSize(1200, 800)
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Left panel for inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Points input
        points_group = QGroupBox("Data Points")
        points_layout = QVBoxLayout()
        
        # Number of points spinner
        points_input_layout = QHBoxLayout()
        self.points_spinner = QSpinBox()
        self.points_spinner.setRange(2, 10)
        self.points_spinner.setValue(3)
        self.points_spinner.valueChanged.connect(self.update_table)
        points_input_layout.addWidget(QLabel("Number of Points:"))
        points_input_layout.addWidget(self.points_spinner)
        points_layout.addLayout(points_input_layout)
        
        # Points table
        self.points_table = QTableWidget(3, 2)
        self.points_table.setHorizontalHeaderLabels(["x", "y"])
        self.points_table.horizontalHeader().setStretchLastSection(True)
        points_layout.addWidget(self.points_table)
        points_group.setLayout(points_layout)
        left_layout.addWidget(points_group)
        
        # Interpolation controls
        interp_group = QGroupBox("Interpolation")
        interp_layout = QVBoxLayout()
        
        # Method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Lagrange", "Newton's Divided Difference"])
        interp_layout.addWidget(QLabel("Method:"))
        interp_layout.addWidget(self.method_combo)
        
        # X value input
        self.x_input = QLineEdit()
        interp_layout.addWidget(QLabel("Interpolate at x:"))
        interp_layout.addWidget(self.x_input)
        
        # Interpolate button
        self.interpolate_button = QPushButton("Interpolate")
        self.interpolate_button.clicked.connect(self.perform_interpolation)
        self.interpolate_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        interp_layout.addWidget(self.interpolate_button)
        
        interp_group.setLayout(interp_layout)
        left_layout.addWidget(interp_group)
        
        # Result display
        self.result_label = QLabel()
        self.result_label.setStyleSheet("font-size: 14px; color: #2ecc71;")
        left_layout.addWidget(self.result_label)
        
        layout.addWidget(left_panel, 1)
        
        # Right panel for plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        layout.addWidget(right_panel, 2)
        
        self.update_table()

        # Add Details button next to the Interpolate button
        button_layout = QHBoxLayout()
        
        self.interpolate_button = QPushButton("Interpolate")
        self.interpolate_button.clicked.connect(self.perform_interpolation)
        self.interpolate_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        
        self.details_button = QPushButton("Show Details")
        self.details_button.clicked.connect(self.show_details)
        self.details_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        
        button_layout.addWidget(self.interpolate_button)
        button_layout.addWidget(self.details_button)
        interp_layout.addLayout(button_layout)

    def show_details(self):
        try:
            # Get points from table
            n_points = self.points_table.rowCount()
            x_points = []
            y_points = []
            
            for i in range(n_points):
                x_item = self.points_table.item(i, 0)
                y_item = self.points_table.item(i, 1)
                
                if x_item and y_item:
                    x_points.append(float(x_item.text()))
                    y_points.append(float(y_item.text()))
            
            x_points = np.array(x_points)
            y_points = np.array(y_points)
            
            # Create appropriate details object based on selected method
            if self.method_combo.currentText() == "Lagrange":
                details = LagrangeDetails(x_points, y_points)
                dialog = DetailsDialog(lagrange_details=details, parent=self)
            else:
                details = NewtonDetails(x_points, y_points)
                dialog = DetailsDialog(newton_details=details, parent=self)
                
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to show details: {str(e)}")  

            
    def update_table(self):
        n_points = self.points_spinner.value()
        self.points_table.setRowCount(n_points)
        
    def perform_interpolation(self):
        try:
            # Get points from table
            n_points = self.points_table.rowCount()
            x_points = []
            y_points = []
            
            for i in range(n_points):
                x_item = self.points_table.item(i, 0)
                y_item = self.points_table.item(i, 1)
                
                if x_item and y_item:
                    x_points.append(float(x_item.text()))
                    y_points.append(float(y_item.text()))
            
            x_points = np.array(x_points)
            y_points = np.array(y_points)
            
            # Get interpolation point
            x_interp = float(self.x_input.text())
            
            # Perform interpolation
            if self.method_combo.currentText() == "Lagrange":
                result = InterpolationMethods.lagrange_interpolation(x_points, y_points, x_interp)
                self.result_label.setText(f"Interpolated value: {result:.6f}")
            else:
                result, coef = InterpolationMethods.newton_divided_difference(x_points, y_points, x_interp)
                self.result_label.setText(f"Interpolated value: {result:.6f}")
            
            # Update plot
            self.update_plot(x_points, y_points, x_interp, result)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def update_plot(self, x_points, y_points, x_interp, y_interp):
        self.ax.clear()
        
        # Plot original points
        self.ax.scatter(x_points, y_points, color='#2ecc71', label='Data points')
        
        # Plot interpolated point
        self.ax.scatter([x_interp], [y_interp], color='#e74c3c', label='Interpolated point')
        
        # Plot smooth curve through all points
        x_smooth = np.linspace(min(x_points), max(x_points), 200)
        y_smooth = []
        
        for x in x_smooth:
            if self.method_combo.currentText() == "Lagrange":
                y = InterpolationMethods.lagrange_interpolation(x_points, y_points, x)
            else:
                y, _ = InterpolationMethods.newton_divided_difference(x_points, y_points, x)
            y_smooth.append(y)
            
        self.ax.plot(x_smooth, y_smooth, '--', color='#3498db', label='Interpolation curve')
        
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.ax.set_title('Interpolation Result')
        
        # Dark theme for plot
        self.figure.set_facecolor('#353535')
        self.ax.set_facecolor('#353535')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.title.set_color('white')
        
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle('Fusion')
    app.setPalette(DarkPalette())
    
    window = InterpolationGUI()
    window.show()
    sys.exit(app.exec())
