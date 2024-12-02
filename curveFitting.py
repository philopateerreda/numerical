import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from pathlib import Path

class StyleSheet:
    DARK = """
    QMainWindow, QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
    }
    QTableWidget {
        background-color: #333333;
        border: 1px solid #555555;
        border-radius: 5px;
        gridline-color: #444444;
        color: #ffffff;
    }
    QTableWidget::item {
        padding: 5px;
    }
    QHeaderView::section {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 8px;
        border: none;
    }
    QPushButton {
        background-color: #0d47a1;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #1565c0;
    }
    QPushButton:pressed {
        background-color: #0a3d91;
    }
    QTextEdit {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 10px;
    }
    QLabel {
        color: #ffffff;
        font-weight: bold;
    }
    QGroupBox {
        border: 2px solid #555555;
        border-radius: 5px;
        margin-top: 1em;
        padding-top: 10px;
        color: #ffffff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        padding: 0 3px;
    }
    """

class CustomTableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        header = self.horizontalHeader()
        for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

class CurveFittingCalculator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Curve Fitting Calc")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(StyleSheet.DARK)
        
        # Configure matplotlib style for dark theme
        plt.style.use('dark_background')
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header
        title_label = QLabel("Curve Fitting Analysis")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Split layout
        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        
        # Left panel (input)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        input_group = QGroupBox("Data Input")
        input_layout = QVBoxLayout(input_group)
        
        self.table = CustomTableWidget(10, 3)
        self.table.setHorizontalHeaderLabels(['X Value', 'Y Value', 'Include Point'])
        for i in range(10):
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(i, 2, checkbox_item)
        input_layout.addWidget(self.table)
        
        controls_layout = QHBoxLayout()
        calc_button = QPushButton("Calculate Fitting")
        calc_button.setMinimumHeight(40)
        calc_button.clicked.connect(self.calculate_fitting)
        clear_button = QPushButton("Clear Data")
        clear_button.setMinimumHeight(40)
        clear_button.clicked.connect(self.clear_data)
        controls_layout.addWidget(calc_button)
        controls_layout.addWidget(clear_button)
        input_layout.addLayout(controls_layout)
        
        left_layout.addWidget(input_group)
        
        # Right panel (results and plot)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        plot_group = QGroupBox("Visualization")
        plot_layout = QVBoxLayout(plot_group)
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        right_layout.addWidget(plot_group)
        
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        right_layout.addWidget(results_group)
        
        split_layout.addWidget(left_panel, 1)
        split_layout.addWidget(right_panel, 2)
        
        layout.addWidget(split_widget)
        
        self.statusBar().showMessage("Ready to calculate curve fitting")
        
    def clear_data(self):
        for i in range(self.table.rowCount()):
            for j in range(2):
                self.table.setItem(i, j, None)
        self.results_text.clear()
        self.ax.clear()
        self.canvas.draw()
        self.statusBar().showMessage("Data cleared")
    
    def get_table_data(self):
        data = {'x': [], 'y': []}
        for i in range(self.table.rowCount()):
            x_item = self.table.item(i, 0)
            y_item = self.table.item(i, 1)
            include_item = self.table.item(i, 2)
            
            if (x_item and y_item and include_item and 
                x_item.text() and y_item.text() and 
                include_item.checkState() == Qt.CheckState.Checked):
                try:
                    data['x'].append(float(x_item.text()))
                    data['y'].append(float(y_item.text()))
                except ValueError:
                    continue
        return pd.DataFrame(data)
    
    def calculate_first_degree(self, df):
        n = len(df)
        sum_x = df['x'].sum()
        sum_y = df['y'].sum()
        sum_xy = (df['x'] * df['y']).sum()
        sum_x2 = (df['x'] ** 2).sum()
        
        A = np.array([[n, sum_x], [sum_x, sum_x2]])
        b = np.array([sum_y, sum_xy])
        a0, a1 = np.linalg.solve(A, b)
        
        return a0, a1
    
    def calculate_second_degree(self, df):
        n = len(df)
        sum_x = df['x'].sum()
        sum_x2 = (df['x'] ** 2).sum()
        sum_x3 = (df['x'] ** 3).sum()
        sum_x4 = (df['x'] ** 4).sum()
        sum_y = df['y'].sum()
        sum_xy = (df['x'] * df['y']).sum()
        sum_x2y = (df['x'] ** 2 * df['y']).sum()
        
        A = np.array([[n, sum_x, sum_x2],
                     [sum_x, sum_x2, sum_x3],
                     [sum_x2, sum_x3, sum_x4]])
        b = np.array([sum_y, sum_xy, sum_x2y])
        a0, a1, a2 = np.linalg.solve(A, b)
        
        return a0, a1, a2
    
    def calculate_fitting(self):
        self.statusBar().showMessage("Calculating curve fitting...")
        df = self.get_table_data()
        if len(df) < 3:
            self.results_text.setText("Please enter at least 3 valid points")
            self.statusBar().showMessage("Error: Insufficient data points")
            return
        
        try:
            # Calculate fittings
            a0_1, a1_1 = self.calculate_first_degree(df)
            first_degree_eq = f"y = {a1_1:.4f}x + {a0_1:.4f}"
            
            a0_2, a1_2, a2_2 = self.calculate_second_degree(df)
            second_degree_eq = f"y = {a2_2:.4f}x² + {a1_2:.4f}x + {a0_2:.4f}"
            
            # Generate points for plotting
            x_range = np.linspace(min(df['x']), max(df['x']), 100)
            y1 = a1_1 * x_range + a0_1
            y2 = a2_2 * x_range**2 + a1_2 * x_range + a0_2
            
            # Plot
            self.ax.clear()
            self.ax.set_title("Curve Fitting Analysis", pad=20, color='white')
            self.ax.set_xlabel("X Values", color='white')
            self.ax.set_ylabel("Y Values", color='white')
            
            self.ax.scatter(df['x'], df['y'], color='#00ff00', s=100, label='Data points', zorder=3)
            self.ax.plot(x_range, y1, color='#ff4444', linewidth=2, label='First degree', zorder=2)
            self.ax.plot(x_range, y2, color='#44aaff', linewidth=2, label='Second degree', zorder=1)
            
            self.ax.legend(frameon=True, fancybox=True)
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.ax.set_facecolor('#2b2b2b')
            self.figure.patch.set_facecolor('#2b2b2b')
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Results
            results = (
                "Curve Fitting Analysis Results\n"
                "================================\n\n"
                "🔵 First Degree Fitting:\n"
                f"{first_degree_eq}\n\n"
                "🔵 Second Degree Fitting:\n"
                f"{second_degree_eq}\n\n"
                " Calculation Parameters:\n"
                "------------------------\n"
                f"First Degree:\n"
                f"  a₀ = {a0_1:.4f}\n"
                f"  a₁ = {a1_1:.4f}\n\n"
                f"Second Degree:\n"
                f"  a₀ = {a0_2:.4f}\n"
                f"  a₁ = {a1_2:.4f}\n"
                f"  a₂ = {a2_2:.4f}\n\n"
            )
            self.results_text.setText(results)
            self.statusBar().showMessage("Calculation completed successfully")
            
        except np.linalg.LinAlgError:
            self.results_text.setText("Error: Unable to calculate fitting. Please check your data points.")
            self.statusBar().showMessage("Error: Calculation failed")
        except Exception as e:
            self.results_text.setText(f"An error occurred: {str(e)}")
            self.statusBar().showMessage("Error: Calculation failed")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CurveFittingCalculator()
    window.show()
    sys.exit(app.exec())