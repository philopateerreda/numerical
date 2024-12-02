import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QSlider, QLabel, QPushButton, QLineEdit,
                            QGridLayout, QGroupBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class EnhancedLinearFunctionPlotter(QMainWindow):
    COLORS = {
        'background': '#1e1e1e',
        'text': '#ffffff',
        'line': '#61afef',
        'point': '#e06c75',
        'grid': '#3e4451',
        'slider': '#528bff',
        'button': '#528bff',
        'groupbox': '#2c313c'
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Linear Function Plotter")
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.COLORS['background']};
                color: {self.COLORS['text']};
            }}
            QGroupBox {{
                background-color: {self.COLORS['groupbox']};
                border: 1px solid {self.COLORS['text']};
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }}
        """)
        self.setMinimumSize(1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Initialize equation parameters
        self.slope = 1.0
        self.intercept = 0.0
        self.x_value = 5.0

        self.create_plot()
        self.create_equation_input()
        self.create_controls()
        self.update_plot()

    def create_plot(self):
        plot_group = QGroupBox("Function Plot")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.figure.patch.set_facecolor(self.COLORS['background'])
        self.ax.set_facecolor(self.COLORS['background'])
        
        plot_group.setLayout(plot_layout)
        self.layout.addWidget(plot_group)

    def create_equation_input(self):
        equation_group = QGroupBox("Equation Parameters")
        equation_layout = QGridLayout()

        # Slope input
        slope_label = QLabel("Slope (m):")
        self.slope_input = QLineEdit(str(self.slope))
        self.slope_input.setPlaceholderText("Enter slope")
        self.slope_input.returnPressed.connect(self.update_equation)
        equation_layout.addWidget(slope_label, 0, 0)
        equation_layout.addWidget(self.slope_input, 0, 1)

        # Intercept input
        intercept_label = QLabel("Y-intercept (b):")
        self.intercept_input = QLineEdit(str(self.intercept))
        self.intercept_input.setPlaceholderText("Enter y-intercept")
        self.intercept_input.returnPressed.connect(self.update_equation)
        equation_layout.addWidget(intercept_label, 0, 2)
        equation_layout.addWidget(self.intercept_input, 0, 3)

        # Current equation display
        self.equation_label = QLabel("Current equation: y = x + 0")
        self.equation_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        equation_layout.addWidget(self.equation_label, 1, 0, 1, 4)

        # Style inputs
        input_style = f"""
            QLineEdit {{
                background-color: {self.COLORS['background']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['text']};
                padding: 5px;
                font-size: 14px;
                border-radius: 3px;
            }}
        """
        self.slope_input.setStyleSheet(input_style)
        self.intercept_input.setStyleSheet(input_style)

        equation_group.setLayout(equation_layout)
        self.layout.addWidget(equation_group)

    def create_controls(self):
        controls_group = QGroupBox("X-Value Controls")
        controls_layout = QGridLayout()

        # Slider for x value
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {self.COLORS['grid']};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {self.COLORS['slider']};
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }}
        """)
        self.slider.valueChanged.connect(self.update_from_slider)
        controls_layout.addWidget(self.slider, 0, 0, 1, 3)

        # X and Y value displays
        self.x_label = QLabel(f"x = {self.x_value:.2f}")
        self.y_label = QLabel(f"y = {self.calculate_y(self.x_value):.2f}")
        controls_layout.addWidget(self.x_label, 1, 0)
        controls_layout.addWidget(self.y_label, 1, 1)

        # X value input field
        x_input_label = QLabel("Enter x value:")
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("0-10")
        self.x_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.COLORS['background']};
                color: {self.COLORS['text']};
                border: 1px solid {self.COLORS['text']};
                padding: 5px;
                font-size: 14px;
                border-radius: 3px;
            }}
        """)
        self.x_input.returnPressed.connect(self.update_from_input)
        controls_layout.addWidget(x_input_label, 2, 0)
        controls_layout.addWidget(self.x_input, 2, 1)

        # Quit button
        self.quit_button = QPushButton("Quit")
        self.quit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLORS['button']};
                color: {self.COLORS['text']};
                border: none;
                padding: 5px 15px;
                font-size: 14px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: #3a6bbd;
            }}
        """)
        self.quit_button.clicked.connect(self.close)
        controls_layout.addWidget(self.quit_button, 2, 2)

        controls_group.setLayout(controls_layout)
        self.layout.addWidget(controls_group)

    def calculate_y(self, x):
        return self.slope * x + self.intercept

    def update_equation(self):
        try:
            self.slope = float(self.slope_input.text())
            self.intercept = float(self.intercept_input.text())
            self.equation_label.setText(f"Current equation: y = {self.slope}x + {self.intercept}")
            self.update_plot()
        except ValueError:
            self.equation_label.setText("Invalid input! Please enter valid numbers.")

    def update_plot(self):
        self.ax.clear()
        x = np.linspace(0, 10, 100)
        y = self.slope * x + self.intercept
        
        # Plot the line
        self.ax.plot(x, y, linewidth=2, color=self.COLORS['line'])
        
        # Plot the current point
        y_point = self.calculate_y(self.x_value)
        self.ax.scatter(self.x_value, y_point, color=self.COLORS['point'], s=100, zorder=5)

        # Set labels and title
        self.ax.set_title(f"Linear Function: y = {self.slope}x + {self.intercept}", 
                         fontsize=16, color=self.COLORS['text'])
        self.ax.set_xlabel("x", fontsize=12, color=self.COLORS['text'])
        self.ax.set_ylabel("y", fontsize=12, color=self.COLORS['text'])

        # Style the plot
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.COLORS['text'])

        self.ax.grid(color=self.COLORS['grid'], linestyle='--', linewidth=0.5)
        self.ax.tick_params(colors=self.COLORS['text'])

        # Add point annotation
        self.ax.annotate(f'({self.x_value:.2f}, {y_point:.2f})', 
                        (self.x_value, y_point),
                        xytext=(5, 5),
                        textcoords='offset points',
                        color=self.COLORS['text'],
                        fontweight='bold')

        self.canvas.draw()
        
        # Update labels
        self.x_label.setText(f"x = {self.x_value:.2f}")
        self.y_label.setText(f"y = {y_point:.2f}")

    def update_from_slider(self, value):
        self.x_value = value / 10
        self.update_plot()

    def update_from_input(self):
        try:
            x_input = float(self.x_input.text())
            if 0 <= x_input <= 10:
                self.x_value = x_input
                self.slider.setValue(int(x_input * 10))
                self.update_plot()
            else:
                self.x_input.setText("Must be 0-10")
        except ValueError:
            self.x_input.setText("Invalid input")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = EnhancedLinearFunctionPlotter()
    main_window.show()
    sys.exit(app.exec())