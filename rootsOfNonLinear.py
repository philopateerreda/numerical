
import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout,
    QHBoxLayout, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QWidget, QGroupBox, QSpinBox, QMessageBox, QProgressBar
)
from PyQt6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Callable, List, Tuple
import warnings

import sympy

from typing import Callable, List, Tuple
import numpy as np

class NumericalMethods:
    """A class containing numerical methods for solving nonlinear equations.
    
    This class implements various root-finding algorithms,
    Bisection, and Secant methods. Each method returns detailed iteration history
    and includes robust error handling and convergence checks.
    """

    @staticmethod
    def bisection(
        func: Callable[[float], float],
        a: float,
        b: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        fixed_digits: int = 6
    ) -> List[Tuple[int, float, float, float, float]]:
        """Implements the Bisection method for finding roots of nonlinear equations.
        
        Args:
            func: The function for which we want to find the root
            a: Left endpoint of interval
            b: Right endpoint of interval
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            fixed_digits: Number of fixed digits for convergence check
            
        Returns:
            List of tuples containing (iteration, a, b, c, f(c))
            
        Raises:
            ValueError: If the method fails to converge or invalid interval is provided
        """
        if not callable(func):
            raise TypeError("func must be callable")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if a >= b:
            raise ValueError("Left endpoint must be less than right endpoint")
            
        fa, fb = func(a), func(b)
        
        if fa * fb >= 0:
            raise ValueError(
                "Function must have opposite signs at interval endpoints"
            )
            
        results = []
        
        for i in range(max_iter):
            c = (a + b) / 2
            fc = func(c)
            
            results.append((i + 1, a, b, c, fc))
            
            if abs(fc) < tolerance or abs(b - a) < 10**(-fixed_digits):
                return results
                
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
                
        raise ValueError(
            f"Failed to converge within {max_iter} iterations. Current interval: [{a}, {b}]"
        )

    @staticmethod
    def secant(
        func: Callable[[float], float],
        x0: float,
        x1: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        fixed_digits: int = 6
    ) -> List[Tuple[int, float, float, float, float]]:
        """Implements the Secant method for finding roots of nonlinear equations.
        
        Args:
            func: The function for which we want to find the root
            x0: First initial guess
            x1: Second initial guess
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            fixed_digits: Number of fixed digits for convergence check
            
        Returns:
            List of tuples containing (iteration, x0, x1, x_next, f(x1))
            
        Raises:
            ValueError: If the method fails to converge or encounters numerical issues
        """
        if not callable(func):
            raise TypeError("func must be callable")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if abs(x1 - x0) < tolerance:
            raise ValueError("Initial guesses must be different")
            
        results = []
        
        for i in range(max_iter):
            fx0, fx1 = func(x0), func(x1)
            
            if abs(fx1 - fx0) < 1e-10:
                raise ValueError(
                    f"Denominator too close to zero at iteration {i + 1}. Method fails."
                )
                
            x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            results.append((i + 1, x0, x1, x_next, fx1))
            
            if abs(fx1) < tolerance:
                return results
                
            if abs(x_next - x1) < 10**(-fixed_digits):
                return results
                
            if abs(x_next) > 1e6:
                raise ValueError(
                    f"Solution diverging at iteration {i + 1}. Try different initial values."
                )
                
            x0, x1 = x1, x_next
            
        raise ValueError(
            f"Failed to converge within {max_iter} iterations. Last value: x = {x1}"
        )
    @staticmethod
    def simple_iteration(
        func: Callable[[float], float],
        x0: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        fixed_digits: int = 6
    ) -> List[Tuple[int, float, float]]:
        """Implements the Simple Iteration method (Fixed Point Iteration) for solving nonlinear equations.
        
        The method works by rewriting f(x) = 0 as x = g(x) and iterating x_{n+1} = g(x_n).
        Convergence requires that |g'(x)| < 1 in the neighborhood of the solution.
        
        Args:
            func: The iteration function g(x) where the fixed point is sought
            x0: Initial guess
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            fixed_digits: Number of fixed digits for convergence check
            
        Returns:
            List of tuples containing (iteration, x_current, x_next)
            
        Raises:
            TypeError: If func is not callable
            ValueError: If method fails to converge or invalid parameters are provided
            
        Note:
            The function passed should be in the form g(x) where x = g(x) is the fixed point.
            This is different from the function f(x) where we seek f(x) = 0.
        """
        # Input validation
        if not callable(func):
            raise TypeError("func must be callable")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
            
        x = float(x0)  # Ensure x0 is float
        results = []
        
        for i in range(max_iter):
            try:
                x_next = func(x)
            except (ValueError, ZeroDivisionError) as e:
                raise ValueError(
                    f"Error in iteration function at x = {x}: {str(e)}"
                )
                
            # Store current iteration results
            results.append((i + 1, x, x_next))
            
            # Check for convergence using both absolute and relative errors
            abs_error = abs(x_next - x)
            rel_error = abs_error / (abs(x_next) + 1e-10)  # Avoid division by zero
            
            if abs_error < tolerance or rel_error < 10**(-fixed_digits):
                return results
                
            # Check for divergence
            if abs(x_next) > 1e6:
                raise ValueError(
                    f"Solution diverging at iteration {i + 1}. "
                    f"Last values: x_{i} = {x:.6f}, x_{i+1} = {x_next:.6f}. "
                    "Try different initial guess or iteration function."
                )
                
            # Check for oscillation
            if i > 4 and len(set(result[1] for result in results[-4:])) <= 2:
                raise ValueError(
                    f"Method appears to be oscillating at iteration {i + 1}. "
                    "The iteration function might not satisfy convergence conditions."
                )
                
            x = x_next
            
        raise ValueError(
            f"Failed to converge within {max_iter} iterations. "
            f"Last values: x_{max_iter-1} = {x:.6f}, x_{max_iter} = {x_next:.6f}"
        )


class CustomLineEdit(QLineEdit):
    def __init__(self, placeholder: str = "", validator=None):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
            }
            QLineEdit:hover {
                background-color: #323232;
            }
        """)
        if validator:
            self.setValidator(validator)


class CustomComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QComboBox {
                background-color: #2b2b2b;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                color: #ffffff;
                font-size: 14px;
            }
            QComboBox:hover {
                background-color: #323232;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)


class AnimatedProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        self.setTextVisible(True)
        self.setMinimum(0)
        self.setMaximum(100)


class NonLinearSolverApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.numerical_methods = NumericalMethods()
        self.setWindowTitle("Advanced Non-Linear Equation Solver")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_theme()
        self.init_ui()

    def create_function(self, expr: str) -> Callable[[float], float]:
        try:
            from sympy import sympify, lambdify
            x = sympy.symbols('x')
            func = lambdify(x, sympify(expr), modules=['numpy'])
            return func
        except (ImportError, sympy.SympifyError, TypeError) as e:
            warnings.warn("Sympy not found or invalid expression. Using less safe eval(). Install sympy for improved security.")
            namespace = {
                'np': np,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt
            }
            return lambda x: eval(expr, namespace, {'x': x})


    def setup_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-size: 14px;
            }
            QGroupBox {
                background-color: #252525;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 1em;
                padding-top: 1em;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QTableWidget {
                background-color: #252525;
                alternate-background-color: #2d2d2d;
                gridline-color: #3d3d3d;
                border: none;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                padding: 5px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
            }
            QScrollBar {
                background-color: #252525;
            }
            QScrollBar:vertical {
                width: 12px;
            }
            QScrollBar:horizontal {
                height: 12px;
            }
            QLabel {
                color: #cccccc;
            }
            QSpinBox {
                background-color: #2b2b2b;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                color: white;
            }
            QToolTip {
                background-color: #2d2d2d;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 5px;
            }
        """)
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()

        func_layout = QHBoxLayout()
        func_layout.addWidget(QLabel("Function f(x):"))
        self.func_input = CustomLineEdit("Enter function (e.g., x**2 - 4)")
        func_layout.addWidget(self.func_input)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = CustomComboBox()
        self.method_combo.addItems(["Bisection", "Secant", "Simple Iteration"])
        self.method_combo.currentTextChanged.connect(self.update_parameter_labels)
        method_layout.addWidget(self.method_combo)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Parameter 1:"))
        self.param1_input = CustomLineEdit("Initial guess")
        param_layout.addWidget(self.param1_input)
        param_layout.addWidget(QLabel("Parameter 2:"))
        self.param2_input = CustomLineEdit("Second guess/bound")
        param_layout.addWidget(self.param2_input)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Tolerance:"))
        self.tolerance_input = CustomLineEdit("1e-6")
        settings_layout.addWidget(self.tolerance_input)
        settings_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setRange(1, 1000)
        self.max_iter_spinbox.setValue(100)
        settings_layout.addWidget(self.max_iter_spinbox)
        settings_layout.addWidget(QLabel("Fixed Digits:"))
        self.digits_spinbox = QSpinBox()
        self.digits_spinbox.setRange(1, 15)
        self.digits_spinbox.setValue(6)
        settings_layout.addWidget(self.digits_spinbox)

        input_layout.addLayout(func_layout)
        input_layout.addLayout(method_layout)
        input_layout.addLayout(param_layout)
        input_layout.addLayout(settings_layout)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        button_layout = QHBoxLayout()
        solve_button = QPushButton("Solve")
        solve_button.clicked.connect(self.solve_equation)
        button_layout.addWidget(solve_button)

        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setVisible(False)
        button_layout.addWidget(self.progress_bar)
        main_layout.addLayout(button_layout)

        results_group = QGroupBox("Results")
        results_layout = QHBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Iteration", "Value 1", "Value 2", "Next Value", "f(x)"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        figure = Figure(figsize=(8, 6))
        self.ax = figure.add_subplot(111)
        self.plot_canvas = FigureCanvas(figure)
        plot_layout.addWidget(self.plot_canvas)
        results_layout.addWidget(plot_widget)

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.update_parameter_labels()

    def update_parameter_labels(self):
        method = self.method_combo.currentText()

        if method == "Bisection":
            self.param1_input.setPlaceholderText("Left bound (a)")
            self.param2_input.setPlaceholderText("Right bound (b)")
            self.param2_input.setEnabled(True)
        elif method == "Secant":
            self.param1_input.setPlaceholderText("First guess (x0)")
            self.param2_input.setPlaceholderText("Second guess (x1)")
            self.param2_input.setEnabled(True)
        else:  # Simple Iteration
            self.param1_input.setPlaceholderText("Initial guess (x0)")
            self.param2_input.setEnabled(False)

    def solve_equation(self):
        try:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)
            QApplication.processEvents()

            func_expr = self.func_input.text()
            method = self.method_combo.currentText()
            tolerance = float(self.tolerance_input.text())
            max_iter = self.max_iter_spinbox.value()
            fixed_digits = self.digits_spinbox.value()

            self.progress_bar.setValue(20)
            QApplication.processEvents()

            func = self.create_function(func_expr)

            self.progress_bar.setValue(40)
            QApplication.processEvents()

            results = None

            if method == "Bisection":
                a = float(self.param1_input.text())
                b = float(self.param2_input.text())
                results = self.numerical_methods.bisection(func, a, b, tolerance, max_iter, fixed_digits)
            elif method == "Secant":
                x0 = float(self.param1_input.text())
                x1 = float(self.param2_input.text())
                results = self.numerical_methods.secant(func, x0, x1, tolerance, max_iter, fixed_digits)
            else:  # Simple Iteration
                x0 = float(self.param1_input.text())
                results = self.numerical_methods.simple_iteration(func, x0, tolerance, max_iter, fixed_digits)

            self.progress_bar.setValue(60)
            QApplication.processEvents()

            if results:
                self.display_results(results, method)
                self.plot_function(func, results, method)
                QMessageBox.information(self, "Success", f"Solution found in {len(results)} iterations!")

            self.progress_bar.setValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))

        except (ValueError, TypeError, NameError) as e:
            QMessageBox.critical(self, "Error", str(e))
            self.progress_bar.setVisible(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")
            self.progress_bar.setVisible(False)

    def display_results(self, results: List[tuple], method: str):
        self.results_table.setRowCount(0)
        for row_data in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            if method == "Bisection":
                iter_num, a, b, x, fx = row_data
                data = [str(iter_num), f"{a:.6f}", f"{b:.6f}", f"{x:.6f}", f"{fx:.6f}"]
            elif method == "Secant":
                iter_num, x0, x1, x_next, fx = row_data
                data = [str(iter_num), f"{x0:.6f}", f"{x1:.6f}", f"{x_next:.6f}", f"{fx:.6f}"]
            elif method == "Simple Iteration":
                iter_num, x, x_next = row_data
                data = [str(iter_num), f"{x:.6f}", "-", f"{x_next:.6f}", "-"]
            for col, value in enumerate(data):
                self.results_table.setItem(row, col, QTableWidgetItem(value))

    def plot_function(self, func: Callable[[float], float], results: List[tuple], method: str):
        self.ax.clear()
        x_values = []
        if method == "Bisection":
            x_values = [result[1] for result in results] + [result[2] for result in results]
        elif method == "Secant":
            x_values = [result[1] for result in results] + [result[2] for result in results]
        elif method == "Simple Iteration":
            x_values = [result[1] for result in results]

        x_min, x_max = min(x_values), max(x_values)
        margin = (x_max - x_min) * 0.5
        x = np.linspace(x_min - margin, x_max + margin, 1000)
        y = [func(xi) for xi in x]

        self.ax.plot(x, y, 'b-', label='f(x)')
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        if results:
            last_x = results[-1][3]
            self.ax.plot(last_x, func(last_x), 'ro', label='Root')

        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_title(f'Function Plot with {method} Method')
        self.plot_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NonLinearSolverApp()
    window.show()
    sys.exit(app.exec())
