import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTableWidget, QTableWidgetItem, QComboBox, 
                            QMessageBox, QGridLayout, QGroupBox, QDialog,
                            QTextEdit)
import numpy as np
from typing import Callable, List, Tuple
from dataclasses import dataclass

@dataclass
class Function:
    name: str
    func: Callable
    deriv: Callable
    default_x0: float

class FunctionInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Custom Function")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        self.name_input = QLineEdit()
        layout.addWidget(QLabel("Function Name:"))
        layout.addWidget(self.name_input)
        
        self.func_input = QTextEdit()
        layout.addWidget(QLabel("Function Expression (use 'x' as variable):"))
        layout.addWidget(self.func_input)
        
        self.deriv_input = QTextEdit()
        layout.addWidget(QLabel("Derivative Expression (use 'x' as variable):"))
        layout.addWidget(self.deriv_input)
        
        self.x0_input = QLineEdit()
        layout.addWidget(QLabel("Default x₀:"))
        layout.addWidget(self.x0_input)
        
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        layout.addLayout(button_box)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

class NewtonRaphsonSolver:
    @staticmethod
    def solve(func: Callable, dfunc: Callable, x0: float, 
             tolerance: float = 1e-6, max_iter: int = 100) -> List[Tuple]:
        x = float(x0)
        results = []
        
        for i in range(max_iter):
            try:
                fx = func(x)
                dfx = dfunc(x)
                
                if abs(dfx) < 1e-10:
                    if abs(fx) < tolerance:
                        results.append((i + 1, x, fx, 0.0))
                        return results
                    dfx = dfx + np.sign(dfx) * 1e-10 if dfx != 0 else 1e-10
                
                x_next = x - fx / dfx
                error = abs(x_next - x)
                
                results.append((i + 1, x, fx, error))
                
                if abs(fx) < tolerance or error < tolerance:
                    return results
                
                if abs(x_next) > 1e6:
                    raise ValueError("Method diverging - try different initial guess")
                    
                x = x_next
                
            except Exception as e:
                raise ValueError(f"Error in iteration {i + 1}: {str(e)}")
                
        raise ValueError(f"Failed to converge within {max_iter} iterations")

class NewtonRaphsonGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Newton-Raphson Method Solver")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        self.functions = [
            Function("x² - 4", lambda x: x**2 - 4, lambda x: 2*x, 1.0),
            Function("sin(x)", lambda x: np.sin(x), lambda x: np.cos(x), 0.0),
            Function("e^x - 3", lambda x: np.exp(x) - 3, lambda x: np.exp(x), 1.0)
        ]
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout()
        
        self.function_combo = QComboBox()
        for func in self.functions:
            self.function_combo.addItem(func.name)
        input_layout.addWidget(QLabel("Select Function:"), 0, 0)
        input_layout.addWidget(self.function_combo, 0, 1)
        
        self.x0_input = QLineEdit()
        input_layout.addWidget(QLabel("Initial Guess (x₀):"), 1, 0)
        input_layout.addWidget(self.x0_input, 1, 1)
        
        self.tolerance_input = QLineEdit("1e-6")
        input_layout.addWidget(QLabel("Tolerance:"), 2, 0)
        input_layout.addWidget(self.tolerance_input, 2, 1)
        
        self.max_iter_input = QLineEdit("100")
        input_layout.addWidget(QLabel("Max Iterations:"), 3, 0)
        input_layout.addWidget(self.max_iter_input, 3, 1)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        self.add_function_button = QPushButton("Add Function")
        self.add_function_button.clicked.connect(self.add_function)
        main_layout.addWidget(self.add_function_button)
        
        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_equation)
        self.solve_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        main_layout.addWidget(self.solve_button)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Iteration", "x", "f(x)", "Error"])
        self.table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table)

    def add_function(self):
        dialog = FunctionInputDialog(self)
        if dialog.exec():
            try:
                name = dialog.name_input.text()
                func_expr = dialog.func_input.toPlainText()
                deriv_expr = dialog.deriv_input.toPlainText()
                default_x0 = float(dialog.x0_input.text())
                
                func = eval(f"lambda x: {func_expr}")
                deriv = eval(f"lambda x: {deriv_expr}")
                
                test_x = 1.0
                func(test_x)
                deriv(test_x)
                
                self.functions.insert(0, Function(name, func, deriv, default_x0))
                self.function_combo.insertItem(0, name)
                self.function_combo.setCurrentIndex(0)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Invalid function definition: {str(e)}")

    def solve_equation(self):
        try:
            selected_func = self.functions[self.function_combo.currentIndex()]
            x0 = float(self.x0_input.text())
            tolerance = float(self.tolerance_input.text())
            max_iter = int(self.max_iter_input.text())
            
            results = NewtonRaphsonSolver.solve(
                selected_func.func,
                selected_func.deriv,
                x0,
                tolerance,
                max_iter
            )
            
            self.table.setRowCount(len(results))
            for i, (iter_num, x, fx, error) in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(f"{iter_num}"))
                self.table.setItem(i, 1, QTableWidgetItem(f"{x:.8f}"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{fx:.8f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{error:.8f}"))
                
            final_x = results[-1][1]
            QMessageBox.information(
                self,
                "Solution Found",
                f"Root found: x = {final_x:.8f}\n"
                f"f(x) = {results[-1][2]:.8f}\n"
                f"Iterations: {len(results)}"
            )
            
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = NewtonRaphsonGUI()
    window.show()
    sys.exit(app.exec())
