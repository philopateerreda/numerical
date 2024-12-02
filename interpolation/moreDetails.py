import numpy as np
from typing import List, Tuple
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QTableWidget,
                            QTableWidgetItem, QTextEdit, QLabel, QWidget)
from PyQt6.QtCore import Qt

class LagrangeDetails:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        self.x_points = x_points
        self.y_points = y_points
        self.n = len(x_points)
        
    def get_basis_polynomials(self) -> List[str]:
        basis_polynomials = []
        for i in range(self.n):
            terms = []
            for j in range(self.n):
                if i != j:
                    terms.append(f"(x - {self.x_points[j]:.2f})/({self.x_points[i]:.2f} - {self.x_points[j]:.2f})")
            basis_polynomials.append(" * ".join(terms))
        return basis_polynomials
    
    def get_full_polynomial(self) -> str:
        terms = []
        for i in range(self.n):
            basis = self.get_basis_polynomials()[i]
            terms.append(f"{self.y_points[i]:.2f} * ({basis})")
        return " + ".join(terms)
    
    def evaluate_basis_at_point(self, x: float) -> List[float]:
        basis_values = []
        for i in range(self.n):
            value = 1.0
            for j in range(self.n):
                if i != j:
                    value *= (x - self.x_points[j])/(self.x_points[i] - self.x_points[j])
            basis_values.append(value)
        return basis_values

class NewtonDetails:
    def __init__(self, x_points: np.ndarray, y_points: np.ndarray):
        self.x_points = x_points
        self.y_points = y_points
        self.n = len(x_points)
        self.divided_diff_table = self._compute_divided_difference_table()
    
    def _compute_divided_difference_table(self) -> np.ndarray:
        table = np.zeros((self.n, self.n))
        table[:,0] = self.y_points
        
        for j in range(1, self.n):
            for i in range(self.n - j):
                table[i,j] = (table[i+1,j-1] - table[i,j-1]) / (self.x_points[i+j] - self.x_points[i])
        
        return table
    
    def get_polynomial_terms(self) -> List[str]:
        terms = [f"{self.divided_diff_table[0,0]:.4f}"]
        product = ""
        
        for i in range(1, self.n):
            product += f"(x - {self.x_points[i-1]:.2f})"
            terms.append(f"{self.divided_diff_table[0,i]:.4f}{product}")
            
        return terms
    
    def get_formatted_table(self) -> List[List[str]]:
        headers = ["x", "f(x)"]
        for i in range(1, self.n):
            headers.append(f"f[{i}]")
            
        table_data = []
        for i in range(self.n):
            row = [f"{self.x_points[i]:.4f}"]
            for j in range(self.n-i):
                row.append(f"{self.divided_diff_table[i,j]:.4f}")
            table_data.append(row)
            
        return [headers] + table_data

class DetailsDialog(QDialog):
    def __init__(self, lagrange_details=None, newton_details=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interpolation Details")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        if lagrange_details:
            lagrange_tab = QWidget()
            lagrange_layout = QVBoxLayout(lagrange_tab)
            
            basis_label = QLabel("Lagrange Basis Polynomials:")
            basis_text = QTextEdit()
            basis_text.setReadOnly(True)
            basis_text.setPlainText("\n".join([
                f"L_{i}(x) = {poly}"
                for i, poly in enumerate(lagrange_details.get_basis_polynomials())
            ]))
            lagrange_layout.addWidget(basis_label)
            lagrange_layout.addWidget(basis_text)
            
            poly_label = QLabel("Complete Lagrange Polynomial:")
            poly_text = QTextEdit()
            poly_text.setReadOnly(True)
            poly_text.setPlainText(f"P(x) = {lagrange_details.get_full_polynomial()}")
            lagrange_layout.addWidget(poly_label)
            lagrange_layout.addWidget(poly_text)
            
            tabs.addTab(lagrange_tab, "Lagrange Details")
            
        if newton_details:
            newton_tab = QWidget()
            newton_layout = QVBoxLayout(newton_tab)
            
            table_label = QLabel("Divided Difference Table:")
            table = QTableWidget()
            formatted_table = newton_details.get_formatted_table()
            table.setRowCount(len(formatted_table)-1)
            table.setColumnCount(len(formatted_table[0]))
            table.setHorizontalHeaderLabels(formatted_table[0])
            
            for i, row in enumerate(formatted_table[1:]):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(i, j, item)
            
            newton_layout.addWidget(table_label)
            newton_layout.addWidget(table)
            
            poly_label = QLabel("Newton's Polynomial Terms:")
            poly_text = QTextEdit()
            poly_text.setReadOnly(True)
            poly_text.setPlainText(" + ".join(newton_details.get_polynomial_terms()))
            newton_layout.addWidget(poly_label)
            newton_layout.addWidget(poly_text)
            
            tabs.addTab(newton_tab, "Newton Details")
        
        layout.addWidget(tabs)
