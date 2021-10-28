import numpy as np
from main import Matrix

a = np.array([[0, 2, 3],
              [0, 0, 6],
              [7, 8, 9],
              [10, 11, 12]])
u = Matrix(a)
print(f"Row echelon form of a:\n{u.gauss_jordan_elimination()}")

c = np.array([[2, 4, -2],[4, 9, -3], [-2, -3, 7]])
c = Matrix(c)
print("\nMatrix test for Gauss Elimination: ")
print("Original matrix:\n", c.matrix)
print(f"Reduced row echelon form of c:\n{c.gauss_jordan_elimination()}")

d = np.array([[1, 1, 0, 0],
            [0, 1, 1, 4],
            [0, 0, 0, 1]])
d = Matrix(d)
print(f"Reduced row echelon form of d:\n{d.gauss_jordan_elimination()}")