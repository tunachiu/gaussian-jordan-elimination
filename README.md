# gaussian-jordan-elimination
A code for Gaussian Jordan Elimination Algorithm for all cases
import numpy as np


def __interchange__(matrix, row1, row2):
    """Interchange two rows"""
    temp = matrix[row1].copy()
    matrix[row1] = matrix[row2]
    matrix[row2] = temp


def __multiply__(matrix, row, number):
    """Multiply a with a non-zero constant"""
    matrix[row] = matrix[row] * number


def __replace__(matrix, row1, row2, k):
    """Replace a row by the sum of that row and a multiple of another row."""
    temp = matrix[row2].copy()
    matrix[row1] = matrix[row1] + temp * k


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.x, self.y = self.matrix.shape

    def gauss_jordan_elimination(self):
        """Carry a matrix to the row echelon form"""
        self.matrix = self.matrix.astype(float)
        for i in range(self.x):  # check row-wise
            if np.all(self.matrix[i] == 0):
                __interchange__(self.matrix, i, -1)  # nếu dòng đó = 0 thì đổi chỗ dòng đó vs dòng cuối
            else:
                for j in range(self.y):  # check từng phần tử của row đó
                    if np.all(self.matrix[i:, j] == 0):
                        continue  # bỏ qua cột đó nếu các entry phía dưới đều đã bằng 0
                    else:
                        t = i + 1
                        while self.matrix[i, j] == 0:  # nếu pivot = 0 thì đổi chỗ vs dòng dưới
                            __interchange__(self.matrix, i, t)
                            t += 1
                    __multiply__(self.matrix, i, 1 / self.matrix[i, j])  # Nhân cả dòng i sao cho pivot = 1
                    for k in range(i + 1, self.x):  # Khử các entry ở dưới pivot về 0
                        ratio = - self.matrix[k, j] / self.matrix[i, j]
                        __replace__(self.matrix, k, i, ratio)
                    break

        """Carry a matrix to reduced row echelon form"""
        for i in range(self.x - 1, -1, -1):  # check theo dòng nhưng giật ngược từ dưới lên
            if np.all(self.matrix[i] == 0):
                continue  # nếu dòng đó = 0 thì bỏ qua
            else:
                for j in range(
                        self.y):  # check từng phần tử của row đó khi nào có phần tử = 1 thì thực hiện khử phía trên
                    if self.matrix[i, j] == 1:
                        for k in range(i - 1, -1, -1):  # Khử các entry ở trên pivot về 0
                            ratio = - self.matrix[k, j] / self.matrix[i, j]
                            __replace__(self.matrix, k, i, ratio)
                        break
        return self.matrix
