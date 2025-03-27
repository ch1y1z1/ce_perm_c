# 导入必要的库
from scipy.sparse import diags
import numpy as np
from ce_perm_c.mini_sci.linsolve import spsolve, sp_perm_c, sp_solve_with_perm_c

# 设置矩阵大小
n = 24000

# 生成复数随机数据作为对角线元素
main_diagonal = np.random.rand(n) + 1j * np.random.rand(n)  # 主对角线
upper_diagonal = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)  # 上对角线
lower_diagonal = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)  # 下对角线

# 创建三对角复数矩阵
matrix = diags(
    [main_diagonal, upper_diagonal, lower_diagonal],
    [0, 1, -1],  # 对角线的位置：0表示主对角线，1表示上对角线，-1表示下对角线
    shape=(n, n),
    format="csc",  # 使用压缩稀疏列（CSC）格式
)

# 生成复数随机向量 b
b = np.random.rand(n) + 1j * np.random.rand(n)

# 计算列置换向量
perm_c = sp_perm_c(matrix, b)


def test_spsolve():
    for _i in range(20):
        # 生成复数随机数据作为对角线元素
        main_diagonal = np.random.rand(n) + 1j * np.random.rand(n)  # 主对角线
        upper_diagonal = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)  # 上对角线
        lower_diagonal = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)  # 下对角线

        # 创建三对角复数矩阵
        matrix = diags(
            [main_diagonal, upper_diagonal, lower_diagonal],
            [0, 1, -1],  # 对角线的位置：0表示主对角线，1表示上对角线，-1表示下对角线
            shape=(n, n),
            format="csc",  # 使用压缩稀疏列（CSC）格式
        )
        # 使用常规方法求解线性方程组
        x = spsolve(matrix, b)

        # 使用带有列置换的方法求解线性方程组
        x_perm = sp_solve_with_perm_c(matrix, b, perm_c)

        # 比较两种方法的结果，确保它们在数值误差范围内相等
        np.testing.assert_allclose(x, x_perm)
