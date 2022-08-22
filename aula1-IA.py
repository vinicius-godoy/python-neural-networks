# Created on Wed Aug 17 22:50:07 2022
# @author: vinicius-godoy

# https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/linear_algebra.py
from linear_algebra import dot

# AND
# 0 * 1 + 0 * 1 = 0 / 0
# 0 * 1 + 1 * 1 = 1 / 0
# 1 * 1 + 0 * 1 = 1 / 0
# 1 * 1 + 1 * 1 = 2 / 1

# OR
# 0 * 1 + 0 * 1 = 0 / 0
# 0 * 1 + 1 * 1 = 1 / 1
# 1 * 1 + 0 * 1 = 1 / 1
# 1 * 1 + 1 * 1 = 2 / 1

# NAND
# 0 * -1 + 0 * -1 =  0 / 1
# 0 * -1 + 1 * -1 = -1 / 1
# 1 * -1 + 0 * -1 = -1 / 1
# 1 * -1 + 1 * -1 = -2 / 0

# NOR
# 0 * -1 + 0 * -1 =  0 / 1
# 0 * -1 + 1 * -1 = -1 / 0
# 1 * -1 + 0 * -1 = -1 / 0
# 1 * -1 + 1 * -1 = -2 / 0

THRESHOLD_AND = 1.5
THRESHOLD_OR = 0.5
THRESHOLD_NAND = -THRESHOLD_AND
THRESHOLD_NOR = -THRESHOLD_OR

def step_function (x, boolFunction):
    if (boolFunction == 'AND'):
        return 1 if x >= THRESHOLD_AND else 0
    elif (boolFunction == 'OR'):
        return 1 if x >= THRESHOLD_OR else 0
    elif (boolFunction == 'NAND'):
        return 1 if x >= THRESHOLD_NAND else 0
    elif (boolFunction == 'NOR'):
        return 1 if x >= THRESHOLD_NOR else 0
    pass

def perceptron_output (weights, bias, x, boolFunction):
    calculation = dot(weights, x) + bias
    return step_function(calculation, boolFunction)

x0 = [0, 0]
x1 = [0, 1]
x2 = [1, 0]
x3 = [1, 1]

weights = [1, 1]
weights2 = [-1, -1]
bias = 0

print("PERCEPTRON BOOLEAN AND")
print("0 AND 0 = ", perceptron_output(weights, bias, x0, 'AND'))
print("0 AND 1 = ", perceptron_output(weights, bias, x1, 'AND'))
print("1 AND 0 = ", perceptron_output(weights, bias, x2, 'AND'))
print("1 AND 1 = ", perceptron_output(weights, bias, x3, 'AND'))

print("PERCEPTRON BOOLEAN OR")
print("0 OR 0 = ", perceptron_output(weights, bias, x0, 'OR'))
print("0 OR 1 = ", perceptron_output(weights, bias, x1, 'OR'))
print("1 OR 0 = ", perceptron_output(weights, bias, x2, 'OR'))
print("1 OR 1 = ", perceptron_output(weights, bias, x3, 'OR'))

print("PERCEPTRON BOOLEAN NAND")
print("0 NAND 0 = ", perceptron_output(weights2, bias, x0, 'NAND'))
print("0 NAND 1 = ", perceptron_output(weights2, bias, x1, 'NAND'))
print("1 NAND 0 = ", perceptron_output(weights2, bias, x2, 'NAND'))
print("1 NAND 1 = ", perceptron_output(weights2, bias, x3, 'NAND'))

print("PERCEPTRON BOOLEAN NOR")
print("0 NOR 0 = ", perceptron_output(weights2, bias, x0, 'NOR'))
print("0 NOR 1 = ", perceptron_output(weights2, bias, x1, 'NOR'))
print("1 NOR 0 = ", perceptron_output(weights2, bias, x2, 'NOR'))
print("1 NOR 1 = ", perceptron_output(weights2, bias, x3, 'NOR'))
