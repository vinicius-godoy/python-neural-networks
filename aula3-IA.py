# Created on Wed Sep 09 20:46:07 2022
# @author: vinicius-godoy

# https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/linear_algebra.py
from linear_algebra import dot
import matplotlib.pyplot as plt

THRESHOLD = 0
LEARNING_RATE = 0.88 # Optimal learning rate for this exercise - 0.88
BIAS = -1
EXEC_CYCLES = 100

EXPECTED_OUTPUT_ZERO = [0]
EXPECTED_OUTPUT_ONE = [1]

def step_function (x):
    return 1 if x >= THRESHOLD else 0

def perceptron_output (weights, input):
    y = dot(weights, input)
    return step_function(y)

def neuron_output (synapses, input, output):
    partialOuput = perceptron_output(synapses, input)
    
    synapsesNum = len(synapses)
    for j in range(synapsesNum):
        synapses[j] = synapses[j] + LEARNING_RATE * (output[0] - partialOuput) * input[j]
        
    output = partialOuput
    return synapses, output

def generalization_test (synapses, input):
    partialOutput = perceptron_output(synapses, input)
    output = partialOutput
    return synapses, output

# Initial weights and inputs based by class
weights = [0.22, -0.33, 0.44]
pattern_zero = [
    [BIAS, 0.1, 0.1],
    [BIAS, 0.1, 0.5],
    [BIAS, 0.3, 0.3],
    [BIAS, 0.3, 0.0],
    [BIAS, 0.6, 0.3],
    [BIAS, 0.2, 0.6],
    [BIAS, 0.8, 0.1],
    [BIAS, 0.4, 0.4],
    [BIAS, 0.8, 0.1],
    [BIAS, 0.2, 0.6],
    [BIAS, 0.5, 0.3],
    [BIAS, 0.3, 0.4],
    [BIAS, 0.3, 0.5],
    [BIAS, 0.1, 0.7],
    [BIAS, 0.3, 0.5],
    [BIAS, 0.2, 0.5],
    [BIAS, 0.1, 0.5],
    [BIAS, 0.5, 0.1],
    [BIAS, 0.5, 0.2],
    [BIAS, 0.4, 0.1],
    [BIAS, 0.4, 0.0],
]
pattern_one = [
    [BIAS, 0.6, 0.6],
    [BIAS, 0.8, 0.2],
    [BIAS, 0.9, 0.5],
    [BIAS, 0.4, 0.7],
    [BIAS, 0.1, 0.9],
    [BIAS, 0.7, 0.3],
    [BIAS, 0.6, 0.4],
    [BIAS, 0.6, 0.5],
    [BIAS, 0.6, 0.7],
    [BIAS, 0.6, 0.8],
    [BIAS, 0.6, 0.9],
    [BIAS, 0.6, 1.0],
    [BIAS, 0.7, 0.3],
    [BIAS, 0.7, 0.4],
    [BIAS, 0.7, 0.5],
    [BIAS, 0.7, 0.6],
    [BIAS, 0.7, 0.7],
    [BIAS, 0.7, 0.8],
    [BIAS, 0.8, 0.8],
    [BIAS, 0.8, 0.9],
    [BIAS, 0.8, 1.0],
]

# Weights balacing
for i in range(EXEC_CYCLES):
    print("Cycle ", i + 1)

    rightCount = 0

    for pattern in pattern_zero:
        weights, outputZero = neuron_output(weights, pattern, EXPECTED_OUTPUT_ZERO)
        print(weights,  "output 0 = ", outputZero)
        if (outputZero == EXPECTED_OUTPUT_ZERO[0]):
            rightCount += 1

    for pattern in pattern_one:
        weights, outputOne = neuron_output(weights, pattern, EXPECTED_OUTPUT_ONE)
        print(weights,  "output 1 = ", outputOne)
        if (outputZero == EXPECTED_OUTPUT_ZERO[0]):
            rightCount += 1
    
    if (rightCount == len(pattern_zero) + len(pattern_one)):
        print("\nExecution Completed Successfully in ", i + 1, " cycles!\n")
        break

# Graph Plotting
thresholdX = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
thresholdY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

classZeroX = [0.1, 0.1, 0.3]
classZeroY = [0.1, 0.5, 0.3]
classOneX = [0.6, 0.8, 0.9]
classOneY = [0.6, 0.2, 0.5]

plt.scatter(classZeroX, classZeroY, color = 'green')
plt.scatter(classOneX, classOneY, color = 'red')

plt.plot(thresholdY, thresholdX, color = 'blue', marker = '.', linestyle = '-')
plt.title("Perceptron classification")

plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()

# Generalization Tests
tests = [
    [BIAS, 0.2, 0.4],
    [BIAS, 0.7, 0.8],
    [BIAS, 0.6, 0.3],
    [BIAS, 0.1, 0.9],
    [BIAS, 0.2, 0.6],
    [BIAS, 0.8, 0.1],
]

print("Generalization Tests")
for pattern in tests:
    weights, outputTest = generalization_test(weights, pattern)
    print(weights, " output = ", outputTest)
