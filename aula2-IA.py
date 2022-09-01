# Created on Wed Aug 31 20:48:07 2022
# @author: vinicius-godoy

# https://github.com/joelgrus/data-science-from-scratch/blob/master/scratch/linear_algebra.py
from linear_algebra import dot

THRESHOLD = 0
LEARNING_RATE = 0.1
BIAS = -1
EXEC_CYCLES = 23

INPUT_TWO = [BIAS, 2, 2]
INPUT_FOUR = [BIAS, 4, 4]
EXPECTED_OUTPUT_TWO = [1]
EXPECTED_OUTPUT_FOUR = [0]

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

weights = [0.5441, -0.5562, 0.4074]

for i in range(EXEC_CYCLES):
    print("Cycle ", i + 1)
    weights, outputTwo = neuron_output(weights, INPUT_TWO, EXPECTED_OUTPUT_TWO)
    print(weights,  "output 2 = ", outputTwo)
    weights, outputFour = neuron_output(weights, INPUT_FOUR, EXPECTED_OUTPUT_FOUR)
    print(weights,  "output 4 = ", outputFour, "\n")

    if (outputTwo == EXPECTED_OUTPUT_TWO[0] and outputFour == EXPECTED_OUTPUT_FOUR[0]):
        print("Completed after ", i + 1, " cycles")
        break
