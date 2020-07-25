from __future__ import annotations
from typing import List, Callable, Optional
from neuron import Neuron
from nptyping import NDArray, Float64
import numpy as np

class Layer:
    def __init__(self, 
    previous_layer: Optional[Layer], 
    num_neurons: int,
    learning_rate: float, 
    activation_fx: Callable[[float], float],
    activation_dx: Callable[[float], float]) -> None:
        self.previous_layer: Optional[Layer] = previous_layer
        self.neurons: List[Neuron] = []
        
        # the following could all be one large list comprehension 
        #for i in range(num_neurons):
        #    if previous_layer is None:
        #        random_weights: List[float] = []
        #    else:
        #        random_weights = [random() for _ in range(len(previous_layer.neurons))]
        #    neuron: Neuron = Neuron(random_weights, learning_rate, activation_fx, activation_dx)
        #    self.neurons.append(neuron)
        #self.output_cache: List[float] = [0.0 for _ in range(num_neurons)]

        # Creates random weights and populated N
        # Need to experiment with diff values for weights - 0.5 may get us half way there
        for i in range(num_neurons):
            random_weights: NDArray[Float64] = [np.random.rand(len(previous_layer.neurons))]
            neuron: Neuron = Neuron(random_weights, learning_rate, activation_fx, activation_dx)
            self.neurons.append(neuron)
        self.output_cache: NDArray[Float64] = np.zeros(num_neurons)


    # Updates the Output value on each Neuron
    def outputs(self, inputs: NDArray[Float64]) -> NDArray[Float64]:
        if self.previous_layer is None:
            self.output_cache: NDArray[Float64] = inputs
        else:
            self.output_cache: NDArray[Float64] = [n.output(inputs) for n in self.neurons]
        return self.output_cache


    # Should only be called on output layer
    # Updates error on each Neuron
    def error_output_layer(self, expected: NDArray[Float64] -> None:
        for n in range(len(self.neurons)):
            self.neurons[n].error = self.neurons[n].activation_dx(self.neurons[n].output_cache) * (expected[n] - self.output_cache[n])
        return

    # Should only be called on hidden layer
    # Updates error on each Neuron
    def error_hidden_layer(self, next_layer: Layer) -> None:
        for index, neuron in enumerate(self.neurons):
            next_weights: NDArray[Float64] = [n.weights[index] for n in next_layer.neurons]
            next_errors: NDArray[Float64] = [n.error for n in next_layer.neurons]
            sum_weights_and_errors: Float64 = np.dot(next_weights, np.transpose(next_errors))
            neuron.error: Float64 = neuron.activation_dx(neuron.output_cache) * sum_weights_and_errors
        return