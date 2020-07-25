from typing import List, Callable
from nptyping import NDArray, Float64
import numpy as np

# def foo(array: NDArray[Float64]) -> str:
class Neuron:
    def __init__(self, 
    weights: NDArray[Float64], 
    learning_rate: float, 
    activation_fx: Callable[[float], float], 
    activation_dx: Callable[[float], float]) -> None:
        self.weights: NDArray[Float64] = weights
        self.activation_fx: Callable[[float], float] = activation_fx
        self.activation_dx: Callable[[float], float] = activation_dx
        self.learning_rate: float = learning_rate
        self.output_cache: float = 0.0
        self.error: float = 0.0

    def output(self, inputs: NDArray[Float64]) -> float:
        self.output_cache = np.dot(inputs, np.transpose(self.weights))
        return self.activation_fx(self.output_cache)
