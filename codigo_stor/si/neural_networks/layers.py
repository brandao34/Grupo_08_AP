#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__
    

class DenseLayer (Layer):
    
    def __init__(self, n_units, input_shape = None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        
    def initialize(self, optimizer):
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = None   ## COMPLETE 
        ## multiplicação do x pelo peso menos o biase
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
         # computes the layer input error (the output error from the previous layer),
         # dE/dX, to pass on to the previous layer
         # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
         # input_error = None  ## COMPLETE
         input_error = np.dot(output_error, self.weights.T)
    
         # computes the weight error: dE/dW = X.T * dE/dY
         # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
         # weights_error = None ## COMPLETE
         weights_error = np.dot(self.input.T, output_error)
         # computes the bias error: dE/dB = dE/dY
         # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
         bias_error = np.sum(output_error, axis=0, keepdims=True)
    
         # updates parameters
         self.weights = self.w_opt.update(self.weights, weights_error)
         self.biases = self.b_opt.update(self.biases, bias_error)
         return input_error
 
    def output_shape(self):
         return (self.n_units,) 

    
