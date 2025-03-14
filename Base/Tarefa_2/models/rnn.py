import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, dropout_rate=0.0):
        """
        Initializes the RNN with given parameters.

        Parameters:
        - input_size: Number of features per time step.
        - hidden_size: Number of hidden units.
        - output_size: Number of output units (for binary classification, this is 1).
        - lr: Learning rate.
        - dropout_rate: Fraction of hidden units to drop during training (0.0 means no dropout).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.dropout_rate = dropout_rate

        # Weight initialization:
        # Input to hidden weights
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        # Hidden to hidden weights (recurrent connections)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Bias for hidden layer
        self.b_h = np.zeros((1, hidden_size))
        # Hidden to output weights
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        # Bias for output layer
        self.b_y = np.zeros((1, output_size))

        self.loss_history = []  # To store loss values over epochs

    def _tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def _tanh_deriv(self, x):
        """Derivative of tanh activation function: 1 - tanh(x)^2."""
        return 1 - np.tanh(x) ** 2

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def forward(self, X, training=True):
        """
        Forward pass through the RNN.

        Parameters:
        - X: Input array with shape (batch_size, seq_length, input_size)
        - training: Boolean flag indicating whether we are training (enables dropout)

        Returns:
        - output: Final predictions with shape (batch_size, output_size)
        """
        batch_size, seq_length, _ = X.shape
        # Initialize arrays to store hidden states and pre-activation values
        self.h = np.zeros((batch_size, seq_length, self.hidden_size))
        self.z = np.zeros((batch_size, seq_length, self.hidden_size))

        # Initialize previous hidden state as zeros
        h_prev = np.zeros((batch_size, self.hidden_size))

        # Process each time step
        for t in range(seq_length):
            # Linear combination for hidden state at time t
            self.z[:, t, :] = (
                np.dot(X[:, t, :], self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            )
            h_current = self._tanh(self.z[:, t, :])

            # Apply dropout to the hidden state if in training mode
            if training and self.dropout_rate > 0.0:
                # Create dropout mask: 1 indicates keep neuron, 0 indicates drop
                dropout_mask = (
                    np.random.rand(*h_current.shape) > self.dropout_rate
                ).astype(float)
                h_current *= dropout_mask
                # Inverted dropout scaling to maintain the expected value at test time
                h_current /= 1.0 - self.dropout_rate

            self.h[:, t, :] = h_current
            h_prev = h_current

        # Use the last hidden state to compute the output
        self.output_linear = np.dot(h_prev, self.W_hy) + self.b_y
        self.output = self._sigmoid(self.output_linear)
        return self.output

    def compute_loss(self, y, output):
        """
        Computes binary cross-entropy loss.

        Parameters:
        - y: True labels with shape (batch_size,)
        - output: Predicted outputs with shape (batch_size, 1)

        Returns:
        - loss: Average binary cross-entropy loss over the batch
        """
        epsilon = 1e-15  # Small constant for numerical stability
        y = y.reshape(-1, 1)
        loss = -np.mean(
            y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon)
        )
        return loss

    def backward(self, X, y, output):
        """
        Performs backpropagation through time (BPTT) to compute gradients.

        Parameters:
        - X: Input array with shape (batch_size, seq_length, input_size)
        - y: True labels with shape (batch_size,)
        - output: Predicted outputs from the forward pass

        Returns:
        - dW_xh: Gradient for input-to-hidden weights
        - dW_hh: Gradient for hidden-to-hidden weights
        - db_h: Gradient for hidden biases
        - dW_hy: Gradient for hidden-to-output weights
        - db_y: Gradient for output biases
        """
        batch_size, seq_length, _ = X.shape
        y = y.reshape(-1, 1)

        # Gradients for output layer
        d_output = output - y  # (batch_size, 1)
        dW_hy = np.dot(self.h[:, -1, :].T, d_output) / batch_size
        db_y = np.sum(d_output, axis=0, keepdims=True) / batch_size

        # Initialize gradients for recurrent weights and biases
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        # Gradient from the output layer to the last hidden state
        dh = np.dot(d_output, self.W_hy.T)  # (batch_size, hidden_size)
        dh_next = np.zeros((batch_size, self.hidden_size))

        # Backpropagation through time (iterate backwards over time steps)
        for t in reversed(range(seq_length)):
            # Total gradient for the hidden state at time t
            dh_total = dh + dh_next
            dz = dh_total * self._tanh_deriv(self.z[:, t, :])

            # Gradients for the input-to-hidden weights
            dW_xh += np.dot(X[:, t, :].T, dz) / batch_size

            # Determine the previous hidden state (zero if t == 0)
            h_prev = (
                self.h[:, t - 1, :]
                if t > 0
                else np.zeros((batch_size, self.hidden_size))
            )
            dW_hh += np.dot(h_prev.T, dz) / batch_size
            db_h += np.sum(dz, axis=0, keepdims=True) / batch_size

            # Propagate gradient to previous time step
            dh_next = np.dot(dz, self.W_hh.T)

        return dW_xh, dW_hh, db_h, dW_hy, db_y

    def train(self, X, y, epochs=1000, print_loss=True):
        """
        Trains the RNN over a specified number of epochs.

        Parameters:
        - X: Input array with shape (batch_size, seq_length, input_size)
        - y: Labels with shape (batch_size,)
        - epochs: Number of training iterations
        - print_loss: Flag to print loss and accuracy at intervals
        """
        for epoch in range(epochs):
            # Forward pass (with dropout enabled during training)
            output = self.forward(X, training=True)

            # Compute and store loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)

            # Optionally print loss and accuracy every 100 epochs (or on the final epoch)
            if print_loss and (epoch % 100 == 0 or epoch == epochs - 1):
                predictions = self.predict(X)
                accuracy = np.mean(predictions.flatten() == y.flatten())
                print(f"Epoch {epoch}: Loss = {loss}, Accuracy = {accuracy*100:.2f}%")

            # Backward pass and weight update
            dW_xh, dW_hh, db_h, dW_hy, db_y = self.backward(X, y, output)
            self.W_xh -= self.lr * dW_xh
            self.W_hh -= self.lr * dW_hh
            self.b_h -= self.lr * db_h
            self.W_hy -= self.lr * dW_hy
            self.b_y -= self.lr * db_y

    def predict(self, X):
        """
        Predicts the output for a given input.

        Parameters:
        - X: Input array with shape (batch_size, seq_length, input_size)

        Returns:
        - Binary predictions with shape (batch_size, output_size)
        """
        # Disable dropout during prediction
        output = self.forward(X, training=False)
        return (output > 0.5).astype(int)
