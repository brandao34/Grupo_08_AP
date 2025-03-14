import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, dropout_rate=0.0):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = lr
        self.dropout_rate = (
            dropout_rate  # Fraction of neurons to drop (0.0 means no dropout)
        )
        self.loss_history = []  # To monitor loss over epochs

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X, training=True):
        # First layer forward
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._relu(self.z1)

        # Apply dropout during training
        if training and self.dropout_rate > 0.0:
            # Create dropout mask: 1 means keep, 0 means drop.
            self.dropout_mask = (
                np.random.rand(*self.a1.shape) > self.dropout_rate
            ).astype(float)
            # Apply inverted dropout scaling so that no scaling is needed at test time.
            self.a1 *= self.dropout_mask
            self.a1 /= 1.0 - self.dropout_rate
        else:
            # If not training, no dropout mask is used.
            self.dropout_mask = None

        # Second layer forward
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self._sigmoid(self.z2)
        return self.output

    def compute_loss(self, y, output):
        # Binary cross-entropy loss with a small epsilon for numerical stability
        epsilon = 1e-15
        y = y.reshape(-1, 1)
        loss = -np.mean(
            y * np.log(output + epsilon) + (1 - y) * np.log(1 - output + epsilon)
        )
        return loss

    def backward(self, X, y, output):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        # Compute gradients for the output layer
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Backpropagate to the hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self._relu_deriv(self.z1)
        # If dropout was applied, propagate the mask
        if self.dropout_rate > 0.0 and self.dropout_mask is not None:
            dz1 *= self.dropout_mask

        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=1000, print_loss=True):
        for epoch in range(epochs):
            # Forward pass with dropout enabled
            output = self.forward(X, training=True)

            # Compute and store loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)

            # Optionally print loss and accuracy at intervals
            if print_loss and (epoch % 100 == 0 or epoch == epochs - 1):
                predictions = self.predict(X)
                accuracy = np.mean(predictions.flatten() == y.flatten())
                print(f"Epoch {epoch}: Loss = {loss}, Accuracy = {accuracy*100:.2f}%")

            # Backward pass and weight update
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        # During prediction, disable dropout by setting training=False
        output = self.forward(X, training=False)
        return (output > 0.5).astype(int)
