import numpy as np

class NeuralNetwork_UP:
    def __init__(
        self,
        layer_sizes,               # Ex: [input_size, hidden1, hidden2, ..., output_size]
        activation_functions,      # Ex: ['relu', 'tanh', 'sigmoid']
        loss_fn='binary_crossentropy',
        optimizer='adam',
        lr=0.001,
        dropout_rate=0.0,
        l2_lambda=0.01,
        batch_size=32,
        early_stopping_patience=3
    ):
        # Inicialização da arquitetura
        self.layer_sizes = layer_sizes
        
        # Ensure we have the right number of activation functions
        if len(activation_functions) != len(layer_sizes) - 1:
            raise ValueError(f"Expected {len(layer_sizes)-1} activation functions, got {len(activation_functions)}")
        
        self.activations = activation_functions
        self.loss_fn = loss_fn
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        
        # Inicialização dos parâmetros
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2./layer_sizes[i])  # He initialization
            )
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        
        # Configuração do otimizador
        self.optimizer = optimizer.lower()
        if self.optimizer in ['adam', 'rmsprop']:
            self._init_optimizer_params()

        # Históricos
        self.loss_history = {'train': [], 'val': []}
        self.accuracy_history = {'train': [], 'val': []}

    def _init_optimizer_params(self):
        # Parâmetros para Adam/RMSprop
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def _activation(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to prevent overflow
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        return Z

    def _activation_deriv(self, Z, activation):
        if activation == 'relu':
            return (Z > 0).astype(float)
        elif activation == 'sigmoid':
            s = self._activation(Z, 'sigmoid')
            return s * (1 - s)
        elif activation == 'tanh':
            return 1 - np.tanh(Z)**2
        return np.ones_like(Z)

    def forward(self, X, training=True):
        self.layer_outputs = [X]
        self.layer_z = []
        self.dropout_masks = []
        
        # Propagação através de todas as camadas
        for i in range(len(self.weights)):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self._activation(z, self.activations[i])
            
            # Aplicar dropout apenas em camadas ocultas durante o treino
            if training and (i < len(self.weights)-1) and (self.dropout_rate > 0):
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                a *= mask
                a /= (1.0 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                # Add empty mask for consistency in backprop
                if training and i < len(self.weights)-1:
                    self.dropout_masks.append(None)
            
            self.layer_z.append(z)
            self.layer_outputs.append(a)
            
        return self.layer_outputs[-1]

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevenir log(0)
        
        if self.loss_fn == 'binary_crossentropy':
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_fn == 'mse':
            loss = np.mean((y_true - y_pred)**2)
        elif self.loss_fn == 'categorical_crossentropy':
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Adicionar regularização L2
        if self.l2_lambda > 0:
            l2_penalty = sum(np.sum(w**2) for w in self.weights) * (self.l2_lambda / 2)
            loss += l2_penalty
            
        return loss

    def backward(self, X, y_true):
        m = y_true.shape[0]  # batch size
        gradients = []
        
        # Gradiente da função de perda
        if self.loss_fn == 'binary_crossentropy':
            delta = (self.layer_outputs[-1] - y_true) / m
        elif self.loss_fn == 'mse':
            delta = 2 * (self.layer_outputs[-1] - y_true) / m
        elif self.loss_fn == 'categorical_crossentropy':
            delta = (self.layer_outputs[-1] - y_true) / m
        
        # Retropropagação
        for i in reversed(range(len(self.weights))):
            # Gradientes dos pesos e biases
            dW = np.dot(self.layer_outputs[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Adicionar gradiente da regularização L2
            if self.l2_lambda > 0:
                dW += self.l2_lambda * self.weights[i]
            
            # Guardar gradientes
            gradients.append((dW, db))
            
            # Calcular delta para próxima camada (se não for a primeira camada)
            if i > 0:
                # This is the line where the error occurred before
                # We need to make sure the dimensions align correctly
                delta = np.dot(delta, self.weights[i].T)
                delta *= self._activation_deriv(self.layer_z[i-1], self.activations[i-1])
                
                # Aplicar máscara de dropout se existir
                if i-1 < len(self.dropout_masks) and self.dropout_masks[i-1] is not None:
                    delta *= self.dropout_masks[i-1]
        
        # Return gradients in correct order (from first to last layer)
        return list(reversed(gradients))

    def _update_weights(self, gradients, t):
        for i, (dW, db) in enumerate(gradients):
            if self.optimizer == 'adam':
                # Atualizar momentos
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dW
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (dW**2)
                
                # Correção de viés
                m_hat = self.m[i] / (1 - self.beta1**(t+1))
                v_hat = self.v[i] / (1 - self.beta2**(t+1))
                
                # Atualização
                self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.biases[i] -= self.lr * db
                
            else:  # SGD padrão
                self.weights[i] -= self.lr * dW
                self.biases[i] -= self.lr * db

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        best_val_loss = float('inf')
        no_improvement = 0
        
        # Convert labels to appropriate shape if they're not already
        if len(y_train.shape) == 1 or y_train.shape[1] == 1:
            if self.layer_sizes[-1] == 1:  # Binary classification
                y_train = y_train.reshape(-1, 1)
                if y_val is not None:
                    y_val = y_val.reshape(-1, 1)
        
        for epoch in range(epochs):
            # Mini-batches
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward e backward
                y_pred = self.forward(X_batch)
                gradients = self.backward(X_batch, y_batch)
                self._update_weights(gradients, epoch)
            
            # Cálculo das métricas
            train_pred = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, train_pred)
            self.loss_history['train'].append(train_loss)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_pred)
                self.loss_history['val'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                if no_improvement >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Logging
            if epoch % 10 == 0:
                if self.layer_sizes[-1] == 1:  # Binary classification
                    acc = np.mean((train_pred > 0.5).astype(int) == y_train)
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Acc = {acc:.4f}")
                    if X_val is not None:
                        val_acc = np.mean((val_pred > 0.5).astype(int) == y_val)
                        print(f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
                else:  # Multi-class
                    acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_train, axis=1))
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Acc = {acc:.4f}")
                    if X_val is not None:
                        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
                        print(f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    def predict(self, X):
        y_pred = self.forward(X, training=False)
        if self.layer_sizes[-1] == 1:  # Binary classification
            return (y_pred > 0.5).astype(int)
        else:  # Multi-class
            return np.argmax(y_pred, axis=1)