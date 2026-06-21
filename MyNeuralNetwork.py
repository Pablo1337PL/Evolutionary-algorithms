import numpy as np
from ActivationFunctions import ActivationFunction, Sigmoid, Linear, Softmax


class MyNeuralNetwork:
    def __init__(self, 
                input_size: int, hidden_layers: list[int] = None, output_size: int = 1, 
                activation_function: ActivationFunction = None,
                last_activation_function: ActivationFunction = None,
                RANDOM_SEED: int = None):
        """
        hidden_layers: None jeśli nie warstw ukrytych, w przeciwnym razie lista z liczbą neuronów w każdej warstwie ukrytej
        """
        self.beta1 = 0.9   # Zanikanie dla Momentum i 1. momentu Adam
        self.beta2 = 0.999 # Zanikanie dla RMSProp i 2. momentu Adam
        self.epsilon = 1e-8 # Ochrona przed dzieleniem przez 0
        
        
        self.RANDOM_SEED = RANDOM_SEED
        if self.RANDOM_SEED is not None:
            np.random.seed(self.RANDOM_SEED)

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        if activation_function is None:
            self.activation_function = Sigmoid()
        else:
            self.activation_function = activation_function

        if last_activation_function is None:
            self.last_activation_function = Linear()
        else:
            self.last_activation_function = last_activation_function
        

        if hidden_layers is None:
            self.layers = [input_size, output_size]
        else:
            self.layers = [input_size] + hidden_layers + [output_size]

        if len(self.layers) < 2:
            raise ValueError("Sieć musi mieć co najmniej dwie warstwy (wejściową i wyjściową).")

        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            weight_matrix = [[np.random.uniform(-1, 1) for _ in range(self.layers[i + 1])] for _ in range(self.layers[i])]
            self.weights.append(weight_matrix)
        
        self.biases = [[np.random.uniform(-1, 1) for _ in range(self.layers[i + 1])] for i in range(len(self.layers) - 1)]

    def set_weights(self, weights, biases):
        if len(weights) != len(self.weights):
            raise ValueError("Liczba macierzy wag musi odpowiadać liczbie warstw minus jeden.")
        for i in range(len(weights)):
            if len(weights[i]) != self.layers[i] or len(weights[i][0]) != self.layers[i + 1]:
                raise ValueError(f"Macierz wag dla warstwy {i} ma nieprawidłowy rozmiar.")
        self.weights = weights
        self.biases = biases
    
    def save_weights(self, filename="NN_weights.txt"):
        with open(filename, mode="w", encoding="utf-8") as f:
            for w in range(len(self.weights)):
                for row in self.weights[w]:
                    f.write(" ".join(f"{weight:.8f}" for weight in row) + "\n")
                
                f.write(" ".join(f"{bias:.8f}" for bias in self.biases[w]) + "\n\n")
        print(f"Pomyślnie zapisano wagi do pliku: {filename}")


    def read_weights(self, filename="NN_weights.txt"):
        with open(filename, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            
        new_weights = []
        new_biases = []
        
        current_block = []
        for line in lines:
            line = line.strip()
            
            if line == "":
                if current_block:
                    w_lines = current_block[:-1]
                    w_matrix = [[float(val) for val in r.split()] for r in w_lines]
                    
                    b_line = current_block[-1]
                    b_vector = [float(val) for val in b_line.split()]
                    
                    new_weights.append(np.array(w_matrix))
                    new_biases.append(np.array(b_vector))
                    
                    current_block = []
            else:
                current_block.append(line)
                
        if current_block:
            w_matrix = [[float(val) for val in r.split()] for r in current_block[:-1]]
            b_vector = [float(val) for val in current_block[-1].split()]
            new_weights.append(np.array(w_matrix))
            new_biases.append(np.array(b_vector))
            
        if len(new_weights) != len(self.weights):
            raise ValueError(f"Błąd struktury: Plik zawiera {len(new_weights)} warstw, a sieć oczekuje {len(self.weights)}.")
            
        self.weights = new_weights
        self.biases = new_biases
        print(f"Pomyślnie załadowano wagi z pliku: {filename}")


    def forward(self, data, dropout_masks=None):
        activations = data.copy()
        
        for w in range(len(self.weights)):
            weight_matrix = self.weights[w]
            
            weighted_sum = np.dot(activations, weight_matrix) + self.biases[w] # X^T * W + b
            # analogia do regresji liniowej : x_i ^ T * beta + b
            if w != len(self.weights) - 1: 
                activations = self.activation_function.function(weighted_sum)
                if dropout_masks is not None and dropout_masks[w] is not None:
                    activations = activations * dropout_masks[w] 
            else:
                activations = self.last_activation_function.function(weighted_sum)
            
        return activations
    
    def forward_all_activations(self, data, dropout_masks=None):
        # Wymuszamy macierz 2D (batch_size, features)
        activations = np.atleast_2d(data)
        
        all_activations = [activations] 
        
        for w in range(len(self.weights)):
            weighted_sum = np.dot(activations, self.weights[w]) + self.biases[w]
            
            if w != len(self.weights) - 1:  
                activations = self.activation_function.function(weighted_sum)
                # dropout na warstwie ukrytej
                if dropout_masks is not None and dropout_masks[w] is not None:
                    activations = activations * dropout_masks[w]
            else:
                activations = self.last_activation_function.function(weighted_sum)
            
            all_activations.append(activations)
            
        return all_activations


    def initialize_weights(self, weights_initiation_method='uniform'):
        if weights_initiation_method is None:
            return

        # Iterujemy po warstwach (indeks 'w')
        for w in range(len(self.weights)):
            w_array = np.array(self.weights[w])
            n_in = w_array.shape[0]  # liczba wierszy (wejść)
            n_out = w_array.shape[1]  # liczba kolumn (neuronów)
            
            if weights_initiation_method == 'uniform':
                new_weights = np.random.uniform(0, 1, size=(n_in, n_out))
                new_biases = np.random.uniform(0, 1, size=(n_out,))
                
            elif weights_initiation_method == 'he':
                std = np.sqrt(2.0 / n_in)
                new_weights = np.random.normal(0, std, size=(n_in, n_out))
                new_biases = np.random.normal(0, std, size=(n_out,))  # biases are 1D
                
            elif weights_initiation_method == 'xavier':
                std = np.sqrt(2.0 / (n_in + n_out))
                new_weights = np.random.normal(0, std, size=(n_in, n_out))
                new_biases = np.random.normal(0, std, size=(n_out,))  # biases are 1D
                
            else:
                raise ValueError("Nieznana metoda inicjalizacji")

            self.weights[w] = new_weights
            self.biases[w] = new_biases


    def back_propagation(self, error, activations, learning_rate, 
                         optimizer='sgd', m_w=None, m_b=None, v_w=None, v_b=None, t=0,
                         l1=False, l2=False, 
                         l1_lambda=0.0001, l2_lambda=0.0001, n_samples=1,
                         dropout_masks=None):
        
        batch_size = error.shape[0]
        delta = error * self.last_activation_function.derivative(activations[-1])
        L = len(self.weights)

        for w in range(L - 1, -1, -1):
            a_prev = activations[w]
            
            grad_w = np.dot(a_prev.T, delta) / batch_size
            grad_b = np.sum(delta, axis=0) / batch_size

            # Dodaj gradienty regularyzacyjne
            w_array = np.array(self.weights[w])
            if l1:
                grad_w += (l1_lambda / n_samples) * np.sign(w_array)
            if l2:
                grad_w += (2 * l2_lambda / n_samples) * w_array

            if w > 0:
                W = np.array(self.weights[w])
                delta = np.dot(delta, W.T) * self.activation_function.derivative(activations[w])
                
                if dropout_masks is not None and dropout_masks[w-1] is not None:
                    delta = delta * dropout_masks[w-1]

            if optimizer == 'adam':
                m_w[w] = self.beta1 * m_w[w] + (1 - self.beta1) * grad_w
                m_b[w] = self.beta1 * m_b[w] + (1 - self.beta1) * grad_b
                
                v_w[w] = self.beta2 * v_w[w] + (1 - self.beta2) * (grad_w ** 2)
                v_b[w] = self.beta2 * v_b[w] + (1 - self.beta2) * (grad_b ** 2)
                
                m_hat_w = m_w[w] / (1 - self.beta1 ** t)
                m_hat_b = m_b[w] / (1 - self.beta1 ** t)
                v_hat_w = v_w[w] / (1 - self.beta2 ** t)
                v_hat_b = v_b[w] / (1 - self.beta2 ** t)
                
                self.weights[w] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[w] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            elif optimizer == 'rmsprop':
                v_w[w] = self.beta2 * v_w[w] + (1 - self.beta2) * (grad_w ** 2)
                v_b[w] = self.beta2 * v_b[w] + (1 - self.beta2) * (grad_b ** 2)
                
                self.weights[w] -= learning_rate * grad_w / (np.sqrt(v_w[w]) + self.epsilon)
                self.biases[w] -= learning_rate * grad_b / (np.sqrt(v_b[w]) + self.epsilon)

            elif optimizer == 'momentum':
                m_w[w] = grad_w + m_w[w] * self.beta1
                m_b[w] = grad_b + m_b[w] * self.beta1

                self.weights[w] -= learning_rate * m_w[w]
                self.biases[w] -= learning_rate * m_b[w]

            elif optimizer == 'sgd':
                self.weights[w] -= learning_rate * grad_w
                self.biases[w] -= learning_rate * grad_b


    def train(self, X_train, y_train, 
              epochs, learning_rate=0.01, weights_initiation_method='uniform', 
              batch_size=32, optimizer='sgd', save_weights=False, verbose=False,
              early_stopping=False, X_val=None, y_val=None, patience=50, min_delta=1e-4,
              l1=False, l2=False, l1_lambda=0.0001, l2_lambda=0.0001, 
              dropout=False, dropout_rate=0.2):
        """
        Args:
            X_train, y_train: Dane treningowe
            epochs: Maksymalna liczba epok trenowania
            learning_rate: Learning rate
            weights_initiation_method: Metoda inicjalizacji wag
            batch_size: Rozmiar batcha
            optimizer: Optymalizator (sgd, momentum, rmsprop, adam)
            save_weights: Czy zapisać wagi
            verbose: Czy drukować informacje
            early_stopping: Czy włączyć early stopping
            X_val, y_val: Dane walidacyjne
            patience: Liczba epok bez poprawy zanim się zatrzymamy
            min_delta: Minimalna zmiana loss by liczyć to za poprawę
            l1: Czy włączyć regularyzację L1
            l2: Czy włączyć regularyzację L2
            dropout: Czy włączyć dropout
            l1_lambda: Siła regularyzacji L1 (default 0.0001)
            l2_lambda: Siła regularyzacji L2 (default 0.0001)
            dropout_rate: Procent neuronów do wyłączenia podczas dropout (default 0.5 = 50%)
        """
        
        self.initialize_weights(weights_initiation_method)
        
        optimizer = optimizer.lower()
        if optimizer not in ['sgd', 'momentum', 'rmsprop', 'adam']:
            raise ValueError("Nieznany optymalizator! Wybierz: 'sgd', 'momentum', 'rmsprop' lub 'adam'.")

        m_w = [np.zeros_like(w) for w in self.weights] if optimizer in ['momentum', 'adam'] else None
        m_b = [np.zeros_like(b) for b in self.biases] if optimizer in ['momentum', 'adam'] else None
        
        v_w = [np.zeros_like(w) for w in self.weights] if optimizer in ['rmsprop', 'adam'] else None
        v_b = [np.zeros_like(b) for b in self.biases] if optimizer in ['rmsprop', 'adam'] else None

        loss_history = []
        val_loss_history = []
        weights_history = []
        
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, self.output_size)
        
        if early_stopping or (X_val is not None and y_val is not None):
            early_stopping = True
            if X_val is None or y_val is None:
                raise ValueError("Dane walidacyjne (X_val, y_val) są wymagane przy włączonym early stopping.")
            X_val = np.array(X_val)
            y_val = np.array(y_val).reshape(-1, self.output_size)
        else:
            X_val = None
            y_val = None
        
        n_samples = len(X_train)
        
        if batch_size == 0 or batch_size > n_samples:
            batch_size = n_samples

        if save_weights:
            w_copy = [w.copy() for w in self.weights]
            b_copy = [b.copy() for b in self.biases]
            weights_history.append((w_copy, b_copy))

        # Zmienne dla early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        best_epoch = 0

        t = 0

        for epoch in range(epochs):
            epoch_loss = 0 
            
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                t += 1
                
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                
                dropout_masks = None
                if dropout:
                    dropout_masks = []
                    for layer_idx in range(len(self.weights) - 1):
                        mask_shape = (X_batch.shape[0], self.layers[layer_idx + 1])
                        mask = np.random.binomial(1, 1 - dropout_rate, size=mask_shape) / (1 - dropout_rate)
                        dropout_masks.append(mask)
                    dropout_masks.append(None)
                
                activations = self.forward_all_activations(X_batch, dropout_masks=dropout_masks)
                
                predictions = activations[-1]
                error = predictions - y_batch
                
                if isinstance(self.last_activation_function, Softmax):
                    epoch_loss += -np.sum(y_batch * np.log(predictions + 1e-8))
                    loss_type = "Cross-Entropy"
                else: 
                    epoch_loss += np.sum(error ** 2)
                    loss_type = "MSE"
                
                if l1 or l2:
                    for w in self.weights:
                        w_array = np.array(w)
                        if l1:
                            epoch_loss += l1_lambda * np.sum(np.abs(w_array))
                        if l2:
                            epoch_loss += l2_lambda * np.sum(w_array ** 2)
                
                self.back_propagation(error, activations, learning_rate, optimizer, 
                                      m_w, m_b, v_w, v_b, t,
                                      l1=l1, l2=l2, l1_lambda=l1_lambda, l2_lambda=l2_lambda,
                                      n_samples=n_samples, dropout_masks=dropout_masks)

            mean_loss = epoch_loss / n_samples
            loss_history.append(mean_loss)
            
            if early_stopping:
                val_predictions = self.forward(X_val, dropout_masks=None)
                if isinstance(self.last_activation_function, Softmax):
                    val_loss = -np.sum(y_val * np.log(val_predictions + 1e-8)) / len(X_val)
                else:
                    val_loss = np.sum((val_predictions - y_val) ** 2) / len(X_val)
                
                val_loss_history.append(val_loss)
                
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1
        
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly Stopping! Brak poprawy przez {patience} epok.")
                        print(f"Przywracanie wag z epoki {best_epoch}")
                    
                    if best_weights is not None:
                        self.weights = best_weights
                        self.biases = best_biases
                    break
            
            if (epoch + 1) % 100 == 0:
                if verbose:
                    if early_stopping:
                        print(f"Epoka [{epoch + 1}/{epochs}] - Train: {mean_loss:.6f}  {loss_type}, Val: {val_loss:.6f}")
                    else:
                        print(f"Epoka [{epoch + 1}/{epochs}] - Loss: {mean_loss:.6f}  {loss_type}")
                if save_weights:
                    w_copy = [w.copy() for w in self.weights]
                    b_copy = [b.copy() for b in self.biases]
                    weights_history.append((w_copy, b_copy))

        self.save_weights()
        return loss_history, val_loss_history, weights_history
