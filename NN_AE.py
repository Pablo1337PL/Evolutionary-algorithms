import numpy as np

from ActivationFunctions import Softmax


# ---------------------------------------------------------------------------
# Weight adapters: flatten / restore the weights of a MyNeuralNetwork instance
# so that an evolutionary algorithm can treat the whole network as a single
# real-valued vector (chromosome).
# ---------------------------------------------------------------------------
def layer_shapes(net):
    """List of (weight_shape, bias_shape) for every layer of the network."""
    shapes = []
    for i in range(len(net.layers) - 1):
        n_in, n_out = net.layers[i], net.layers[i + 1]
        shapes.append(((n_in, n_out), (n_out,)))
    return shapes


def count_params(net):
    """Total number of trainable parameters (weights + biases) = chromosome length."""
    total = 0
    for (w_shape, b_shape) in layer_shapes(net):
        total += w_shape[0] * w_shape[1] + b_shape[0]
    return total


def flatten_weights(net):
    """Flatten all weight matrices and bias vectors into one 1-D float vector."""
    parts = []
    for i in range(len(net.weights)):
        parts.append(np.asarray(net.weights[i], dtype=float).ravel())
        parts.append(np.asarray(net.biases[i], dtype=float).ravel())
    return np.concatenate(parts)


def set_weights_from_vector(net, vector):
    """Restore network weights/biases from a flat vector and apply them."""
    vector = np.asarray(vector, dtype=float)
    new_weights, new_biases = [], []
    offset = 0
    for (w_shape, b_shape) in layer_shapes(net):
        w_size = w_shape[0] * w_shape[1]
        b_size = b_shape[0]
        w = vector[offset:offset + w_size].reshape(w_shape)
        offset += w_size
        b = vector[offset:offset + b_size].reshape(b_shape)
        offset += b_size
        new_weights.append(w)
        new_biases.append(b)
    net.set_weights(new_weights, new_biases)


def nn_loss(net, X, y, task):
    """
    Objective value: error of the network on a data set.
      - classification -> mean cross-entropy (expects softmax output + one-hot y),
      - regression     -> mean squared error.
    """
    predictions = np.atleast_2d(net.forward(X))
    y = np.asarray(y, dtype=float).reshape(predictions.shape)
    if task == "classification":
        return float(-np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1)))
    return float(np.mean((predictions - y) ** 2))


class EvolutionaryNNTrainer:
    """
    Trains the weights of a (fixed-architecture) MyNeuralNetwork using an
    evolutionary algorithm instead of back-propagation.

    The chromosome is the flattened weight vector of the network; fitness is the
    training error (cross-entropy for classification, MSE for regression), which
    the algorithm minimizes. Standard operators are used: tournament selection,
    BLX-alpha (or single-point) crossover, gaussian mutation, and elitism.
    """

    def __init__(self, net, X_train, y_train, task="classification",
                 population_size=100, generations=300,
                 crossover_rate=0.8, mutation_rate=0.1, sigma=0.1, sigma_decay=1.0,
                 tournament_size=3, crossover="blx", blx_alpha=0.1,
                 init_std=0.5, elitism=1, seed=None):

        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if crossover not in ("blx", "one_point"):
            raise ValueError("crossover must be 'blx' or 'one_point'")

        self.net = net
        self.X_train = np.asarray(X_train, dtype=float)
        self.task = task

        y_train = np.asarray(y_train, dtype=float)
        if task == "regression":
            y_train = y_train.reshape(-1, net.output_size)
        self.y_train = y_train

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.tournament_size = tournament_size
        self.crossover = crossover
        self.blx_alpha = blx_alpha
        self.init_std = init_std
        self.elitism = max(1, elitism)

        self.rng = np.random.default_rng(seed)
        self.n_params = count_params(net)

        # Population of real-valued weight vectors (genotypes).
        self.population = self.rng.normal(0.0, self.init_std,
                                          (self.population_size, self.n_params))
        self.fitness_scores = np.zeros(self.population_size)

        self.best_vector = None
        self.best_loss = np.inf
        self.history = []

    # ------------------------------------------------------------------ helpers
    def _objective(self, vector):
        """Decode a chromosome into the network and return its training error."""
        set_weights_from_vector(self.net, vector)
        return nn_loss(self.net, self.X_train, self.y_train, self.task)

    def _evaluate(self, population):
        return np.array([self._objective(ind) for ind in population])

    def _tournament_selection(self):
        idx = self.rng.choice(self.population_size, self.tournament_size, replace=False)
        return self.population[idx[np.argmin(self.fitness_scores[idx])]]

    def _crossover_op(self, parent1, parent2):
        if self.rng.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.crossover == "blx":
            # BLX-alpha: sample each gene from an interval extended by alpha*range.
            lo = np.minimum(parent1, parent2)
            hi = np.maximum(parent1, parent2)
            d = hi - lo
            low = lo - self.blx_alpha * d
            high = hi + self.blx_alpha * d
            child1 = self.rng.uniform(low, high)
            child2 = self.rng.uniform(low, high)
            return child1, child2

        # Single-point crossover.
        point = self.rng.integers(1, self.n_params)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def _gaussian_mutation(self, child, sigma):
        mask = self.rng.random(self.n_params) < self.mutation_rate
        if np.any(mask):
            child = child.copy()
            child[mask] += self.rng.normal(0.0, sigma, np.count_nonzero(mask))
        return child

    # --------------------------------------------------------------------- main
    def run(self, verbose=False):
        """Run the evolution and return (best_vector, best_loss, history)."""
        self.fitness_scores = self._evaluate(self.population)
        sigma = self.sigma

        for gen in range(self.generations):
            order = np.argsort(self.fitness_scores)
            gen_best = self.fitness_scores[order[0]]
            if gen_best < self.best_loss:
                self.best_loss = gen_best
                self.best_vector = self.population[order[0]].copy()
            self.history.append(self.best_loss)

            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen [{gen + 1}/{self.generations}] best loss = {self.best_loss:.6f}")

            # Elitism: carry over the best individuals unchanged.
            new_population = [self.population[order[i]].copy() for i in range(self.elitism)]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover_op(parent1, parent2)

                new_population.append(self._gaussian_mutation(child1, sigma))
                if len(new_population) < self.population_size:
                    new_population.append(self._gaussian_mutation(child2, sigma))

            self.population = np.array(new_population)
            self.fitness_scores = self._evaluate(self.population)
            sigma *= self.sigma_decay

        # Make sure the very last population is reflected in the best result.
        final = np.argmin(self.fitness_scores)
        if self.fitness_scores[final] < self.best_loss:
            self.best_loss = self.fitness_scores[final]
            self.best_vector = self.population[final].copy()
        self.history.append(self.best_loss)

        return self.best_vector, self.best_loss, self.history

    def apply_best(self):
        """Load the best found weights into the network and return it."""
        if self.best_vector is None:
            raise RuntimeError("Call run() before apply_best().")
        set_weights_from_vector(self.net, self.best_vector)
        return self.net

    def predict(self, X):
        """Predict with the best found weights (class indices / regression values)."""
        self.apply_best()
        out = np.atleast_2d(self.net.forward(np.asarray(X, dtype=float)))
        if self.task == "classification":
            return np.argmax(out, axis=1)
        return out
