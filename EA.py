import numpy as np

class EvolutionaryAlgorithm:
    """
    A class implementing an Evolutionary Algorithm for minimizing 
    an unknown multi-dimensional objective function.
    """
    def __init__(self, objective_function, dimensions, bounds, population_size=100, 
                 generations=200, crossover_rate=0.8, mutation_rate=0.2, sigma=0.1, tournament_size=3):
        
        self.objective_function = objective_function # The function we want to optimize
        self.dimensions = dimensions                 # Number of dimensions (variables)
        self.bounds = bounds                         # Tuple (min, max) defining the search space
        self.population_size = population_size       # Number of individuals in the population
        self.generations = generations               # Stopping condition (number of iterations)
        self.crossover_rate = crossover_rate         # Probability of crossover
        self.mutation_rate = mutation_rate           # Probability of mutation
        self.sigma = sigma                           # Mutation strength (standard deviation)
        self.tournament_size = tournament_size       # Number of individuals in a single tournament
        
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], 
                                           (self.population_size, self.dimensions))
        self.fitness_scores = np.zeros(self.population_size)

    def _evaluate(self, population):
        """Calculates the fitness value for all individuals in the population."""
        return np.array([self.objective_function(individual) for individual in population])

    def _tournament_selection(self):
        indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        best_index = indices[np.argmin(self.fitness_scores[indices])]
        return self.population[best_index]

    def _averaging_crossover(self, parent1, parent2):
        """Averaging crossover (using a weighted average)."""
        if np.random.rand() < self.crossover_rate:
            weight = np.random.rand()
            child1 = weight * parent1 + (1 - weight) * parent2
            child2 = (1 - weight) * parent1 + weight * parent2
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _single_point_crossover(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dimensions - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _gaussian_mutation(self, child):
        """Adds a disturbance from a normal distribution to the individual's genes."""
        for i in range(self.dimensions):
            if np.random.rand() < self.mutation_rate:
                child[i] += np.random.normal(0, self.sigma)
        
        # Prevent the individual from leaving the allowed bounds
        return np.clip(child, self.bounds[0], self.bounds[1])

    def run(self):
        self.fitness_scores = self._evaluate(self.population)
        
        best_history = []

        for gen in range(self.generations):
            new_population = []
            
            best_index = np.argmin(self.fitness_scores)
            best_individual = self.population[best_index]
            best_score = self.fitness_scores[best_index]
            
            new_population.append(best_individual.copy())
            best_history.append(best_score)

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
        
                child1, child2 = self._single_point_crossover(parent1, parent2)
                
                child1 = self._gaussian_mutation(child1)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    child2 = self._gaussian_mutation(child2)
                    new_population.append(child2)
            
            self.population = np.array(new_population)
            self.fitness_scores = self._evaluate(self.population)

        final_best_index = np.argmin(self.fitness_scores)
        return self.population[final_best_index], self.fitness_scores[final_best_index], best_history
