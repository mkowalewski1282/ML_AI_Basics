import numpy as np
import matplotlib.pyplot as plt
import random

def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

class Chromosome:
    def __init__(self, length, array=None):
        if array is None:
            self.array = [random.randint(0, 1) for _ in range(length)]
        else:
            self.array = array

    def decode(self, lower_bound, upper_bound, aoi):
        decoded_values = []
        for i in range(0, len(self.array), len(aoi)):
            gene = self.array[i:i + len(aoi)]
            decoded_value = min_max_norm(int("".join(map(str, gene)), 2), 0, 2 ** len(gene) - 1, aoi[0], aoi[1])
            decoded_values.append(decoded_value)
        return decoded_values

    def mutation(self, probability):
        for i in range(len(self.array)):
            if random.random() < probability:
                self.array[i] = 1 - self.array[i]

    def crossover(self, other):
        point = random.randint(1, len(self.array) - 1)
        child1_array = self.array[:point] + other.array[point:]
        child2_array = other.array[:point] + self.array[point:]
        return Chromosome(len(self.array), child1_array), Chromosome(len(self.array), child2_array)

class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi, population_size=1000,
                 tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps
        self.population_size = population_size
        self.population = [Chromosome(chromosome_length) for _ in range(population_size)]
        self.best_solution = None

    def eval_objective_func(self, chromosome):
        decoded_values = chromosome.decode(0, self.chromosome_length, self.aoi)
        return self.objective_function(*decoded_values)

    def tournament_selection(self):
        selected_parents = []
        for _ in range(self.tournament_size):
            candidates = random.sample(self.population, self.tournament_size)
            selected = min(candidates, key=lambda chromosome: self.eval_objective_func(chromosome))
            selected_parents.append(selected)
        return selected_parents

    def reproduce(self, parents):
        children = []
        if random.random() < self.crossover_probability:
            parent1, parent2 = parents
            child1, child2 = parent1.crossover(parent2)
            children.extend([child1, child2])
        return children

    def plot_func(self, trace):
        X = np.arange(-2, 3, 0.05)
        Y = np.arange(-4, 2, 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = 1.5 - np.exp(-X ** (2) - Y ** (2)) - 0.5 * np.exp(-(X - 1) ** (2) - (Y + 2) ** (2))
        plt.figure()
        plt.contour(X, Y, Z, 10)
        cmaps = [[ii / len(trace), 0, 0] for ii in range(len(trace))]
        plt.scatter([x[0] for x in trace], [x[1] for x in trace], c=cmaps)
        plt.show()

    def run(self):
        trace = []
        for step in range(self.num_steps):
            new_population = []
            for _ in range(self.population_size // 2):
                parents = self.tournament_selection()
                children = self.reproduce(parents)
                for child in children:
                    child.mutation(self.mutation_probability)
                    new_population.append(child)
            self.population = new_population
            best_chromosome = min(self.population, key=lambda chromosome: self.eval_objective_func(chromosome))
            trace.append(best_chromosome.decode(0, self.chromosome_length, self.aoi))
        self.best_solution = best_chromosome.decode(0, self.chromosome_length, self.aoi)
        self.plot_func(trace)

        return self.best_solution

# Define the objective function
def objective_function(x, y):
    return 1.5 - np.exp(-x**2 - y**2) - 0.5 * np.exp(-(x - 1)**2 - (y + 2)**2)

# Define the range for decoding
aoi = [-2, 2]

# Create a GeneticAlgorithm instance and run the optimization
ga = GeneticAlgorithm(chromosome_length=4, obj_func_num_args=2, objective_function=objective_function, aoi=aoi)
result = ga.run()

print("Optimal solution:", result)
# import numpy as np
# import matplotlib.pyplot as plt
# import random

# def min_max_norm(val, min_val, max_val, new_min, new_max):
#     return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min

# class Chromosome:
#     def __init__(self, length, array=None):
#         if array is None:
#             self.array = [random.randint(0, 1) for _ in range(length)]
#         else:
#             self.array = array

#     def decode(self, lower_bound, upper_bound, aoi):
#         decoded_values = []
#         for i in range(0, len(self.array), len(aoi)):
#             gene = self.array[i:i + len(aoi)]
#             decoded_value = min_max_norm(int("".join(map(str, gene)), 2), 0, 2 ** len(gene) - 1, aoi[0], aoi[1])
#             decoded_values.append(decoded_value)
#         return decoded_values

#     def mutation(self, probability):
#         for i in range(len(self.array)):
#             if random.random() < probability:
#                 self.array[i] = 1 - self.array[i]

#     def crossover(self, other):
#         point = random.randint(1, len(self.array) - 1)
#         child1_array = self.array[:point] + other.array[point:]
#         child2_array = other.array[:point] + self.array[point:]
#         return Chromosome(len(self.array), child1_array), Chromosome(len(self.array), child2_array)

# class GeneticAlgorithm:
#     def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi, population_size=1000,
#                  tournament_size=2, mutation_probability=0.05, crossover_probability=0.8, num_steps=30):
#         assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
#         self.chromosome_length = chromosome_length
#         self.obj_func_num_args = obj_func_num_args
#         self.bits_per_arg = int(chromosome_length / obj_func_num_args)
#         self.objective_function = objective_function
#         self.aoi = aoi
#         self.tournament_size = tournament_size
#         self.mutation_probability = mutation_probability
#         self.crossover_probability = crossover_probability
#         self.num_steps = num_steps
#         self.population_size = population_size
#         self.population = [Chromosome(chromosome_length) for _ in range(population_size)]
#         self.best_solution = None

#     def eval_objective_func(self, chromosome):
#         decoded_values = chromosome.decode(0, self.chromosome_length, self.aoi)
#         return self.objective_function(*decoded_values)

#     def tournament_selection(self):
#         selected_parents = []
#         for _ in range(self.tournament_size):
#             candidates = random.sample(self.population, self.tournament_size)
#             selected = min(candidates, key=lambda chromosome: self.eval_objective_func(chromosome))
#             selected_parents.append(selected)
#         return selected_parents

#     def reproduce(self, parents):
#         children = []
#         if random.random() < self.crossover_probability:
#             parent1, parent2 = parents
#             child1, child2 = parent1.crossover(parent2)
#             children.extend([child1, child2])
#         return children

#     def run(self):
#         trace = []
#         for step in range(self.num_steps):
#             new_population = []
#             for _ in range(self.population_size // 2):
#                 parents = self.tournament_selection()
#                 children = self.reproduce(parents)
#                 for child in children:
#                     child.mutation(self.mutation_probability)
#                     new_population.append(child)
#             self.population = new_population
#             best_chromosome = min(self.population, key=lambda chromosome: self.eval_objective_func(chromosome))
#             trace.append(best_chromosome.decode(0, self.chromosome_length, self.aoi))
#         self.best_solution = best_chromosome.decode(0, self.chromosome_length, self.aoi)
#         self.plot_func(trace)

#         return self.best_solution

# # Define the objective function
# def objective_function(x, y):
#     return 1.5 - np.exp(-x**2 - y**2) - 0.5 * np.exp(-(x - 1)**2 - (y + 2)**2)

# # Define the range for decoding
# aoi = [-2, 2]

# # Create a GeneticAlgorithm instance and run the optimization
# ga = GeneticAlgorithm(chromosome_length=16, obj_func_num_args=2, objective_function=objective_function, aoi=aoi)
# result = ga.run()

# print("Optimal solution:", result)