import numpy as np
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz

# Fuzzy universe
clothing = np.arange(0, 3, 0.1)

# Fitness function
def fitness_function(individual):
    light_peak, medium_peak, heavy_peak = individual

    # Ensure ascending order and within bounds
    if not (0 <= light_peak <= medium_peak <= heavy_peak <= 2):
        return -1000,  # large penalty

    cl_light = fuzz.trimf(clothing, [0, 0, light_peak])
    cl_medium = fuzz.trimf(clothing, [0, medium_peak, 2])
    cl_heavy = fuzz.trimf(clothing, [heavy_peak, 2, 2])

    # Example target score
    target_score = 1.5

    # Aggregate
    aggregated = np.fmax(cl_light, np.fmax(cl_medium, cl_heavy))
    score = fuzz.defuzz(clothing, aggregated, 'centroid')

    # Fitness: smaller difference from target is better
    fitness = -abs(score - target_score)
    return fitness,

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 2)  # generate within 0-2
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
pop = toolbox.population(n=20)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)

# Best solution
best_ind = tools.selBest(pop, 1)[0]
print("Best fuzzy peaks found:", best_ind)
