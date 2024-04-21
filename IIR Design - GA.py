import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import random


# Define constants
POPULATION_SIZE = 1000
MUTATION_RATE = 0.6
MUT_SIGMA = 0.1
NUM_GENERATIONS = 1000
CROSSOVER_PERCENTAGE = 0.7
ELITE_PERCENTAGE = 0.1


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def to_db(magnitude):
    return 20 * np.log10(abs(magnitude))

def generate_population(chromosome_size, limits):
    population = np.zeros((POPULATION_SIZE, chromosome_size))
    lower_limit = [x[0] for x in limits]
    upper_limit = [x[1] for x in limits]
    for i in range(0, POPULATION_SIZE):
        population[i] = np.random.uniform(lower_limit, upper_limit, size=chromosome_size)
    return population

def evaluate_fitness(population, freqs, fs, target_freq_response):
    fitness = []
    for chromosome in population:
        b0, b1, b2, a1, a2, b0_2, b1_2, b2_2, a1_2, a2_2 = chromosome
        b = [b0, b1, b2]
        a = [1, a1, a2]
        b2 = [b0_2, b1_2, b2_2]
        a2 = [1, a1_2, a2_2]
        _, h = freqz(b,a, worN=freqs, fs=fs)
        _, h2 = freqz(b2,a2, worN=freqs, fs=fs)
        h_total = h * h2
        h_total = to_db(h_total)
        error = rmse(target_freq_response, h_total)
        fitness.append(-error)
    return fitness

def plot_top_n(frequencies, fs, population, target_freq_response, num):
    plt.figure(1)
    plt.clf()
    plt.semilogx(frequencies, target_freq_response, label='Desired Response')
    for i in range(0, num):
        chromosome=population[i]
        b0, b1, b2, a1, a2, b0_2, b1_2, b2_2, a1_2, a2_2 = chromosome
        b = [b0, b1, b2]
        a = [1, a1, a2]
        b2 = [b0_2, b1_2, b2_2]
        a2 = [1, a1_2, a2_2]
        _, h = freqz(b,a, worN=freqs, fs=fs)
        _, h2 = freqz(b2,a2, worN=freqs, fs=fs)
        h_total = h * h2
        h_total = to_db(h_total)
        plt.semilogx(frequencies, h_total, linestyle='--', label='Chromosome ' + str(i+1))
    plt.title('Magnitude Response of IIR Filter and Desired Response')
    plt.xlabel('Frequency [radians/sample]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    plt.grid()
    plt.show(block=False)
        
def blendCrossover(parent1, parent2, chromosome_size, limits):
    offspring = np.zeros(chromosome_size)
    alpha = np.random.uniform(0,1)
    for i in range(0, chromosome_size):
        offspring[i] = alpha * parent1[i] + (1-alpha) * parent2[i]
        offspring[i] = min(max(offspring[i], limits[i][0]), limits[i][1] )
    return offspring
        
def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    # Calculate the probability of selection for each individual
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    # Select two individuals using roulette wheel selection
    selected_parents_indices = random.choices(range(len(population)), weights=probabilities, k=2)
    # Return the selected parents
    selected_parents = [population[i] for i in selected_parents_indices]
    return selected_parents       

def reproduce_population(population, fitness, chromosome_size, limits):
    n_crossover = POPULATION_SIZE * CROSSOVER_PERCENTAGE
    n_offspring = (int)(n_crossover / 2)
    offsprings = np.zeros((n_offspring, chromosome_size))
    for i in range(0, n_offspring):
        parent1, parent2 = roulette_wheel_selection(population, fitness)
        offsprings[i] = blendCrossover(parent1, parent2, chromosome_size, limits)
    return offsprings

def uniformMutation(chromosome, chromosome_size, limits):
    mut_chromosome = chromosome.copy()
    gene = np.random.randint(0, chromosome_size-1)
    mut_chromosome[gene] += np.random.uniform(limits[gene][0], limits[gene][1]) * MUT_SIGMA 
    return mut_chromosome
    
def mutatePopulation(population, chromosome_size, limits, n_mutants):
    indexes = [np.random.randint(0, POPULATION_SIZE-1) for _ in range(0, n_mutants-1)]
    mutated_population = population.copy()
    for i in indexes:
        mutated_population[i] = uniformMutation(mutated_population[i], chromosome_size, limits)
    return mutated_population
    
    
load_f_m = np.loadtxt('C:\\Users\\jhvaz\\Documents\\Faculdade\\5º Ano\\Tese\\Flow\\3. LTI Filter Design\\Response\\H1_freq_chirp_1V_1s.txt')
# Define frequency points (in Hz)
freqs = load_f_m[0] #np.linspace(0, 10000, 1000)


# Define magnitude response (in dB)
mags = np.zeros_like(freqs)  # Initialize as all zeros
center_freq = 3000  # Center frequency of the bandpass filter
bandwidth = 3000  # Bandwidth of the bandpass filter
gain = 0  # Peak gain in dB
attenuation = -3

# Calculate the magnitude response of the bandpass filter
for i, freq in enumerate(freqs):
    if center_freq - bandwidth / 2 <= freq <= center_freq + bandwidth / 2:
        # Inside the passband
        mags[i] = gain
    else:
        # Outside the passband
        mags[i] = attenuation

mags = load_f_m[1]
        
fs=44.1e3
chromosome_size = 10
limits = [  (-1, 1),    #b0
            (-2, 2),    #b1
            (-2, 2),    #b2
            (-1, 1),    #a1
            (-1, 1),    #a2
            (-1, 1),    #b0
            (-2, 2),    #b1
            (-2, 2),    #b2
            (-1, 1),    #a1
            (-1, 1)     #a2
            ]

#%% Population Generation
population = generate_population(chromosome_size, limits)



#%% GA Loop
for gen in range(0, NUM_GENERATIONS):
    
    #%% Fitness 1
    fitness = evaluate_fitness(population, freqs, 44.1e3, mags)
    sorted_indices = np.argsort(fitness)[::-1]
    fitness.sort(reverse=True)
    population = population[sorted_indices]
    #plot_top_n(freqs, fs, population, mags, 5)
    elitist = population[0]
    
    #%% Crossover
    n_crossover = POPULATION_SIZE * CROSSOVER_PERCENTAGE
    n_offspring = (int)(n_crossover / 2)
    offsprings = reproduce_population(population, fitness, chromosome_size, limits)
    
    #%% Mutation
    n_mutants = int(POPULATION_SIZE * MUTATION_RATE)
    mutated_population = mutatePopulation(population, chromosome_size, limits, n_mutants)
    
    #%% New  Population
    
    new_population = np.concatenate((mutated_population, offsprings), axis=0)
    fitness = evaluate_fitness(new_population, freqs, 44.1e3, mags)
    sorted_indices = np.argsort(fitness)[::-1]
    fitness.sort(reverse=True)
    new_population = new_population[sorted_indices]
    population = new_population[0:POPULATION_SIZE]
    population[POPULATION_SIZE-1] = elitist
    plot_top_n(freqs, fs, population, mags, 5)
    print("generation " + str(gen) + " | best fitness: " + str(fitness[0]))


    
    




print(fitness)


# %%









