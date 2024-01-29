import random
import numpy as np
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, n_bits):
        self.genome = 0  # Representing the individual with an integer
        self.n_bits = n_bits

    def randomize(self):
        self.genome = random.getrandbits(self.n_bits)

    def mutate2(self):
        mutation_point = random.randint(0, self.n_bits - 1)
        self.genome ^= (1 << mutation_point)  # Flip the bit at mutation_point


    def mutate(self):
        for i in range(self.n_bits):
            if random.random() < 1 / self.n_bits:  # Probability of 1/n for each bit
                self.genome ^= (1 << i)  # Flip the i-th bit

def one_max(individual):
    #counts the number of ones 
    return bin(individual.genome).count('1')



def leading_ones(individual):
    # Count the number of leading 1s
    return len(bin(individual.genome)[2:].split('0', 1)[0])

def jump_k(individual, k ):
    # Jump function
    ones_count = one_max(individual)
    if ones_count in range(0, individual.n_bits - k + 1 ) or ones_count == individual.n_bits:
        return k + ones_count
    else:
        return individual.n_bits - ones_count


def run_one_plus_one_ea(fitness_function, n_bits):
    individual = Individual(n_bits)
    individual.randomize()
    fitness = fitness_function(individual)
    
    evaluations = 0
    while fitness < n_bits:  # Assuming maximization until all bits are 1s
        new_individual = Individual(n_bits)
        new_individual.genome = individual.genome
        new_individual.mutate()
        new_fitness = fitness_function(new_individual)
        evaluations += 1
        if new_fitness >= fitness:  # In (1+1) EA, accept equal or better solutions
            individual = new_individual
            fitness = new_fitness

    return evaluations,fitness


def task2_part1():

    def run_experiment(problem_size, num_trials, fitness_function):
        results = []
        for _ in range(num_trials):
            evaluations = run_one_plus_one_ea(fitness_function, problem_size)
            results.append(evaluations)
        return np.mean(results)

    problem_sizes = np.arange(10, 101, 10) 
    num_trials = 100  # Number of trials per problem size

    # Testing on OneMax
    one_max_avg_runtimes = [run_experiment(size, num_trials, one_max) for size in problem_sizes]

    # Testing on LeadingOnes
    leading_ones_avg_runtimes = [run_experiment(size, num_trials, leading_ones) for size in problem_sizes]
    theoretical_one_max = problem_sizes * np.log(problem_sizes)  # O(n log n) for OneMax
    theoretical_leading_ones = problem_sizes ** 2  # O(n^2) for LeadingOnes

    # Plotting empirical results and theoretical bounds
    plt.figure(figsize=(12, 8))

    # Empirical results
    plt.plot(problem_sizes, one_max_avg_runtimes, label='OneMax (Empirical)', marker='o', color='blue')
    plt.plot(problem_sizes, leading_ones_avg_runtimes, label='LeadingOnes (Empirical)', marker='x', color='orange')

    # Theoretical bounds
    plt.plot(problem_sizes, theoretical_one_max, label='OneMax (Theoretical: O(n log n))', linestyle='--', color='blue')
    plt.plot(problem_sizes, theoretical_leading_ones, label='LeadingOnes (Theoretical: O(n^2))', linestyle='--', color='orange')
    # Theoretical bounds can be plotted here for comparison if available

    plt.xlabel('Problem Size (n)')
    plt.ylabel('Average Runtime (Number of Evaluations)')
    plt.title('Empirical Runtime of the (1+1) EA on OneMax and LeadingOnes')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_mu_plus_one_ea(mu, problem_size,fitness_function):
    # Initialize population
    population = [Individual(problem_size) for _ in range(mu)]
    for ind in population:
        ind.randomize()

    # Find the best individual in the initial population
    fitnesses = [fitness_function(ind) for ind in population]
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    evaluations = mu

    while best_fitness < problem_size:
        # Select and mutate one offspring
        offspring = Individual(problem_size)
        offspring.genome = random.choice(population).genome
        offspring.mutate()

        # Evaluate offspring
        offspring_fitness = fitness_function(offspring)
        evaluations += 1

        # Replace the worst individual if offspring is better
        if offspring_fitness >= worst_fitness:
            
            worst_index = fitnesses.index(worst_fitness)
            population[worst_index] = offspring
            fitnesses[worst_index] = offspring_fitness
            best_fitness = max(best_fitness,offspring_fitness)
            worst_fitness = min(fitnesses)
            

    return evaluations
def task2_part_2():
    mu_values = [1, 2, 5, 10, 50]
    problem_sizes = [10, 20, 30, 40, 50]
    num_trials = 100

    results = {mu: [] for mu in mu_values}
    for mu in mu_values:
        for size in problem_sizes:
            trial_results = [run_mu_plus_one_ea(mu, size,one_max) for _ in range(num_trials)]
            avg_runtime = np.mean(trial_results)
            results[mu].append(avg_runtime)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    for mu, runtimes in results.items():
        plt.plot(problem_sizes, runtimes, label=f'μ={mu}', marker='o')

    plt.xlabel('Problem Size (n)')
    plt.ylabel('Average Runtime (Number of Evaluations)')
    plt.title('Average Runtime of (μ+1) EA on OneMax for Various μ Values')
    plt.legend()
    plt.grid(True)
    plt.show()

task2_part_2()

def uniform_crossover(ind1, ind2):
    child = Individual(ind1.n_bits)
    child.genome = 0
    for i in range(ind1.n_bits):
        child.genome |= (random.choice([ind1.genome, ind2.genome]) >> i & 1) << i
    return child



def run_mu_lambda_ga(mu, lambda_, n_bits, pc, fitness_function,k=None):
    population = [Individual(n_bits) for _ in range(mu)]
    for individual in population:
        individual.randomize()

    best_fitness = max(fitness_function(individual) for individual in population)
    evaluations = mu  # Initial population fitness evaluations
    optimal_fit = optimal_fitness(fitness_function, n_bits, k)
    while best_fitness < optimal_fit:
        offspring = []
        for _ in range(lambda_):
            if random.random() < pc:
                # Perform uniform crossover
                parent1, parent2 = random.sample(population, 2)
                child = uniform_crossover(parent1, parent2)
            else:
                # Copy and mutate
                child = Individual(n_bits)
                child.genome = random.choice(population).genome
            child.mutate()
            offspring.append(child)

        # Combine, evaluate fitness, and select the best mu individuals
        evaluations += lambda_
        population.extend(offspring)
        population.sort(key=fitness_function, reverse=True)
        population = population[:mu]

        best_fitness = fitness_function(population[0])

    return evaluations
k_value = 2
def jump_fitness(individual):
    return jump_k(individual, k_value)


def optimal_fitness(fitness_func, n_bits, k=None):
    if fitness_func == one_max:
        return n_bits
    elif fitness_func == jump_fitness:
        return n_bits + k
    else:
        return n_bits   # we must have leading_ones as a fitness_function
#k_value= 3
#mu, lambda_ = 100,100
#n_bits = 10
#pc = 0.5  # Crossover probability

#print(run_mu_lambda_ga(mu,lambda_,n_bits,pc,jump_fitness,k_value))
    
def hamming_distance(individual1, individual2):
    """Calculate the Hamming distance between two individuals."""
    x = individual1.genome ^ individual2.genome
    return bin(x).count('1')

def calculate_diversity(population, d):
    """Calculate the diversity of the population at distance d."""
    diversity_count = 0
    n = len(population)

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(population[i], population[j]) == 2 * d:
                diversity_count += 1

    return diversity_count

def experiment(n, k, mu, lambda_, pc, max_iterations):
    """Run the (μ + λ) GA on Jump_k and measure diversity over time."""
    # Initialize the population on the plateau of Jump_k
    population = [Individual(n) for _ in range(mu)]
    for individual in population:
        # Initialize each individual with n-k ones and k zeros
        individual.genome = (1 << (n - k)) - 1
        # Randomly shuffle the k zeros within the genome
        zero_positions = random.sample(range(n), k)
        for pos in zero_positions:
            individual.genome &= ~(1 << pos)

    diversity_over_time = []
    best_fitness = 0
    evaluations = 0

    for iteration in range(max_iterations):
        # Run one iteration of the GA
        offspring = []
        for _ in range(lambda_):
            if random.random() < pc:
                parent1, parent2 = random.sample(population, 2)
                child = uniform_crossover(parent1, parent2)
            else:
                child = Individual(n)
                child.genome = random.choice(population).genome
            child.mutate()
            offspring.append(child)

        population.extend(offspring)
        population.sort(key=lambda ind: jump_k(ind, k), reverse=True)
        population = population[:mu]

        # Update best fitness and evaluations
        current_best_fitness = jump_k(population[0], k)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
        evaluations += lambda_

        # Calculate and record diversity for each distance d
        diversity_record = {}
        for d in range(k + 1):
            diversity_record[d] = calculate_diversity(population, d)
        diversity_over_time.append(diversity_record)

        # Check for termination condition
        if best_fitness >= n + k:
            break
    print(iteration)
    return diversity_over_time, evaluations, iteration


def plot():
        # Plotting
        n_values = [100]
        k_values = [4]
        mu_values = [100]
        lambda_values = [50]
        pc_values = [0.5]

        # Conducting experiments and collecting results
        results = {}
        for n in n_values:
            for k in k_values:
                for mu in mu_values:
                    for lambda_ in lambda_values:
                        for pc in pc_values:
                            diversity_timeline, evaluations, last_iteration = experiment(n, k, mu, lambda_, pc, 2000)
                            results[(n, k, mu, lambda_, pc)] = diversity_timeline
        # Visualizing the diversity over time for selected parameter combinations
        for n, k, mu, lambda_, pc in results:
            diversity_timeline = results[(n, k, mu, lambda_, pc)]
            # Extract the diversity for each distance
            distances = range(k + 1)  # Assuming 'k' is the maximum distance of interest
            diversity_per_distance = {d: [iteration.get(d, 0) for iteration in diversity_timeline] for d in distances}

            # Plotting diversity over time for each distance
            plt.figure(figsize=(10, 6))
            for d, diversity_values in diversity_per_distance.items():
                plt.plot(diversity_values, label=f'Distance {d}')

            plt.xlabel('Iteration')
            plt.ylabel('Diversity at Distance k')
            plt.title(f'Diversity Over Time (n={n}, k={k}, mu={mu}, lambda={lambda_}, pc={pc})')
            plt.legend()
            plt.grid(True)
            plt.show()

print('new')