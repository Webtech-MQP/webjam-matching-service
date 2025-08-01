import random
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Candidate and Problem Definition ---

class Candidate:
    def __init__(self, id, skill, role):
        self.id = id
        self.skill = skill  # A numerical score for capability
        self.role = role    # A categorical attribute for diversity

    def __repr__(self):
        return f"Candidate(ID: {self.id}, Skill: {self.skill}, Role: '{self.role}')"

# --- Configuration ---
NUM_CANDIDATES = 40
NUM_TEAMS = 5
POPULATION_SIZE = 8
GENERATIONS = 15
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
LOCAL_SEARCH_ITERATIONS = 10

# --- Generate a pool of random candidates ---
CANDIDATES = [
    Candidate(id=i, skill=random.randint(5, 10), role=random.choice(['Engineer', 'Designer', 'QA', 'PM']))
    for i in range(NUM_CANDIDATES)
]

# Create PyTorch tensors for candidate skills
candidate_skills = torch.tensor([c.skill for c in CANDIDATES], dtype=torch.float32, device=device)
role_to_idx = {'Engineer': 0, 'Designer': 1, 'QA': 2, 'PM': 3}
candidate_roles = torch.tensor([role_to_idx[c.role] for c in CANDIDATES], dtype=torch.long, device=device)

# --- 2. Fitness Functions ---

def get_teams_from_chromosome(chromosome):
    """Converts a chromosome into a dictionary of teams."""
    teams = defaultdict(list)
    for candidate_idx, team_id in enumerate(chromosome):
        teams[team_id].append(CANDIDATES[candidate_idx])
    return teams

def calculate_capability_pytorch(chromosome_tensor):
    """Objective 1: Maximize the sum of skills in all teams using PyTorch."""
    total_capability = 0.0
    for team_id in range(NUM_TEAMS):
        team_mask = (chromosome_tensor == team_id)
        if team_mask.sum() > 0:
            team_capability = candidate_skills[team_mask].sum()
            total_capability += team_capability.item()
    return total_capability

def calculate_diversity_pytorch(chromosome_tensor):
    """Objective 2: Maximize the number of unique roles in all teams using PyTorch."""
    total_diversity = 0
    for team_id in range(NUM_TEAMS):
        team_mask = (chromosome_tensor == team_id)
        if team_mask.sum() > 0:
            team_roles = candidate_roles[team_mask]
            unique_roles = torch.unique(team_roles)
            total_diversity += len(unique_roles)
    return total_diversity

def evaluate_fitness(chromosome):
    """Calculates both fitness scores for a given chromosome using PyTorch."""
    chromosome_tensor = torch.tensor(chromosome, dtype=torch.long, device=device)
    capability = calculate_capability_pytorch(chromosome_tensor)
    diversity = calculate_diversity_pytorch(chromosome_tensor)
    return torch.tensor([capability, diversity], dtype=torch.float32, device=device)

# --- 3. Core NSGA-II Functions ---

def non_dominated_sort(population_fitness):
    """
    Sorts the population into Pareto fronts using PyTorch operations.
    Returns a list of fronts, where each front is a list of indices.
    """
    pop_size = len(population_fitness)
    dominating_counts = torch.zeros(pop_size, dtype=torch.int32, device=device)
    dominated_sets = [[] for _ in range(pop_size)]
    fronts = [[]]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            p_fitness = population_fitness[i]
            q_fitness = population_fitness[j]

            # Check for dominance (maximization problem)
            p_dominates_q = torch.all(p_fitness >= q_fitness) and torch.any(p_fitness > q_fitness)
            q_dominates_p = torch.all(q_fitness >= p_fitness) and torch.any(q_fitness > p_fitness)

            if p_dominates_q:
                dominating_counts[j] += 1
                dominated_sets[i].append(j)
            elif q_dominates_p:
                dominating_counts[i] += 1
                dominated_sets[j].append(i)

    # The first front contains all non-dominated individuals
    fronts[0] = [i for i, count in enumerate(dominating_counts) if count == 0]

    front_idx = 0
    while front_idx < len(fronts) and len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_sets[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)
        front_idx += 1

    return fronts

def crowding_distance(fitness_values):
    """
    Calculates the crowding distance for a set of points in a front using PyTorch.
    """
    pop_size = fitness_values.shape[0]
    if pop_size == 0:
        return torch.zeros(0, device=device)

    distances = torch.zeros(pop_size, device=device)
    num_objectives = fitness_values.shape[1]

    for i in range(num_objectives):
        sorted_indices = torch.argsort(fitness_values[:, i])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        if pop_size > 2:
            f_min = fitness_values[sorted_indices[0], i]
            f_max = fitness_values[sorted_indices[-1], i]
            if f_max > f_min:
                # Normalize and add distances
                distances[sorted_indices[1:-1]] += \
                    (fitness_values[sorted_indices[2:], i] - fitness_values[sorted_indices[:-2], i]) / (f_max - f_min)

    return distances

# --- 4. Genetic Operators ---

def crossover(parent1, parent2):
    """Performs uniform crossover."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    child1, child2 = parent1[:], parent2[:]
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def mutate(chromosome):
    """Randomly reassigns a candidate to a different team."""
    if random.random() < MUTATION_RATE:
        idx_to_mutate = random.randint(0, len(chromosome) - 1)
        chromosome[idx_to_mutate] = random.randint(0, NUM_TEAMS - 1)
    return chromosome

# --- 5. Hybrid Component: Local Search (Hill Climbing) ---

def local_search(chromosome):
    """
    Performs a simple hill-climbing search to fine-tune a solution using PyTorch.
    """
    current_fitness = evaluate_fitness(chromosome)

    for _ in range(LOCAL_SEARCH_ITERATIONS):
        # Create a neighbor by making a small change
        neighbor = chromosome[:]
        gene_to_move = random.randint(0, len(neighbor) - 1)
        current_team = neighbor[gene_to_move]

        # Find a new team that is not the current one
        possible_new_teams = list(set(range(NUM_TEAMS)) - {current_team})
        if not possible_new_teams:
            continue
        new_team = random.choice(possible_new_teams)
        neighbor[gene_to_move] = new_team

        neighbor_fitness = evaluate_fitness(neighbor)

        # Dominance check (maximization)
        if (torch.all(neighbor_fitness >= current_fitness) and torch.any(neighbor_fitness > current_fitness)):
            # If neighbor dominates, accept it
            chromosome = neighbor
            current_fitness = neighbor_fitness

    return chromosome

# --- 6. Team Display Functions ---

def display_teams(chromosome, title="Teams"):
    """Display detailed information about the teams formed by a chromosome."""
    teams = get_teams_from_chromosome(chromosome)

    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    total_capability = 0
    total_diversity = 0

    for team_id in range(NUM_TEAMS):
        members = teams.get(team_id, [])

        if not members:
            print(f"\nTeam {team_id + 1}: (Empty)")
            continue

        team_capability = sum(c.skill for c in members)
        unique_roles = set(c.role for c in members)
        team_diversity = len(unique_roles)

        total_capability += team_capability
        total_diversity += team_diversity

        print(f"\nTeam {team_id + 1}: ({len(members)} members)")
        print(f"  Capability Score: {team_capability}")
        print(f"  Diversity Score: {team_diversity}")
        print(f"  Roles: {', '.join(sorted(unique_roles))}")
        print(f"  Members:")

        # Sort members by role for better display
        sorted_members = sorted(members, key=lambda x: (x.role, x.skill), reverse=True)
        for member in sorted_members:
            print(f"    • {member.role:<10} | Skill: {member.skill} | ID: {member.id}")

    print(f"\n{'='*60}")
    print(f"TOTAL CAPABILITY: {total_capability}")
    print(f"TOTAL DIVERSITY:  {total_diversity}")
    print(f"{'='*60}")

def display_pareto_solutions(pareto_solutions, pareto_fitness):
    """Display all Pareto optimal solutions."""
    print(f"\n{'='*80}")
    print(f"{'PARETO OPTIMAL SOLUTIONS':^80}")
    print(f"{'='*80}")

    for i, (solution, fitness) in enumerate(zip(pareto_solutions, pareto_fitness)):
        print(f"\nSolution {i+1}: Capability={fitness[0]:.1f}, Diversity={fitness[1]:.1f}")
        display_teams(solution, f"Solution {i+1} - Teams")

# --- 7. Main Algorithm ---

def main():
    """Main execution loop for the Hybrid Genetic Algorithm."""
    # Initialize population: list of chromosomes
    population = [
        [random.randint(0, NUM_TEAMS - 1) for _ in range(NUM_CANDIDATES)]
        for _ in range(POPULATION_SIZE)
    ]

    print("Starting Hybrid GA with PyTorch...")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Candidates: {NUM_CANDIDATES}")
    print(f"Teams: {NUM_TEAMS}")

    for gen in range(GENERATIONS):
        # Evaluate fitness of the entire population using PyTorch
        fitness_values = torch.stack([evaluate_fitness(ind) for ind in population])

        # --- Parent Selection ---
        mating_pool = []
        for _ in range(POPULATION_SIZE):
            i1, i2 = random.sample(range(POPULATION_SIZE), 2)
            mating_pool.append(random.choice([population[i1], population[i2]]))

        # --- Crossover and Mutation to create offspring ---
        offspring_population = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1] if i+1 < len(mating_pool) else mating_pool[0]
            child1, child2 = crossover(parent1, parent2)
            offspring_population.append(mutate(child1))
            offspring_population.append(mutate(child2))

        # --- LOCAL SEARCH (Hybrid Step) ---
        refined_offspring = [local_search(child) for child in offspring_population]

        # --- Survivor Selection (Elitism using NSGA-II principles) ---
        combined_population = population + refined_offspring
        combined_fitness = torch.stack([evaluate_fitness(ind) for ind in combined_population])

        fronts = non_dominated_sort(combined_fitness)

        next_population = []
        front_idx = 0
        while front_idx < len(fronts) and len(next_population) + len(fronts[front_idx]) <= POPULATION_SIZE:
            front_indices = fronts[front_idx]
            for index in front_indices:
                next_population.append(combined_population[index])
            front_idx += 1

        # If the last front is too big, use crowding distance to select
        if len(next_population) < POPULATION_SIZE and front_idx < len(fronts):
            last_front_indices = fronts[front_idx]
            last_front_fitness = combined_fitness[last_front_indices]
            distances = crowding_distance(last_front_fitness)

            # Sort by distance (descending)
            sorted_by_crowding = sorted(zip(last_front_indices, distances),
                                      key=lambda x: float(x[1]), reverse=True)

            remaining_space = POPULATION_SIZE - len(next_population)
            for i in range(remaining_space):
                next_population.append(combined_population[sorted_by_crowding[i][0]])

        population = next_population

        if (gen + 1) % 10 == 0:
            print(f"Generation {gen + 1}/{GENERATIONS} complete.")

    # --- Final Result ---
    final_fitness = torch.stack([evaluate_fitness(ind) for ind in population])
    final_fronts = non_dominated_sort(final_fitness)

    # The Pareto front is the first front of the final population
    pareto_front_indices = final_fronts[0]
    pareto_solutions = [population[i] for i in pareto_front_indices]
    pareto_fitness = final_fitness[pareto_front_indices].cpu().numpy()

    print("\nHybrid GA with PyTorch finished.")
    print(f"Found {len(pareto_solutions)} optimal solutions on the Pareto front.")

    # --- Visualization ---
    plt.figure(figsize=(12, 8))
    plt.scatter(pareto_fitness[:, 0], pareto_fitness[:, 1], c='blue', s=100, alpha=0.7, edgecolors='black')
    for i, (x, y) in enumerate(pareto_fitness):
        plt.annotate(f'Sol {i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.title('Pareto Front: Team Capability vs. Diversity', fontsize=16, fontweight='bold')
    plt.xlabel('Total Team Capability (Sum of Skills)', fontsize=12)
    plt.ylabel('Total Team Diversity (Sum of Unique Roles)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Display Results ---

    # Find the most balanced solution (highest sum of normalized objectives)
    normalized_fitness = pareto_fitness / pareto_fitness.max(axis=0)
    balanced_solution_idx = (normalized_fitness.sum(axis=1)).argmax()
    best_balanced_solution = pareto_solutions[balanced_solution_idx]

    display_teams(best_balanced_solution, "MOST BALANCED SOLUTION")

    # Display all Pareto solutions if there aren't too many
    if len(pareto_solutions) <= 8:
        display_pareto_solutions(pareto_solutions, pareto_fitness)
    else:
        print(f"\nFound {len(pareto_solutions)} Pareto optimal solutions. Showing most balanced solution above.")
        print("Run with fewer generations or smaller population to see all solutions.")

    # Display candidate pool summary
    print(f"\n{'='*60}")
    print(f"{'CANDIDATE POOL SUMMARY':^60}")
    print(f"{'='*60}")
    role_counts = defaultdict(int)
    skill_sum = 0
    for candidate in CANDIDATES:
        role_counts[candidate.role] += 1
        skill_sum += candidate.skill

    print(f"Total Candidates: {NUM_CANDIDATES}")
    print(f"Average Skill: {skill_sum/NUM_CANDIDATES:.2f}")
    print("Role Distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count} candidates")

if __name__ == '__main__':
    main()
