import numpy as np

class GA:
    def __init__(self):
        pass

    def matrix_to_vector(self, matrices): #returns list(matrix) of vectors for list(matrix) of 3D matrices
        if len(matrices.shape) != 4:
            raise Exception('Input should be 4D matrices (array of 3D matrices)')
        vectors = [m.flatten() for m in matrices]
        return np.asarray(vectors)

    def vector_to_matrix(self, vectors, matrices):
        if len(matrices.shape) != 4:
            raise Exception('Input matrix should be 4D matrices (array of 3D matrices)')
        new_matrices = [v.reshape(matrices.shape[1], matrices.shape[2], matrices.shape[3]) for v in vectors]
        return np.asarray(new_matrices)

    def select_mating_pool(self, population, fitness, num_parents): 
        parents = np.zeros((num_parents, population.shape[1]))
        for idx in range(num_parents):
            max_id = np.where(fitness==np.max(fitness))[0][0]
            parents[idx,:] = population[max_id,:]
            fitness[max_id] = -9999999
        return parents

    def crossover(self, parents, num_offsprings):
        offsprings = np.zeros((num_offsprings, parents.shape[1]))
        crossover_point = np.uint32(parents.shape[1]/2)
        for k in range(num_offsprings):
            parent_1 = k % parents.shape[0]
            parent_2 = (k+1) % parents.shape[0]
            offsprings[k, :crossover_point] = parents[parent_1, :crossover_point]
            offsprings[k, crossover_point:] = parents[parent_2, crossover_point:]
        return offsprings

    def mutate(self, offsprings, mutation_percent):
        num_of_cells_to_mutate = np.uint32((mutation_percent/100) * offsprings.shape[1])
        for idx in range(offsprings.shape[0]):
            mutation_indices = np.random.randint(0, offsprings.shape[1], num_of_cells_to_mutate)
            random_val = np.random.uniform(-1.0,1.0,1)
            offsprings[idx, mutation_indices] += random_val
        return offsprings
