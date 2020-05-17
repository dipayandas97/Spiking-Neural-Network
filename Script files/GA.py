import numpy as np

class GA:
    def __init__(self):
        pass

    def matrix_to_vector(self, matrices): #input list of matrices (for a single model)
        vectors = np.concatenate([m.flatten() for m in matrices])
        return np.asarray(vectors)

    def vector_to_matrix(vector, dummy_matrices): #vector shaped: (L,) (for a single model)
        synapse_list = []
        
        lens = []
        for s in dummy_matrices:
            lens.append(s.shape[0] * s.shape[1])
        if np.sum(lens) != vector.shape[0]:
            raise Exception('Cant reshape vector of shape',vector.shape,'to matrices having elements:,',lens)

        for l in range(len(dummy_matrices)):
            start_id = int(np.sum(lens[:l]))
            end_id = int(np.sum(lens[:l+1]))
            s = vector[start_id:end_id]
            synapse = s.reshape(dummy_matrices[l].shape[0], dummy_matrices[l].shape[1])
            synapse_list.append(synapse)
        
        return synapse_list

    def select_mating_pool(self, population, fitness, num_parents, mode='min'): 
        parents = np.zeros((num_parents, population.shape[1]))
        for idx in range(num_parents):
            if mode=='min':
                min_id = np.where(fitness==np.min(fitness))[0][0]
                parents[idx,:] = population[min_id,:]
                fitness[min_id] = 9999999
            elif mode=='max':
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
