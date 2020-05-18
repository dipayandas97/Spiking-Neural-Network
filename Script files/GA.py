import numpy as np

class GA:
    def __init__(self):
        pass

    def matrix_to_vector(self, matrices): #input list of matrices (for a single model)
        vectors = np.concatenate([m.flatten() for m in matrices])
        return np.asarray(vectors)

    def vector_to_matrix(self, vector, dummy_matrices): #vector shaped: (L,) (for a single model)
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
        if len(parents.shape)>1:
            offsprings = np.zeros((num_offsprings, parents.shape[1]))
            crossover_point = np.uint32(parents.shape[1]/2)
            for k in range(num_offsprings):
                parent_1 = k % parents.shape[0]
                parent_2 = (k+1) % parents.shape[0]
                offsprings[k, :crossover_point] = parents[parent_1, :crossover_point]
                offsprings[k, crossover_point:] = parents[parent_2, crossover_point:]
        else:
            offsprings = np.zeros((num_offsprings,))
            sample_ids = np.random.randint(0,parents.shape[0], size=(num_offsprings,))
            offsprings = parents[sample_ids]
        return offsprings

    def mutate(self, offsprings, perturbation_range, mutation_percent):
        mutated_offsprings = np.zeros(offsprings.shape)        
        if len(offsprings.shape)>1:
            num_of_cells_to_mutate = np.uint32((mutation_percent/100) * offsprings.shape[1])
            for idx in range(offsprings.shape[0]):
                randomness = np.zeros(offsprings.shape[1])
                mutation_indices = np.random.randint(0, offsprings.shape[1], num_of_cells_to_mutate)
                
                if type(perturbation_range[0]) == type(1.):
                    for m in mutation_indices:
                        randomness[m] = np.random.uniform(perturbation_range[0],perturbation_range[1],1)         #[-R,R]
                elif type(perturbation_range[0]) == type(1):
                    for m in mutation_indices:
                        random_val = 0
                        while random_val==0:                                                                       
                            random_val = np.random.randint(perturbation_range[0],1+perturbation_range[1],1)      #{-I,I}
                        randomness[m] = random_val
                        
                mutated_offsprings[idx] += randomness
        else:
            num_of_cells_to_mutate = np.uint32((mutation_percent/100) * offsprings.shape[0])
            randomness = np.zeros(offsprings.shape[0])
            mutation_indices = np.random.randint(0, offsprings.shape[0], num_of_cells_to_mutate)

            if type(perturbation_range[0]) == type(1.):
                for m in mutation_indices:
                    randomness[m] = np.random.uniform(perturbation_range[0],perturbation_range[1],1)         #[-R,R]
            elif type(perturbation_range[0]) == type(1):
               for m in mutation_indices:
                        random_val = 0
                        while random_val==0:                                                                       
                            random_val = np.random.randint(perturbation_range[0],1+perturbation_range[1],1)      #{-I,I}
                        randomness[m] = random_val
                           
            mutated_offsprings += randomness   
            
        return mutated_offsprings
