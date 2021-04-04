import sys
import numpy as np

import matplotlib.pyplot as plt
import random

class Generation(object): #gen class for initiating all values
    def __init__(self, best_candidate, best_val, average):
        self.best_candidate = best_candidate
        self.best_value = best_val
        self.average = average

graph = {}
def eggholderfunction(candidate): #egg holder function
    return -(candidate[1]+47)*np.sin(np.sqrt(np.abs(candidate[0]/2+(candidate[1]+47))))-candidate[0]*np.sin(np.sqrt(np.abs(candidate[0]-(candidate[1]+47))))

def holdertablefunction(candidate):#holder table function
    return -np.abs(np.sin(candidate[0])*np.cos(candidate[1])*np.exp(np.abs(1-(np.sqrt(candidate[0]**2+candidate[1]**2)/np.pi))))

def initialization(n, limits): # func for initialization
    initial_vectors = []
    for i in range(n):
        x_rand = random.uniform(limits[0], limits[1])
        y_rand = random.uniform(limits[2], limits[3])
        initial_vectors.append(np.array([x_rand, y_rand])) 
    return initial_vectors

def get_F(_min=-2, _max=2): #getting random f  b/w -2 and 2
    return random.uniform(_min, _max )

def check_limits(z, limits): # limit check for vector 

    if limits[0] <= z[0] <= limits[1] and limits[2] <= z[1] <= limits[3]:
        return True
    else:
        return False

def elitism(function, parent_vectors, trial_vectors):
    selected_vectors = []
    _best_value = sys.maxsize #max value
    _average = 0

    for parent, trial in zip(parent_vectors, trial_vectors):
        
        parent_val = function(parent)
        trial_val = function(trial)

        if parent_val < trial_val:
            selected = parent
            _average += parent_val
            if parent_val < _best_value:
                _best_value = parent_val
                _best_candidate = selected
                
        else:
            selected = trial
            _average += trial_val
            if trial_val < _best_value:
                _best_value = trial_val
                _best_candidate = selected
        
        selected_vectors.append(selected)
    _average /= len(trial_vectors)
    return selected_vectors, Generation(_best_candidate, _best_value, _average)

def plot(function_name, generations, pop_size, gens_count):#function to plot all the cases
    global graph
    x = [i for i in range(len(generations))]
    y_best = [i.best_value for i in generations]
    y_avg = [i.average for i in generations]
    graph[pop_size] = {'y_best' : y_best, 'y_avg': y_avg}

    if(len(graph) == len(population_size)):
        n = len(population_size)/2
        plt.suptitle(function_name.title()+"\n#Generations: {}".format(gens_count), fontsize=16)
        for index, pop_size in enumerate(population_size):
            
            plt.subplot(n, n, index+1)
            plt.title("Population size: {}".format(pop_size))
            plt.plot(x, graph[pop_size]['y_avg'], label='Average value')
            plt.legend()
            plt.plot(x, graph[pop_size]['y_best'], label='Best value')
            plt.legend()
            plt.xlabel('Number of generations -- >')
            plt.ylabel('Function value -->')
            plt.legend()
        graph = {}
        plt.show()
    
def DIE(function, limits, pop_size, gens_count):#function for differential evolution
    
    parent_vectors = initialization(pop_size, limits) #initialize parent vector
    best_of_all_generations = [] 
    no_of_gens = gens_count
    function_name = function.__name__.title() #get function name
    while(gens_count-1):#looping through all generations
        
        trial_vectors = [] #trail vect
        F = get_F()
        gens_count -= 1
        for index, candidate in enumerate(parent_vectors):            
            while(pop_size-1):
                parents_remove = parent_vectors[:]
                parents_remove.pop(index)
                r1, r2, r3 = random.sample(parents_remove, 3)#generating 3 random vectors excluding the ith vector
                mutant = candidate + 0.5*(r1 - candidate) + F*(r2 - r3)#mutant_vetor
                z = []
                for j in range(len(mutant)):
                    Crp = random.random()
                    if Crp <= 0.8: # cross over prob
                        z.append(mutant[j])
                    else:
                        z.append(candidate[j])      
                if(check_limits(z, limits)):#checkbounds
                    break
            trial_vectors.append(np.array(z))
        new_gen, best_of_generation = elitism(function, parent_vectors, trial_vectors)
        best_of_all_generations.append(best_of_generation)
        parent_vectors = new_gen[:]
    print ('Function: {}\n#Generations: {}\nPopulation size: {}\nBest value: {}\nBest candidate: {}\n'.format(function_name, no_of_gens, pop_size, best_of_generation.best_value, best_of_generation.best_candidate))

    plot(function_name, best_of_all_generations, pop_size, no_of_gens)
   
population_size = [20, 50, 100, 200]
num_of_gens = [50, 100, 200]
egg_limits = [-512, 512, -512, 512]
holder_limits = [-10, 10, -10, 10]
functions = [{'function': eggholderfunction, 'limits': egg_limits},
                 {'function': holdertablefunction, 'limits': holder_limits}]

for func in functions:
  for gen in num_of_gens:
    for pop in population_size:
      DIE(func['function'], func['limits'], pop, gen)