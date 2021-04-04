

import random
import math
import numpy as np
import matplotlib.pyplot as plt
graph_store = {}
population_size = [20, 50, 100, 200]
def func1(x):

    return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[0]/2+(x[1]+47))))-x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47))))

def func2(x):
      
    return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-(np.sqrt(x[0]**2+x[1]**2)/np.pi))))

def ensure_bounds(vec, bounds):

    vec_new = []
    # cycle through each variable in vector 
    for i in range(len(vec)):

        # variable exceedes the minimum boundary
        if vec[i] < bounds[i][0]:
            vec_new.append(bounds[i][0])

        # variable exceedes the maximum boundary
        if vec[i] > bounds[i][1]:
            vec_new.append(bounds[i][1])

        # the variable is fine
        if bounds[i][0] <= vec[i] <= bounds[i][1]:
            vec_new.append(vec[i])
        
    return vec_new


#--- MAIN ---------------------------------------------------------------------+

def main(cost_func, bounds, popsize, mutate, recombination, maxiter):

    #--- INITIALIZE A POPULATION (step #1) ----------------+
    function_name = cost_func.__name__.title()
    function_name = function_name.replace('_', ' ')
    population = []
    for i in range(0,popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0],bounds[j][1]))
        population.append(indv)
            
    # cycle through each generation (step #2)
    for i in range(1,maxiter+1):
        print ('GENERATION:',i)

        gen_scores = [] # score keeping

        # cycle through each individual in the population
        for j in list(range(0, popsize)):

            #--- MUTATION (step #3.A) ---------------------+
            
            # select three random vector index positions [0, popsize), not including current vector (j)
            canidates =list( range(0,popsize))
            canidates.remove(j)
            random_index = random.sample(canidates, 3)

            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]     # target individual

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # multiply x_diff by the mutation factor (F) and add to x_1
            v_donor = [x_1_i + mutate * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            #--- RECOMBINATION (step #3.B) ----------------+

            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random() #random value between 0 to 1
                if crossover <= recombination:
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])
                    
            #--- GREEDY SELECTION (step #3.C) -------------+

            score_trial  = cost_func(v_trial)
            score_target = cost_func(x_t)

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                print ('   >',score_trial, v_trial)

            else:
                print ('   >',score_target, x_t)
                gen_scores.append(score_target)

        #--- SCORE KEEPING --------------------------------+

        gen_avg = sum(gen_scores) / popsize                         # current generation avg. fitness
        gen_best = min(gen_scores)                                  # fitness of best individual
        gen_sol = population[gen_scores.index(min(gen_scores))]     # solution of best individual

        print ('      > GENERATION AVERAGE:',gen_avg)
        print ('      > GENERATION BEST:',gen_best)
        print ('         > BEST SOLUTION:',gen_sol,'\n')
        global graph_store
        x = [50 ,100,200]
    
        graph_store[popsize] = {'y_best' : gen_best, 'y_avg': gen_avg}

        if(len(graph_store) == len(population_size)):
            n = len(population_size)/2
            plt.suptitle(function_name.title()+"\n#Generations: {}".format(maxiter), fontsize=16)
            for index, pop_size in enumerate(population_size):
            
                plt.subplot(n, n, index+1)
                plt.title("Population size: {}".format(pop_size))
                plt.plot(x, graph_store[pop_size]['y_avg'], label='Average value')
                plt.legend()
                plt.plot(x, graph_store[pop_size]['y_best'], label='Best value')
                plt.legend()
                plt.xlabel('Number of generations -- >')
                plt.ylabel('Function value -->')
                plt.legend()
            
            graph_store = {}
            plt.show()

    return gen_sol
population_size = [20, 50, 100, 200]
num_of_gens = [50, 100, 200]             # Cost function
bound1 = [(-512,512),(-512,512)]            # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]
                       # Population size, must be >= 4
mutate = 0.5                        # Mutation factor [0,2]
recombination = 0.8                 # Recombination rate [0,1]
                      # Max number of generations (maxiter)
bound2 = [(-10,10),(-10,10)]            # Bounds [(x1_min, x1_max), (x2_min, x2_max),...]

functions = [{'function': func1, 'limits': bound1},
                 {'function': func2, 'limits': bound2}]
for func in functions:
    for gen in num_of_gens:
        for pop in population_size:
            print('name of function ',func,'\n')
            print('name of generation',gen,'\n')
            print('name of population',pop,'\n')

            
            main(func['function'], func['limits'], pop, mutate,recombination ,gen)