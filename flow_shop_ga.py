import numpy as np
import math
import time
import random
import itertools
import queue
import pandas as pd
import csv

from gantt import plot_job_scheduling, plot_fitness



#Fitness List
fitness = []

fit_individuals = []
penalised_individuals = {}

f = open("result.txt", "w")

# number of jobs
n = 50

# number of machines
m = 20

# ith job's processing time at jth machine 
data = np.loadtxt(open("b.csv", "rb"), delimiter=",")
cost = np.array(data)
cost = np.transpose(cost)

temp = []


prec_constraints = [[2,3], [1,2]] #2 before 3 and 1 before 2

rj = {0:{1:95}} # Release Dates  <Machine>: <Job>:  Release_Time

#Initialisation of population is done by creating a random list of jobs
def initialization(Npop):
    pop = []
    for i in range(Npop):
        p = list(np.random.permutation(n))
        while p in pop:
            p = list(np.random.permutation(n))
        pop.append(p)
    
    return pop


#Calculation of the Fitness Function
#Input is the job sequence - j1->j2->j3
def calculateObj(sol, generation, impose_penalty):
    time = 0
    completion_times = []

    machine_times = [0] * m  # Keep track of the completion time for each machine

    for job in sol:
        machine_times[0] += cost[job][0]  # Assign the job to the first machine
        for i in range(1, m):
            machine_times[i] = max(machine_times[i], machine_times[i-1]) + cost[job][i]

    makespan = machine_times[-1]

    if(impose_penalty):
        makespan = impose_penalty_precedence(sol, generation, makespan)
        makespan = impose_penalty_release_time(sol, generation, makespan)
   
    return makespan  # Return the completion time of the last machine



def impose_penalty_precedence(sol, generation, makespan):
    #Imposing Penalty Function - Checking Precedence Constraints
    for i in range(len(prec_constraints)):
        if(sol.index(prec_constraints[i][0]) > sol.index(prec_constraints[i][1])):
            makespan +=50
            #Keeping track of penalised individuals
            penalised_individuals[generation].append(sol)

    return makespan

def impose_penalty_release_time(sol, generation, makespan):
    #rj = {2:{2:10}} 
    completion_times  = compute_final_times(sol, cost)

    for i in rj.keys():
        for j in rj[i].keys():
            job = i
            mc = j
            r_time = rj[i][j]      
            if(completion_times[job][mc][0] < r_time):
                makespan+=100
                penalised_individuals[generation].append(sol)

    return makespan


def compute_final_times(job_seq, cost):
    time = 0
    machine_times = [0] * m

    in_out_time = [0, 0]
    comp_times = {}
    temp = []

    for i in range(len(job_seq)):
        for j in range(m):
            temp.append([0,0])
        comp_times[i] = temp
        temp = []

    for job in range(len(job_seq)):
        machine_times[0] += cost[job_seq[job]][0]  # Assign the job to the first machine
    
        in_out_time[1] = machine_times[0] 
        comp_times[job][0] = in_out_time

        in_out_time = [0, 0]
        if(job < len(job_seq)-1):
            in_out_time[0] = machine_times[0]
            comp_times[job+1][0] = in_out_time

        for i in range(1, m):
            in_out = [0, 0]
            in_out[0] = max(machine_times[i], machine_times[i-1])
            in_out[1] = max(machine_times[i], machine_times[i-1]) + cost[job_seq[job]][i]
            comp_times[job][i] = in_out
            
            machine_times[i] = max(machine_times[i], machine_times[i-1]) + cost[job_seq[job]][i]
            in_out= [0,0]

    return comp_times     


# Implementing tournament selection operator for selection of parent individuals from the population. That there will be two parents after selection
def tournament_selection(pop, generation):
    #Calculate Objective function for each individual in the population
    popObj = []
    for i in pop:
        popObj.append(calculateObj(i, generation, 1))

    parent = []
    parents = []
    for q in range(Npop):
        for p in range(2):
            #Randomly selecting k individuals as the tournament size and separating into another array. Taking k as 1/2 the pop
            tournament = random.sample(pop, int(len(pop)/2))

            #Sort the tournament based on objective values
            tour_obj = []
            new_tournament = []
            for i in tournament:
                tour_obj.append(calculateObj(i, generation, 0))
            tour_obj.sort()

            for i in tour_obj:
                for j in tournament:
                    if(calculateObj(j, generation, 0) == i):
                        new_tournament.append(j)

            tournament = new_tournament
            parent.append(tournament[0])
        parents.append(parent)
        parent = []

    return parents
    
def fully_random_selection(pop):
    parent = []
    parents = []

    unique = []

    for i in range(len(pop)):
        parent_pos = random.randint(0, len(pop)-1)
        while(parent_pos in unique):
            parent_pos = random.randint(0, len(pop)-1)
        parent_pos_2 = random.randint(0, len(pop)-1)
        while(parent_pos_2 == parent_pos):
            parent_pos_2 = random.randint(0, len(pop)-1)
        parent.append(pop[parent_pos])
        parent.append(pop[parent_pos_2])
        parents.append(parent)
        parent = []
    return parents


#One point cross-over
def crossover(parents):
    #Take a random point for performing the crossover
    point = random.randint(1, len(parents[0]))

    #Take the genes till point from parent 1. Then add remaining individuals from parent 2 such that no repetitions occur
    offspring = []
    for i in range(point):
        offspring.append(parents[0][i])

    for i in range(point, len(parents[0])):
        offspring.append(parents[1][i])

    #Checking for repetitions, check from second parent which can be replaced
    unique = []
    duplicate = []
    for i in range(len(offspring)):
        if(offspring[i] not in unique):
            unique.append(offspring[i])
        else:
            j = random.randint(0, len(parents[1])-1)
            while(j in unique):
                j = random.randint(0, len(parents[1])-1)
            if(j not in unique):
                unique.append(j)
                offspring[i] = j
                
    return offspring



# Swapping Mutation - Randomly swap two positions in the job sequence
def mutation(sol):
    pos1 = random.randint(0, len(sol)-1)
    pos2 = random.randint(0, len(sol)-1)
    
    temp = sol[pos1]
    sol[pos1] = sol[pos2]
    sol[pos2] = temp

    return sol


#Population Repair here is performed by first detecting the precedent element and performing random swap with element below antecedent
def population_repair(pop):
    for individual in pop:
        for i in range(len(prec_constraints)):
                #Check if the precedent element is coming after the antecedent
                if(individual.index(prec_constraints[i][0]) > individual.index(prec_constraints[i][1])):
                    precedent_index = individual.index(prec_constraints[i][0])
                    antecedent_index = individual.index(prec_constraints[i][1])
                    #Generate Random index till antecedent
                    rand_ind = random.randint(0, antecedent_index)
                    #Perform Swap
                    individual[precedent_index], individual[rand_ind] = individual[rand_ind], individual[precedent_index]
    
    return pop
            


#Performing Complete Generational Replacement      
def age_based_generation_update(oldPop, newPop):    
    return newPop


def rank_based_generation_update(oldPop, newPop, generation):
    if(len(oldPop)%2 == 0):
        num_old_survivors = len(oldPop)/2
        num_new_survivors = len(oldPop)/2
    else:
        num_old_survivors = len(oldPop)/2
        num_new_survivors = (len(oldPop)/2)+1

    
    old_popObj = []
    new_popObj = []

    old_tot = 0
    new_tot = 0

    surv_old_pop = []
    surv_new_pop = []

    for i in oldPop:
        old_popObj.append(calculateObj(i, generation, 1))
        old_tot+=calculateObj(i, generation, 1)
    

    for i in newPop:
        new_popObj.append(calculateObj(i, generation, 1))
        new_tot+=calculateObj(i, generation, 1)

    wt = []

    for i in oldPop:
        wt.append(calculateObj(i, generation, 1)/old_tot)
    
    surv_old_pop = random.choices(oldPop, weights=wt, k=int(num_old_survivors))
    

    wt = []
    for i in newPop:
        wt.append(calculateObj(i, generation, 1)/new_tot)
    
    surv_new_pop = random.choices(newPop, weights=wt, k=int(num_new_survivors))
   

    return surv_new_pop+surv_old_pop
    
# Returns best solution's index number, best solution's objective value and average objective value of the given population.
def findBestSolution(pop, generation):
    bestObj = calculateObj(pop[0], generation, 1)
    avgObj = bestObj
    bestInd = 0
    for i in range(1, len(pop)):
        tObj = calculateObj(pop[i], generation, 1)
        avgObj = avgObj + tObj
        if tObj < bestObj:
            bestObj = tObj
            bestInd = i
            
    return bestInd, bestObj, avgObj/len(pop)



# Number of population
Npop = 50
# Probability of crossover
Pc = 0.8
# Probability of mutation
Pm = 0.2
# Stopping number for generation
total_generations= 200

# Start Timer
t1 = time.time()

# Creating the initial population
population = initialization(Npop)


print("Started Solver at {}".format(t1))
#Genetic Algo
for i in range(total_generations):

    #Initialising penalty individuals
    penalised_individuals[i] = []

    #Initialising fit individuals
    fit_individuals.append([])

    # Selecting parents
    parents = fully_random_selection(population)

    childs = []
    
    # Apply crossover
    for p in parents:
        r = random.random()
        if r < Pc:
            childs.append(crossover(p))
        else:
            if r < 0.5:
                childs.append(p[0])
            else:
                childs.append(p[1])
    
    
    # Apply mutation 
    for c in childs:
        r = random.random()
        if r < Pm:
            c = mutation(c)
    
    
    #Repair Population
    childs = population_repair(childs)


    bestInd, bestObj, avgObj = findBestSolution(population, i)

    # Update the population
    population = rank_based_generation_update(population, childs, i)
    

    fitness.append(bestObj)
    fit_individuals[i].append(population[bestInd])

    #Break after detecting same makespan - and check penalised individuals
    if(len(fitness)>10):
        flag = 0
        penalty_flag = 0
        start = fitness[i]
        for j in range(i, i-5, -1):
            if(start!=fitness[j]):
                flag = 1
            for k in fit_individuals[j]:
                if(k in penalised_individuals[j]):
                    penalty_flag = 1

        if(flag == 0 and penalty_flag==0):
            f.write("Break at generation {}".format(i) + "\n")
            break  


# Stop Timer
t2 = time.time()


# Results Time
bestSol, bestObj, avgObj = findBestSolution(population, i)
    
#print("Population:")
#print(population)
#print() 

print("Solution:")
print(population[bestSol])
f.write("Solution: {}".format(population[bestSol]) + "\n")

print() 

print("Objective Makespan Value:")
print(bestObj)
f.write("Objective Makespan Value: {}".format(bestObj) + "\n")
print()

print("Average Objective Value of Population:")
print("%.2f" %avgObj)
f.write("Objective Avg Value: {}".format(avgObj) + "\n")
print()

"""print("%Gap:")
G = 100 * (bestObj-optimalObjective) / optimalObjective
print("%.2f" %G)
print()"""

print("CPU Time (s)")
timePassed = (t2-t1)
print("%.2f" %timePassed)
f.write("Time: {}".format(timePassed) + "\n")

f.write("Penalised Individuals: {}".format(penalised_individuals) + "\n")



#completion_times = compute_final_times(population[bestSol], cost)

comp_times = compute_final_times(population[bestSol], cost)
#print(comp_times)

plot_job_scheduling(comp_times, n, m)


# Call the function to plot the graph
plot_fitness(range(len(fitness)), fitness, n, m, total_generations, bestObj)
