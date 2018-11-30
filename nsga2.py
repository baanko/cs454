import math
import random
import sys
import copy
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List
import numpy as np
from plotly.validators.surface.contours.x import project

""" Important parts on MOCell:
    S: Number of Skills
    T: Number of Tasks
    E: Number of Employees
    Solution class object: Make a 2-dimensional employee allocation matrix and save each employee's working hour for each task. It's form is 1-dimensional array with size T * E. It also saves the size T * E and the location (the index of the individual).
    SBX crossover: Implemented that operator
    Mutation: change 1 variable among T*E elements in the employee allocation matrix for 2 offspring genes
    Selecting parent: pick 2 among the 8 neighbors of the current population. Neighbor doesn't occur after that in the remaining evaluation cycle.
    crossover: define SBX operator and apply execute function to the parents and put them into the offspring
    Solve TPG: ...
    Replacement on the current population: replacing the individual's parameter (employee allocation matrix) by offspring[0]'s one after comparing fitness of the original one and the offspring[0] (and add that offspring[0] to archive) (Note that picking individual from current population is done by shallow copy (referencing), not by deep copy (which should be copied by copy.deepcopy(nameOfObject)).

     Changes to adapt to NSGA-II:
    - do non-dominated sort on initial population based on the fitness. (Note that pareto-front on single-objective task is just a sorting)
    - Assign crowding distance : non needed for single object task, but if it becomes MO, do that
    - Selection for crossover/mutation: use crowded comparison operator for binary tournament selection - select currentPopulation times. (Find other implementations)
    - crossover: do on the offspring by SBX
    - mutation:
    - recombination: pick the best currentPopulation genes (it performs non-dominated sort and crowding distance computation), and go to the selection process.
    """

#####################instance_generator.py###############################3
class Task(object):
    """docstring for Task"""
  def __init__(self, taskId, cost, skills):
      super(Task, self).__init__()
    self.taskId = taskId
    self.cost = cost
    self.skills = skills
class Employee(object):
    """docstring for Employee"""
  def __init__(self, employeeId, salary, skills):
      super(Employee, self).__init__()
    self.employeeId = employeeId
    self.salary = salary
    self.skills = skills

# set the number of objectives (fitnesses)
num_objectives = 1

# randomize the number of skills between 5 and 10 inclusively
S = random.randint(7,10)

# randomize the number of tasks between 5 and 10 inclusively
T = random.randint(5,10)

tasks = []

# generate tasks
for i in range(1, T+1):
  # sample a cost from a normal distribution with mu=10 and std=5
  cost = round(np.random.normal(10,5))

  skills = []

  # randomize number of skills required for this task
  numSkills = random.randint(2,3)


  # randomize skills for this task
  for j in range(numSkills):
    r = random.randint(1, S)
    # remove duplication
    while (r in skills):
      r = random.randint(1, S)
    skills.append(r)

  # maintain the skill list as sorted
  skills.sort()
  # add task instance to task list
  tasks.append(Task(i, cost, skills))

# randomize the rate of edge/task in the Task Precedence Graph
evRate = np.random.normal(1.5,0.5)
numEdge = round(evRate * float(T))

TPG = []
countEdge = 0

# randomize a number of edges (a, b) in which a < b
while countEdge != numEdge:
  a = random.randint(1,T-1)
  b = random.randint(a+1,T)

  # dont add the edge into TPG if its already in there (avoid overlaps)
  if (a,b) not in TPG:
    TPG.append((a,b))
    countEdge += 1

# randomize number of employees
E = random.randint(10,15)
employees = []

# randomize the salary and skills for each employees
for i in range(1,E+1):
  salary = round(np.random.normal(10000,1000))
  skills = []

  numSkills = random.randint(6,7)
  # because of the malicious usage of i, append operation inserted the wrong i for the Employee object. Changed from i to j.
  for j in range(numSkills):
    r = random.randint(1,S)
    # remove duplication
    while (r in skills):
      r = random.randint(1,S)
    skills.append(r)

  # maintain the skill list as sorted
  skills.sort()
  employees.append(Employee(i, salary, skills))

'''if __name__ == "__main__":
    for t in tasks:
        print(t.taskId, t.cost, t.skills)
    for ed in TPG:
        print(ed)
    for e in employees:
        print(e.employeeId, e.salary, e.skills)
'''
#####################instance_generator.py###############################3

S = TypeVar('S')
R = TypeVar('R')
class Operator(Generic[S, R]):
    """ Class representing operator """

    __metaclass__ = ABCMeta

    @abstractmethod
    def execute(self, source: S) -> R:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class Solution(Generic[S]):
    """ Class representing solutions """

    __metaclass__ = ABCMeta

    def __init__(self, number_of_objectives: int, number_of_variables: int, variables: List[float]):
        '''
        Constructor. Parameters:
        number_of_variables
        variables: time (in hour) that each employee has been allocated for that hour for such task
        Other elements:
        rank: rank for NSGA-II
        distance: distance for crowding-distance
        '''
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables
        self.lower_bound=[0.0 for _ in range(self.number_of_variables)]
        self.upper_bound =[1.0 for _ in range(self.number_of_variables)]

        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.variables = variables
        self.attributes = {}
        self.rank = sys.maxsize
        self.distance = 0.0

    def __copy__(self):
        '''
        Copying operator
        '''
        new_solution = Solution(
                self.number_of_variables,
                self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.rank = self.rank
        new_solution.distance = self.distance

        return new_solution

    def evaluate(self):
        # TODO: adapt fitness evaluation for NSGA-II

        fitness=[]
        for test in range(2):
            object = individual
            if test==1:
                object = offspring[0]
            undt =0
            for i in range(T):
                k=0
                for j in range(E):
                    k=k+object.variables[i*E+j]

                if k==0:
                    undt=undt+1

            reqsk=0
            for i in tasks:
                s = set([])
                for j in employees:
                    if object.variables[(i.taskId-1)*E+j.employeeId-1]>0:
                        s = s.union(set(j.skills))

                s = set(i.skills)-s

                reqsk = reqsk+len(s)



            solvable= 1
            projectduration=0
            unfinished =copy.deepcopy(tasks)

            TPG2 = copy.deepcopy(TPG)
            totaloverwork=0
            while (TPG)!=0:
                V=[]
                depended = []
                for tpg in TPG:
                    if tpg[1] not in depended:
                        depended.append(tpg[1])

                for f in unfinished:
                    if f.taskId not in depended:
                        V.append(f)
                overwork=0

                if (len(V)==0):
                    solvable=0
                    break
                dedication=[]

                ratio=[]
                dedicationj=[]
                i=0
                for v in V:
                    d=0
                    for e in employees:
                        ded = object.variables[(v.taskId-1)*E+e.employeeId-1]
                        dedication.append(ded)
                        d = d+ded
                    if d==0:
                        solvable=0
                        break
                    dedicationj.append(d)


                    ratio.append(v.cost/d)
                    i=i+1
                dedsum=0
                for e in employees:
                    for p in range(i):
                        dedsum=dedsum+dedication[p*E+e.employeeId-1]
                    if dedsum>1:
                        overwork=overwork+dedsum-1

                t = min(ratio)
                projectduration = projectduration+t
                i=0
                deleted=[]
                if solvable==0:
                    break
                for j in V:
                    for un in unfinished:
                        if un.taskId == j.taskId:
                            un.cost = un.cost - t*dedicationj[i]
                            if un.cost<=0.001:
                                deleted.append(j.taskId)

                    i=i+1
                totaloverwork = totaloverwork +overwork*t


                for j in unfinished:
                    if j.taskId in deleted:
                        del unfinished[unfinished.index(j)]
                for tpg in TPG:
                    if (tpg[0] in deleted) or (tpg[1] in deleted):
                        del TPG[TPG.index(tpg)]


            projectcost=0
            tkj=[]
            Pei=[]
            for task in tasks:
                sum=0
                for employee in employees:
                    sum=sum+object.variables[(task.taskId-1)*E+employee.employeeId-1]
                tkj.append(task.cost/sum)
            for employee in employees:
                Pei.append(employee.salary)
            for employee in employees:
                for task in tasks:
                    projectcost = projectcost+object.variables[(task.taskId-1)*E+employee.employeeId-1]*tkj[task.taskId-1]*Pei[employee.employeeId-1]


            if totaloverwork>0:
                fitness.append( 1/(projectcost*0.000001+projectduration*0.1+100+10*undt+10*reqsk+0.1*totaloverwork))
        if fitness[0]<fitness[1]:
            print("fitness", fitness[1])
            individual.variables = offspring[0].variables
            individual.location = offspring[0].location
            archive.append(individual)
        # evaluate solution and update objective values.
        raise NotImplementedError("Solution class have to be implemented")

    '''
    def crossover(self, other):
        # crossover using such operator
        raise NotImplementedError("Solution class have to be implemented")
        '''

    '''
    def mutate(self):
        # mutate with such mutation operator
        raise NotImplementedError("Solution class have to be implemented")
    '''

    def __rshift__(self, other):
        '''
        True if this solution dominates the other (">>" operator)
        '''
        dominates = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False

            elif self.objectives[i] < other.objectives[i]:
                dominates = True

        return dominates

    def __lshift__(self, other):
        '''
        True if the other solution dominates this solution ("<<" operator)
        '''
        return other >> self

def crowded_comparison(s1, s2):
    # compare s1 and s2 based on crowded comparison
    # if s1 has a priority, return -1. else if s2 has a priority, return 1. else, return 0
    if s1.rank < s2.rank:
        return 1
    elif s1.rank > s2.rank:
        return -1
    elif s1.distance > s2.distance:
        return 1
    elif s1.distance < s2.distance:
        return -1
    else:
        return 0

class Crossover(Operator[List[S], List[R]]):
    """ Class representing crossover operator. """

    __metaclass__ = ABCMeta

    def __init__(self, probability: float):
        if probability > 1.0:
            raise Exception('The probability is greater than one: {}'.format(probability))
        elif probability < 0.0:
            raise Exception('The probability is lower than zero: {}'.format(probability))

        self.probability = probability

    @abstractmethod
    def execute(self, source: S) -> R:
        pass

class SBX(Crossover[Solution, Solution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(SBX, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))

        offspring = [copy.copy(parents[0]), copy.copy(parents[1])]
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]

                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1

                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[1].upper_bound[i]

                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))

                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))

                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))

                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound

                        if random.random() <= 0.5:
                            offspring[0].variables[i] = c2
                            offspring[1].variables[i] = c1
                        else:
                            offspring[0].variables[i] = c1
                            offspring[1].variables[i] = c2
                    else:
                        offspring[0].variables[i] = value_x1
                        offspring[1].variables[i] = value_x2
                else:
                    offspring[0].variables[i] = value_x1
                    offspring[1].variables[i] = value_x2
        return offspring

def mutation(Solution1, Solution2):
    i=random.randint(0, T*E-1)
    j =random.randint(0,10)/10
    Solution1.variables[i] = j
    i=random.randint(0, T*E-1)
    j =random.randint(0,10)/10
    Solution2.variables[i] = j
    return [Solution1, Solution2]

class NSGA2:
    '''
    Implement parts of NSGA-II
    '''

    current_evaluated_objective = 0

    def __init__(self, num_objectives, num_variables, crossover_rate = 0.9):
        '''
        Constructor. Parameters: number of objectives, number of variables (size of the employee allocation matrix, crossover_rate (default 90%)
        '''
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.crossover_rate = crossover_rate

        random.seed()

    def run(self, P, population_size, num_generations):
        # Run the NSGA-II instance

        for s in P:
            s.evaluate()

        Q = []

        for i in range(num_generations):
            # print("Iteration ", i)

            # combine parent and offspring
            R = []
            R.extend(P)
            R.extend(Q)

            fronts = self.fast_nondominated_sort(R) # front construction

            del P[:] # make parent P empty

            front = []
            for front in fronts.values(): # fill parent until it reaches the size
                if len(front) == 0: # Assert the non-emptyness of the front
                    break

                elif len(front) + len(P) > population_size:
                    break

                self.assign_crowding_dist(front) # assign crowding distance
                P.extend(front)

                '''
                if len(P) >= population_size: # parent reaches the size
                    break
                '''
            self.sort_crowdingdist(front) # sort by crowding_distance
            P.append(front[:(population_size-len(P))]) # add first (population_size - len(P)) elements from the last front

            Q = self.make_offspring(P)

    def sort_objective(self, P, obj_idx):
        # sort the popoulation (or the front) by obj_idx'th objective
        P.sort(key=lambda x: x.objectives[obj_idx])

    def sort_crowdingdist(self, P):
        # sort the population (or the front) by decreasing order of crowding distance. note that rank order should be preserved.
        P.sort(key=lambda x:x.distance, reverse=True)
        P.sort(key=lambda x:x.rank)

    def make_offspring(self, P):
        # make offspring by crossover and mutation
        # selection of parents will be done by random
        Q = []
        crossover = SBX(probability=crossover_rate, distribution_index=20) # initiate crossover instance

        while len(Q) < len(P):
            parents = [None, None]

            while parents[0] == parents[1]:
                # Additionally on the random choice, select two and pick one by the order of crowded comparison operaor
                for i in range(2):
                    s1 = random.choice(P)
                    s2 = s1
                    while s1 is s2:
                        s2 = random.choice(P)

                    if crowded_comparison(s1, s2) > 0:
                        selected_solutions[i] = s1
                    else:
                        selected_solutions[i] = s2

            if random.random() < self.crossover_rate:

                child = crossover.execute(parents)

                child = mutation(child[0], child[1])

                child[0].evaluate()
                Q.append(child)
                if len(Q) < len(P):
                    child[1].evaluate()
                    Q.append(child[1])

        return Q

    def fast_nondominated_sort(self, P):
        # discover pareto fronts in P based on non-domination criterion
        fronts = {}

        S = {} # Set of dominated instance of each instance p
        n = {} # number of dominating instance of each instance p
        for s in P: # use each instance of P as an index
            S[s] = []
            n[s] = 0

        fronts[1] = [] # pareto fronts will be constructed from rank 1

        for p in P: # with each instance p,
            for q in P: # compare each instance q for p
                if p is q: # p and q are instances of P, so is works as reference check
                    continue

                if p >> q: # p dominates q
                    S[p].append[q]

                elif p << q: # q dominates p
                    n[p] += 1

            if n[p] == 0:
                p.rank = 1 # rank of this instance is 1
                fronts[1].append(p) # insert this instance to rank 1 front

        i = 1
        while len(fronts[i]) != 0: # see each dominated instance of each front
            next_front = []
            for p in fronts[i]:
                for q in S[p]: # find a dominates set q by p
                    n[q] -= 1 # domination is recovered
                    if n[q] == 0: # no other dominates q anymore
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts[i] = next_front # fill the next front

        return fronts # return the front

    def assign_crowding_distance(self, front):
        '''
        assign a crowding distance for each solution in each front.
        note that front is called as a reference, so changing the value here affects the entire population
        '''
        for p in front:
            p.distance = 0 # distance initialize

        for obj_index in range(self.number_of_objectives):
            self.sort_objective(front, obj_index) # sort by each objective

            front[0].distance = front[len(front) - 1].distance = float('inf') # boundary points are always selected
            for i in range(1, len(front) - 1):
                front[i].distance += (front[i+1].distance - front[i-1].distance) # Originally added value should be devided by the difference between the maximum possible fitness value and the minimum possible fitness value, but since we will regulate fitness as smaller than 1, just ignore.


#initialize currentpopulation
crossover_rate=0.9
populationSize=50
archiveSize=15
maxEvaluations=1000
feedback=20
currentpopulation = []
solution = []
currentPopulation = []
evaluations = 0
archive = []

for i in range(0, populationSize):
    temp=[]
    for i in range(T):
        temp2=[]
        for j in range(E):
            ded = random.randint(0,10)
            ded = ded/10
            temp2.append(ded)
        temp = temp+temp2
    individual = Solution(variables=temp, number_of_variables=T*E, number_of_objectives=num_objectives)
    currentPopulation.append(individual)

#iteration
while (evaluations < maxEvaluations):
    for popul in range(0, len(currentPopulation)):
        individual = currentPopulation[popul]
        parents = [Solution(variables=[], number_of_variables=T*E, number_of_objectives=num_objectives) for ii in range(2)]
        offspring = [Solution(variables=[], number_of_variables=T*E, number_of_objectives=num_objectives) for ii in range(2)]

        neighbors[popul] = getEightNeighbors(neighborhood, currentPopulation, popul)

        ind = random.randint(0, len(neighbors[popul])-1)
        parents[0] = neighbors[popul][ind]

        if len(archive)>0:
            ind = random.randint(0, len(archive)-1)
            parents[1] = archive[ind]
        else:
            ind = random.randint(0, len(neighbors[popul])-1)
            parents[1] = neighbors[popul][ind]

        crossover = SBX(probability=crossover_rate, distribution_index=20)
        offspring = crossover.execute(parents)

        offspring = mutation(offspring[0], offspring[1])
        evaluations = evaluations+1

        fitness=[]
        for test in range(2):
            object = individual
            if test==1:
                object = offspring[0]
            undt =0
            for i in range(T):
                k=0
                for j in range(E):
                    k=k+object.variables[i*E+j]

                if k==0:
                    undt=undt+1

            reqsk=0
            for i in tasks:
                s = set([])
                for j in employees:
                    if object.variables[(i.taskId-1)*E+j.employeeId-1]>0:
                        s = s.union(set(j.skills))

                s = set(i.skills)-s

                reqsk = reqsk+len(s)



            solvable= 1
            projectduration=0
            unfinished =copy.deepcopy(tasks)

            TPG2 = copy.deepcopy(TPG)
            totaloverwork=0
            while (TPG)!=0:
                V=[]
                depended = []
                for tpg in TPG:
                    if tpg[1] not in depended:
                        depended.append(tpg[1])

                for f in unfinished:
                    if f.taskId not in depended:
                        V.append(f)
                overwork=0

                if (len(V)==0):
                    solvable=0
                    break
                dedication=[]

                ratio=[]
                dedicationj=[]
                i=0
                for v in V:
                    d=0
                    for e in employees:
                        ded = object.variables[(v.taskId-1)*E+e.employeeId-1]
                        dedication.append(ded)
                        d = d+ded
                    if d==0:
                        solvable=0
                        break
                    dedicationj.append(d)


                    ratio.append(v.cost/d)
                    i=i+1
                dedsum=0
                for e in employees:
                    for p in range(i):
                        dedsum=dedsum+dedication[p*E+e.employeeId-1]
                    if dedsum>1:
                        overwork=overwork+dedsum-1

                t = min(ratio)
                projectduration = projectduration+t
                i=0
                deleted=[]
                if solvable==0:
                    break
                for j in V:
                    for un in unfinished:
                        if un.taskId == j.taskId:
                            un.cost = un.cost - t*dedicationj[i]
                            if un.cost<=0.001:
                                deleted.append(j.taskId)

                    i=i+1
                totaloverwork = totaloverwork +overwork*t


                for j in unfinished:
                    if j.taskId in deleted:
                        del unfinished[unfinished.index(j)]
                for tpg in TPG:
                    if (tpg[0] in deleted) or (tpg[1] in deleted):
                        del TPG[TPG.index(tpg)]


            projectcost=0
            tkj=[]
            Pei=[]
            for task in tasks:
                sum=0
                for employee in employees:
                    sum=sum+object.variables[(task.taskId-1)*E+employee.employeeId-1]
                tkj.append(task.cost/sum)
            for employee in employees:
                Pei.append(employee.salary)
            for employee in employees:
                for task in tasks:
                    projectcost = projectcost+object.variables[(task.taskId-1)*E+employee.employeeId-1]*tkj[task.taskId-1]*Pei[employee.employeeId-1]


            if totaloverwork>0:
                fitness.append( 1/(projectcost*0.000001+projectduration*0.1+100+10*undt+10*reqsk+0.1*totaloverwork))
        if fitness[0]<fitness[1]:
            print("fitness", fitness[1])
            individual.variables = offspring[0].variables
            individual.location = offspring[0].location
            archive.append(individual)

for i in range(archiveSize):
    print(archive[len(archive)-1-i].variables)


