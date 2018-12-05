import math
import random
import copy
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, List
import numpy as np

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
  def __init__(self, employeeId, salary, skills, team):
    super(Employee, self).__init__()
    self.employeeId = employeeId
    self.salary = salary
    self.skills = skills
    self.team = team

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
    team = [None]*E
      
    numSkills = random.randint(6,7)
      # because of the malicious usage of i, append operation inserted the wrong i for the Employee object. Changed from i to j.
    for j in range(numSkills):
        r = random.randint(1,S)
        while (r in skills):
            r = random.randint(1,S)
        skills.append(r)

      
    for k in range(1, E+1):
        if k < i:
            team[k-1] = employees[k-1].team[i-1]
        else:
            team[k-1] = random.random() * 2
  # maintain the skill list as sorted
    skills.sort()
    employees.append(Employee(i, salary, skills, team))

'''if __name__ == "__main__":
        for t in tasks:
            print(t.taskId, t.cost, t.skills)
        for ed in TPG:
            print(ed)
        for e in employees:
            print(e.employeeId, e.salary, e.skills, e.team)'''
#####################instance_generator.py###############################3

T=8;
tasks=[];


E=10
employees=[]
cost=[8.0,8.0,8.0,3.0,17.0,20.0,10.0,9.0]
skills_t = [[6, 7],[3, 5, 7],[2, 3],[3, 6, 7],[2, 3, 7],[1, 2],[2, 3],[1, 6, 7]]

salary = [9465.0,9475.0,11501.0,9769.0,10749.0,10288.0,8571.0,10903.0,11037.0,9828.0]
skills_e = [[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 7],[1, 2, 3, 4, 5, 6, 7],[2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 7],[1, 2, 3, 4, 5, 7],[1, 2, 3, 4, 5, 6, 7],[2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7]]

team = [[1.9850026010223878, 0.18212375051252394, 0.7176406839199339, 1.5340873914952124, 0.14863581676990179, 1.3932135973019686, 0.47077045345645807, 1.4890159454850462, 0.27824517847103625, 1.5964701089741966]
,[0.18212375051252394, 0.15133877966497167, 0.7778187939774182, 0.7981114960025331, 0.8472192924749398, 1.3930634011990006, 0.4909995217013994, 1.1151725310028653, 1.2789818710617091, 1.802499326729509]
,[0.7176406839199339, 0.7778187939774182, 1.174265498685762, 0.8618080817973071, 0.6399809057368213, 0.9260073334361711, 0.10667661971917908, 0.6483140981769688, 0.9415352155373198, 0.2792483512974877]
,[1.5340873914952124, 0.7981114960025331, 0.8618080817973071, 1.4340801631563556, 1.2175021462906137, 1.6176868217802627, 0.2228937417974326, 1.2943922301988497, 1.2756081176813578, 1.3051895787048884]
,[0.14863581676990179, 0.8472192924749398, 0.6399809057368213, 1.2175021462906137, 0.31192937924613173, 0.47800990295685675, 0.4226701520364715, 1.0721857998635547, 0.27603041409724094, 1.3357600021901013]
,[1.3932135973019686, 1.3930634011990006, 0.9260073334361711, 1.6176868217802627, 0.47800990295685675, 0.10904442155462712, 1.3622364050254925, 1.4000271252322727, 0.48386533282966626, 0.35237321377639286]
,[0.47077045345645807, 0.4909995217013994, 0.10667661971917908, 0.2228937417974326, 0.4226701520364715, 1.3622364050254925, 0.4157121650281814, 1.4348420931759798, 1.1016468388691938, 1.9628272200314718]
,[1.4890159454850462, 1.1151725310028653, 0.6483140981769688, 1.2943922301988497, 1.0721857998635547, 1.4000271252322727, 1.4348420931759798, 0.08120641201010037, 0.8233022739244853, 0.07368181520046768]
,[0.27824517847103625, 1.2789818710617091, 0.9415352155373198, 1.2756081176813578, 0.27603041409724094, 0.48386533282966626, 1.1016468388691938, 0.8233022739244853, 1.6408448059389065, 0.3105475664853563]
,[1.5964701089741966, 1.802499326729509, 0.2792483512974877, 1.3051895787048884, 1.3357600021901013, 0.35237321377639286, 1.9628272200314718, 0.07368181520046768, 0.3105475664853563, 0.7910222561839355]
]
TPG = [(1, 7), (3, 4), (2, 5), (4, 5), (1, 6), (3, 5), (2, 6)]

for i in range(1,T+1):
    tasks.append(Task(i, cost[i-1], skills_t[i-1]))
for j in range(1,E+1):
    employees.append(Employee(j, salary[j-1], skills_e[j-1], team[j-1]))

def neighborhood(size):
    structure = [[0]*8 for i in range(size)]
    #for i in range(0,size):
    #    for j in range(0, 2):
    #            structure[i][j]=[]
                
    rowsize = int(math.sqrt(size))

    for i in range(0, size):
        if (i >rowsize-1):
            structure[i][0] = i=rowsize
        else:
            structure[i][0] = (i-rowsize+size)%size
    
        if (i+1)%rowsize==0:
            structure[i][2] = i-rowsize+1
        else:
            structure[i][2] = i+1;
            
        if i % rowsize ==0:
            structure[i][3] = i+rowsize-1
        else:
            structure[i][3] = i-1;
            
        structure[i][1] = (i+rowsize)%size;
        
        
    for i in range(0, size):
        structure[i][6] = structure[structure[i][0]][2]
        structure[i][4] = structure[structure[i][0]][3]
        structure[i][7] = structure[structure[i][1]][2]
        structure[i][5] = structure[structure[i][1]][3]
    
    return structure

def getEightNeighbors(struc, list, num):
    neighbors = []
    for i in range(8):
        index = struc[num][i]
        neighbors.append(list[index])
    return neighbors;

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

    def __init__(self, number_of_variables: int, location:int, variables: List[float]):
        self.number_of_objectives = 1;
        self.number_of_variables = number_of_variables
        self.location = location
        self.lower_bound=[0.0 for _ in range(self.number_of_variables)]
        self.upper_bound =[1.0 for _ in range(self.number_of_variables)]
        
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.variables = variables
        self.attributes = {}

    def __copy__(self):
        new_solution = Solution(
            self.number_of_variables,
            self.number_of_objectives,
            self.location)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        return new_solution
    
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



populationSize=64
archiveSize=15
maxEvaluations=1000

currentpopulation = [] 
neighborhood = neighborhood(populationSize)
neighbors = [Solution(variables=[], number_of_variables=T*E, location=0) for i in range(populationSize)]
solution = []
currentPopulation = []
evaluations = 0
archive = []

#initialize currentpopulation
for i in range(0, populationSize):
    temp=[]
    for i in range(T):
        temp2=[]
        for j in range(E):
            ded = random.randint(0,10)
            ded= ded/10
            temp2.append(ded)
        temp = temp+temp2
    individual = Solution(variables=temp, number_of_variables=T*E, location=i)
    currentPopulation.append(individual)
  
#iteration    
while (evaluations < maxEvaluations):
    for popul in range(0, len(currentPopulation)):
        individual = currentPopulation[popul]
      
        parents =  [Solution(variables=[], number_of_variables=T*E, location=0) for ii in range(2)]
        offspring = [Solution(variables=[], number_of_variables=T*E, location=0) for ii in range(2)]
        
        neighbors[popul] = getEightNeighbors(neighborhood, currentPopulation, popul)
       
        ind = random.randint(0, len(neighbors[popul])-1)
        parents[0] = neighbors[popul][ind]
    
        if len(archive)>0:
            if len(archive)>archiveSize:
                ind = random.randint(0, archiveSize-1);
                parents[1] = archive[ind+len(archive)-archiveSize]
            else:
                ind = random.randint(0, len(archive)-1)
                parents[1] = archive[ind]
        else:
            ind = random.randint(0, len(neighbors[popul])-1)
            parents[1] = neighbors[popul][ind]        

        crossover = SBX(probability=0.9, distribution_index=20)
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

            while (TPG2)!=0:
                V=[]
                depended = []
             
                for tpg in TPG2:
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

                            if un.cost<=0.000001:
                                deleted.append(j.taskId)
                    
                    i=i+1
                totaloverwork = totaloverwork +overwork*t
    
    
                for j in unfinished:
                    if j.taskId in deleted:
                        del unfinished[unfinished.index(j)]
                for tpg in TPG2:
                    if (tpg[0] in deleted) or (tpg[1] in deleted):
                        del TPG2[TPG2.index(tpg)]
                    
               
            projectcost=0 
            tkj=[]
            Pei=[]
            for task in tasks:
                sum=0
                efficiency=0
                
                for employee in employees:
                    sum=sum+object.variables[(task.taskId-1)*E+employee.employeeId-1]
                ratio_sum=0
                for em in range(0,E-1):
                    for em2 in range(em,E):
                        num=0;
                        for sk in task.skills:
                            if sk in employees[em].skills or sk in employees[em2].skills:
                                num=num+1;
                        efficiency = efficiency+employees[em].team[em2]*num/len(task.skills)
                        ratio_sum=ratio_sum+num/len(task.skills)
                
                tkj.append(task.cost+efficiency/(sum*ratio_sum))
            for employee in employees:
                Pei.append(employee.salary)
            for employee in employees:
                for task in tasks:
                    projectcost = projectcost+object.variables[(task.taskId-1)*E+employee.employeeId-1]*tkj[task.taskId-1]*Pei[employee.employeeId-1]

            q=projectcost*0.000001+projectduration*0.1;
            r=100+10*undt+10*reqsk+0.1*totaloverwork;

            if totaloverwork>0:
                fitness.append( 1/(q+r))
        if fitness[0]<fitness[1]:
            if (evaluations%10==0):
                print(fitness[1])
            individual.variables = offspring[0].variables
            individual.location = offspring[0].location
            archive.append(individual)   
        else:
            if (evaluations%10==0):
                print(fitness[0])
for i in range(archiveSize):
    print(archive[len(archive)-1-i].variables)
        
            