import numpy as np
import random

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
  for i in range(numSkills):
    skills.append(random.randint(1,S))

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
  for i in range(numSkills):
    skills.append(random.randint(1,S))

  employees.append(Employee(i, salary, skills))
