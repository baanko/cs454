import numpy as np
import random
class Task(object):
    """docstring for Task"""
    def __init__(self, taskId, cost, skills):
        super(Task, self).__init__()
        self.taskId = taskId
        self.cost = cost
        self.skills = skills
        self.team = []
        self.ready = 0
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
        # remove duplication
        while (r in skills):
            r = random.randint(1,S)
        skills.append(r)

    #randomize team efficiency
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
    print(e.employeeId, e.salary, e.skills)'''

fit = 0
operation_limit = 1000

complete = []
ready = []
time_stamp = [0]
time = 0
employee_available = list(employees)
task_in_progress = []
def check_ready(task, complete, tpg):
    ready = True
    for i in range(0, len(tpg)):
        if tpg[i][1] == task:
            if not tpg[i][0] in complete:
                break
    return ready
def update_ready(tasks, complete, TPG):
    for i in range(0, len(tasks)):
        if check_ready(tasks[i].taskId, complete, TPG):
            ready.append(tasks[i].taskId)
            tasks[i].ready += 1
    
def task_complete(time, employee_available, task_in_progress, complete):
    remove = []
    for i in range(0, len(task_in_progress)):
        if task_in_progress[i][1] == time:
            complete.append(task_in_progress[i][0].taskId)
            employee_available.extend(task_in_progress[i][0].team)
            remove.append(i)
    for i in range(0, len(remove)):
        del task_in_progress[remove[len(remove) - i - 1]]

def time_taken(task):
    size = len(task.team)
    efficiency = 0
    total = 0
    skill = len(task.skills)
    if size == 1:
        return task.cost*task.team[0].team[task.team[0].employeeId - 1]
    for i in range(1, size-1):
        for j in range(i, size):
            num = 0
            task.team[i-1]
            for k in range(0, skill):
                task.team[i-1].skills
                if skill[k] in task.team[i-1].skills or skill[k] in task.team[j]:
                    num += 1
            total += 1
            efficiency += task.team[i-1].team[k] * num/skill
    efficiency = size*efficiency
    return task.cost*efficiency

def fitness_eval(answer):
    cost = 0
    for i in range(0, len(answer[0])):
        time = time_taken(answer[0][i])
        for j in range(0, len(answer[0][i].team)):
            cost += time*answer[0][i].team[j].salary
    return 1/(cost*0.000001 + answer[1]*0.1)
def select_task(ready, task_pheromone):
    total = 0
    for i in range(0, len(ready)):
        total += task_pheromone[ready[i]-1]
    r = random.random()*total
    count = 0
    for i in range(0, len(ready)):
        count += task_pheromone[ready[i]-1]
        if count > r:
            return ready[i]

def fill_employee(taskId, employee_available, employee_pheromone, task_in_progress, time):
    for i in range(0, len(tasks)):
        if tasks[i].taskId == taskId:
            task = tasks[i]
            break
    task.team = []
    skill = list(task.skills)
    while not len(skill) == 0:
        c = False
        for i in range(0, len(employee_available)):
            if employee_available not in task.team:
                if skill[0] in employee_available[i].skills:
                    c = True
                    if random.random() < employee_pheromone[employee_available[i].employeeId - 1][task.taskId - 1]:
                        task.team.append(employee_available[i])
                        del skill[0]
                        for k in range(0, len(employee_available[i].skills)):
                            if employee_available[i].skills[k] in skill:
                                skill.remove(employee_available[i].skills[k])
                        break
        if not c:
            return False
    for i in range(0, len(task.team)):
        employee_available.remove(task.team[i])
    task_in_progress.append([task, time_taken(task) + time])
    return time_taken(task)
                        
                

    
    
def update_pheromone(task_pheromone, employee_pheromone, answer, fit):
    for i in range(0, len(answer[0])):
        if not answer[0][i].ready == 0:
            task_pheromone[answer[0][i].taskId - 1] += 1/answer[0][i].ready
        answer[0][i].ready = 0
        for j in range(0, len(answer[0][i].team)):
            employee_pheromone[answer[0][i].team[j].employeeId -1][answer[0][i].taskId -1] += 1/fit
    return
    
population = 10
task_pheromone = [1]*len(tasks)
employee_pheromone = [[1]*len(tasks)]*len(employees)
while (fit < operation_limit):
    best = 0
    best_answer = []
    for ant in range(0, population):
        print(fit)
        while not len(complete) == len(tasks): 
            print(time_stamp)
            time = time_stamp[0]
            del time_stamp[0]
            task_complete(time, employee_available, task_in_progress, complete)
            update_ready(tasks, complete, TPG)
            success = True
            while (not len(ready) == 0):
                task = select_task(ready, task_pheromone)
                success = fill_employee(task, employee_available, employee_pheromone, task_in_progress, time)
                if (success):
                    ready.remove(task)
                    time_stamp.append(success + time)
                    time_stamp.sort()
                else:
                    ready.remove(task)
        answer = [tasks, time]
        fitness = fitness_eval(answer)
        fit += 1
        if fitness > best:
            best = fitness
            best_answer = answer
    print(best)
    update_pheromone(task_pheromone, employee_pheromone, best_answer, fit)
        

