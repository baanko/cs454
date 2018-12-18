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
        self.nextTask = []
class Employee(object):
    """docstring for Employee"""
    def __init__(self, employeeId, salary, skills, team):
        super(Employee, self).__init__()
        self.employeeId = employeeId
        self.salary = salary
        self.skills = skills
        self.team = team

tskill = [[1, 4, 6],
[2, 6],
[1, 5],
[2, 4],
[5, 8],
[1, 7],
[1, 5],
[1, 2, 5],
[1, 4, 8],
[2, 5],
[1, 3, 4],
[4, 5],
[2, 7],
[1, 3, 4],
[4, 5, 6],
[1, 6, 8],
[4, 6, 8],
[3, 4],
[1, 7],
[1, 4, 5]]

tcost = [7.0,
9.0,
14.0,
0.0,
4.0,
7.0,
11.0,
3.0,
10.0,
10.0,
15.0,
6.0,
12.0,
3.0,
16.0,
13.0,
11.0,
15.0,
5.0,
7.0]

eskill = [[7, 5, 8, 6, 2, 1],
[5, 2, 7, 3, 1, 6],
[3, 1, 4, 6, 8, 2],
[8, 5],
[3, 5],
[3, 8],
[6, 8],
[8, 5],
[8, 5],
[5]]

salary = [20083.0,
20166.0,
19913.0,
6812.0,
6803.0,
6587.0,
6545.0,
6758.0,
6751.0,
3424.0]

team = [[1.919160459324181, 0.8529930058987774, 0.6105265724064588, 1.5010990351880813, 1.4737285755229392, 1.9675735939480248, 0.1831727590154817, 0.9982875270053955, 0.850982708607745, 1.2302698665466596],
[0.8529930058987774, 1.0178536981814996, 0.7443126578016201, 0.8110855780253616, 1.0869935099851775, 0.8605644559996153, 1.6984309782840035, 1.5580634090438945, 1.278925006997476, 0.2657116013970331],
[0.6105265724064588, 0.7443126578016201, 0.12860598778900778, 1.8246011974464753, 1.937169812233241, 1.920160692187233, 0.46460506137107593, 1.045549072882408, 0.5083078529831011, 1.2390202507646226],
[1.5010990351880813, 0.8110855780253616, 1.8246011974464753, 1.2132507586466625, 1.8846159694705114, 1.064289079722209, 0.29171712827122565, 1.017032361095393, 1.9918732588553136, 0.6282521636720813],
[1.4737285755229392, 1.0869935099851775, 1.937169812233241, 1.8846159694705114, 1.8407430066416839, 0.41710662906842577, 1.3730613298861083, 0.08176663019576558, 0.8560084652854885, 1.5257655036969004],
[1.9675735939480248, 0.8605644559996153, 1.920160692187233, 1.064289079722209, 0.41710662906842577, 1.8180659735001437, 1.505047434524321, 1.1970331956284794, 1.231134646058145, 1.3498197851117766],
[0.1831727590154817, 1.6984309782840035, 0.46460506137107593, 0.29171712827122565, 1.3730613298861083, 1.505047434524321, 1.4008707195065058, 0.006179390362634818, 0.617790706474473, 1.5180016606145765],
[0.9982875270053955, 1.5580634090438945, 1.045549072882408, 1.017032361095393, 0.08176663019576558, 1.1970331956284794, 0.006179390362634818, 1.7668636256879924, 1.0191324549880407, 1.4488840042073303],
[0.850982708607745, 1.278925006997476, 0.5083078529831011, 1.9918732588553136, 0.8560084652854885, 1.231134646058145, 0.617790706474473, 1.0191324549880407, 1.2346639410058655, 0.2542021710943949],
[1.2302698665466596, 0.2657116013970331, 1.2390202507646226, 0.6282521636720813, 1.5257655036969004, 1.3498197851117766, 1.5180016606145765, 1.4488840042073303, 0.2542021710943949, 0.2689696007215854]]

TPG = [(13, 20), (5, 7), (9, 20), (19, 20), (8, 12), (8, 17), (15, 16), (3, 4), (12, 17), (4, 16), (4, 13), (3, 9), (8, 15), (10, 13), (15, 18), (12, 18), (17, 19), (13, 15), (17, 20), (6, 16), (9, 15), (6, 13), (4, 5), (18, 20), (13, 19), (17, 18), (7, 10), (10, 16), (13, 16), (1, 3), (1, 6), (18, 19), (12, 13), (7, 15), (16, 20), (7, 19), (14, 16), (9, 11), (14, 20), (11, 20), (6, 7), (3, 11), (9, 10), (4, 8)]

# randomize the number of skills between 5 and 10 inclusively
S = random.randint(7,10)

# randomize the number of tasks between 5 and 10 inclusively
#T = random.randint(5,10)

tasks = []
employees = []

for i in range(0, len(tskill)):
    tasks.append(Task(i + 1, tcost[i], tskill[i]))
for i in range(0, len(eskill)):
    employees.append(Employee(i + 1, salary[i], eskill[i], team[i]))
'''if __name__ == "__main__":
    for t in tasks:
    print(t.taskId, t.cost, t.skills)
    for ed in TPG:
    print(ed)
    for e in employees:
    print(e.employeeId, e.salary, e.skills)'''

fit = 0
operation_limit = 10000


def check_ready(task, complete, tpg):
    ready = True
    for i in range(0, len(tpg)):
        if task in complete:
            ready = False
            break        
        if tpg[i][1] == task:
            if not tpg[i][0] in complete:
                ready = False
                break
    return ready

def update_ready(tasks, complete, TPG, task_in_progress):
    ready = []
    for i in range(0, len(tasks)):
        c = True
        for j in range(0, len(task_in_progress)):
            if task_in_progress[j][0].taskId == i + 1:
                c = False
                break
        if c and check_ready(tasks[i].taskId, complete, TPG):
            ready.append(tasks[i].taskId)
            tasks[i].ready += 1
    return ready
    
def task_complete(time, employee_available, task_in_progress, complete):
    removed = []
    task = []
    for i in range(0, len(task_in_progress)):
        if task_in_progress[i][1] <= time:
            if not task_in_progress[i][0].taskId in complete:
                complete.append(task_in_progress[i][0].taskId)
                task.append(task_in_progress[i][0])
            #print(len(task_in_progress[i][0].team))
            for k in range (0, len(task_in_progress[i][0].team)):
                cnt = 0
                if not task_in_progress[i][0].team[k] in employee_available:
                    employee_available.append(task_in_progress[i][0].team[k])
                    cnt += 1
            removed.append(task_in_progress[i])
    for i in range(0, len(removed)):
        task_in_progress.remove(removed[i])
    return task

def time_taken(task):
    size = len(task.team)
    efficiency = 0.0
    total = 0.0
    skill = len(task.skills)
        
        
    if size == 1:
        return task.cost*task.team[0].team[task.team[0].employeeId - 1]
    for i in range(1, size+1):
        for j in range(i, size):
            num = 0.0
            for k in range(0, skill):
                task.team[i-1].skills
                if task.skills[k] in task.team[i-1].skills or task.skills[k] in task.team[j].skills:
                    num += 1
            efficiency += task.team[i-1].team[k] * num/skill
            total += num/skill
    efficiency = efficiency/total
    efficiency = size*efficiency
    return task.cost*efficiency

def fitness_eval(answer):
    cost = 0
    for i in range(0, len(answer[0])):
        time = time_taken(answer[0][i])
        for j in range(0, len(answer[0][i].team)):
            cost += time*answer[0][i].team[j].salary
    #print("time = " + str(answer[1]))
    return 1/(cost*0.000001 + answer[1]*0.1)

def select_task(ready, task_pheromone, complete, tasks):
    total = 0
    if len(complete) > 0:
        seed_task = complete[0]
    else:
        for x in range(0, len(tasks)):
            if tasks[x].taskId == ready[0]:
                seed_task = tasks[x]
                break
    for i in range(0, len(ready)):
        total += task_pheromone[seed_task.taskId - 1][ready[i]-1]
    r = random.random()*total
    count = 0
    for i in range(0, len(ready)):
        count += task_pheromone[seed_task.taskId - 1][ready[i]-1]
        if count >= r:
            ind = ready[i]
            ready.remove(ind)
            for m in range(0, len(tasks)):
                if tasks[m].taskId == ind:
                    t = tasks[m]
                    break
            for j in range(0, len(complete)):
                complete[j].nextTask.append(t)
            return ind

def fill_employee(taskId, employee_available, employee_pheromone, task_in_progress, time, time_stamp):
    for i in range(0, len(tasks)):
        if tasks[i].taskId == taskId:
            task = tasks[i]
            break
    team = []
    
    skill = list(task.skills)

    while not len(skill) == 0:
        c = False
        suitable_employee = []
        total = 0
        for i in range(0, len(employee_available)):
            if employee_available[i] not in team:
                if skill[0] in employee_available[i].skills:
                    suitable_employee.append(employee_available[i])
                    total += employee_pheromone[employee_available[i].employeeId - 1][task.taskId - 1]
        if len(suitable_employee) == 0:
            return False
        ran = random.random()*total
        #print(ran)
        cnt = 0
        del skill[0]
        for i in range(0, len(suitable_employee)):
            cnt += employee_pheromone[suitable_employee[i].employeeId - 1][task.taskId - 1]
            if cnt >= ran:
                team.append(suitable_employee[i])
                for k in range(0, len(suitable_employee[i].skills)):
                    if suitable_employee[i].skills[k] in skill:
                        skill.remove(suitable_employee[i].skills[k])    
                break;
         #           c = True
                    #if random.random() < employee_pheromone[employee_available[i].employeeId - 1][task.taskId - 1]:
                    #    team.append(employee_available[i])
                    #    del skill[0]
                    #    for k in range(0, len(employee_available[i].skills)):
                    #        if employee_available[i].skills[k] in skill:
                    #            skill.remove(employee_available[i].skills[k])
                    #    break
        #if not c:
        #    return False
    
    for i in range(0, len(team)):
        employee_available.remove(team[i])
 
        
    task.team = team
    t = time_taken(task)
    task_in_progress.append([task, time_taken(task) + time])
    time_stamp.append(time_taken(task) + time)
    
    return t
                        
                

    
    
def update_pheromone(task_pheromone, employee_pheromone, answer, fit):
    for i in range(0, len(answer[0])):
        if not answer[0][i].ready == 0:
            task_pheromone[answer[0][i].taskId - 1][answer[0][i].taskId - 1] += 1/answer[0][i].ready
        answer[0][i].ready = 0
        for k in range(0, len(answer[0][i].nextTask)):
            task_pheromone[answer[0][i].taskId - 1][answer[0][i].nextTask[k].taskId - 1] += 1
        for j in range(0, len(answer[0][i].team)):
            employee_pheromone[answer[0][i].team[j].employeeId -1][answer[0][i].taskId -1] += 1/fit
    for i in range (0, len(task_pheromone)):
        task_pheromone[i] = [j*0.9 for j in task_pheromone[i]]
    for i in range(0, len(employee_pheromone)):
        employee_pheromone[i] = [j*0.9 for j in employee_pheromone[i]]
    return


population = 10
task_pheromone = [[1]*len(tasks)]*len(tasks)
employee_pheromone = [[1]*len(tasks)]*len(employees)
print(len(employees))
print(len(tasks))
best_so_far = 0
best_answer_so_far = [] 
while (fit < operation_limit):
    best = 0
    best_answer = []
    for ant in range(0, population):
        
        complete = []
        ready = []
        time_stamp = [0]
        time = 0
        employee_available = list(employees)
        task_in_progress = []        
        while not len(complete) == len(tasks): 
            time_stamp.sort()
            time = time_stamp[0];
            del time_stamp[0]
            #print("employee available : " + str(len(employee_available)))
            task_completed = task_complete(time, employee_available, task_in_progress, complete)
            #print("employee available after complete : " + str(len(employee_available))) 
            
            ready = update_ready(tasks, complete, TPG, task_in_progress)

            success = True
            selected_task = [];  
            while (not len(ready) == 0):
                task = select_task(ready, task_pheromone, task_completed, tasks)
                success = fill_employee(task, employee_available, employee_pheromone, task_in_progress, time, time_stamp)
        answer = [tasks, time]
        fitness = fitness_eval(answer)
        fit += 1
        #print(answer[1])
        #print(answer[0][0].team)
        #print(best)
        if fitness > best:
            best = fitness
            best_answer = answer
    if best > best_so_far:
        best_so_far = best
        best_answer_so_far = best_answer
    print(best_so_far)
    
    update_pheromone(task_pheromone, employee_pheromone, best_answer, fit)
        

