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

tskill = [[2, 6, 8], [3, 9], [2, 3, 6], [2, 4], [2, 6, 8], [1, 2, 9], [4, 7, 8]]
tcost = [12.0,1.0,9.0,14.0,8.0,6.0,7.0]
eskill = [[1, 3, 5, 6, 7, 8],[1, 4, 5, 6, 8, 9],[1, 2, 4, 5, 6, 7, 9],[1, 2, 4, 6, 7, 8, 9],[1, 2, 4, 6, 8, 9],[1, 2, 3, 5, 6, 8, 9],[2, 3, 4, 5, 7, 8],[1, 2, 3, 4, 7, 8]
,[1, 2, 4, 5, 6, 7, 8]
,[1, 2, 3, 5, 8, 9]
,[1, 2, 3, 5, 6, 8, 9]
,[1, 2, 4, 5, 7, 9]
,[1, 3, 4, 6, 7, 8]
,[1, 2, 3, 5, 6, 8]]
salary = [10152.0
,11469.0
,8586.0
,10173.0
,9545.0
,9075.0
,10225.0
,8976.0
,10114.0
,8520.0
,9423.0
,10032.0
,10196.0
,11377.0]

team = [[1.0036171434428822, 1.7764880785447332, 0.6409764456084752, 0.5627298810651225, 0.6008653641933352, 0.6422669456291143, 0.7233384177229021, 0.341741956140444, 1.753052649500598, 0.9692969613718136, 0.184668729004692, 1.9662166657317277, 0.20013686010112464, 0.6975489947120348]
,[1.7764880785447332, 0.16743328851100725, 0.5875061594786228, 1.6633807624800303, 1.9278273506256511, 1.5820821920221362, 1.880185472856609, 1.0529567614224362, 0.4200054182141406, 0.36314081979398294, 0.2584708715531019, 0.5968874609455306, 0.5351233624483911, 0.33821728374203386]
,[0.6409764456084752, 0.5875061594786228, 0.5937030358211395, 1.3882318900700326, 0.23989540741426096, 1.144257120430726, 0.8894131535921579, 0.9583811732258942, 0.6491871111091034, 0.04914496354167652, 1.7725830814272312, 1.4154829362627201, 0.4551518632068212, 1.610603473155036]
,[0.5627298810651225, 1.6633807624800303, 1.3882318900700326, 1.7068243732792625, 0.6640074830668443, 1.5146061679633764, 0.9992712370696397, 1.2599232234719322, 0.8847026343932194, 1.8420588237219002, 0.6997621763665176, 0.4959272378981425, 1.5512650499197334, 0.5155710690953523]
,[0.6008653641933352, 1.9278273506256511, 0.23989540741426096, 0.6640074830668443, 0.6084911383198244, 1.8666347630761384, 1.6283072425784317, 1.5939142385596816, 1.5275593868705701, 0.5713652678280785, 1.578632463719388, 0.1606669506482945, 1.7623013468581825, 1.7323939734351246]
,[0.6422669456291143, 1.5820821920221362, 1.144257120430726, 1.5146061679633764, 1.8666347630761384, 0.8877610262682276, 1.3446074725546286, 1.5581447308771517, 0.23347129019243873, 0.06424377271466941, 1.727047414047409, 0.3147815475172371, 0.4295287244263424, 0.1871076613335183]
,[0.7233384177229021, 1.880185472856609, 0.8894131535921579, 0.9992712370696397, 1.6283072425784317, 1.3446074725546286, 0.47693430487188215, 0.43211429193013084, 1.402375421308326, 1.9095572565299759, 0.010033427557095242, 1.7544337641150465, 1.9709673109575423, 0.6993488616540915]
,[0.341741956140444, 1.0529567614224362, 0.9583811732258942, 1.2599232234719322, 1.5939142385596816, 1.5581447308771517, 0.43211429193013084, 1.1268000777209028, 0.07445298435671321, 1.2163092146870054, 1.5957340956046693, 1.9190097977308835, 1.6202130704291107, 0.7880715759466592]
,[1.753052649500598, 0.4200054182141406, 0.6491871111091034, 0.8847026343932194, 1.5275593868705701, 0.23347129019243873, 1.402375421308326, 0.07445298435671321, 1.4876424720380945, 1.0672180044558468, 0.9311281894945533, 1.7805786703509867, 0.7273657304604229, 1.0738298370884578]
,[0.9692969613718136, 0.36314081979398294, 0.04914496354167652, 1.8420588237219002, 0.5713652678280785, 0.06424377271466941, 1.9095572565299759, 1.2163092146870054, 1.0672180044558468, 1.6417867705358382, 0.565993534420048, 0.7697465738331621, 0.34365394309729536, 1.2557527159445387]
,[0.184668729004692, 0.2584708715531019, 1.7725830814272312, 0.6997621763665176, 1.578632463719388, 1.727047414047409, 0.010033427557095242, 1.5957340956046693, 0.9311281894945533, 0.565993534420048, 1.2276239758762066, 0.30708781253978157, 0.45773082296445144, 0.3336088218360487]
,[1.9662166657317277, 0.5968874609455306, 1.4154829362627201, 0.4959272378981425, 0.1606669506482945, 0.3147815475172371, 1.7544337641150465, 1.9190097977308835, 1.7805786703509867, 0.7697465738331621, 0.30708781253978157, 1.326667150245837, 0.9147828663253321, 0.7180311878919661]
,[0.20013686010112464, 0.5351233624483911, 0.4551518632068212, 1.5512650499197334, 1.7623013468581825, 0.4295287244263424, 1.9709673109575423, 1.6202130704291107, 0.7273657304604229, 0.34365394309729536, 0.45773082296445144, 0.9147828663253321, 1.7476315075701054, 1.5076122475518585]
,[0.6975489947120348, 0.33821728374203386, 1.610603473155036, 0.5155710690953523, 1.7323939734351246, 0.1871076613335183, 0.6993488616540915, 0.7880715759466592, 1.0738298370884578, 1.2557527159445387, 0.3336088218360487, 0.7180311878919661, 1.5076122475518585, 0.45801406780667264]]
TPG = [(4, 5), (5, 6), (1, 6), (4, 7), (6, 7), (2, 3), (2, 7), (5, 7), (2, 4), (3, 4)]
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
operation_limit = 1000


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
    print("TIP")
    for i in range(0, len(task_in_progress)):
        print("ID : " + str(task_in_progress[i][0].taskId))
        print("employee : " + str(len(task_in_progress[i][0].team)))
    print("employee available : " + str(len(employee_available)))
    for i in range(0, len(task_in_progress)):
        if task_in_progress[i][1] <= time:
            if not task_in_progress[i][0].taskId in complete:
                complete.append(task_in_progress[i][0].taskId)
                task.append(task_in_progress[i][0])
            #print(len(task_in_progress[i][0].team))
            print("employee available : " + str(len(employee_available)))
            for k in range (0, len(task_in_progress[i][0].team)):
                cnt = 0
                if not task_in_progress[i][0].team[k] in employee_available:
                    employee_available.append(task_in_progress[i][0].team[k])
                    cnt += 1
            print("added : " + str(cnt))
            print("employee available after : " + str(len(employee_available)))
            removed.append(task_in_progress[i])
    for i in range(0, len(removed)):
        task_in_progress.remove(removed[i])
    return task

def time_taken(task):
    size = len(task.team)
    efficiency = 0
    total = 0
    skill = len(task.skills)
        
        
    if size == 1:
        return task.cost*task.team[0].team[task.team[0].employeeId - 1]
    for i in range(1, size+1):
        for j in range(i, size):
            num = 0
            for k in range(0, skill):
                task.team[i-1].skills
                if task.skills[k] in task.team[i-1].skills or task.skills[k] in task.team[j].skills:
                    num += 1
            efficiency += task.team[i-1].team[k] * num/skill
            total += num/skill
    efficiency = efficiency/(2*total)
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
    return


population = 10
task_pheromone = [[1]*len(tasks)]*len(tasks)
employee_pheromone = [[1]*len(tasks)]*len(employees)
print(len(employees))
print(len(tasks))
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
    print(best_answer[1])
    update_pheromone(task_pheromone, employee_pheromone, best_answer, fit)
        

