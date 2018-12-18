from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax
import random

class Task(object):
  """docstring for Task"""
  def __init__(self, taskId, cost, skills):
    super(Task, self).__init__()
    self.taskId = taskId
    self.cost = cost
    self.skills = deepcopy(skills)
    self.workload = cost

  def __str__(self):
    return ''.join(['Task #', str(self.taskId), ' has workload=', str(self.workload),
        ' and requires skills ', str(self.skills)])

class Employee(object):
  """docstring for Employee"""
  def __init__(self, employeeId, salary, skills, team):
    super(Employee, self).__init__()
    self.employeeId = employeeId
    self.salary = salary
    self.skills = skills   
    self.team = team

  def __str__(self):
    return ''.join(['Employee #', str(self.employeeId), ' has salary=', str(self.salary),
        ' and skills ', str(self.skills),'\nteam efficiency=', str(self.team)])

# These numbers will be updated based on instance generator
# DATASET 2
# task_skills = [[6, 7], [3, 5, 7], [2, 3], [3, 6, 7], [2, 3, 7], [1, 2], [2, 3], [1, 6, 7]]
# task_costs = [8.0, 8.0, 8.0, 3.0, 17.0, 20.0, 10.0, 9.0]
# employees_skills = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]
# employees_salaries = [9465.0, 9475.0, 11501.0, 9769.0, 10749.0, 10288.0, 8571.0, 10903.0, 11037.0, 9828.0]
# employees_team =[[1.9850026010223878, 0.18212375051252394, 0.7176406839199339, 1.5340873914952124, 0.14863581676990179, 1.3932135973019686, 0.47077045345645807, 1.4890159454850462, 0.27824517847103625, 1.5964701089741966], [0.18212375051252394, 0.15133877966497167, 0.7778187939774182, 0.7981114960025331, 0.8472192924749398, 1.3930634011990006, 0.4909995217013994, 1.1151725310028653, 1.2789818710617091, 1.802499326729509], [0.7176406839199339, 0.7778187939774182, 1.174265498685762, 0.8618080817973071, 0.6399809057368213, 0.9260073334361711, 0.10667661971917908, 0.6483140981769688, 0.9415352155373198, 0.2792483512974877], [1.5340873914952124, 0.7981114960025331, 0.8618080817973071, 1.4340801631563556, 1.2175021462906137, 1.6176868217802627, 0.2228937417974326, 1.2943922301988497, 1.2756081176813578, 1.3051895787048884], [0.14863581676990179, 0.8472192924749398, 0.6399809057368213, 1.2175021462906137, 0.31192937924613173, 0.47800990295685675, 0.4226701520364715, 1.0721857998635547, 0.27603041409724094, 1.3357600021901013], [1.3932135973019686, 1.3930634011990006, 0.9260073334361711, 1.6176868217802627, 0.47800990295685675, 0.10904442155462712, 1.3622364050254925, 1.4000271252322727, 0.48386533282966626, 0.35237321377639286], [0.47077045345645807, 0.4909995217013994, 0.10667661971917908, 0.2228937417974326, 0.4226701520364715, 1.3622364050254925, 0.4157121650281814, 1.4348420931759798, 1.1016468388691938, 1.9628272200314718], [1.4890159454850462, 1.1151725310028653, 0.6483140981769688, 1.2943922301988497, 1.0721857998635547, 1.4000271252322727, 1.4348420931759798, 0.08120641201010037, 0.8233022739244853, 0.07368181520046768], [0.27824517847103625, 1.2789818710617091, 0.9415352155373198, 1.2756081176813578, 0.27603041409724094, 0.48386533282966626, 1.1016468388691938, 0.8233022739244853, 1.6408448059389065, 0.3105475664853563], [1.5964701089741966, 1.802499326729509, 0.2792483512974877, 1.3051895787048884, 1.3357600021901013, 0.35237321377639286, 1.9628272200314718, 0.07368181520046768, 0.3105475664853563, 0.7910222561839355]]
# TPG = [(1, 7), (3, 4), (2, 5), (4, 5), (1, 6), (3, 5), (2, 6)]

task_skills = [[4, 6, 8], [1, 4, 7], [1, 5, 7], [8, 9], [2, 8, 9], [1, 3], [5, 7, 10], [1, 3, 5], [5, 8], [4, 5]]

task_costs = [6.0, 16.0, 17.0, 2.0, 6.0, 13.0, 17.0, 9.0, 10.0, 10.0]

employees_skills = [[7, 1, 2], [1, 2, 8], [2, 3, 9], [6, 5, 7], [5, 9, 3], [3, 5, 1], [5, 3, 8], [8, 6, 10], [8, 6, 4], [8, 10, 7], [9, 2, 3], [8, 2, 5], [10, 6, 2], [5, 9, 6], [9, 6, 3], [2, 4, 10], [3, 4, 7], [7, 1, 10], [8, 7, 10], [7, 10, 8]]

employees_salaries = [10067.0, 10052.0, 9901.0, 9851.0, 9994.0, 10057.0, 10043.0, 10031.0, 10003.0, 9763.0, 10190.0, 9873.0, 9955.0, 10033.0, 9984.0, 9933.0, 10078.0, 9885.0, 9863.0, 9955.0]

employees_team = [[0.9123262001663148, 0.2780352614660919, 0.6718208966503987, 1.4270184199705842, 0.6301479204599949, 0.05734667104627844, 0.30307452656253586, 1.9108379667802688, 0.1329149186794487, 0.12238775977397975, 0.9406683006175796, 0.698637686837122, 1.263730842643979, 0.780585021438263, 1.6040657299995216, 1.479132215528421, 1.1119428709976984, 1.5838702111565053, 1.8350456739276702, 0.7438427761307431], [0.2780352614660919, 0.40777936658563374, 0.6601122021353976, 1.6814625330225539, 0.7462330848536314, 0.6499797542605661, 1.1725214337195164, 1.7664714602345242, 0.5293636164431126, 0.9909352110120491, 0.6489757460119356, 1.3288828962906207, 0.7263738696802196, 0.08070184081819742, 0.9541326977842304, 1.3538021496479524, 0.6257316716067656, 0.9491485131445485, 0.45549190276710094, 1.425145602178398], [0.6718208966503987, 0.6601122021353976, 1.1622075744133427, 1.3942455056947662, 0.985029116707469, 1.6176319041001463, 0.8733843166594912, 0.130197865154823, 1.8503319381371426, 0.41073367934327853, 0.6167348677854265, 0.8959945698844873, 1.7468991301508683, 0.6423813662493298, 0.34012543740993517, 0.760902520684466, 0.5436557065233603, 1.5186881797405878, 1.2117252930963565, 1.2788383613073497], [1.4270184199705842, 1.6814625330225539, 1.3942455056947662, 1.3729552267550695, 1.5098031131115761, 1.304582704937704, 1.5485977326150475, 0.7602411966890699, 0.5984122821813023, 0.2226629829271285, 0.6384387564458627, 0.43378842878352963, 0.826754336032353, 0.9040259436861318, 0.9078718500045613, 1.0585770298385204, 1.8887818171359654, 1.130240247887277, 1.0487975171780797, 1.379290817589792], [0.6301479204599949, 0.7462330848536314, 0.985029116707469, 1.5098031131115761, 0.643004062558137, 1.9564451932496432, 0.43155065727790487, 0.42160037575383247, 0.18822697003639544, 0.36040248472478065, 0.10134926801413036, 1.48949066783836, 1.1268980765652983, 1.636673793211623, 1.4639509886196655, 0.3517487260599865, 0.8685472861128505, 0.9003839149846815, 0.3667716307103812, 1.9232677124357145], [0.05734667104627844, 0.6499797542605661, 1.6176319041001463, 1.304582704937704, 1.9564451932496432, 0.47425392899355945, 1.5047313481549436, 0.22449846577614752, 0.9736676083724463, 0.22266049354961193, 0.49536638073269024, 0.8158538659242363, 0.9927288412763491, 1.846180449639219, 1.8019985012977273, 1.0255557496395586, 0.9240407118431713, 0.1933383329439442, 1.0891294446343713, 0.18179440415546155], [0.30307452656253586, 1.1725214337195164, 0.8733843166594912, 1.5485977326150475, 0.43155065727790487, 1.5047313481549436, 1.6507474206079633, 0.047357123128821677, 0.38765085020595813, 0.7073841251223045, 0.6890964439003437, 1.641919734748158, 0.789543478491971, 0.6822520856242409, 1.8784524152608153, 0.7621982061670785, 0.10663333909058315, 0.6017594509788085, 1.7466182123888017, 0.594538969462644], [1.9108379667802688, 1.7664714602345242, 0.130197865154823, 0.7602411966890699, 0.42160037575383247, 0.22449846577614752, 0.047357123128821677, 1.642181866146154, 0.25794489049871205, 0.6198777820986121, 0.7592770398116735, 0.7105300386728663, 1.5386864110307115, 1.5725701058908284, 1.2378810413856502, 1.7331444638406808, 0.04373618869458862, 1.1644161157128188, 1.0020351576541149, 1.485500245956987], [0.1329149186794487, 0.5293636164431126, 1.8503319381371426, 0.5984122821813023, 0.18822697003639544, 0.9736676083724463, 0.38765085020595813, 0.25794489049871205, 0.6711505991115012, 1.5129825115386477, 0.44205284572294, 0.5402717190044144, 0.4801576213051848, 0.3515668509735186, 0.48717561979879753, 1.075270946759471, 0.08661605934107741, 1.4173035334499853, 0.05196405820597927, 1.085275031433634], [0.12238775977397975, 0.9909352110120491, 0.41073367934327853, 0.2226629829271285, 0.36040248472478065, 0.22266049354961193, 0.7073841251223045, 0.6198777820986121, 1.5129825115386477, 0.38761560528540673, 1.6604948793484622, 1.1576363824612368, 1.8756722375573096, 0.5432593415554605, 1.0200086773178394, 1.7152034380602856, 0.5560644211508665, 1.995884016338117, 0.8099471215274359, 1.9432715561602873], [0.9406683006175796, 0.6489757460119356, 0.6167348677854265, 0.6384387564458627, 0.10134926801413036, 0.49536638073269024, 0.6890964439003437, 0.7592770398116735, 0.44205284572294, 1.6604948793484622, 1.0459835015484524, 1.3082151753545455, 0.5764030098139237, 0.007681780370129454, 0.5329674377011049, 1.2292261345386295, 1.9633675689219714, 0.2480400904491329, 0.3822276064301682, 1.34561108417454], [0.698637686837122, 1.3288828962906207, 0.8959945698844873, 0.43378842878352963, 1.48949066783836, 0.8158538659242363, 1.641919734748158, 0.7105300386728663, 0.5402717190044144, 1.1576363824612368, 1.3082151753545455, 1.8824948653019293, 1.3735388711964773, 1.7013747634963405, 1.7767077874434045, 0.4305060919125254, 1.6197697890807978, 1.125216508619317, 0.5662428955854066, 0.07431072544233652], [1.263730842643979, 0.7263738696802196, 1.7468991301508683, 0.826754336032353, 1.1268980765652983, 0.9927288412763491, 0.789543478491971, 1.5386864110307115, 0.4801576213051848, 1.8756722375573096, 0.5764030098139237, 1.3735388711964773, 0.44085603971192966, 0.2778906934563863, 0.7936152628006516, 0.42054899376501553, 0.8433436463208752, 1.2298385131021157, 1.160874057218917, 0.9130898424686569], [0.780585021438263, 0.08070184081819742, 0.6423813662493298, 0.9040259436861318, 1.636673793211623, 1.846180449639219, 0.6822520856242409, 1.5725701058908284, 0.3515668509735186, 0.5432593415554605, 0.007681780370129454, 1.7013747634963405, 0.2778906934563863, 1.9992360547309185, 1.1231663296994459, 1.7511447405631675, 1.9287390487919385, 1.8735399322312458, 0.3704825277668258, 1.4135943262309494], [1.6040657299995216, 0.9541326977842304, 0.34012543740993517, 0.9078718500045613, 1.4639509886196655, 1.8019985012977273, 1.8784524152608153, 1.2378810413856502, 0.48717561979879753, 1.0200086773178394, 0.5329674377011049, 1.7767077874434045, 0.7936152628006516, 1.1231663296994459, 1.3979730869174738, 1.5924010528504797, 0.5977777143409049, 1.5706390680686362, 0.41231858116986997, 0.5742464392570528], [1.479132215528421, 1.3538021496479524, 0.760902520684466, 1.0585770298385204, 0.3517487260599865, 1.0255557496395586, 0.7621982061670785, 1.7331444638406808, 1.075270946759471, 1.7152034380602856, 1.2292261345386295, 0.4305060919125254, 0.42054899376501553, 1.7511447405631675, 1.5924010528504797, 0.5865702434611837, 1.3657734568890358, 1.378948780201084, 0.8577097039626922, 0.9866557411609935], [1.1119428709976984, 0.6257316716067656, 0.5436557065233603, 1.8887818171359654, 0.8685472861128505, 0.9240407118431713, 0.10663333909058315, 0.04373618869458862, 0.08661605934107741, 0.5560644211508665, 1.9633675689219714, 1.6197697890807978, 0.8433436463208752, 1.9287390487919385, 0.5977777143409049, 1.3657734568890358, 1.6032837788881154, 1.5240350003773948, 0.6733002146784746, 1.3231189015252334], [1.5838702111565053, 0.9491485131445485, 1.5186881797405878, 1.130240247887277, 0.9003839149846815, 0.1933383329439442, 0.6017594509788085, 1.1644161157128188, 1.4173035334499853, 1.995884016338117, 0.2480400904491329, 1.125216508619317, 1.2298385131021157, 1.8735399322312458, 1.5706390680686362, 1.378948780201084, 1.5240350003773948, 1.3909726414780792, 0.08446435803641528, 1.621497375726149], [1.8350456739276702, 0.45549190276710094, 1.2117252930963565, 1.0487975171780797, 0.3667716307103812, 1.0891294446343713, 1.7466182123888017, 1.0020351576541149, 0.05196405820597927, 0.8099471215274359, 0.3822276064301682, 0.5662428955854066, 1.160874057218917, 0.3704825277668258, 0.41231858116986997, 0.8577097039626922, 0.6733002146784746, 0.08446435803641528, 0.13072624925688703, 1.3126413182037686], [0.7438427761307431, 1.425145602178398, 1.2788383613073497, 1.379290817589792, 1.9232677124357145, 0.18179440415546155, 0.594538969462644, 1.485500245956987, 1.085275031433634, 1.9432715561602873, 1.34561108417454, 0.07431072544233652, 0.9130898424686569, 1.4135943262309494, 0.5742464392570528, 0.9866557411609935, 1.3231189015252334, 1.621497375726149, 1.3126413182037686, 1.9882925305387054]]

TPG = [(5, 7), (8, 9), (1, 2), (4, 5), (6, 10), (8, 10), (1, 4), (3, 7)]

num_tasks = len(task_skills)
num_employees = len(employees_skills)

tasks = [Task(i, task_costs[i-1], task_skills[i-1]) for i in range(1, num_tasks+1)]
employees = [Employee(i, employees_salaries[i-1], employees_skills[i-1], employees_team[i-1]) for i in range(1, num_employees+1)]


# for task in tasks:
#   print(task)
# for employee in employees:
#   print(employee)
# =========================================================
fitness_ftn_call_count = 0
# =========================================================

class Schedule(object):
  """docstring for Schedule"""
  def __init__(self, solution_matrix):
    super(Schedule, self).__init__()
    global tasks
    global employees
    global TPG
    global num_tasks
    global num_employees

    # matrix of size MxN = list of M list of length N
    # in which M is # tasks and N is # num_employees
    self.solution_matrix = deepcopy(solution_matrix)

    self.fitness_value = 0

    # can this schedule be used as a candidate solution
    self.is_candidate = True

    # Calculate each task's team of assigned employee efficiency =====
    self.task_efficiency = []
    for idx, task in enumerate(self.solution_matrix):
      # if no employees are assigned to this task,
      # then it is not considered a candidate solution
      if sum(x > 0 for x in task) == 0:
        self.is_candidate = False
        return

      # if there is only one employee assigned to the task,
      # the the efficiency of the team is the efficiency of that employee with himself
      if sum(x > 0 for x in task) == 1:
        k = 0
        while True:
          if task[k] != 0:
            break
          k += 1
        self.task_efficiency.append(employees[k].team[k])
        continue

      efficiency = 0
      for i in range(0, num_employees-1):
        if task[i] == 0:
          continue  # this employee is not assigned to the task

        for j in range(i+1, num_employees):
          if task[j] == 0:
            continue  # this employee is not assigned to the task

          # count number of required skills these 2 employees have
          num_skills = 0
          total = 0.0
          for skill in tasks[idx].skills:
            if skill in employees[i].skills or skill in employees[j].skills:
              num_skills += 1

          efficiency += employees[i].team[j] * num_skills / len(tasks[idx].skills)
          total += float(num_skills)/len(tasks[idx].skills)

      if total != 0:
        efficiency /= total
      else:
        efficiency = 1
      efficiency *= sum(x > 0 for x in task)
      self.task_efficiency.append(efficiency)
    # ================================================================

    # calculate task durations matrix =====
    self.task_durations = []
    for i in range(num_tasks):
      if (sum(self.solution_matrix[i]) != 0):
        self.task_durations.append(tasks[i].workload / sum(self.solution_matrix[i]) * self.task_efficiency[i])
      else:
        self.task_durations.append(333333)
        # if there is a task with no employee assigned to it,
        # then it is not a feasible solution, because project duration cannot 
        # be calculated 
        # mark as not candidate, fitness 0
        self.is_candidate = False
        return 
    # =====================================

    # calculate the project cost =====
    self.project_cost = 0
    for i in range(num_tasks):
      for j in range(num_employees):
        self.project_cost += (employees[j].salary * self.solution_matrix[i][j] * self.task_durations[i])
    # ================================

    # calculate undt = #tasks not assigned to any employee =====
    self.undt = 0
    for i in range(num_tasks):
      if sum(self.solution_matrix[i]) == 0:
        self.undt += 1
    # ==========================================================

    # calculate reqsk = #skills unavailable from employees assigned to tasks ===
    self.reqsk = 0
    for i in range(num_tasks):
      # set of skills required for this task
      temp = set()
      temp.update(tasks[i].skills)

      # for each employee with working hours per day on task more than 0,
      # remove their skills from this set
      for j in range(num_employees):
        if self.solution_matrix[i][j] > 0:
          for skill in employees[j].skills:
            temp.discard(skill)

      self.reqsk += len(temp)
    # ==========================================================================

    # calculate project duration and overwork =====
    self.project_duration = 0
    self.overwork = 0

    # make a copy of TPG
    temp_TPG = deepcopy(TPG)

    # make a set of all task_id in TPG
    unfinished_tasks = list(range(1, num_tasks+1))
    unfinished_tasks_workload = []
    for i in range(num_tasks):
      unfinished_tasks_workload.append(tasks[i].workload)

    # print('unfinished_tasks', unfinished_tasks)
    # print('unfinished_tasks_workload', unfinished_tasks_workload)
    # print('temp_TPG', temp_TPG)

    while len(unfinished_tasks) > 0:
      unfinished_indep_tasks = deepcopy(unfinished_tasks)

      # Create set of unfinished tasks not depending on any tasks
      # Remove tasks that is second element of an edge in TPG
      for edge in temp_TPG:
        if edge[1] in unfinished_indep_tasks:
          unfinished_indep_tasks.remove(edge[1])

      # print('temp_TPG', temp_TPG)
      # print('unfinished_indep_tasks', unfinished_indep_tasks)
      # print('unfinished_tasks', unfinished_tasks)
      # print('unfinished_tasks_workload', unfinished_tasks_workload)

      if len(unfinished_indep_tasks) == 0:
        print("Unsolvable: Error calculating project duration and overwork")
        self.is_candidate = False
        return

      dedication = []
      dedication_j = []
      for task_id in unfinished_indep_tasks:
        # Note that task_id in unfinished_indep_tasks starts from 1 
        # but for matrices it starts from 0 --> indexing carefully
        dedication.append(deepcopy(self.solution_matrix[task_id-1]))
        dedication_j.append(sum(dedication[-1]))

        if dedication_j[-1] == 0:
          # supposed to be return (INF, INF)
          self.overwork = 333333
          self.project_duration = 333333
          return
      # if (5 in unfinished_indep_tasks):
      #   print('dedication', dedication)
      # calculate the dedication of each employee
      for i in range(num_employees):
        dedication_i = 0
        for task in dedication:
          dedication_i += task[i]

        if dedication_i > 1:
          # print('overwork increases')
          self.overwork = self.overwork + dedication_i - 1

      # t = min([unfinished_tasks_workload[unfinished_indep_tasks[j]-1]/float(dedication_j[j]) for j in range(len(unfinished_indep_tasks))])
      t = 333333
      done_task_id = 0
      for j in range(len(unfinished_indep_tasks)):
        idx = unfinished_tasks.index(unfinished_indep_tasks[j])
        # print('debug', unfinished_indep_tasks[j], unfinished_tasks_workload[idx], dedication_j[j], self.task_efficiency[unfinished_indep_tasks[j]-1])
        temp = unfinished_tasks_workload[idx] / dedication_j[j] * self.task_efficiency[unfinished_indep_tasks[j]-1]
        # print('temp', temp)
        if t > temp:
          t = temp
          done_task_id = j

      self.project_duration += t
      # print('t', t)

      finished = []
      overwork_increase_margin = self.overwork * t
      for j in range(len(unfinished_indep_tasks)):
        idx = unfinished_tasks.index(unfinished_indep_tasks[j])

        if j == done_task_id:
          unfinished_tasks_workload[idx] = 0
        else:  
          unfinished_tasks_workload[idx] -= (t * dedication_j[j] / self.task_efficiency[unfinished_indep_tasks[j]-1])

        if unfinished_tasks_workload[idx] <= 0:
          # this task is done
          finished.append(idx)

        self.overwork += overwork_increase_margin

      # CLEAN UP FINISHED TASKS IN TPG, UNFINISHED_TASK, UNFINISHED_TASK_WORKLOAD
      # print('finished', finished)
      finished.reverse()
      for idx in finished:
        del unfinished_tasks_workload[idx]

        i = 0
        while i < len(temp_TPG):
          if unfinished_tasks[idx] in temp_TPG[i]:
            del temp_TPG[i]
            continue
          i += 1

        del unfinished_tasks[idx]
    # =============================================

    # calculate fitness score =====
    q = 10**(-6) * self.project_cost + 0.1 * self.project_duration
    r = 0

    if (self.undt > 0 or self.reqsk > 0 or self.overwork > 0):
      # print('not good solution: undt =', self.undt, '; reqsk =', self.reqsk, '; overwork =', self.overwork)
      r = 100 + 10 * self.undt + 10 * self.reqsk + 0.1 * self.overwork
      self.fitness_value = 1.0/(q + r)
    else:
      self.fitness_value = 1.0/q
    # =============================

  def __eq__(self, other):
    return self.solution_matrix == other.solution_matrix

  def __ne__(self, other):
    return self.solution_matrix != other.solution_matrix

  # def __lt__(self, other):


  def __str__(self):
    result = ''
    for task in self.solution_matrix:
      result += '\t'.join([str(i) for i in task])
      result += '\n'
    result += 'project_duration = ' + str(self.project_duration) + '\n'
    result += 'project_cost = ' + str(self.project_cost) + '\n'
    result += 'fitness_value = ' + str(self.fitness_value) + '\n'
    result += 'overwork = ' + str(self.overwork) + '\n'
    result += 'task_efficiency = ' + str(self.task_efficiency) + '\n'
    return result

class TabuSearch:
  """
  Conducts tabu search
  """
  __metaclass__ = ABCMeta

  cur_steps = None

  tabu_size = None
  tabu_list = None

  # A state is a solution
  initial_state = None
  current = None
  best = None

  max_steps = None
  max_score = None

  best_call_count = None

  def __init__(self, initial_state, tabu_size, max_steps, max_score=None):
    """
    :param initial_state: initial state, should implement __eq__ or __cmp__
    :param tabu_size: number of states to keep in tabu list
    :param max_steps: maximum number of steps to run algorithm for
    :param max_score: score to stop algorithm once reached
    """
    self.initial_state = initial_state

    if isinstance(tabu_size, int) and tabu_size > 0:
      self.tabu_size = tabu_size
    else:
      raise TypeError('Tabu size must be a positive integer')

    if isinstance(max_steps, int) and max_steps > 0:
      self.max_steps = max_steps
    else:
      raise TypeError('Maximum steps must be a positive integer')

    if max_score is not None:
      if isinstance(max_score, (int, float)):
        self.max_score = float(max_score)
      else:
        raise TypeError('Maximum score must be a numeric type')

  def __str__(self):
    global fitness_ftn_call_count

    return ('TABU SEARCH BEST RESULT SO FAR: \n' +
        'CURRENT STEPS: %d \n' +
        'CURRENT #FITNESS FTN CALL: %d \n'
        'BEST SCORE: %f \n' +
        'BEST MEMBER: \n%s \n\n') % \
         (self.cur_steps, self.best_call_count, self._score(self.best), str(self.best))

  def __repr__(self):
    return self.__str__()

  def _clear(self):
    """
    Resets the variables that are altered on a per-run basis of the algorithm
    :return: None
    """
    self.cur_steps = 0
    self.best_call_count = 0
    self.tabu_list = deque(maxlen=self.tabu_size)
    self.current = self.initial_state
    self.best = self.initial_state

  @abstractmethod
  def _score(self, state):
    """
    Returns objective function value of a state
    :param state: a state
    :return: objective function value of state
    """
    pass

  @abstractmethod
  def _neighborhood(self):
    """
    Returns list of all members of neighborhood of current state, given self.current
    :return: list of members of neighborhood
    """
    pass

  def _best(self, neighborhood):
    """
    Finds the best member of a neighborhood
    :param neighborhood: a neighborhood
    :return: best member of neighborhood
    """
    return neighborhood[argmax([self._score(x) for x in neighborhood])]

  def printbestsolution(self, output, best_solution):
    global fitness_ftn_call_count

    self.best_solution = fitness_ftn_call_count
    print("#fitness", fitness_ftn_call_count)
    print('Best is \n', best_solution)   
    output.write(str(fitness_ftn_call_count) + ',' + str(best_solution.fitness_value) + '\n')

  def run(self, verbose=True):
    """
    Conducts tabu search
    :param verbose: indicates whether or not to print progress regularly
    :return: best state and objective function value of best state
    """
    global fitness_ftn_call_count

    self._clear()
    output = open('dataset1-fixed-50.csv', 'w+')
    output.write('0,' + str(self.initial_state.fitness_value) + '\n')

    # for i in range(self.max_steps):
    while True:
      if fitness_ftn_call_count >= self.max_steps:
        break

      self.cur_steps += 1

      # if ((i + 1) % 100 == 0) and verbose:
      #   print(self)

      neighborhood = self._neighborhood()
      fitness_ftn_call_count += len(neighborhood)
      neighborhood_best = self._best(neighborhood)

      while True:
        if all([x in self.tabu_list for x in neighborhood]):
          print("TERMINATING - NO SUITABLE NEIGHBORS")
          output.close()
          return self.best, self._score(self.best)
        if neighborhood_best in self.tabu_list:
          if self._score(neighborhood_best) > self._score(self.best):
            self.tabu_list.append(neighborhood_best)
            self.best = deepcopy(neighborhood_best)
            self.printbestsolution(output, self.best)
            break
          else:
            neighborhood.remove(neighborhood_best)
            neighborhood_best = self._best(neighborhood)
        else:
          self.tabu_list.append(neighborhood_best)
          self.current = neighborhood_best
          
          if self._score(self.current) > self._score(self.best):
            self.best = deepcopy(self.current)
            self.printbestsolution(output, self.best)
          break
      
      # if self.cur_steps % 10 == 0:
      #   self.printbestsolution(output, self.best)
      #   print("Step", self.cur_steps)
      #   print('Current is \n', self.current)
      #   print('Best is \n', self.best)   
      #   output.write(str(self.cur_steps) + ',' + str(self.best.fitness_value) + '\n') 

      if self.max_score is not None and self._score(self.best) > self.max_score:
        print("TERMINATING - REACHED MAXIMUM SCORE")
        output.close()
        return self.best, self._score(self.best)
    output.close()
    print("TERMINATING - REACHED MAXIMUM STEPS")
    return self.best, self._score(self.best)

class MyTabuSearch(TabuSearch):
  def _score(self, state):
    """
    Returns objective function value of a state
    :param state: a state
    :return: objective function value of state
    """
    # global fitness_ftn_call_count
    # fitness_ftn_call_count += 1
    return state.fitness_value

  def _neighborhood(self):
    """
    Returns list of all members of neighborhood of current state, given self.current
    :return: list of members of neighborhood
    """
    global num_tasks, num_employees
    current_solution = deepcopy(self.current.solution_matrix)
    # print(self.current)
    # print(current_solution)

    # check each entry in solution matrix up/down by 0.1 to form neighbors
    neighbors = []
    # dist = random.randint(1, 9) / 10.0
    dist = 0.1
    # print(dist)

    for i in range(num_tasks):
      for j in range(num_employees):
        new_neighbor = deepcopy(current_solution)
        new_neighbor[i][j] += dist
        new_neighbor[i][j] = round(new_neighbor[i][j],1)
        if new_neighbor[i][j] <= 1:
          neighbors.append(Schedule(new_neighbor))
        # print(neighbors[-1])

        new_neighbor[i][j] -= dist

        new_neighbor[i][j] -= dist
        new_neighbor[i][j] = round(new_neighbor[i][j],1)
        if new_neighbor[i][j] >= 0:
          temp = Schedule(new_neighbor)
          if temp.is_candidate:
            neighbors.append(temp)

        # print('current is ', self.current)

    # print('===============')
    # for e in neighbors:
    #   print(e)
    # print('===============')
    # print(self.current)
    # print('===============')
    return neighbors

# generate initial solution
initial_sol = []

# for i in range(num_tasks):
#   initial_sol.append([0.0]*num_employees)
#   initial_sol[i][i] = 0.1

temp = list(range(num_employees))
random.shuffle(temp)
max_init_dedication = float(num_tasks) / num_employees

for i in range(num_tasks):
  initial_sol.append([0.0]*num_employees)
  initial_sol[-1][temp[i%num_employees]] = random.randint(1, 5) / 10.0

  if i == len(temp)-1:
    random.shuffle(temp)

tabu = MyTabuSearch(Schedule(initial_sol), 50, 10000)
print(tabu.initial_state)
tabu.run()
print(tabu)