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
task_skills = [[6, 7], [3, 5, 7], [2, 3], [3, 6, 7], [2, 3, 7], [1, 2], [2, 3], [1, 6, 7]]
task_costs = [8.0, 8.0, 8.0, 3.0, 17.0, 20.0, 10.0, 9.0]
employees_skills = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]
employees_salaries = [9465.0, 9475.0, 11501.0, 9769.0, 10749.0, 10288.0, 8571.0, 10903.0, 11037.0, 9828.0]
employees_team =[[1.9850026010223878, 0.18212375051252394, 0.7176406839199339, 1.5340873914952124, 0.14863581676990179, 1.3932135973019686, 0.47077045345645807, 1.4890159454850462, 0.27824517847103625, 1.5964701089741966], [0.18212375051252394, 0.15133877966497167, 0.7778187939774182, 0.7981114960025331, 0.8472192924749398, 1.3930634011990006, 0.4909995217013994, 1.1151725310028653, 1.2789818710617091, 1.802499326729509], [0.7176406839199339, 0.7778187939774182, 1.174265498685762, 0.8618080817973071, 0.6399809057368213, 0.9260073334361711, 0.10667661971917908, 0.6483140981769688, 0.9415352155373198, 0.2792483512974877], [1.5340873914952124, 0.7981114960025331, 0.8618080817973071, 1.4340801631563556, 1.2175021462906137, 1.6176868217802627, 0.2228937417974326, 1.2943922301988497, 1.2756081176813578, 1.3051895787048884], [0.14863581676990179, 0.8472192924749398, 0.6399809057368213, 1.2175021462906137, 0.31192937924613173, 0.47800990295685675, 0.4226701520364715, 1.0721857998635547, 0.27603041409724094, 1.3357600021901013], [1.3932135973019686, 1.3930634011990006, 0.9260073334361711, 1.6176868217802627, 0.47800990295685675, 0.10904442155462712, 1.3622364050254925, 1.4000271252322727, 0.48386533282966626, 0.35237321377639286], [0.47077045345645807, 0.4909995217013994, 0.10667661971917908, 0.2228937417974326, 0.4226701520364715, 1.3622364050254925, 0.4157121650281814, 1.4348420931759798, 1.1016468388691938, 1.9628272200314718], [1.4890159454850462, 1.1151725310028653, 0.6483140981769688, 1.2943922301988497, 1.0721857998635547, 1.4000271252322727, 1.4348420931759798, 0.08120641201010037, 0.8233022739244853, 0.07368181520046768], [0.27824517847103625, 1.2789818710617091, 0.9415352155373198, 1.2756081176813578, 0.27603041409724094, 0.48386533282966626, 1.1016468388691938, 0.8233022739244853, 1.6408448059389065, 0.3105475664853563], [1.5964701089741966, 1.802499326729509, 0.2792483512974877, 1.3051895787048884, 1.3357600021901013, 0.35237321377639286, 1.9628272200314718, 0.07368181520046768, 0.3105475664853563, 0.7910222561839355]]
# task_skills = [[6, 7], [3, 5, 7], [2, 3]]
# task_costs = [8.0, 8.0, 8.0]
# employees_skills = [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6, 7]]
# employees_salaries = [9465.0, 9475.0, 11501.0]

num_tasks = len(task_skills)
num_employees = len(employees_skills)

tasks = [Task(i, task_costs[i-1], task_skills[i-1]) for i in range(1, num_tasks+1)]
employees = [Employee(i, employees_salaries[i-1], employees_skills[i-1], employees_team[i-1]) for i in range(1, num_employees+1)]
TPG = [(1, 7), (3, 4), (2, 5), (4, 5), (1, 6), (3, 5), (2, 6)]
# TPG = [(1, 2), (1, 3)]

# for task in tasks:
#   print(task)
# for employee in employees:
#   print(employee)
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
      if sum(x > 0 for x in task) == 0:
        self.is_candidate = False
        return

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
          for skill in tasks[idx].skills:
            if skill in employees[i].skills or skill in employees[j].skills:
              num_skills += 1

          efficiency += employees[i].team[j] * num_skills / len(tasks[idx].skills)
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
    return ('TABU SEARCH: \n' +
        'CURRENT STEPS: %d \n' +
        'BEST SCORE: %f \n' +
        'BEST MEMBER: \n%s \n\n') % \
         (self.cur_steps, self._score(self.best), str(self.best))

  def __repr__(self):
    return self.__str__()

  def _clear(self):
    """
    Resets the variables that are altered on a per-run basis of the algorithm
    :return: None
    """
    self.cur_steps = 0
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

  def run(self, verbose=True):
    """
    Conducts tabu search
    :param verbose: indicates whether or not to print progress regularly
    :return: best state and objective function value of best state
    """
    self._clear()
    output = open('result-rnd-100-iter3.csv', 'w+')
    output.write('0,' + str(self.initial_state.fitness_value) + '\n')

    for i in range(self.max_steps):
      self.cur_steps += 1

      if ((i + 1) % 100 == 0) and verbose:
        print(self)

      neighborhood = self._neighborhood()
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
            break
          else:
            neighborhood.remove(neighborhood_best)
            neighborhood_best = self._best(neighborhood)
        else:
          self.tabu_list.append(neighborhood_best)
          self.current = neighborhood_best
          
          if self._score(self.current) > self._score(self.best):
            self.best = deepcopy(self.current)
          break
      
      if self.cur_steps % 10 == 0:
        print("Step", self.cur_steps)
        print('Current is \n', self.current)
        print('Best is \n', self.best)   
        output.write(str(self.cur_steps) + ',' + str(self.best.fitness_value) + '\n')
        # if True:
        #   input()   

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
    dist = random.randint(1, 9) / 10.0
    # dist = 0.1
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

initial_sol = []
for i in range(num_tasks):
  initial_sol.append([0.0]*num_employees)
  initial_sol[i][i] = 0.1

tabu = MyTabuSearch(Schedule(initial_sol), 100, 1000)
# print(tabu.initial_state)
tabu.run()
print(tabu)