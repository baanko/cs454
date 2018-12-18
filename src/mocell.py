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
    
    
#####################data1###############################3




T=7;
tasks=[];

E=14;
employees=[]

cost=[12.0,1.0,9.0,14.0,8.0,6.0,7.0]
skills_t = [[2, 6, 8],[3, 9],[2, 3, 6],[2, 4],[2, 6, 8],[1, 2, 9],[4, 7, 8]    ]

salary=[10152.0,11469.0,8586.0,10173.0,9545.0,9075.0,10225.0,8976.0,10114.0,8520.0,9423.0,10032.0,10196.0,11377.0    ]
skills_e = [[1, 3, 5, 6, 7, 8],[1, 4, 5, 6, 8, 9],[1, 2, 4, 5, 6, 7, 9],[1, 2, 4, 6, 7, 8, 9],[1, 2, 4, 6, 8, 9],
[1, 2, 3, 5, 6, 8, 9],[2, 3, 4, 5, 7, 8],[1, 2, 3, 4, 7, 8],[1, 2, 4, 5, 6, 7, 8],[1, 2, 3, 5, 8, 9],[1, 2, 3, 5, 6, 8, 9],[1, 2, 4, 5, 7, 9],
[1, 3, 4, 6, 7, 8],[1, 2, 3, 5, 6, 8]    ]

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
,[0.6975489947120348, 0.33821728374203386, 1.610603473155036, 0.5155710690953523, 1.7323939734351246, 0.1871076613335183, 0.6993488616540915, 0.7880715759466592, 1.0738298370884578, 1.2557527159445387, 0.3336088218360487, 0.7180311878919661, 1.5076122475518585, 0.45801406780667264]
    ]

TPG=[(4, 5), (5, 6), (1, 6), (4, 7), (6, 7), (2, 3), (2, 7), (5, 7), (2, 4), (3, 4)]



#######################data2######################
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




################################data3#####################################

T=8;
tasks=[];


E=11
employees=[]
cost=[8.0,15.0,9.0,10.0,8.0,16.0,6.0,2.0]
skills_t = [[5, 6],[1, 5, 7],[3, 5],[5, 8],[2, 5, 7],[1, 2],[2, 4, 7],[1, 5]]

salary = [9012.0,11902.0,9623.0,8649.0,9590.0,10970.0,9349.0,10353.0,11031.0,10286.0,9974.0,]
skills_e = [[1, 2, 3, 5, 6, 7, 8],[1, 2, 4, 5, 6, 7, 8],[1, 2, 3, 5, 6, 8],[1, 2, 3, 5, 7, 8],[2, 3, 4, 5, 7, 8],
[1, 2, 4, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 8],[1, 2, 4, 5, 6, 7, 8],[1, 2, 3, 5, 6, 7, 8],[1, 2, 3, 4, 5, 6, 8],[1, 3, 4, 5, 6, 7, 8]]

team = [[0.5881127469773131, 1.0060132252551688, 0.6111452296830593, 0.7359805517679081, 0.3450844324559834, 0.9607247354567401, 1.7471967223606983, 0.9115577102192169, 1.5854988729852633, 0.5082469458048671, 0.306417606334503]
,[1.0060132252551688, 0.23747296254889294, 0.9714731573801312, 0.4486066044857986, 1.530087865407075, 1.5348284051311676, 0.3022538909062382, 0.12818874913061729, 0.5075307457924187, 0.3063170059492277, 1.499101257298709]
,[0.6111452296830593, 0.9714731573801312, 0.5327607846926927, 0.29387474261542446, 1.2702332725563528, 0.025456398440780514, 1.2312323665167106, 1.7610423544111082, 0.04783640696065672, 1.5316257919724812, 1.7964664480192978]
,[0.7359805517679081, 0.4486066044857986, 0.29387474261542446, 0.27510301118227654, 1.4430841261547784, 1.0849606344262788, 1.860073864254358, 1.6829534474261563, 0.37889522489937555, 0.13686585479293978, 1.4326473727636748]
,[0.3450844324559834, 1.530087865407075, 1.2702332725563528, 1.4430841261547784, 0.6887774894339658, 0.679230457978329, 0.6230869523746156, 0.34084001002042963, 0.10075296870250128, 0.3663361568519301, 0.37863779900985417]
,[0.9607247354567401, 1.5348284051311676, 0.025456398440780514, 1.0849606344262788, 0.679230457978329, 0.7955927491812345, 0.0015997655049768689, 1.1056718302404496, 0.5140201741609518, 1.5814078797167568, 0.8296968407701542]
,[1.7471967223606983, 0.3022538909062382, 1.2312323665167106, 1.860073864254358, 0.6230869523746156, 0.0015997655049768689, 1.9928249806058962, 0.8833016680865122, 0.049814197807386495, 0.39099008863284546, 1.8001246690067136]
,[0.9115577102192169, 0.12818874913061729, 1.7610423544111082, 1.6829534474261563, 0.34084001002042963, 1.1056718302404496, 0.8833016680865122, 0.46503051185694577, 1.2350189847581774, 1.8386025913520903, 1.921215764368499]
,[1.5854988729852633, 0.5075307457924187, 0.04783640696065672, 0.37889522489937555, 0.10075296870250128, 0.5140201741609518, 0.049814197807386495, 1.2350189847581774, 1.6392381842238724, 0.3460524494293278, 0.2634061684730349]
,[0.5082469458048671, 0.3063170059492277, 1.5316257919724812, 0.13686585479293978, 0.3663361568519301, 1.5814078797167568, 0.39099008863284546, 1.8386025913520903, 0.3460524494293278, 0.4916324444900779, 1.1988161367686676]
,[0.306417606334503, 1.499101257298709, 1.7964664480192978, 1.4326473727636748, 0.37863779900985417, 0.8296968407701542, 1.8001246690067136, 1.921215764368499, 0.2634061684730349, 1.1988161367686676, 1.4453569755670017]
]
TPG = [(7, 8), (6, 7), (2, 6), (2, 5), (4, 7), (2, 7), (3, 4), (3, 5), (1, 8), (2, 8)]

#############################data4###########################################


T=5;
tasks=[];


E=13
employees=[]
cost=[5.0,2.0,17.0,7.0,11.0]
skills_t = [[3, 4, 5] , [1, 3, 4],[2, 5, 7],[2, 3],[2, 4]]

salary = [11877.0,12108.0,7911.0,10369.0,10218.0,10271.0,9590.0,10961.0,9920.0,9784.0,11074.0,10465.0,9808.0]
skills_e = [[1, 2, 3, 4, 5, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 5, 6, 7],[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6, 7],
[1, 3, 4, 5, 6, 7],[1, 2, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7],
[1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7]]

team = [[1.5989485302187447, 0.18697106567780097, 0.5602142352255448, 0.6536274074560822, 1.5879497560612141, 1.596730567722217, 0.21796045291467792, 0.9237216313171064, 1.0957615428476102, 1.8138556552782763, 0.3683537586318826, 1.1372393842703834, 1.2674731613044354]
,[0.18697106567780097, 0.6100411103823289, 1.787385221815625, 1.8289725495273288, 1.2524150039488984, 1.6388035284762534, 0.3810867214590834, 1.40020525092311, 0.17693632403486292, 1.3270635785517664, 0.9547825596439099, 0.2299080157367912, 0.7168059423777056]
,[0.5602142352255448, 1.787385221815625, 0.9762829435012133, 0.7905044253241638, 0.8654890576351639, 0.058674178976287905, 1.1789679575154672, 1.9591626184666335, 0.2052757234441649, 0.629082616080815, 0.7929445009072287, 1.0522925583852483, 0.31997390245089163]
,[0.6536274074560822, 1.8289725495273288, 0.7905044253241638, 0.5684402564934896, 0.05207347786540928, 0.9522924534433195, 1.674133686487598, 1.935936060759014, 1.5624750837196737, 0.7792017570993361, 0.8797800253747545, 1.5743570157353322, 0.4587776376456243]
,[1.5879497560612141, 1.2524150039488984, 0.8654890576351639, 0.05207347786540928, 0.3301120694990738, 0.07430648145484797, 1.1811437523964103, 1.380816477028848, 0.8685295966021378, 0.3547312048147897, 1.9624494568871282, 1.3495837274598557, 0.15211292724875403]
,[1.596730567722217, 1.6388035284762534, 0.058674178976287905, 0.9522924534433195, 0.07430648145484797, 0.018690427192354475, 0.9162877448055471, 1.728127705495793, 0.9384336619008551, 0.542011235990935, 1.0430239766136, 1.7421041131622885, 1.3490629101962246]
,[0.21796045291467792, 0.3810867214590834, 1.1789679575154672, 1.674133686487598, 1.1811437523964103, 0.9162877448055471, 0.49798906661061326, 0.7831239434462856, 1.7236664042230412, 0.39519381004299126, 0.4378964415921036, 0.532079075620083, 1.218460579239217]
,[0.9237216313171064, 1.40020525092311, 1.9591626184666335, 1.935936060759014, 1.380816477028848, 1.728127705495793, 0.7831239434462856, 0.006010587219586361, 0.8003456016167014, 1.3478246806160459, 1.5911090395902026, 0.5695913408399798, 1.3026161056638403]
,[1.0957615428476102, 0.17693632403486292, 0.2052757234441649, 1.5624750837196737, 0.8685295966021378, 0.9384336619008551, 1.7236664042230412, 0.8003456016167014, 1.1889618528192505, 0.26917143008371824, 0.5607679157297394, 1.2522560483426426, 0.39396993217096465]
,[1.8138556552782763, 1.3270635785517664, 0.629082616080815, 0.7792017570993361, 0.3547312048147897, 0.542011235990935, 0.39519381004299126, 1.3478246806160459, 0.26917143008371824, 0.3552991276493185, 1.5041371119121325, 1.8454450103799793, 1.9638409863136457]
,[0.3683537586318826, 0.9547825596439099, 0.7929445009072287, 0.8797800253747545, 1.9624494568871282, 1.0430239766136, 0.4378964415921036, 1.5911090395902026, 0.5607679157297394, 1.5041371119121325, 1.086483865977052, 0.7393571334065794, 0.7240236456090523]
,[1.1372393842703834, 0.2299080157367912, 1.0522925583852483, 1.5743570157353322, 1.3495837274598557, 1.7421041131622885, 0.532079075620083, 0.5695913408399798, 1.2522560483426426, 1.8454450103799793, 0.7393571334065794, 0.8464244541422694, 0.373044798165749]
,[1.2674731613044354, 0.7168059423777056, 0.31997390245089163, 0.4587776376456243, 0.15211292724875403, 1.3490629101962246, 1.218460579239217, 1.3026161056638403, 0.39396993217096465, 1.9638409863136457, 0.7240236456090523, 0.373044798165749, 1.0183501661759051]
]
TPG =[(3, 4), (2, 3), (3, 5), (2, 4), (1, 5), (4, 5), (1, 4), (1, 3), (2, 5), (1, 2)]







#############################################################################

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
maxEvaluations=5000
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
            ded = random.randint(0,5)
            ded= ded/10.0
            temp2.append(ded)
        #summ=sum(temp2);
        #for temppp in range(len(temp2)):
        #    temp2[temppp]=temp2[temppp]/summ;
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
                
                for e in employees:
                    dedsum=0
                    for p in V:
                        dedsum=dedsum+object.variables[(p.taskId-1)*E+e.employeeId-1]
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
    
                efficiency=0
                
                #for employee in employees:
                #    sum=sum+object.variables[(task.taskId-1)*E+employee.employeeId-1]
                ratio_sum=0
                for em in range(0,E-1):
                    for em2 in range(em,E):
                        num=0;
                        for sk in task.skills:
                            if sk in employees[em].skills or sk in employees[em2].skills:
                                num=num+1;
                        efficiency = efficiency+employees[em].team[em2]*num/len(task.skills)
                        ratio_sum=ratio_sum+num/len(task.skills)
                
                tkj.append(task.cost*efficiency/(2*ratio_sum))
            for employee in employees:
                Pei.append(employee.salary)
            for employee in employees:
                for task in tasks:
                    projectcost = projectcost+object.variables[(task.taskId-1)*E+employee.employeeId-1]*tkj[task.taskId-1]*Pei[employee.employeeId-1]

            q=projectcost*0.000001+projectduration*0.1;
            
            r=100+10*undt+10*reqsk+0.1*totaloverwork;
            #print("projectcost", projectcost*0.000001, "projecturation", projectduration*0.1 )
            if totaloverwork>1 or undt>0.001 or reqsk>0.001:
                fitness.append( 1/(q+r))
            else:
                fitness.append(1/q)
  
 
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
        
            