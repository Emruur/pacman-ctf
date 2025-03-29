# from myTeam import *

# def createTeam(firstIndex, secondIndex, isRed,
# 							 first = 'TreeSearch', second = 'TreeSearch'):

# 	return [eval(first)(firstIndex, random_rolls = False, custom_heuristic = True, rave = True), eval(second)(secondIndex, random_rolls = False, custom_heuristic = True, rave = True)]

from raveAgent import *

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'TreeSearch', second = 'TreeSearch'):

	return [eval(first)(firstIndex, custom_heuristic = True), eval(second)(secondIndex, custom_heuristic = True)]