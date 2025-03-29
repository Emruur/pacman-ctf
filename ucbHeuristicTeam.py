from myTeam import *

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'TreeSearch', second = 'TreeSearch'):

	return [eval(first)(firstIndex, random_rolls = False, custom_heuristic = True, rave = False), eval(second)(secondIndex, random_rolls = False, custom_heuristic = True, rave = False)]