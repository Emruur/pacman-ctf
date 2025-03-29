from myTeam import *

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'TreeSearch', second = 'TreeSearch'):

	return [eval(first)(firstIndex, random_rolls = True, custom_heuristic = False, rave = False), eval(second)(secondIndex, random_rolls = True, custom_heuristic = False, rave = False)]