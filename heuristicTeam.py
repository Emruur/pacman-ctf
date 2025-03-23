# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent, OffensiveReflexAgent
import random, time, util
from game import Directions
import game
from MCTS import MCTS

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'DummyAgent', second = 'DummyAgent'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.  isRed is True if the red team is being created, and
	will be False if the blue team is being created.

	As a potentially helpful development aid, this function can take
	additional string-valued keyword arguments ("first" and "second" are
	such arguments in the case of this function), which will come from
	the --redOpts and --blueOpts command-line arguments to capture.py.
	For the nightly contest, however, your team will be created without
	any extra arguments, so you should make sure that the default
	behavior is what you want for the nightly contest.
	"""

	# The following line is an example only; feel free to change it.
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(OffensiveReflexAgent):
	"""
	A Dummy agent to serve as an example of the necessary agent structure.
	You should look at baselineTeam.py for more details about how to
	create an agent as this is the bare minimum.
	"""

	def registerInitialState(self, gameState):
		"""
		This method handles the initial setup of the
		agent to populate useful fields (such as what team
		we're on).

		A distanceCalculator instance caches the maze distances
		between each pair of positions, so your agents can use:
		self.distancer.getDistance(p1, p2)

		IMPORTANT: This method may run for at most 15 seconds.
		"""

		'''
		Make sure you do not delete the following line. If you would like to
		use Manhattan distances instead of maze distances in order to save
		on initialization time, please take a look at
		CaptureAgent.registerInitialState in captureAgents.py.
		'''
		CaptureAgent.registerInitialState(self, gameState)
		self.start = gameState.getAgentPosition(self.index)


		'''
		Your initialization code goes here, if you need any.
		'''


	def chooseAction(self, gameState):
		# print(self.evaluate_state(gameState))
		actions = gameState.getLegalActions(self.index)
		# return random.choice(actions)
   
		values = [self.evaluate(gameState, action) for action in actions]
		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v==maxValue]
  
		return random.choice(bestActions)
		# values = [self.evaluate(gameState, a) for a in actions]

		# maxValue = max(values)
		# bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		# return bestActions[0]

	def evaluate(self, gameState, action):
		foodAdvantage = self.getFoodAdvantage(gameState, action)
		foodDistAdvantage = self.getFoodDistAdvantage(gameState, action)
		oppDistAdvantage = self.getOppDistAdvantage(gameState, action)

		# combining
		print(f"({foodAdvantage})+({foodDistAdvantage})+{oppDistAdvantage}")
		score = foodAdvantage + foodDistAdvantage + oppDistAdvantage
		print(f"total score to maximize for {action}: {score}")
		return score
  
	def getFoodAdvantage(self, gameState, action):
		successor = self.getSuccessor(gameState, action)
		ownFoodLeft = len(self.getFood(successor).asList())
		oppFoodLeft = len(self.getFoodYouAreDefending(successor).asList())
		foodAdvantage = ownFoodLeft-oppFoodLeft	
		print(f"food advantage: {foodAdvantage}")
		return foodAdvantage

	def getFoodDistAdvantage(self, gameState, action):
		'''move agents closer to food'''
		successor = self.getSuccessor(gameState, action)
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
  
		foodList = self.getFood(gameState).asList()
		if len(foodList)>0:
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			print(f"closest food {minDistance} away")
		else: 
			minDistance =0
		return -minDistance

	def getOppDistAdvantage(self, gameState, action):
		oppDistAdvantage = 0
		successor = self.getSuccessor(gameState, action)
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		ownTurf = not successor.getAgentState(self.index).isPacman # whether the agent is in their home color area (e.g. red agent in red)
		for opp in enemies:
			oppDist = self.getMazeDistance(myPos, opp.getPosition())
			if ownTurf and opp.isPacman:# if opp is an invader and current agent is "home"
				oppDistAdvantage -= oppDist # minimize distance
				print(f"minimizing {oppDist}")
			elif (not ownTurf) and (not opp.isPacman): # if current agent is the invader
				oppDistAdvantage += oppDist # maximize distance
				print(f"maximizing {oppDist}")
			else: # if agent and and opp are on opposite sides
				print("opp and agent on opposit esides")
		return oppDistAdvantage