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
import random, time, util
from game import Directions
import game
from MCTS import MCTS
from baselineTeam import ReflexCaptureAgent
from math import sqrt, log
import numpy as np
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'TreeSearch', second = 'TreeSearch'):
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

class DummyAgent(CaptureAgent):
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

		'''
		Your initialization code goes here, if you need any.
		'''


	def chooseAction(self, gameState):

		# rollout_method can take:
		#   - "random":         The agent selects a random legal action.
		#   - "reflex":         The agent uses its reflex-based chooseAction method.
		#   - "custom_heuristic": The agent uses its custom heuristic-based chooseAction method.
		
		# game_score can take:
		#   - "reflex_heuristic": The score is computed using evaluateGameState().
		#   - "custom_heursitic": The score is computed using HeuristicAgent.evaluateState().
		#   - Any other value:   The score is taken directly from simulation_state.data.score.

		print(gameState.getScore())
		quit()
		return best_move
	
class Node():
	def __init__(self, parent, action_taken, agent_id, gameState):
		self.agent_id = agent_id
		self.parent = parent
		self.parent_a = action_taken
		self.tot_s = 0
		self.visits = 0
		self.state = gameState
		self.untried_moves = gameState.getLegalActions(agent_id)[:]
		self.children = []

	def expand(self):
		if search_agent.random_rolls:
			a = random.randint(0, len(self.untried_moves)-1)
			self.children.append(Node(self, self.untried_moves[a], (self.agent_id+1)%4, self.state.generateSuccessor(self.agent_id, self.untried_moves.pop(a))))
		else:
			next_states = [Node(self, self.untried_moves[i], self.agent_id+1 % 4, self.state.generateSuccessor(self.agent_id, self.untried_moves[i])) for i in range(len(self.untried_moves))]
			if self.agent_id+1 % 4 in search_agent.getOpponents():
				a = np.argmax([s.opp_heuristic() for s in next_states])
			else:
				a = np.argmax([s.self_heuristic() for s in next_states])
		return self.children[-1]
	
	def child_uct(self, child, c):
		return (child.tot_s / child.visits) + (c * (sqrt((2 * log(self.visits)) / child.visits)))
	
	def bestChild(self, explo_factor=2, root_debug=False):
		best = -99999
		best_c = -1
		for i, c in enumerate(self.children):
			if root_debug:
				print(self.child_uct(c,explo_factor))
				print(f'best: {best}')
			if self.child_uct(c, explo_factor) > best:
				best_c = i
				best = self.child_uct(c, explo_factor)

		if best_c == -1:
			print("error in best child selection")

		return best_c, self.children[best_c]
	
	def backup(self, score):
		self.visits += 1
		self.tot_s += score
		if not self.parent is None:
			self.parent.backup(score)

	def self_heuristic(self):
		global search_agent
		foodList = search_agent.getFood(self.state).asList() 
		if len(foodList) > 0:
			myPos = self.state.getAgentState(self.agent_id).getPosition()
			minDistance = min([search_agent.getMazeDistance(myPos, food) for food in foodList])
			food_d = minDistance
		
		return search_agent.getScore(self.state) * 100 - food_d - sum([self.state.getAgentState(i).numCarrying for i in search_agent.getOpponents(self.state)]) + sum([self.state.getAgentState(i).numCarrying+self.state.getAgentState(i).numReturned for i in search_agent.getTeam(self.state)])

	def opp_heuristic(self):
		global search_agent
		foodList = search_agent.getFood(self.state).asList() 
		if len(foodList) > 0:
			myPos = self.state.getAgentState(self.agent_id).getPosition()
			minDistance = min([search_agent.getMazeDistance(myPos, food) for food in foodList])
			food_d = minDistance
		
		return -search_agent.getScore(self.state) * 100 - food_d + sum([self.state.getAgentState(i).numCarrying for i in search_agent.getOpponents(self.state)]) - sum([self.state.getAgentState(i).numCarrying+self.state.getAgentState(i).numReturned for i in search_agent.getTeam(self.state)])


		
class TreeSearch(CaptureAgent):
	def __init__(self, agent_id, d=100, i=2000, random_rolls=True, beta=0):
		super().__init__(agent_id)
		global search_agent
		search_agent = self
		self.rollout_d = d
		self.rollout_i = i
		self.random_rolls = random_rolls
		self.beta = beta

	def treePolicy(self, node):
		for d in range(self.rollout_d):
			if node.state.isOver():
				return node, d
			elif len(node.untried_moves) > 0:
				return node.expand(), d
			else:
				node = node.bestChild()[1]
		return node, self.rollout_d // 2

	def defaultPolicy(self, node):
		for depth in range(self.rollout_d):
			if node.state.isOver():
				return node
			if len(node.untried_moves) > 0:
				node = node.expand()
			else:
				node = node.bestChild()[1]
		return node
			

	def chooseAction(self, gamestate):
		root = Node(None, None, self.index, gamestate)
		for n in range(self.rollout_i):
			node, depth = self.treePolicy(root)
			end_node = self.defaultPolicy(node)
			
			foodList = self.getFood(end_node.state).asList() 
			if len(foodList) > 0:
				myPos = end_node.state.getAgentState(self.index).getPosition()
				minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
				food_d = minDistance

			#print(super().getScore(end_node.state) * 100 - food_d)
			end_node.backup(super().getScore(end_node.state) * 100 - food_d - sum([end_node.state.getAgentState(i).numCarrying for i in self.getOpponents(end_node.state)]) + sum([end_node.state.getAgentState(i).numCarrying+end_node.state.getAgentState(i).numReturned for i in self.getTeam(end_node.state)]))

		for c in root.children:
			print(c.parent_a, c.tot_s/c.visits)
			
		print(f'action {root.bestChild(0)[1].parent_a} taken')
		
		return root.bestChild(0)[1].parent_a
		
