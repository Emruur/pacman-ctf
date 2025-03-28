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

from baselineTeam import OffensiveReflexAgent
from baselineTeam import DefensiveReflexAgent
from heuristicTeam import HeuristicAgent
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
		a = random.randint(0, len(self.untried_moves)-1)
		self.children.append(Node(self, self.untried_moves[a], (self.agent_id+1)%4, self.state.generateSuccessor(self.agent_id, self.untried_moves.pop(a))))
		'''else:
			next_states = [Node(self, self.untried_moves[i], (self.agent_id+1) % 4, self.state.generateSuccessor(self.agent_id, self.untried_moves[i])) for i in range(len(self.untried_moves))]
			if self.agent_id+1 % 4 in search_agent.getOpponents(next_states[0].state):
				a = np.argmax([s.opp_heuristic() for s in next_states])
			else:
				a = np.argmax([s.self_heuristic() for s in next_states])
			self.untried_moves.pop(a)
			self.children.append(next_states[a])'''
		return self.children[-1]
	
	def child_uct(self, child, c):
		return (child.tot_s / child.visits) + (c * (sqrt((2 * log(self.visits)) / child.visits)))
	
	def bestChild(self, explo_factor=2, root_debug=False):
		global search_agent
		best = -99999
		best_c = -1
		
		for i, c in enumerate(self.children):
			if root_debug:
				print(self.child_uct(c,explo_factor))
				print(f'best: {best}')
			if not search_agent.rave:
				if self.child_uct(c, explo_factor) > best:
					best_c = i
					best = self.child_uct(c, explo_factor)
			else:
				v = (1-search_agent.beta)*  (c.tot_s/c.visits) + search_agent.beta * (search_agent.actions[c.parent_a][1]/search_agent.actions[c.parent_a][0])
				if v > best:
					best_c = i
					best = v

		if best_c == -1:
			print("error in best child selection")

		return best_c, self.children[best_c]
	
	def backup(self, score):
		global search_agent
		self.visits += 1
		self.tot_s += score
		if not self.parent is None:
			search_agent.actions[self.parent_a][0] += 1
			search_agent.actions[self.parent_a][1] += score
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
	def __init__(self, agent_id, d=25, i=100, random_rolls=False, custom_heuristic = True, beta=0.3, rave=False):
		super().__init__(agent_id)
		global search_agent
		search_agent = self
		self.rollout_d = d
		self.rollout_i = i
		self.random_rolls = random_rolls
		self.custom_heuristic = custom_heuristic
		self.beta = beta
		self.actions = directions = {Directions.NORTH: [0, 0],
                   Directions.SOUTH: [0, 0],
                   Directions.EAST:  [0, 0],
                   Directions.WEST:  [0, 0],
                   Directions.STOP:  [0, 0]}
		self.rave = rave

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
		
		if self.custom_heuristic:
			agents = [HeuristicAgent(i) for i in range(4)]
		else:
			agents_defense = [DefensiveReflexAgent(i) for i in range(2)]
			agents_offense = [OffensiveReflexAgent(i) for i in range(2,4)]
			agents = agents_defense + agents_offense
		list(map(lambda agent: agent.registerInitialState(node.state), agents))
		curr_id = node.agent_id
		self_id= curr_id
		simulation_state = node.state

		for depth in range(self.rollout_d):
			if node.state.isOver():
				return node
			else:
				if self.random_rolls:
					a = random.choice(node.state.getLegalActions(node.agent_id)[:])
					node = Node(node, a, (node.agent_id+1)%4, node.state.generateSuccessor(node.agent_id, a))
				else:
					curr_agent = agents[curr_id]
					if random.random() < 0.2:
						legal_actions = simulation_state.getLegalActions(curr_id)
						action = random.choice(legal_actions)
					else:
						action = curr_agent.chooseAction(simulation_state)
      
					simulation_state = curr_agent.getSuccessor(simulation_state, action)
					curr_id = (curr_id + 1) % 4
					node = Node(node, action, curr_id, simulation_state)
     
     
		return node


	def calculate_score(self, state, food_d):
		"""
		Calculate the score based on the current game state and food distance.

		Parameters:
		- state: The current game state.
		- food_d: The distance to the nearest food.

		Returns:
		- Computed score as a float or integer.
		"""
		# Base score from the superclass method, scaled by 100
		base_score = super().getScore(state) * 100

		# Penalty based on the distance to the nearest food
		food_distance_penalty = food_d

		# Penalty for opponents' carried food
		opponents_food_penalty = sum(state.getAgentState(i).numCarrying for i in self.getOpponents(state))

		# Reward for team's carried and returned food
		team_food_reward = sum(state.getAgentState(i).numCarrying + state.getAgentState(i).numReturned for i in self.getTeam(state))

		# Final score calculation
		final_score = base_score - food_distance_penalty - opponents_food_penalty + team_food_reward

		return final_score	

	def calculate_score_extended(self, node, food_d):
		state = node.state
		features = util.Counter()
		# score of the game
		features["score"] = super().getScore(state)

		# amount of food left
		foodList = self.getFood(state).asList()    
		features['foodLeft'] = -len(foodList)#self.getScore(successor)

		# dist to nearest food
		features['distanceToFood'] = food_d
		for i in self.getTeam(state):
			# position heuristics
			myState = state.getAgentState(i)
			myPos = myState.getPosition()
	
			# Computes whether we're on defense (1) or offense (0)
			features['onDefense'] = 1
			if myState.isPacman: features['onDefense'] = 0

			# Computes distance to invaders we can see
			enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
			invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
			features['numInvaders'] = len(invaders)
			if len(invaders) > 0:
				dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
				features['invaderDistance'] = np.mean(min(dists), features['invaderDistance'])

			# action = node.parent_a
			# if action == Directions.STOP: features['stop'] = 1
			# rev = Directions.REVERSE[state.getAgentState(i).configuration.direction]
			# if action == rev: features['reverse'] = 1

			# Compute distance to defenders
			if myState.isPacman:
				defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
				if len(defenders) > 0:
					dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
					features['ghostDistance'] = min(dists)

			weights = {'successorScore': 1000, 'foodLeft': 100, 'distanceToFood': -1,
            'numInvaders': -100, 'onDefense': 1,
            'invaderDistance': -10, 'ghostDistance': 1,'stop': -100, 'reverse': -2}
   
			return features*weights
   
	def chooseAction(self, gamestate):
		self.actions = directions = {Directions.NORTH: [0, 0],
                   Directions.SOUTH: [0, 0],
                   Directions.EAST:  [0, 0],
                   Directions.WEST:  [0, 0],
                   Directions.STOP:  [0, 0]}
		root = Node(None, None, self.index, gamestate)
		for n in range(self.rollout_i):
			node, depth = self.treePolicy(root)
			end_node= self.defaultPolicy(node)
			
			foodList = self.getFood(end_node.state).asList() 
			if len(foodList) > 0:
				myPos = end_node.state.getAgentState(self.index).getPosition()
				minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
				food_d = minDistance

			# score = self.calculate_score(end_node.state, food_d)
			score = self.calculate_score_extended(end_node, food_d)
			end_node.backup(score)

		for c in root.children:
			print(c.parent_a, c.tot_s/c.visits)
			
		print(f'action {root.bestChild(0)[1].parent_a} taken')
		
		return root.bestChild(0)[1].parent_a
		
