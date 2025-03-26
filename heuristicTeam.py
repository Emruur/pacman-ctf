from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from baselineTeam import ReflexCaptureAgent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HeuristicAgent', second = 'HeuristicAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class HeuristicAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  @staticmethod
  def evaluateState(gameState):
      """
      Computes the state's value V(s) as the maximum Q(s, a) over all legal actions.
      A temporary HeuristicAgent instance (with a default index, e.g., 0) is used
      to leverage the existing evaluation logic.
      """
      dummy = HeuristicAgent(0)
      dummy.registerInitialState(gameState)
      legalActions = gameState.getLegalActions(dummy.index)
      if not legalActions:
          return 0
      values = [dummy.evaluate(gameState, action) for action in legalActions]
      return max(values)

  def chooseActionSoftmax(self, gameState, softmax_temp):
    """
    Chooses an action stochastically using softmax over Q-values.
    The probability of selecting an action a is proportional to exp((Q(s, a) - maxQ) / softmax_temp)
    to ensure numerical stability and avoid division by zero.
    """
    import math, random

    actions = gameState.getLegalActions(self.index)
    if not actions:
        return None

    # Compute Q-values for each legal action.
    qValues = [self.evaluate(gameState, action) for action in actions]
    
    # Normalize Q-values by subtracting the maximum Q-value for numerical stability.
    maxQ = max(qValues)
    expValues = [math.exp((q - maxQ) / softmax_temp) for q in qValues]
    total = sum(expValues)
    
    # Fallback to a uniform distribution if total is zero (should rarely happen)
    if total == 0:
        probs = [1.0 / len(expValues)] * len(expValues)
    else:
        probs = [expVal / total for expVal in expValues]
    
    # Stochastically choose an action based on the computed probabilities.
    chosenAction = random.choices(actions, weights=probs, k=1)[0]
    return chosenAction



  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    #print(f"### Agent {self.index} ###")
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    #print(f"Best actions: {bestActions}")
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)  

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    # print(features)
    #print(f"{action}: {features*weights}")
    return features * weights
  
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    
    # score of the game
    features["successorScore"] = gameState.data.score
    
    # amount of food left    
    foodList = self.getFood(successor).asList()    
    features['foodLeft'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    # position heuristics
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    # Compute distance to defenders
    if myState.isPacman:
      defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
      if len(defenders) > 0:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
          features['ghostDistance'] = min(dists)
    
	
 
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1000, 'foodLeft': 100, 'distanceToFood': -1,
            'numInvaders': -100, 'onDefense': 1,
            'invaderDistance': -10, 'ghostDistance': 1,'stop': -100, 'reverse': -2}