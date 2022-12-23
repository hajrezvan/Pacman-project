# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        newCapsule = successorGameState.getCapsules()
        score = successorGameState.getScore()

        postion = positionEstimate(currentGameState.getPacmanPosition(), newPos = newPos)

        foodScore, minDistance = foodScoreEstimate(newFood, newPos)

        ghostScore = ghostStateEstimate(newGhostStates, newPos, minDistance)

        capsuleScore = capsuleScoreEstimate(newCapsule, minDistance, newPos)

        scaredScore = scaredScoreEstimate(newGhostStates, minDistance, newPos)

        scaredTimer = sum(newScaredTimes)

        finalScore = postion + (1.05 * score) + (1.50 * foodScore) + (1.28 * ghostScore) + (1.8 * scaredScore) + (0.5 * capsuleScore) + (1.2 * scaredTimer)   
             
        return finalScore


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return maximum(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return maximum(gameState, agentIndex, depth)[1]
        else:
            return minimum(gameState, agentIndex, depth)[1]

    def maximum(self, gameState, agentIndex, depth):
        bestAction = ("max",    -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,   self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = max(bestAction,    succAction, key = lambda x:x[1])
        return bestAction

    def minimum(self, gameState, agentIndex, depth):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action,self.minimax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])
        return bestAction

        # Ha Ha Ha :) 🤔
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxLevel(gameState,depth,alpha, beta):
            myDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or myDepth==self.depth: 
                return self.evaluationFunction(gameState)
            maximum = -1000000
            actions = gameState.getLegalActions(0)
            ins = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maximum = max (maximum,minLevel(successor,myDepth,1,ins,beta))
                if maximum > beta:
                    return maximum
                ins = max(ins,maximum)
            return maximum
    
    # For 👻
        def minLevel(gameState, depth, agentIndex, alpha, beta):
            minimum = 1000000
            if gameState.isWin() or gameState.isLose(): 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            ins = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                # 😋 Pacman selection
                if agentIndex == (gameState.getNumAgents()-1):
                    minimum = min (minimum, maxLevel(successor,depth,alpha,ins))
                    if minimum < alpha:
                        return minimum
                    ins = min(ins,minimum)
                # 👻 Ghosts Selection 
                else:
                    minimum = min(minimum,minLevel(successor,depth,agentIndex+1,alpha,ins))
                    if minimum < alpha:
                        return minimum
                    ins = min(ins,minimum)
            return minimum

        
        actions = gameState.getLegalActions(0)
        currentScore = -201420
        returnAction = ''

        alpha = -1000000
        beta = 1000000
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            score = minLevel(nextState,0,1,alpha,beta)
            if score > currentScore:
                returnAction = action
                currentScore = score
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState,depth):
            Actions=gameState.getLegalActions(0)
            if len(Actions)==0 or gameState.isWin() or gameState.isLose() or depth==self.depth:
                return (self.evaluationFunction(gameState),None)
            w=-(float("inf"))
            Act=None

            for action in Actions:
                sucsValue=exp_value(gameState.generateSuccessor(0,action),1,depth)
                sucsValue=sucsValue[0]
                if(w<sucsValue):
                    w,Act=sucsValue,action
            return(w,Act)

        def exp_value(gameState,agentID,depth):
            Actions=gameState.getLegalActions(agentID)
            if len(Actions)==0:
                return (self.evaluationFunction(gameState),None)

            l=0
            Act=None
            for action in Actions:
                if(agentID==gameState.getNumAgents() -1):
                    sucsValue=max_value(gameState.generateSuccessor(agentID,action),depth+1)
                else:
                    sucsValue=exp_value(gameState.generateSuccessor(agentID,action),agentID+1,depth)
                sucsValue=sucsValue[0]
                prob=sucsValue/len(Actions)
                l+=prob
            return(l,Act)

        max_value=max_value(gameState,0)[1]
        return max_value

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    foodList = newFood.asList()
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos,pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos,pos))

    numberofPowerPellets = len(successorGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))           
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum (ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)
        
    score += successorGameState.getScore()  + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:    
        score +=   sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else :
        score +=  sumGhostDistance + numberofPowerPellets
    return score

    util.raiseNotDefined()

def foodScoreEstimate(newFood, NewPositions):
    foodDistance = []

    for i in newFood.asList():
      if newFood[i[0]][i[1]]:
        foodDistance.append(manhattanDistance(NewPositions, i))

    foodScore = 0
    minDistance = 0
    if foodDistance:
        minDistance = min(foodDistance)
        if minDistance == 0:
            foodScore = 100
        else:
            foodScore = 10/minDistance
    return foodScore, minDistance

def ghostStateEstimate(newGhostStates, newPos, minDistance):
    activeGhostPositions = [i.getPosition() for i in newGhostStates if i.scaredTimer == 0]
    ghostScore = 100
    maxGhostDistance = -1
    minGhostDistance = -1
    if newPos in activeGhostPositions:
        ghostScore = -1000
    activeGhostDistance = [manhattanDistance(newPos, i) for i in activeGhostPositions]

    if activeGhostDistance:
        maxGhostDistance = max(activeGhostDistance)
        minGhostDistance = min(activeGhostDistance)
        if minGhostDistance <= 1:
            ghostScore = -1000
            minGhostDistance = -1
        else:
            if minGhostDistance < minDistance:
                ghostScore = -100
            ghostScore = maxGhostDistance
    return ghostScore

def capsuleScoreEstimate(newCapsule, minDistance, newPos):
    capsuleScore = 0
    if len(newCapsule) >= 1:
        capsuleDistance = min([manhattanDistance(newPos, i) for i in newCapsule])
    if newPos in newCapsule and capsuleDistance < minDistance:
        capsuleScore = 2000
    return capsuleScore

def scaredScoreEstimate(newGhostStates, minDistance, newPos):
    scaredGhostPositions = [i.getPosition() for i in newGhostStates if i.scaredTimer != 0]
    scaredScore = 0
    if len(scaredGhostPositions) > 0:
        if newPos in scaredGhostPositions:
            scaredScore = 20000
        else:
            scaredGhostDistance = min([manhattanDistance(newPos, i) for i in scaredGhostPositions])
            if scaredGhostDistance < minDistance:
                scaredScore = minDistance - scaredGhostDistance
    return scaredScore

def positionEstimate(currentPos, newPos):
    if currentPos == newPos:
        return -100
    else:
        return 0

# Abbreviation
better = betterEvaluationFunction
