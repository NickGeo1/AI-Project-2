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
import sys

from game import Agent, AgentState

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
        #if the best state is stop, we will be stuck so we take the next best step
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
        newFood = successorGameState.getFood();
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        '''
        idea: we substract the manhatan distance to the closest dot from the current score of the successor.
              Then, we substract a ghost avoid rate between pacman and closest ghost, if that ghost is not scared. If closest ghost is
              scared, we add the difference between the remaining scaretimer and pacman's steps to the ghost

              For the ghost avoid rate, we want something that keeps pacman away from ghosts with more force if
              pacman is close to ghosts rather than far from them. Furthermore, we want something that makes pacman
              avoid to eat the next dot if the ghost is close to that dot. We considered the score gained from the dot
              (+10) to form the avoid rate.
        '''
        #get the manhatan distance between the successor point and the closest dot if the successor point is not dot
        has_food = True if newPos in currentGameState.getFood().asList() else False
        if(not has_food):
            min_food_distance = min([manhattanDistance(newPos,dot) for dot in newFood.asList()])
        else: 
            min_food_distance = 0
      
        ghost_distances_and_points = [(manhattanDistance(newPos,ghostState.getPosition()),ghostState.getPosition()) for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        scared_ghost_distances_and_points = [(manhattanDistance(newPos,ghostState.getPosition()),ghostState.getPosition()) for ghostState in newGhostStates if ghostState.scaredTimer > 0]

        scared_ghost_distances = [sgd[0] for sgd in scared_ghost_distances_and_points]
        scared_ghost_points = [sgp[1] for sgp in scared_ghost_distances_and_points]

        ghost_distances = [gd[0] for gd in ghost_distances_and_points]

        step_difference = 0
        if len(scared_ghost_distances):
            scared_ghost_distances_min = min(scared_ghost_distances)
            scared_ghost_points_min = scared_ghost_points[scared_ghost_distances.index(scared_ghost_distances_min)]
            ghostTimer = [ghostState.scaredTimer for ghostState in newGhostStates if ghostState.getPosition() == scared_ghost_points_min][0]
            step_difference = ghostTimer - manhattanDistance(newPos, scared_ghost_points_min)

        ghost_avoid_rate = 0
        if len(ghost_distances):
            ghost_distances_min = min(ghost_distances)
            if(ghost_distances_min!=0):
                ghost_avoid_rate = 21/ghost_distances_min

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore() - min_food_distance - ghost_avoid_rate + step_difference

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
        #return the minimax-optimal action of pacman.
        #We recursively expand the minimax tree starting from player 0(pacman) and we add one to player index at each tree level
        #We recursively return the optimal action and utility to each parent expanded
        return self.miniMax(gameState,0,self.depth)[1]

    def miniMax(self, gameState, player, depth):

        util_actions = []   #a list where we keep the utils and actions of each successor state

        if(gameState.isWin() or gameState.isLose() or depth == 0): #if this gamestate is Terminal (Win, Lose, or reached the maxdepth)
            return (self.evaluationFunction(gameState), None) #return the evaluation value of the state as util and No action

        if(player == 0):        #if pacman is playing
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                #get successor's utility by recursively calling miniMax for successor state
                #note that we expand the successor for the next player(if we get above max player we get back to 0)
                utility = self.miniMax(successor,(player+1)%gameState.getNumAgents(), depth)[0]
                util_actions.append((utility, action))

            utils = [putil[0] for putil in util_actions]
            actions = [action[1] for action in util_actions]

            max_util = max(utils)
            max_action = actions[utils.index(max_util)]

            return (max_util, max_action) #return max utility and max action

        else: #else any other ghost is playing
            #if the ghost player is the last, we substract one from the remaining depth to search
            if(player == gameState.getNumAgents() - 1): 
                depth -= 1

            for ghostaction in gameState.getLegalActions(player): #for all the ghost successor actions
                ghostsuccessor = gameState.generateSuccessor(player, ghostaction) #generate successor
                #get successor's utility by recursively calling miniMax for successor state
                #note that we expand the successor for the next player(if we get above max player we get back to 0)
                utility = self.miniMax(ghostsuccessor,(player+1)%gameState.getNumAgents(), depth)[0] 
                util_actions.append((utility, ghostaction))

            ghost_utils = [putil[0] for putil in util_actions]
            ghost_actions = [action[1] for action in util_actions]

            min_ghost_util = min(ghost_utils)
            min_action = ghost_actions[ghost_utils.index(min_ghost_util)]
            
            return (min_ghost_util, min_action) #return min utility and min action

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

        return self.miniMaxAlphaBetaPruning(gameState,0,self.depth,-sys.maxsize,sys.maxsize)[1]

    def miniMaxAlphaBetaPruning(self, gameState, player, depth, a, b):

        if(gameState.isWin() or gameState.isLose() or depth == 0): #if this gamestate is Terminal (Win, Lose, or reached the maxdepth)
            return (self.evaluationFunction(gameState), None) #return the evaluation value of the state as util and No action

        if(player == 0):        #if pacman is playing
            utility = -sys.maxsize #initialize utility and max_action
            max_action = 0
            for action in gameState.getLegalActions(0):               
                successor = gameState.generateSuccessor(0, action)  #generate pacman successor

                #If the successor's utility is greater than the current utility update the utility value
                #We recursively call miniMaxAlphaBetaPruning to calculate successor's utility
                utility = max(self.miniMaxAlphaBetaPruning(successor, (player+1)%gameState.getNumAgents(), depth, a, b)[0], utility)
                
                #Note that above the current max node we could have a min node. If the successor of this max node
                #has a utility greater than the current minimum utility of the max nodes at this level, that means we can stop searching this max
                #node because we need to find a smaller value max node for the min parent node.
                #b is the current minimum value of the max nodes
                if (utility > b):
                    return (utility, action)
                #a is the current maximum utility value of the min childs of this max node
                old_a = a
                a = max(a, utility) #we update a with the child utility if child utility > a
                if(a > old_a):
                    max_action = action #If a beacame greater, that means the new min child node has bigger utility
                                        #so we update the max action 

            return (utility, max_action) #return max utility and max action

        elif (player != 0): #else any other ghost is playing
            #if the ghost player is the last, we substract one from the remaining depth to search
            if(player == gameState.getNumAgents() - 1): 
                depth -= 1

            utility = sys.maxsize #initialize utility and min_action
            min_action = 0
            for action in gameState.getLegalActions(player):
                successor = gameState.generateSuccessor(player, action)     
                
                #If the successor's utility is less than the current utility update the utility value
                #We recursively call miniMaxAlphaBetaPruning to calculate successor's utility
                utility = min(self.miniMaxAlphaBetaPruning(successor, (player+1)%gameState.getNumAgents(), depth, a, b)[0], utility)

                #Note that above the current min node we could have a max node. If the successor of this min node
                #has a utility less than the current maximum utility of the min nodes at this level, that means we can stop searching this min
                #node because we need to find a greater value min node for the max parent node.
                #a is the current maximum value of the min nodes
                #In case we have a min node parent this works the same.
                if (utility < a):
                    return (utility, action)
                #b is the current minimum utility value of the max/min childs of this min node
                old_b = b
                b = min(b, utility) #we update b with the child utility if child utility < b
                if(b < old_b):
                    min_action = action #If b beacame smaller, that means the new max/min child node has smaller utility
                                        #so we update the min action

            return (utility, min_action) #return min utility and min action


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
        return self.expectiMax(gameState,0,self.depth)[1]

    def expectiMax(self, gameState, player, depth):

        util_actions = []

        if(gameState.isWin() or gameState.isLose() or depth == 0):
            return (self.evaluationFunction(gameState), None)

        if(player == 0):
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                utility = self.expectiMax(successor,(player+1)%gameState.getNumAgents(), depth)[0]
                util_actions.append((utility, action))

            utils = [putil[0] for putil in util_actions]
            actions = [action[1] for action in util_actions]

            max_util = max(utils)
            max_action = actions[utils.index(max_util)]

            return (max_util, max_action)

        else:
            probability = 0.0

            if(player == gameState.getNumAgents() - 1):
                depth -= 1

            if len(gameState.getLegalActions(player)) != 0:
                probability = 1.0/len(gameState.getLegalActions(player))

            for ghostaction in gameState.getLegalActions(player):
                ghostsuccessor = gameState.generateSuccessor(player, ghostaction)
                utility = self.expectiMax(ghostsuccessor,(player+1)%gameState.getNumAgents(), depth)[0]
                util_actions.append((utility, ghostaction))

            ghost_utils = [putil[0] for putil in util_actions]
            ghost_actions = [action[1] for action in util_actions]

            min_ghost_util = 0.0
            for putil in ghost_utils:
                min_ghost_util += putil*probability

            min_action = ghost_actions[0]

            return (min_ghost_util, min_action)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
