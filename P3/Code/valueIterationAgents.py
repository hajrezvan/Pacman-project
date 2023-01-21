# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for state in self.mdp.getStates(): #get all the states
            self.values[state] = 0.0

        for i in range(self.iterations):   #run for these many iterations
            next_values = self.values.copy() #copy back the old values
            for state in self.mdp.getStates():
                state_values = util.Counter() #values for actions of this state
                for action in self.mdp.getPossibleActions(state):
                    state_values[action] = self.getQValue(state, action)
                next_values[state] = state_values[state_values.argMax()]  #update for each state
            self.values = next_values.copy() #copy back the new values        


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0.0
        for transition in transition_probs:
            Tstate, prob = transition
            QValue += prob * (self.mdp.getReward(state, action, Tstate) + self.discount * self.getValue(Tstate))

        return QValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if (self.mdp.isTerminal(state)):
            return None
        else:
            QValues = util.Counter()
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                QValues[action] = self.computeQValueFromValues(state, action)

            return QValues.argMax()
            
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.values = collections.defaultdict(float)
        states = self.mdp.getStates()
        for state in states:
            self.values[state] = 0

        "*** YOUR CODE HERE ***"
        statesSize = len(states)

        for i in range(self.iterations):
            state = states[i % statesSize]

            if not self.mdp.isTerminal(state):
                A = self.computeActionFromValues(state)
                Q = self.computeQValueFromValues(state, A)
                self.values[state] = Q


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        mdp = self.mdp
        values = self.values
        discount = self.discount
        iterations = self.iterations
        theta = self.theta
        states = mdp.getStates()


        predecessors = {} # dict
        for state in states:
          predecessors[state] = set()

        pq = util.PriorityQueue()

        # computes predecessors and puts initial stuff into pq
        for state in states:
          Q_s = util.Counter()

          for action in mdp.getPossibleActions(state):
            # assigning predecessors
            T = mdp.getTransitionStatesAndProbs(state, action)
            for (nextState, prob) in T:
              if prob != 0:
                predecessors[nextState].add(state)

            # computing Q values for determining diff's for the pq
            Q_s[action] = self.computeQValueFromValues(state, action)

          if not mdp.isTerminal(state): # means: if non terminal state
            maxQ_s = Q_s[Q_s.argMax()]
            diff = abs(values[state] - maxQ_s)
            pq.update(state, -diff)


        # now for the actual iterations
        for i in range(iterations):
          if pq.isEmpty():
            return

          state = pq.pop()

          if not mdp.isTerminal(state):
            Q_s = util.Counter()
            for action in mdp.getPossibleActions(state):
              Q_s[action] = self.computeQValueFromValues(state, action)

            values[state] = Q_s[Q_s.argMax()]

          for p in predecessors[state]:
            Q_p = util.Counter()
            for action in mdp.getPossibleActions(p):
              # computing Q values for determining diff's for the pq
              Q_p[action] = self.computeQValueFromValues(p, action)

            #if not mdp.isTerminal(state): # means: if non terminal state
            maxQ_p = Q_p[Q_p.argMax()]
            diff = abs(values[p] - maxQ_p)
              
            if diff > theta:
              pq.update(p, -diff)


