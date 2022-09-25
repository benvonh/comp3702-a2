import sys
import time
import numpy as np
from constants import *
from environment import *
from state import State
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22
"""


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        # Inputs
        self.S = None
        self.A = ROBOT_ACTIONS
        # Outputs
        self.V = None
        self.PI = None
        # Parameters
        self.k = 0 # time step iteration
        self.epsilon = environment.epsilon # convergence threshold
        self.discount = environment.gamma # discount factor

    # === Values Iteration =============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Values Iteration.
        """
        self.k = 0

        # Initialise set of all states
        self.S = list()
        frontier = [self.environment.get_init_state()]
        while frontier:
            expanding_state = frontier.pop()
            for a in self.A:
                cost, new_state = self.environment.apply_dynamics(expanding_state, a)
                if new_state not in self.S:
                    self.S.append(new_state)
                    frontier.append(new_state)

        # Initialises value list
        self.V = list()
        self.V.append(dict())
        for s in self.S:
            self.V[self.k][s] = 0

    def vi_is_converged(self):
        """
        Check if Values Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        for s in self.S:
            if np.abs(self.V[self.k][s] - self.V[self.k-1][s]) > self.epsilon:
                return False

        return True

    def vi_iteration(self):
        """
        Perform a single iteration of Values Iteration (i.e. loop over the state space once).
        """
        self.k += 1
        self.V[self.k].append(dict())

        for s in self.S:
            values = np.zeros(len(self.A))
            for a in self.A:
                for sd in self.S:
                    P = 1 - self.environment.drift_cw_probs[a] - self.environment.drift_ccw_probs[a] - self.environment.double_move_probs[a]
                    R = 
                    values[a] += P * (R + self.discount * self.V[self.k-1][sd])
            self.V[self.k][s] = np.max(values)

        print('iteration??')

    def vi_plan_offline(self):
        """
        Plan using Values Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        #self.vi_iteration() # FIXME: SUS
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.V[self.k][state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Values Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        policy = np.zeros(len(self.A))
        for a in self.A:
            for sd in self.S:
                policy[a] += self.P[sd][state][a] * (self.R[sd][state][a] + self.discount * self.V[self.k][sd])
        
        return np.argmax(policy)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Values Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================

    def action_noise(self, action):
        pass