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
        self.P = None
        self.R = None
        # Outputs
        self.V = None
        self.PI = None
        # Parameters
        self.k = 0 # time step iteration
        self.epsilon = environment.epsilon # convergence threshold
        self.discount = environment.gamma # discount factor

    # === values Iteration =============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of values Iteration.
        """
        self.k = 0

        self.S = dict()
        for player_rot in ROBOT_ORIENTATIONS:
            for player_row in range(self.environment.n_rows):
                for player_col in range(self.environment.n_cols):
                    for widget_row in range(self.environment.n_rows):
                        for widget_col in range(self.environment.n_cols):
                            for widget_rot in WIDGET_ORIENTS[self.environment.widget_types[0]]:
                                try:
                                    self.S[
                                        State(
                                            self.environment,
                                            (player_row, player_col), player_rot,
                                            ((widget_row, widget_col),), (widget_rot,)
                                        )
                                     ] = 0
                                except Exception as e:
                                    #print(e)
                                    continue

        #self.values = [self.states]
        print(self.S)

    def vi_is_converged(self):
        """
        Check if values Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #vk   = np.array(self.S[self.k].values())
        #vk_1 = np.array(self.S[self.k-1].values())
        #return np.all((np.abs(vk - vk_1) < self.epsilon) == True)
        return True

    def vi_iteration(self):
        """
        Perform a single iteration of values Iteration (i.e. loop over the state space once).
        """
        self.k += 1

    def vi_plan_offline(self):
        """
        Plan using values Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        self.vi_iteration() # FIXME: SUS
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_values(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        #return self.values[self.k][state]
        pass

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by values Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

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
        Retrieve the optimal action for the given state (based on values computed by values Iteration).
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
    def get_one_widget_permutation(self):
        for widget_row in range(self.environment.n_rows):
            for widget_col in range(self.environment.n_cols):
                for widget_rot in WIDGET_ORIENTS[self.environment.widget_types]:
                    yield (widget_row, widget_col), widget_rot
