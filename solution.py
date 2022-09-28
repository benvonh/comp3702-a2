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

    # === Values Iteration =============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Values Iteration.
        """
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

        self.P = dict()
        self.R = dict()
        for s in self.S:
            self.P[s] = dict()
            self.R[s] = dict()
            for a in self.A:
                pr = self.get_pr(s, a)
                self.P[s][a] = pr[0]
                self.R[s][a] = pr[1]

        # Initialises value list
        self.k = 0
        self.V = [dict()]
        for s in self.S:
            self.V[self.k][s] = 0.0

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
                    if sd in self.P[s][a]:
                        values[a] += self.P[s][a][sd] * (self.R[s][a][sd] + self.discount * self.V[self.k-1][sd])

            self.V[self.k][s] = np.max(values)

    def vi_plan_offline(self):
        """
        Plan using Values Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        print('asdfasdfasdf')
        self.vi_initialise()
        self.vi_iteration() # FIXME: SUS
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
                if sd in self.P[state][a]:
                    policy[a] += self.P[state][a][sd] * (self.R[state][a][sd] + self.discount * self.V[self.k][sd])
    
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

    def get_pr(self, state: State, action: int) -> list[dict, dict]:
        pr = [dict(), dict()]

        # nominal
        cost, new_state = self.environment.apply_dynamics(state, action)
        pr[0][new_state] = self.prob(action)[0]
        pr[1][new_state] = cost
        
        # cw, nominal
        _, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        if new_state in pr[0]:
            pr[0][new_state] += self.prob(action)[1]
            pr[1][new_state] += cost
        else:
            pr[0][new_state] = self.prob(action)[1]
            pr[1][new_state] = cost

        # ccw, nominal
        _, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        if new_state in pr[0]:
            pr[0][new_state] += self.prob(action)[2]
            pr[1][new_state] += cost
        else:
            pr[0][new_state] = self.prob(action)[2]
            pr[1][new_state] = cost

        # nominal, double
        cost, new_state = self.environment.apply_dynamics(state, action)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost += cost2
        if new_state in pr[0]:
            pr[0][new_state] += self.prob(action)[3]
            pr[1][new_state] += cost
        else:
            pr[0][new_state] = self.prob(action)[3]
            pr[1][new_state] = cost

        # cw, nominal, double
        _, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        cost2, new_state = self.environment.apply_dynamics(state, action)
        cost += cost2
        if new_state in pr[0]:
            pr[0][new_state] += self.prob(action)[4]
            pr[1][new_state] += cost
        else:
            pr[0][new_state] = self.prob(action)[4]
            pr[1][new_state] = cost

        # cw, nominal, double
        _, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        cost2, new_state = self.environment.apply_dynamics(state, action)
        cost += cost2
        if new_state in pr[0]:
            pr[0][new_state] += self.prob(action)[5]
            pr[1][new_state] += cost
        else:
            pr[0][new_state] = self.prob(action)[5]
            pr[1][new_state] = cost

        return pr

    def prob(self, a: int) -> list[float]:
        """
        0: nominal
        1: cw, nominal
        2: ccw, nominal
        3: nominal, double
        4: cw, nominal, double
        5: ccw, nominal, double
        """
        return [
            # nominal
            (1 - self.environment.drift_cw_probs[a] - self.environment.drift_ccw_probs[a]) * \
            (1 - self.environment.double_move_probs[a]),
            # cw, nominal
            self.environment.drift_cw_probs[a] * (1 - self.environment.double_move_probs[a]),
            # ccw, nominal
            self.environment.drift_ccw_probs[a] * (1 - self.environment.double_move_probs[a]),
            # nominal, double
            (1 - self.environment.drift_cw_probs[a] - self.environment.drift_ccw_probs[a]) * \
            self.environment.double_move_probs[a],
            # cw, nominal, double
            self.environment.drift_cw_probs[a] * self.environment.double_move_probs[a],
            # ccw, nominal, double
            self.environment.drift_ccw_probs[a] * self.environment.double_move_probs[a]
        ]