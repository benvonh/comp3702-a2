from re import A
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
        self.S = list()
        self.A = ROBOT_ACTIONS
        self.P = dict()
        self.R = dict()
        # Outputs
        self.V = None
        self.PI = None
        # Parameters
        self.k = 0 # time step iteration
        self.epsilon = environment.epsilon # convergence threshold
        self.discount = environment.gamma # discount factor
        # The cheese
        self.solution_reward = 0.0
        self.change = False
        # BFS to populate set of states
        frontier = [self.environment.get_init_state()]
        while frontier:
            expanding_state = frontier.pop()
            for a in self.A:
                _, new_state = self.environment.apply_dynamics(expanding_state, a)
                if new_state not in self.S:
                    self.S.append(new_state)
                    frontier.append(new_state)
        # Probability and reward functions
        for s in self.S:
            self.P[s] = dict()
            self.R[s] = dict()
            for a in self.A:
                self.P[s][a], self.R[s][a] = self.get_pr(s, a)

    # === Values Iteration =============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Values Iteration.
        """
        self.k = 1
        self.V = [dict(), dict()]
        for s in self.S:
            self.V[self.k-1][s] = -1.0
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
        self.V.append(dict())

        for s in self.S:
            values = np.zeros(len(self.A))
            for a in self.A:
                for sd in self.P[s][a]:
                    for i in range(len(self.P[s][a][sd])):
                        values[a] += self.P[s][a][sd][i] * \
                            (self.R[s][a][sd][i] + self.discount * self.V[self.k-1][sd])

            self.V[self.k][s] = np.max(values)
        
    def vi_plan_offline(self):
        """
        Plan using Values Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
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
        action = np.zeros(len(self.A))
        for a in self.A:
            for sd in self.P[state][a]:
                for i in range(len(self.P[state][a][sd])):
                    action[a] += self.P[state][a][sd][i] * \
                        (self.R[state][a][sd][i] + self.discount * self.V[self.k][sd])
    
        return np.argmax(action)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.change = True
        self.PI = dict()
        self.V = dict()
        for s in self.S:
            self.V[s] = 10.0
            self.PI[s] = FORWARD

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return not self.change

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        self.change = False
        # policy evaluation
        v = self.V.copy()
        self.V = { s: 0.0 for s in self.S }
        for s in self.S:
            for sd in self.P[s][self.PI[s]]:
                for i in range(len(self.P[s][self.PI[s]][sd])):
                    self.V[s] += self.P[s][self.PI[s]][sd][i] * \
                        (self.R[s][self.PI[s]][sd][i] + self.discount * v[sd])

        # policy improvement
        for s in self.S:
            QBest = self.V[s]
            for a in self.A:
                if a == self.PI[s]:
                    continue
                Qsa = 0.0
                for sd in self.P[s][a]:
                    for i in range(len(self.P[s][a][sd])):
                        Qsa += self.P[s][a][sd][i] * \
                            (self.R[s][a][sd][i] + self.discount * self.V[sd])
                if Qsa > QBest:
                    self.PI[s] = a
                    QBest = Qsa
                    self.change = True

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
        return self.PI[state]

    # === Helper Methods ===============================================================================================

    def get_pr(self, state: State, action: int) -> tuple[dict[list], dict[list]]:
        # probability -> [P1, ...] , reward -> [R1, ...]
        p = dict()
        r = dict()

        # nominal
        cost, new_state = self.environment.apply_dynamics(state, action)
        p[new_state] = [self.prob(action)[0]]
        if self.environment.is_solved(new_state):
            r[new_state] = [self.solution_reward]
        else:
            r[new_state] = [cost]
        
        # cw, nominal
        cost, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost = np.max([cost, cost2])
        if new_state in p:
            p[new_state].append(self.prob(action)[1])
            if self.environment.is_solved(new_state):
                r[new_state].append(self.solution_reward)
            else:
                r[new_state].append(cost)
        else:
            p[new_state] = [self.prob(action)[1]]
            if self.environment.is_solved(new_state):
                r[new_state] = [self.solution_reward]
            else:
                r[new_state] = [cost]

        # ccw, nominal
        cost, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost = np.max([cost, cost2])
        if new_state in p:
            p[new_state].append(self.prob(action)[2])
            if self.environment.is_solved(new_state):
                r[new_state].append(self.solution_reward)
            else:
                r[new_state].append(cost)
        else:
            p[new_state] = [self.prob(action)[2]]
            if self.environment.is_solved(new_state):
                r[new_state] = [self.solution_reward]
            else:
                r[new_state] = [cost]

        # nominal, double
        cost, new_state = self.environment.apply_dynamics(state, action)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost = np.max([cost, cost2])
        if new_state in p:
            p[new_state].append(self.prob(action)[3])
            if self.environment.is_solved(new_state):
                r[new_state].append(self.solution_reward)
            else:
                r[new_state].append(cost)
        else:
            p[new_state] = [self.prob(action)[3]]
            if self.environment.is_solved(new_state):
                r[new_state] = [self.solution_reward]
            else:
                r[new_state] = [cost]

        # cw, nominal, double
        cost, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost3, new_state = self.environment.apply_dynamics(new_state, action)
        cost = np.max([cost, cost2, cost3])
        if new_state in p:
            p[new_state].append(self.prob(action)[4])
            if self.environment.is_solved(new_state):
                r[new_state].append(self.solution_reward)
            else:
                r[new_state].append(cost)
        else:
            p[new_state] = [self.prob(action)[4]]
            if self.environment.is_solved(new_state):
                r[new_state] = [self.solution_reward]
            else:
                r[new_state] = [cost]

        # ccw, nominal, double
        cost, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost3, new_state = self.environment.apply_dynamics(new_state, action)
        cost = np.max([cost, cost2, cost3])
        if new_state in p:
            p[new_state].append(self.prob(action)[5])
            if self.environment.is_solved(new_state):
                r[new_state].append(self.solution_reward)
            else:
                r[new_state].append(cost)
        else:
            p[new_state] = [self.prob(action)[5]]
            if self.environment.is_solved(new_state):
                r[new_state] = [self.solution_reward]
            else:
                r[new_state] = [cost]

        return p, r

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
