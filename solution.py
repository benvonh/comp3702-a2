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
        # The cheese
        self.collision_penalty = 0
        self.hazard_penalty = 0

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

    def get_pr(self, state: State, action: int) -> list[dict[list], dict[list]]:
        # probability -> P, reward -> [R1, ...]
        pr = [dict(), dict()]

        # nominal
        cost, new_state = self.environment.apply_dynamics(state, action)
        pr[0][new_state] = [self.prob(action)[0]]
        if self.environment.is_solved(new_state):
            pr[1][new_state] = [0]
        else:
            pr[1][new_state] = [cost]
        
        # cw, nominal
        _, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        if new_state in pr[0]:
            pr[0][new_state].append(self.prob(action)[1])
            if self.environment.is_solved(new_state):
                pr[1][new_state].append(0)
            else:
                pr[1][new_state].append(0)
        else:
            pr[0][new_state] = [self.prob(action)[1]]
            if self.environment.is_solved(new_state):
                pr[1][new_state] = [0]
            else:
                pr[1][new_state] = [cost]

        # ccw, nominal
        _, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        if new_state in pr[0]:
            pr[0][new_state].append(self.prob(action)[2])
            if self.environment.is_solved(new_state):
                pr[1][new_state].append(0)
            else:
                pr[1][new_state].append(0)
        else:
            pr[0][new_state] = [self.prob(action)[2]]
            if self.environment.is_solved(new_state):
                pr[1][new_state] = [0]
            else:
                pr[1][new_state] = [cost]

        # nominal, double
        cost, new_state = self.environment.apply_dynamics(state, action)
        cost2, new_state2 = self.environment.apply_dynamics(new_state, action)
        cost += cost2 - self.nominal_movement_cost(new_state, action)
        if new_state2 in pr[0]:
            pr[0][new_state2].append(self.prob(action)[3])
            if self.environment.is_solved(new_state2):
                pr[1][new_state2].append(0)
            else:
                pr[1][new_state2].append(0)
        else:
            pr[0][new_state2] = [self.prob(action)[3]]
            if self.environment.is_solved(new_state2):
                pr[1][new_state2] = [0]
            else:
                pr[1][new_state2] = [cost]

        # cw, nominal, double
        _, new_state = self.environment.apply_dynamics(state, SPIN_RIGHT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        cost2, new_state = self.environment.apply_dynamics(new_state, action)
        cost += cost2 - self.nominal_movement_cost(new_state, action)
        if new_state2 in pr[0]:
            pr[0][new_state2].append(self.prob(action)[4])
            if self.environment.is_solved(new_state2):
                pr[1][new_state2].append(0)
            else:
                pr[1][new_state2].append(0)
        else:
            pr[0][new_state2] = [self.prob(action)[4]]
            if self.environment.is_solved(new_state2):
                pr[1][new_state2] = [0]
            else:
                pr[1][new_state2] = [cost]

        # cw, nominal, double
        _, new_state = self.environment.apply_dynamics(state, SPIN_LEFT)
        cost, new_state = self.environment.apply_dynamics(new_state, action)
        cost2, new_state2 = self.environment.apply_dynamics(new_state, action)
        cost += cost2 - self.nominal_movement_cost(new_state, action)
        if new_state2 in pr[0]:
            pr[0][new_state2].append(self.prob(action)[5])
            if self.environment.is_solved(new_state2):
                pr[1][new_state2].append(0)
            else:
                pr[1][new_state2].append(0)
        else:
            pr[0][new_state2] = [self.prob(action)[5]]
            if self.environment.is_solved(new_state2):
                pr[1][new_state2] = [0]
            else:
                pr[1][new_state2] = [cost]

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

    def nominal_movement_cost(self, state: State, movement: int) -> float:
        if movement == SPIN_LEFT or movement == SPIN_RIGHT:
            cost = ACTION_BASE_COST[movement]
            return -1 * cost
        else:
            forward_direction = state.robot_orient
            # get coordinates of position forward of the robot
            forward_robot_posit = get_adjacent_cell_coords(state.robot_posit, forward_direction)
            if movement == FORWARD:
                move_direction = state.robot_orient
                new_robot_posit = forward_robot_posit
            else:
                move_direction = {ROBOT_UP: ROBOT_DOWN,
                                  ROBOT_DOWN: ROBOT_UP,
                                  ROBOT_UP_LEFT: ROBOT_DOWN_RIGHT,
                                  ROBOT_UP_RIGHT: ROBOT_DOWN_LEFT,
                                  ROBOT_DOWN_LEFT: ROBOT_UP_RIGHT,
                                  ROBOT_DOWN_RIGHT: ROBOT_UP_LEFT}[state.robot_orient]
                new_robot_posit = get_adjacent_cell_coords(state.robot_posit, move_direction)

            # test for out of bounds
            nr, nc = new_robot_posit
            if (not 0 <= nr < self.environment.n_rows) or (not 0 <= nc < self.environment.n_cols):
                return -1 * self.collision_penalty

            # test for robot collision with obstacle
            if self.environment.obstacle_map[nr][nc]:
                return -1 * self.collision_penalty

            # test for robot collision with hazard
            if self.environment.hazard_map[nr][nc]:
                return -1 * self.hazard_penalty

            # check if the new position overlaps with a widget
            widget_cells = [widget_get_occupied_cells(self.environment.widget_types[i], state.widget_centres[i],
                                                      state.widget_orients[i]) for i in range(self.environment.n_widgets)]

            # check for reversing collision
            for i in range(self.environment.n_widgets):
                if movement == REVERSE and new_robot_posit in widget_cells[i]:
                    # this action causes a reversing collision with a widget
                    return -1 * self.collision_penalty

            # check if the new position moves a widget
            for i in range(self.environment.n_widgets):
                if forward_robot_posit in widget_cells[i]:
                    # this action pushes or pulls a widget
                    cost = ACTION_BASE_COST[movement] + ACTION_PUSH_COST[movement]

                    # get movement type - always use forward direction
                    widget_move_type = widget_get_movement_type(forward_direction, forward_robot_posit,
                                                                state.widget_centres[i])

                    # apply movement to the widget
                    if widget_move_type == TRANSLATE:
                        # translate widget in movement direction
                        new_centre = get_adjacent_cell_coords(state.widget_centres[i], move_direction)
                        new_cells = widget_get_occupied_cells(self.environment.widget_types[i], new_centre,
                                                              state.widget_orients[i])
                        # test collision for each cell of the widget
                        for (cr, cc) in new_cells:
                            # check collision with boundary
                            if (not 0 <= cr < self.environment.n_rows) or (not 0 <= cc < self.environment.n_cols):
                                # new widget position is invalid - collides with boundary
                                return -1 * self.collision_penalty

                            # check collision with obstacles
                            if self.environment.obstacle_map[cr][cc]:
                                # new widget position is invalid - collides with an obstacle
                                return -1 * self.collision_penalty

                            # check collision with hazards
                            if self.environment.hazard_map[cr][cc]:
                                # new widget position is invalid - collides with an obstacle
                                return -1 * self.hazard_penalty

                            # check collision with other widgets
                            for j in range(self.environment.n_widgets):
                                if j == i:
                                    continue
                                if (cr, cc) in widget_cells[j]:
                                    # new widget position is invalid - collides with another widget
                                    return -1 * self.collision_penalty

                        return -1 * cost

                    else:  # widget_move_type == SPIN_CW or widget_move_type == SPIN_CCW
                        # rotating a widget while reversing is not possible
                        if movement == REVERSE:
                            return -1 * self.collision_penalty

                        # rotate widget about its centre
                        if self.environment.widget_types[i] == WIDGET3:
                            if widget_move_type == SPIN_CW:
                                new_orient = {VERTICAL: SLANT_RIGHT,
                                              SLANT_RIGHT: SLANT_LEFT,
                                              SLANT_LEFT: VERTICAL}[state.widget_orients[i]]
                            else:
                                new_orient = {VERTICAL: SLANT_LEFT,
                                              SLANT_LEFT: SLANT_RIGHT,
                                              SLANT_RIGHT: VERTICAL}[state.widget_orients[i]]
                        elif self.environment.widget_types[i] == WIDGET4:
                            # CW and CCW are symmetric for this case
                            new_orient = {UP: DOWN, DOWN: UP}[state.widget_orients[i]]
                        else:  # self.environment.widget_types[i] == WIDGET5
                            if widget_move_type == SPIN_CW:
                                new_orient = {HORIZONTAL: SLANT_LEFT,
                                              SLANT_LEFT: SLANT_RIGHT,
                                              SLANT_RIGHT: HORIZONTAL}[state.widget_orients[i]]
                            else:
                                new_orient = {HORIZONTAL: SLANT_RIGHT,
                                              SLANT_RIGHT: SLANT_LEFT,
                                              SLANT_LEFT: HORIZONTAL}[state.widget_orients[i]]
                        new_cells = widget_get_occupied_cells(self.environment.widget_types[i], state.widget_centres[i], new_orient)

                        # check collision with the new robot position
                        if new_robot_posit in new_cells:
                            # new widget position is invalid - collides with the robot
                            return -1 * self.collision_penalty

                        # test collision for each cell of the widget
                        for (cr, cc) in new_cells:
                            # check collision with boundary
                            if (not 0 <= cr < self.environment.n_rows) or (not 0 <= cc < self.environment.n_cols):
                                # new widget position is invalid - collides with boundary
                                return -1 * self.collision_penalty

                            # check collision with obstacles
                            if self.environment.obstacle_map[cr][cc]:
                                # new widget position is invalid - collides with an obstacle
                                return -1 * self.collision_penalty

                            # check collision with hazard
                            if self.environment.hazard_map[cr][cc]:
                                # new widget position is invalid - collides with an obstacle
                                return -1 * self.hazard_penalty

                            # check collision with other widgets
                            for j in range(self.environment.n_widgets):
                                if j == i:
                                    continue
                                if (cr, cc) in widget_cells[j]:
                                    # new widget position is invalid - collides with another widget
                                    return -1 * self.collision_penalty

                        return -1 * cost

            # this action does not collide and does not push or pull any widgets
            cost = ACTION_BASE_COST[movement]
            return -1 * cost


def get_adjacent_cell_coords(posit, direction):
    """
    Return the coordinates of the cell adjacent to the given position in the given direction.
    orientation.
    :param posit: position
    :param direction: direction (element of ROBOT_ORIENTATIONS)
    :return: (row, col) of adjacent cell
    """
    r, c = posit
    if direction == ROBOT_UP:
        return r - 1, c
    elif direction == ROBOT_DOWN:
        return r + 1, c
    elif direction == ROBOT_UP_LEFT:
        if c % 2 == 0:
            return r - 1, c - 1
        else:
            return r, c - 1
    elif direction == ROBOT_UP_RIGHT:
        if c % 2 == 0:
            return r - 1, c + 1
        else:
            return r, c + 1
    elif direction == ROBOT_DOWN_LEFT:
        if c % 2 == 0:
            return r, c - 1
        else:
            return r + 1, c - 1
    else:   # direction == ROBOT_DOWN_RIGHT
        if c % 2 == 0:
            return r, c + 1
        else:
            return r + 1, c + 1


def widget_get_occupied_cells(w_type, centre, orient):
    """
    Return a list of cell coordinates which are occupied by this widget (useful for checking if the widget is in
    collision and how the widget should move if pushed or pulled by the robot).

    :param w_type: widget type
    :param centre: centre point of the widget
    :param orient: orientation of the widget
    :return: [(r, c) for each cell]
    """
    occupied = [centre]
    cr, cc = centre

    # cell in UP direction
    if ((w_type == WIDGET3 and orient == VERTICAL) or
            (w_type == WIDGET4 and orient == UP) or
            (w_type == WIDGET5 and (orient == SLANT_LEFT or orient == SLANT_RIGHT))):
        occupied.append((cr - 1, cc))

    # cell in DOWN direction
    if ((w_type == WIDGET3 and orient == VERTICAL) or
            (w_type == WIDGET4 and orient == DOWN) or
            (w_type == WIDGET5 and (orient == SLANT_LEFT or orient == SLANT_RIGHT))):
        occupied.append((cr + 1, cc))

    # cell in UP_LEFT direction
    if ((w_type == WIDGET3 and orient == SLANT_LEFT) or
            (w_type == WIDGET4 and orient == DOWN) or
            (w_type == WIDGET5 and (orient == SLANT_LEFT or orient == HORIZONTAL))):
        if cc % 2 == 0:
            # even column - row decreases
            occupied.append((cr - 1, cc - 1))
        else:
            # odd column - row stays the same
            occupied.append((cr, cc - 1))

    # cell in UP_RIGHT direction
    if ((w_type == WIDGET3 and orient == SLANT_RIGHT) or
            (w_type == WIDGET4 and orient == DOWN) or
            (w_type == WIDGET5 and (orient == SLANT_RIGHT or orient == HORIZONTAL))):
        if cc % 2 == 0:
            # even column - row decreases
            occupied.append((cr - 1, cc + 1))
        else:
            # odd column - row stays the same
            occupied.append((cr, cc + 1))

    # cell in DOWN_LEFT direction
    if ((w_type == WIDGET3 and orient == SLANT_RIGHT) or
            (w_type == WIDGET4 and orient == UP) or
            (w_type == WIDGET5 and (orient == SLANT_RIGHT or orient == HORIZONTAL))):
        if cc % 2 == 0:
            # even column - row stays the same
            occupied.append((cr, cc - 1))
        else:
            # odd column - row increases
            occupied.append((cr + 1, cc - 1))

    # cell in DOWN_RIGHT direction
    if ((w_type == WIDGET3 and orient == SLANT_LEFT) or
            (w_type == WIDGET4 and orient == UP) or
            (w_type == WIDGET5 and (orient == SLANT_LEFT or orient == HORIZONTAL))):
        if cc % 2 == 0:
            # even column - row stays the same
            occupied.append((cr, cc + 1))
        else:
            # odd column - row increases
            occupied.append((cr + 1, cc + 1))

    return occupied


def widget_get_movement_type(robot_orient, forward_robot_posit, centre):
    """
    Test if the given forward robot position and widget type, position and rotation results in a translation. Assumes
    that new_robot_posit overlaps with the given widget (implying that new_robot_posit overlaps or is adjacent to
    the widget centre).

    If the robot is reversing and this function returns a rotation movement type then the action is invalid.

    :param robot_orient: robot orientation
    :param forward_robot_posit: (row, col) new robot position
    :param centre: widget centre position
    :return: True if translation
    """
    # simple case --> new posit == centre is always translation
    if forward_robot_posit == centre:
        return TRANSLATE

    # if direction between new_robot_posit and centre is the same as robot_orient, then move is a translation
    nr, nc = forward_robot_posit
    cr, cc = centre

    # these directions do not depend on even/odd column
    if nr == cr - 1 and nc == cc:
        direction = ROBOT_DOWN
    elif nr == cr + 1 and nc == cc:
        direction = ROBOT_UP
    elif nr == cr - 1 and nc == cc - 1:
        direction = ROBOT_DOWN_RIGHT
    elif nr == cr - 1 and nc == cc + 1:
        direction = ROBOT_DOWN_LEFT
    elif nr == cr + 1 and nc == cc - 1:
        direction = ROBOT_UP_RIGHT
    elif nr == cr + 1 and nc == cc + 1:
        direction = ROBOT_UP_LEFT

    # these directions split based on even/odd
    elif nr == cr and nc == cc - 1:
        direction = ROBOT_UP_RIGHT if cc % 2 == 0 else ROBOT_DOWN_RIGHT
    else:  # nr == cr and nc == cc + 1
        direction = ROBOT_UP_LEFT if cc % 2 == 0 else ROBOT_DOWN_LEFT

    if direction == robot_orient:
        return TRANSLATE
    elif ((robot_orient == ROBOT_UP and (direction == ROBOT_DOWN_RIGHT or direction == ROBOT_UP_RIGHT)) or
          (robot_orient == ROBOT_DOWN and (direction == ROBOT_DOWN_LEFT or direction == ROBOT_UP_LEFT)) or
          (robot_orient == ROBOT_UP_LEFT and (direction == ROBOT_UP_RIGHT or direction == ROBOT_UP)) or
          (robot_orient == ROBOT_UP_RIGHT and (direction == ROBOT_DOWN or direction == ROBOT_DOWN_RIGHT)) or
          (robot_orient == ROBOT_DOWN_LEFT and (direction == ROBOT_UP or direction == ROBOT_UP_LEFT)) or
          (robot_orient == ROBOT_DOWN_RIGHT and (direction == ROBOT_DOWN_LEFT or direction == ROBOT_DOWN))):
        return SPIN_CW
    else:
        return SPIN_CCW