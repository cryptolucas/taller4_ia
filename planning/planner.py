from __future__ import annotations

from collections.abc import Callable
from collections import deque
from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    initial_state = problem.initial_state

    if hasattr(problem, "isGoalState"):
        if problem.isGoalState(initial_state):
            return []
    else:
        goal = problem.goal if hasattr(problem, "goal") else problem.goal_state
        if goal.issubset(initial_state):
            return []

    frontier = [(initial_state, [])]
    visited = set()
    visited.add(initial_state)

    while frontier:
        state, plan = frontier.pop(0)

        for next_state, action, cost in problem.getSuccessors(state):
            if next_state not in visited:
                new_plan = plan + [action]

                if hasattr(problem, "isGoalState"):
                    if problem.isGoalState(next_state):
                        return new_plan
                else:
                    goal = problem.goal if hasattr(problem, "goal") else problem.goal_state
                    if goal.issubset(next_state):
                        return new_plan

                visited.add(next_state)
                frontier.append((next_state, new_plan))

    return []
    
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    goal_set = frozenset(goal_set)
    add_list = frozenset(action.add_list)
    del_list = frozenset(action.del_list)
    precond_pos = frozenset(action.precond_pos)

    # La acción debe aportar al menos un fluente del objetivo actual.
    if not (add_list & goal_set):
        return None

    # La acción no puede borrar algo que debe seguir siendo verdad.
    if del_list & goal_set:
        return None

    # REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
    return (goal_set - add_list) | precond_pos
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    from planning.pddl import get_all_groundings

    initial_state = problem.initial_state
    goal = problem.goal

    if goal.issubset(initial_state):
        return []

    static_preds = {"MedicalPost", "Adjacent", "Pickable"}

    medical_posts: frozenset = frozenset(
        f[1] for f in initial_state if f[0] == "MedicalPost"
    )
    obj_initial: dict[str, set] = {}
    for f in initial_state:
        if f[0] == "At" and f[1] != "robot":
            obj_initial.setdefault(f[1], set()).add(f[2])

    def valid_obj_loc(obj: str, loc) -> bool:
        return loc in obj_initial.get(obj, set()) or loc in medical_posts

    def simplify(goal_desc: frozenset) -> frozenset | None:
        result = set()
        robot_locs: set = set()
        held_objs: set = set()

        for f in goal_desc:
            if f[0] in static_preds:
                if f not in initial_state:
                    return None
            elif f[0] == "Free":
                pass
            else:
                if f[0] == "At" and f[1] != "robot":
                    if not valid_obj_loc(f[1], f[2]):
                        return None
                result.add(f)
                if f[0] == "At" and f[1] == "robot":
                    robot_locs.add(f[2])
                elif f[0] == "Holding" and f[1] == "robot":
                    held_objs.add(f[2])

        if len(robot_locs) > 1 or len(held_objs) > 1:
            return None

        return frozenset(result)

    all_actions = get_all_groundings(problem.domain, problem.objects)

    from collections import defaultdict
    add_index: dict = defaultdict(list)
    for a in all_actions:
        for f in a.add_list:
            add_index[f].append(a)

    start_goal = simplify(goal)
    if start_goal is None:
        return []
    if start_goal.issubset(initial_state):
        return []

    parent: dict = {start_goal: None}
    frontier = deque([start_goal])

    def reconstruct(g: frozenset, last_action: Action) -> list[Action]:
        plan = [last_action]
        node = g
        while parent[node] is not None:
            prev_goal, act = parent[node]
            plan.append(act)
            node = prev_goal
        return plan

    while frontier:
        current_goal = frontier.popleft()

        seen_actions: set = set()
        for f in current_goal:
            for action in add_index.get(f, []):
                if id(action) in seen_actions:
                    continue
                seen_actions.add(id(action))

                new_goal_raw = regress(current_goal, action)
                if new_goal_raw is None:
                    continue

                new_goal = simplify(new_goal_raw)
                if new_goal is None:
                    continue

                if new_goal in parent:
                    continue

                parent[new_goal] = (current_goal, action)

                if new_goal.issubset(initial_state):
                    return reconstruct(current_goal, action)

                frontier.append(new_goal)

    return []
    ### End of your code ###

# ---------------------------------------------------------------------------
# Punto 4 – A* Planner 
# ---------------------------------------------------------------------------

def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    initial_state = problem.initial_state
    
    if hasattr(problem, "isGoalState"):
        is_goal = problem.isGoalState
    else:
        goal_target = problem.goal if hasattr(problem, "goal") else problem.goal_state
        is_goal = lambda s: goal_target.issubset(s)
        
    if is_goal(initial_state):
        return []
        
    frontier = PriorityQueue()
    goal = problem.goal if hasattr(problem, "goal") else problem.goal_state
    
    h_initial = heuristic(initial_state, goal, problem.domain, problem.objects)
    frontier.push((initial_state, []), h_initial)
    
    g_cost = {initial_state: 0}
    
    while not frontier.isEmpty():
        state, plan = frontier.pop()
        
        if len(plan) > g_cost.get(state, float('inf')):
            continue
            
        if is_goal(state):
            return plan
            
        for next_state, action, cost in problem.getSuccessors(state):
            new_g_cost = g_cost[state] + cost
            
            if next_state not in g_cost or new_g_cost < g_cost[next_state]:
                g_cost[next_state] = new_g_cost
                h = heuristic(next_state, goal, problem.domain, problem.objects)
                f = new_g_cost + h
                frontier.push((next_state, plan + [action]), f)
                
    return []


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
forwardSearch = forwardBFS   # alias for CLI compatibility
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
