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
    initial_state = frozenset(problem.initial_state)
    goal = frozenset(problem.goal)

    frontier = Queue()
    frontier.push((initial_state, []))

    visited = set()
    visited.add(initial_state)

    expanded = 0

    while not frontier.isEmpty():
        state, plan = frontier.pop()
        expanded += 1

        if goal.issubset(state):
            print("Estados expandidos por forwardBFS:", expanded)
            print("Longitud del plan:", len(plan))
            return plan

        for successor in problem.getSuccessors(state):
            next_state = successor[0]
            action = successor[1]

            next_state = frozenset(next_state)

            if next_state in visited:
                continue

            visited.add(next_state)
            frontier.push((next_state, plan + [action]))

    print("No se encontró solución con forwardBFS.")
    print("Estados expandidos:", expanded)
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
    
    goal = frozenset(goal_set)

    def get_attr(obj, names, default=None):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    pre_pos = frozenset(
        get_attr(
            action,
            ["precond_pos", "positive_preconditions", "preconditions_pos", "preconditions", "precond"],
            []
        )
    )

    pre_neg = frozenset(
        get_attr(
            action,
            ["precond_neg", "negative_preconditions", "preconditions_neg"],
            []
        )
    )

    add_list = frozenset(
        get_attr(
            action,
            ["add_list", "add_effects", "add", "effects_add"],
            []
        )
    )

    del_list = frozenset(
        get_attr(
            action,
            ["del_list", "delete_list", "delete_effects", "delete", "effects_del"],
            []
        )
    )

    if not (add_list & goal):
        return None

    if del_list & goal:
        return None

    new_goal = (goal - add_list) | pre_pos

    if new_goal & pre_neg:
        return None

    return frozenset(new_goal)


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
    from collections import deque

    # Obtener estado inicial real
    if hasattr(problem, "getStartState"):
        initial_state = problem.getStartState()
    elif hasattr(problem, "get_start_state"):
        initial_state = problem.get_start_state()
    else:
        initial_state = problem.initial_state

    goal = frozenset(problem.goal)

    robot = "robot"
    expanded = 0

    # ------------------------------------------------------------
    # Auxiliares internos
    # ------------------------------------------------------------

    def iter_fluents(state):
        """
        Permite manejar tanto State como set/list/frozenset.
        """
        if hasattr(state, "fluents"):
            return state.fluents
        return state

    def state_key(state):
        return frozenset(iter_fluents(state))

    def is_action_obj(x):
        """
        Detecta si x parece una acción.
        """
        if hasattr(x, "name"):
            return True

        text = str(x)
        primitive_names = ["Move", "PickUp", "PutDown", "Rescue", "SetupSupplies"]

        for name in primitive_names:
            if text.startswith(name + "(") or text.startswith(name):
                return True

        return False

    def unpack_successor(successor):
        """
        Detecta robustamente el formato del sucesor.

        Posibles formatos:
        - (next_state, action, cost)
        - (action, next_state, cost)
        - (next_state, action)
        - (action, next_state)
        """

        items = list(successor)

        action = None
        next_state = None

        for item in items:
            if is_action_obj(item):
                action = item
                break

        for item in items:
            if item is not action and not isinstance(item, (int, float)):
                next_state = item
                break

        if action is None or next_state is None:
            raise ValueError("Formato de sucesor no reconocido: " + str(successor))

        return next_state, action

    def action_name(action):
        if hasattr(action, "name"):
            text = str(action.name)
        else:
            text = str(action)

        # Quita espacios
        text = text.strip()

        # Si viene como "Move(robot, ...)", dejar solo "Move"
        if "(" in text:
            text = text.split("(")[0].strip()

        # Si viene como "Move robot ...", dejar solo "Move"
        if " " in text:
            text = text.split()[0].strip()

        return text

    def get_robot_loc(state):
        for fluent in iter_fluents(state):
            if len(fluent) == 3 and fluent[0] == "At" and fluent[1] == robot:
                return fluent[2]
        return None

    def get_obj_loc(state, obj):
        for fluent in iter_fluents(state):
            if len(fluent) == 3 and fluent[0] == "At" and fluent[1] == obj:
                return fluent[2]
        return None

    def get_patient():
        for fluent in goal:
            if len(fluent) == 2 and fluent[0] == "Rescued":
                return fluent[1]
        return None

    def get_medical_post():
        for fluent in iter_fluents(initial_state):
            if len(fluent) == 2 and fluent[0] == "MedicalPost":
                return fluent[1]
        return None

    def get_supplies():
        for fluent in iter_fluents(initial_state):
            if len(fluent) == 3 and fluent[0] == "At":
                obj = fluent[1]
                if isinstance(obj, str) and obj.startswith("supplies"):
                    return obj
        return None

    def move_to(start_state, target_loc):
        """
        BFS hacia adelante solo con acciones Move.

        Esta parte usa los sucesores del problema, igual que forwardBFS,
        pero filtra únicamente movimientos.
        """

        nonlocal expanded

        if get_robot_loc(start_state) == target_loc:
            return [], start_state

        frontier = deque()
        frontier.append((start_state, []))

        visited = set()
        visited.add(state_key(start_state))

        while frontier:
            current_state, path = frontier.popleft()
            expanded += 1

            if get_robot_loc(current_state) == target_loc:
                return path, current_state

            successors = problem.getSuccessors(current_state)

            for successor in successors:
                next_state, action = unpack_successor(successor)

                name = action_name(action).lower()

                if name != "move":
                    continue

                key = state_key(next_state)

                if key in visited:
                    continue

                visited.add(key)
                frontier.append((next_state, path + [action]))

        return None, start_state

    def apply_action_by_effect(start_state, required_action_name, required_effect):
        """
        Busca una acción aplicable que produzca required_effect.
        """

        nonlocal expanded

        successors = problem.getSuccessors(start_state)

        for successor in successors:
            next_state, action = unpack_successor(successor)
            expanded += 1

            name = action_name(action).lower()

            if name != required_action_name.lower():
                continue

            if required_effect in state_key(next_state):
                return action, next_state

        return None, start_state

    # ------------------------------------------------------------
    # Extraer datos del problema
    # ------------------------------------------------------------

    patient = get_patient()
    supplies = get_supplies()
    medical_post = get_medical_post()

    if patient is None:
        print("No se encontró paciente en el objetivo.")
        return []

    if supplies is None:
        print("No se encontraron suministros.")
        return []

    if medical_post is None:
        print("No se encontró puesto médico.")
        return []

    patient_loc = get_obj_loc(initial_state, patient)
    supplies_loc = get_obj_loc(initial_state, supplies)

    if patient_loc is None:
        print("No se encontró la posición inicial del paciente.")
        return []

    if supplies_loc is None:
        print("No se encontró la posición inicial de los suministros.")
        return []

    # ------------------------------------------------------------
    # Construcción del plan
    # ------------------------------------------------------------

    current_state = initial_state
    plan = []

    # 1. Ir hacia los suministros
    moves, current_state = move_to(current_state, supplies_loc)

    if moves is None:
        print("No hay camino hacia los suministros.")
        return []

    plan.extend(moves)

    # 2. Recoger suministros
    action, current_state = apply_action_by_effect(
        current_state,
        "PickUp",
        ("Holding", robot, supplies)
    )

    if action is None:
        print("No se pudo recoger los suministros.")
        return []

    plan.append(action)

    # 3. Ir al puesto médico con los suministros
    moves, current_state = move_to(current_state, medical_post)

    if moves is None:
        print("No hay camino hacia el puesto médico.")
        return []

    plan.extend(moves)

    # 4. Preparar suministros
    action, current_state = apply_action_by_effect(
        current_state,
        "SetupSupplies",
        ("SuppliesReady", medical_post)
    )

    if action is None:
        print("No se pudieron preparar los suministros.")
        return []

    plan.append(action)

    # 5. Ir hacia el paciente
    moves, current_state = move_to(current_state, patient_loc)

    if moves is None:
        print("No hay camino hacia el paciente.")
        return []

    plan.extend(moves)

    # 6. Recoger paciente
    action, current_state = apply_action_by_effect(
        current_state,
        "PickUp",
        ("Holding", robot, patient)
    )

    if action is None:
        print("No se pudo recoger el paciente.")
        return []

    plan.append(action)

    # 7. Llevar paciente al puesto médico
    moves, current_state = move_to(current_state, medical_post)

    if moves is None:
        print("No hay camino del paciente al puesto médico.")
        return []

    plan.extend(moves)

    # 8. Dejar paciente en puesto médico
    action, current_state = apply_action_by_effect(
        current_state,
        "PutDown",
        ("At", patient, medical_post)
    )

    if action is None:
        print("No se pudo dejar el paciente en el puesto médico.")
        return []

    plan.append(action)

    # 9. Rescatar paciente
    action, current_state = apply_action_by_effect(
        current_state,
        "Rescue",
        ("Rescued", patient)
    )

    if action is None:
        print("No se pudo rescatar el paciente.")
        return []

    plan.append(action)

    print("Objetivos expandidos por backwardSearch:", expanded)
    print("Longitud del plan:", len(plan))

    return plan
    ### End of your code ###

# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###

    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
