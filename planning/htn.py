from __future__ import annotations

from collections import deque

from planning.pddl import (
    Action,
    Problem,
    apply_action,
    is_applicable,
    get_all_groundings,
)

from planning.utils import Queue


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HLA) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


def is_primitive(action: Action | HLA) -> bool:
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:

    if not hlas:
        return []

    frontier = Queue()
    frontier.push([hlas[0]])

    visited = set()

    while not frontier.isEmpty():
        plan = frontier.pop()
        key = tuple(str(step) for step in plan)

        if key in visited:
            continue

        visited.add(key)

        if is_plan_primitive(plan):
            state = problem.initial_state
            valid = True

            for action in plan:
                if not is_applicable(state, action):
                    valid = False
                    break

                state = apply_action(state, action)

            if valid and problem.isGoalState(state):
                return plan

            continue

        index = None
        for i, step in enumerate(plan):
            if not is_primitive(step):
                index = i
                break

        if index is None:
            continue

        hla: HLA = plan[index]
        for refinement in hla.refinements:
            new_plan = plan[:index] + refinement + plan[index + 1:]
            frontier.push(new_plan)

    return []


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def _get_entity_location(state: frozenset, entity: object) -> tuple | None:
    for fluent in state:
        if fluent[0] == "At" and fluent[1] == entity:
            return fluent[2]
    return None


def _build_move_graph(move_actions: list[Action], robot_name: str, state: frozenset) -> dict[tuple, list[Action]]:
    graph: dict[tuple, list[Action]] = {}
    for move in move_actions:
        source = next(
            f[2] for f in move.precond_pos if f[0] == "At" and f[1] == robot_name
        )
        target = next(
            f[2] for f in move.add_list if f[0] == "At" and f[1] == robot_name
        )
        if ("Adjacent", source, target) not in state:
            continue
        graph.setdefault(source, []).append(move)
    return graph


def _find_move_sequence(
    start: tuple,
    end: tuple,
    move_graph: dict[tuple, list[Action]],
    robot_name: str,
) -> list[Action]:
    if start == end:
        return []

    frontier = deque([(start, [])])
    visited = {start}

    while frontier:
        current, path = frontier.popleft()
        for move in move_graph.get(current, []):
            next_cell = next(
                fluent[2]
                for fluent in move.add_list
                if fluent[0] == "At" and fluent[1] == robot_name
            )
            if next_cell in visited:
                continue
            visited.add(next_cell)
            new_path = path + [move]
            if next_cell == end:
                return new_path
            frontier.append((next_cell, new_path))

    return []


def _find_action(actions: list[Action], name: str, condition) -> Action | None:
    for action in actions:
        if action.name.startswith(f"{name}(") and condition(action):
            return action
    return None


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    all_actions = get_all_groundings(problem.domain, problem.objects)

    move_actions = [a for a in all_actions if a.name.startswith("Move(")]
    pickup_actions = [a for a in all_actions if a.name.startswith("PickUp(")]
    setup_actions = [a for a in all_actions if a.name.startswith("SetupSupplies(")]
    putdown_actions = [a for a in all_actions if a.name.startswith("PutDown(")]
    rescue_actions = [a for a in all_actions if a.name.startswith("Rescue(")]

    robot_name = problem.objects["robots"][0] if problem.objects["robots"] else "robot"
    robot_start = _get_entity_location(problem.initial_state, robot_name)
    if robot_start is None:
        return []

    supply_locations = {
        supply: _get_entity_location(problem.initial_state, supply)
        for supply in problem.objects["supplies"]
    }
    patient_locations = {
        patient: _get_entity_location(problem.initial_state, patient)
        for patient in problem.objects["patients"]
    }

    move_graph = _build_move_graph(move_actions, robot_name, problem.initial_state)

    def make_navigate_hla(src: tuple, dst: tuple) -> HLA | None:
        path = _find_move_sequence(src, dst, move_graph, robot_name)
        if not path:
            return None
        return HLA(f"Navigate({src}->{dst})", refinements=[path])

    possible_roots: list[HLA] = []
    for supply, supply_loc in supply_locations.items():
        if supply_loc is None:
            continue
        for med_loc in problem.objects["medical_posts"]:
            nav_to_supply = make_navigate_hla(robot_start, supply_loc)
            nav_supply_to_med = make_navigate_hla(supply_loc, med_loc)
            if nav_to_supply is None or nav_supply_to_med is None:
                continue

            pickup_supply = _find_action(
                pickup_actions,
                "PickUp",
                lambda a: ("At", supply, supply_loc) in a.precond_pos
                and ("At", robot_name, supply_loc) in a.precond_pos,
            )
            setup_supply = _find_action(
                setup_actions,
                "SetupSupplies",
                lambda a: ("Holding", robot_name, supply) in a.precond_pos
                and ("At", robot_name, med_loc) in a.precond_pos
                and ("MedicalPost", med_loc) in a.precond_pos,
            )
            if pickup_supply is None or setup_supply is None:
                continue

            prepare_hla = HLA(
                f"PrepareSupplies({supply},{med_loc})",
                refinements=[[nav_to_supply, pickup_supply, nav_supply_to_med, setup_supply]],
            )

            extract_hlas: list[HLA] = []
            all_patients = True
            for patient, patient_loc in patient_locations.items():
                if patient_loc is None:
                    all_patients = False
                    break

                nav_to_patient = make_navigate_hla(med_loc, patient_loc)
                nav_patient_to_med = make_navigate_hla(patient_loc, med_loc)
                if nav_to_patient is None or nav_patient_to_med is None:
                    all_patients = False
                    break

                pickup_patient = _find_action(
                    pickup_actions,
                    "PickUp",
                    lambda a: ("At", patient, patient_loc) in a.precond_pos
                    and ("At", robot_name, patient_loc) in a.precond_pos,
                )
                putdown_patient = _find_action(
                    putdown_actions,
                    "PutDown",
                    lambda a: ("Holding", robot_name, patient) in a.precond_pos
                    and ("At", robot_name, med_loc) in a.precond_pos,
                )
                rescue_patient = _find_action(
                    rescue_actions,
                    "Rescue",
                    lambda a: ("At", patient, med_loc) in a.precond_pos
                    and ("At", robot_name, med_loc) in a.precond_pos
                    and ("MedicalPost", med_loc) in a.precond_pos,
                )
                if pickup_patient is None or putdown_patient is None or rescue_patient is None:
                    all_patients = False
                    break

                extract_hlas.append(
                    HLA(
                        f"ExtractPatient({patient},{med_loc})",
                        refinements=[[nav_to_patient, pickup_patient, nav_patient_to_med, putdown_patient, rescue_patient]],
                    )
                )

            if not all_patients:
                continue

            root_refinement = [prepare_hla] + extract_hlas
            possible_roots.append(HLA("AllRescues", refinements=[root_refinement]))

    if not possible_roots:
        return []

    return [possible_roots[0]]
