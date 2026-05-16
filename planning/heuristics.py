from __future__ import annotations

from planning.pddl import ActionSchema, State, Objects


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0

# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic 
# ---------------------------------------------------------------------------

def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    from planning.pddl import get_all_groundings
    
    unsatisfied = goal - state
    if not unsatisfied:
        return 0.0
        
    cache = objects.setdefault("_ign_precond_cache", {})
    if unsatisfied in cache:
        return cache[unsatisfied]
    
    if "_ign_precond_actions" not in objects:
        all_actions = get_all_groundings(domain, objects)
        objects["_ign_precond_actions"] = [a.add_list for a in all_actions if a.add_list & goal]
        
    useful_add_lists = objects["_ign_precond_actions"]
    
    actions_count = 0.0
    curr_unsatisfied = unsatisfied
    
    while curr_unsatisfied:
        best_coverage = 0
        best_add = None
        
        for add_list in useful_add_lists:
            coverage = len(add_list & curr_unsatisfied)
            if coverage > best_coverage:
                best_coverage = coverage
                best_add = add_list
                
        if best_coverage == 0:
            cache[unsatisfied] = float('inf')
            return float('inf')
            
        curr_unsatisfied = curr_unsatisfied - best_add
        actions_count += 1.0
        
    cache[unsatisfied] = actions_count
    return actions_count

# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic 
# ---------------------------------------------------------------------------

def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    from planning.pddl import get_all_groundings
    
    cache = objects.setdefault("_ign_del_cache", {})
    if state in cache:
        return cache[state]
    
    if "_all_actions_cache" not in objects:
        objects["_all_actions_cache"] = get_all_groundings(domain, objects)
        
    current_state = state
    steps = 0.0
    
    available_actions = objects["_all_actions_cache"]
    
    while not goal.issubset(current_state):
        best_action = None
        best_score = -1
        best_new_fluents = frozenset()
        
        unsatisfied = goal - current_state
        next_available = []
        
        for action in available_actions:
            new_fluents = action.add_list - current_state
            

            if not new_fluents:
                continue 
                
            next_available.append(action)
            
            if action.precond_pos.issubset(current_state) and action.precond_neg.isdisjoint(current_state):
                goal_added = len(new_fluents & unsatisfied)
                score = (goal_added * 10000) + len(new_fluents)
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_new_fluents = new_fluents
                    
        if best_action is None:
            cache[state] = float('inf')
            return float('inf')
            
        current_state = current_state | best_new_fluents
        steps += 1.0
        available_actions = next_available # Reducimos la lista para el siguiente ciclo
        
    cache[state] = steps
    return steps