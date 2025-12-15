import pm4py


from collections import defaultdict
from typing import List, Dict, Set, Tuple

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.semantics import ClassicSemantics
from typing import List, Optional, Tuple, Dict, Set

semantics = ClassicSemantics()

def calculate_fintess_alignements(event_log_xes, net, im, fm):
    alignments = pm4py.conformance_diagnostics_alignments(event_log_xes, net, im, fm)
    fitness_values = [alignment.get("fitness") for alignment in alignments]
    fitness = sum(fitness_values) / len(fitness_values)
    return fitness


def analyze_transitions_with_end_labels(
    net: PetriNet,
    final_marking: Marking
) -> Tuple[
    List[PetriNet.Transition],
    Dict[PetriNet.Transition, Set[str]]
]:
    visible_transitions: List[PetriNet.Transition] = []
    silent_transition_map: Dict[PetriNet.Transition, Set[str]] = defaultdict(set)

    for t in net.transitions:
        if t.label is not None:
            visible_transitions.append(t)

    for silent_t in net.transitions:
        if silent_t.label is not None:
            continue  # skip visible transitions

        visited = set()
        to_visit = set()

        for arc in net.arcs:
            if arc.source == silent_t:
                to_visit.add(arc.target)

        while to_visit:
            node = to_visit.pop()
            if node in visited:
                continue
            visited.add(node)

            if isinstance(node, PetriNet.Place):
                if node in final_marking and final_marking[node] > 0:
                    silent_transition_map[silent_t].add("END")

                for arc in net.arcs:
                    if arc.source == node and isinstance(arc.target, PetriNet.Transition):
                        target_t = arc.target
                        if target_t.label is not None:
                            silent_transition_map[silent_t].add(target_t.label)
                        else:
                            for arc2 in net.arcs:
                                if arc2.source == target_t:
                                    to_visit.add(arc2.target)
                                    
    return visible_transitions, silent_transition_map


def visualize_petri_net(net, im, fm):
    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.view(gviz)


def execute_transition(
    t: Optional[PetriNet.Transition],
    net: PetriNet,
    marking: Marking
) -> Marking:
    """
    Executes the given transition (if not None) and returns the new marking.

    Raises an error if the transition is not enabled.

    Parameters
    ----------
    t : PetriNet.Transition or None
        The transition to execute. If None, returns the marking unchanged.
    net : PetriNet
        The Petri net
    marking : Marking
        The current marking

    Returns
    -------
    Marking
        The updated marking
    """
    if t is None:
        return marking

    enabled = semantics.enabled_transitions(net, marking)
    if t not in enabled:
        raise ValueError(f"The given transition {t.name} is not enabled in the current marking.")

    return semantics.execute(t, net, marking)


def get_enabled_transitions(
    net: PetriNet,
    marking: Marking
) -> List[PetriNet.Transition]:
    """
    Returns the list of currently enabled transitions.

    Parameters
    ----------
    net : PetriNet
        The Petri net
    marking : Marking
        The current marking

    Returns
    -------
    List[PetriNet.Transition]
        Enabled transitions at the given marking
    """
    return list(semantics.enabled_transitions(net, marking))


def split_enabled_transitions(
    enabled: List[PetriNet.Transition],
    visible_transitions: List[PetriNet.Transition],
    silent_transition_map: Dict[PetriNet.Transition, Set[str]]
) -> Tuple[List[PetriNet.Transition], List[PetriNet.Transition]]:
    """
    Splits the currently enabled transitions into visible and silent, based on provided transition information.

    Parameters
    ----------
    enabled : List[PetriNet.Transition]
        Currently enabled transitions (from marking)
    visible_transitions : List[PetriNet.Transition]
        All known visible transitions
    silent_transition_map : Dict[PetriNet.Transition, Set[str]]
        Mapping of known silent transitions to visible labels

    Returns
    -------
    Tuple[List[PetriNet.Transition], List[PetriNet.Transition]]
        (enabled_visible_transitions, enabled_silent_transitions)
    """
    enabled_visible = []
    enabled_silent = []

    for t in enabled:
        if t.label is not None and t in visible_transitions:
            enabled_visible.append(t)
        elif t.label is None and t in silent_transition_map:
            enabled_silent.append(t)

    return enabled_visible, enabled_silent


def get_enabled_tokens_and_transitions(
        net: PetriNet,
        current_marking: Marking,
        visible_transitions: List[PetriNet.Transition],
        silent_transition_map: Dict[PetriNet.Transition, Set[str]],
        concept_index: Dict[str, int]
    ) -> Tuple[
        List[int],                    # valid_tokens
        List[PetriNet.Transition]     # currently_enabled_transitions
    ]:
        """
        Returns the valid concept indexes and currently enabled transitions.

        Parameters
        ----------
        net : PetriNet
        current_marking : Marking
        visible_transitions : list of visible transitions
        silent_transition_map : dict mapping silent transitions to visible labels
        concept_index : dict mapping labels to concept indexes

        Returns
        -------
        Tuple[List[int], List[PetriNet.Transition]]
            valid_tokens and currently_enabled_transitions
        """
        enabled_transitions = get_enabled_transitions(net, current_marking)

        enabled_visible, enabled_silent = split_enabled_transitions(
            enabled_transitions,
            visible_transitions,
            silent_transition_map
        )

        enabled_mapped_visible = [
            concept_index[t.label]
            for t in enabled_visible
            if t.label in concept_index
        ]
        
        enabled_silent_reachable_labels = [
            [concept_index[label] for label in silent_transition_map.get(silent_t, set()) if label in concept_index]
            for silent_t in enabled_silent
        ]

        valid_tokens = enabled_mapped_visible + enabled_silent_reachable_labels
        currently_enabled_transitions = enabled_visible + enabled_silent

        return valid_tokens, currently_enabled_transitions


def get_filtered_probabilities(valid_tokens, prediction_output):
    filtered_probabilities = []
    for token in valid_tokens:
        if isinstance(token, list):
            prob_sum = sum(prediction_output[t] for t in token)
            filtered_probabilities.append(prob_sum)
        else:
            filtered_probabilities.append(prediction_output[token])
    return filtered_probabilities