import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
def filtered_trace_miner(log, top_k=50):
    """Trace Model over the top-K variants (0.0 memorization pole)."""
    from collections import defaultdict
    net = PetriNet("Filtered Trace Model")
    p_start, p_end = PetriNet.Place("start"), PetriNet.Place("end")
    net.places.update([p_start, p_end])
    variant_counts = defaultdict(int)
    for t in log:
        variant_counts[tuple(e["concept:name"] for e in t)] += 1
    top_variants = [v for v, c in sorted(variant_counts.items(),
                                         key=lambda i: i[1], reverse=True)[:top_k]]
    for i, variant in enumerate(top_variants):
        prev = p_start
        for j, act in enumerate(variant):
            t = PetriNet.Transition(f"t_{i}_{j}", act)
            net.transitions.add(t)
            petri_utils.add_arc_from_to(prev, t, net)
            if j == len(variant) - 1:
                petri_utils.add_arc_from_to(t, p_end, net)
            else:
                p_next = PetriNet.Place(f"p_{i}_{j}")
                net.places.add(p_next)
                petri_utils.add_arc_from_to(t, p_next, net)
                prev = p_next
    im, fm = Marking(), Marking()
    im[p_start] = 1; fm[p_end] = 1
    return net, im, fm

def flower_miner(log):
    net = PetriNet("Flower Model")
    p_mid = PetriNet.Place("mid")
    net.places.add(p_mid)
    activities = set(e["concept:name"] for t in log for e in t)
    for act in activities:
        t = PetriNet.Transition(f"t_{act}", act)
        net.transitions.add(t)
        petri_utils.add_arc_from_to(p_mid, t, net)
        petri_utils.add_arc_from_to(t, p_mid, net)
    im, fm = Marking(), Marking()
    im[p_mid] = 1; fm[p_mid] = 1
    return net, im, fm
MINERS = {
    "Trace_Filtered":       filtered_trace_miner,
    "Alpha":                lambda l: pm4py.discover_petri_net_alpha(l),
    "Alpha+":               lambda l: pm4py.discover_petri_net_alpha_plus(l),
    "Heuristics":           lambda l: pm4py.discover_petri_net_heuristics(l),
    "Heuristics_Strict":    lambda l: pm4py.discover_petri_net_heuristics(l, dependency_threshold=0.99),
    "Inductive_Strict":     lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.0),
    "Inductive_Infrequent": lambda l: pm4py.discover_petri_net_inductive(l, noise_threshold=0.2),
    "Flower":               flower_miner,
}