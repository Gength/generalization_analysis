"""
Export a PM4Py Petri net to ProcessM-compatible JSON.
"""
import json

def export_pnml_to_processm_json(net, im, fm, path):
    """Convert PM4Py Petri net (net, im, fm) to ProcessM bridge JSON."""
    all_places = sorted(net.places, key=lambda p: str(p.name or ""))
    place_index = {p: i for i, p in enumerate(all_places)}
    
    transitions = []
    for t in net.transitions:
        tname = t.name or "__silent__"
        is_silent = tname.startswith("_") or tname.startswith("tau") or tname == "__silent__"
        
        in_place_indices = sorted(set(
            place_index[a.source] for a in net.arcs if a.target == t and a.source in place_index
        ))
        out_place_indices = sorted(set(
            place_index[a.target] for a in net.arcs if a.source == t and a.target in place_index
        ))
        
        transitions.append({
            "name": tname,
            "isSilent": is_silent,
            "inPlaces": in_place_indices,
            "outPlaces": out_place_indices,
        })
    
    im_indices = [place_index[p] for p in im if p in place_index]
    fm_indices = [place_index[p] for p in fm if p in place_index]
    
    result = {
        "numPlaces": len(all_places),
        "initialMarkingIndices": im_indices,
        "finalMarkingIndices": fm_indices,
        "transitions": transitions,
    }
    
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    
    return result
