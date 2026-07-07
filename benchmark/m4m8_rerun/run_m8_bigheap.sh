#!/bin/bash
cd "$HOME/m8_attempt"
export DISPLAY=:0
export LD_LIBRARY_PATH="$HOME/m8_attempt/lp:$LD_LIBRARY_PATH"
CP="out/production/AutomataConformance:Libraries/*"
BASE="$HOME/m8_run"; mkdir -p "$BASE/results_big"
for miner in "Alpha+" Heuristics Inductive_Infrequent; do
  d="$BASE/cell_$miner"
  (
    START=$(date +%s)
    timeout 3660 java -Xmx16g -Djava.library.path="$HOME/m8_attempt/lp" -cp "$CP" \
      au.unimelb.patternBasedGeneralization.PatternGeneralizationCommandLineTool \
      "$d/" sepsis.xes model.pnml patternbasedgeneralization global PartialMatching 0.02 60 MINUTES \
      > "$BASE/results_big/$miner.out" 2>&1
    echo "___EXIT=$? WALL=$(( $(date +%s) - START ))s" >> "$BASE/results_big/$miner.out"
  ) &
done
wait
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/results_big/_DONE"
