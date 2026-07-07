#!/bin/bash
cd "$HOME/m8_attempt"
export DISPLAY=:0
export LD_LIBRARY_PATH="$HOME/m8_attempt/lp:$LD_LIBRARY_PATH"
CP="out/production/AutomataConformance:Libraries/*"
BASE="$HOME/m8_run"; LOG="$BASE/sepsis.xes"
mkdir -p "$BASE/results"
for miner in Alpha "Alpha+" Heuristics Heuristics_Strict Inductive_Infrequent Inductive_Strict Trace_Filtered Flower; do
  d="$BASE/cell_$miner"; mkdir -p "$d"
  ln -sf "$LOG" "$d/sepsis.xes"
  cp "$HOME/genbench/benchmark/models/D1/$miner.pnml" "$d/model.pnml"
  (
    START=$(date +%s)
    timeout 3660 java -Xmx4g -Djava.library.path="$HOME/m8_attempt/lp" -cp "$CP" \
      au.unimelb.patternBasedGeneralization.PatternGeneralizationCommandLineTool \
      "$d/" sepsis.xes model.pnml patternbasedgeneralization global PartialMatching 0.02 60 MINUTES \
      > "$BASE/results/$miner.out" 2>&1
    echo "___EXIT=$? WALL=$(( $(date +%s) - START ))s" >> "$BASE/results/$miner.out"
  ) &
done
wait
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BASE/results/_DONE"
