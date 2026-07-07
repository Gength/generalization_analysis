#!/bin/bash
cd "$HOME/m8_attempt"
export DISPLAY=:0
export LD_LIBRARY_PATH="$HOME/m8_attempt/lp:$LD_LIBRARY_PATH"
CP="out/production/AutomataConformance:Libraries/*"
LP="$HOME/m8_attempt/lp"
ROOT="$HOME/m45_run"
GEN="$HOME/genbench"
MINERS=(Alpha "Alpha+" Heuristics Heuristics_Strict Inductive_Infrequent Inductive_Strict Trace_Filtered Flower)
declare -A HEAP=( [D2]=8g [D3]=20g [D4]=36g [D5]=20g )
declare -A CONC=( [D2]=8 [D3]=3 [D4]=2 [D5]=3 )
declare -A GZ=( [D2]="data/BPI-Challenge_2013/Incident_Management_Log.xes.gz" [D3]="data/BPI-Challenge_2017/BPI Challenge 2017.xes.gz" [D4]="data/BPI-Challenge_2018/BPI Challenge 2018.xes.gz" [D5]="data/BPI-Challenge_2019/BPI_Challenge_2019.xes.gz" )

run_cell() {
  local method="$1" heap="$2" d="$3"; local START=$(date +%s)
  if [ "$method" = "M4" ]; then
    timeout 3660 java -Xmx${heap} -Djava.library.path="$LP" -cp "$CP" au.unimelb.evaluation.M4Progress "$d/" log.xes model.pnml > "$d/out.txt" 2>&1
  else
    timeout 3660 java -Xmx${heap} -Djava.library.path="$LP" -cp "$CP" au.unimelb.patternBasedGeneralization.PatternGeneralizationCommandLineTool "$d/" log.xes model.pnml patternbasedgeneralization global PartialMatching 0.02 60 MINUTES > "$d/out.txt" 2>&1
  fi
  echo "___EXIT=$? WALL=$(( $(date +%s) - START ))s" >> "$d/out.txt"
}

for ds in D2 D3 D4 D5; do
  base="$ROOT/$ds"; mkdir -p "$base"
  echo "[$(date -u +%H:%M:%SZ)] staging $ds"
  gunzip -c "$GEN/${GZ[$ds]}" > "$base/log.xes"
  for method in M4 M8; do
    echo "[$(date -u +%H:%M:%SZ)] $ds $method (heap=${HEAP[$ds]} conc=${CONC[$ds]})"
    for miner in "${MINERS[@]}"; do
      while [ "$(jobs -rp | wc -l)" -ge "${CONC[$ds]}" ]; do sleep 3; done
      d="$base/$method/cell_$miner"; mkdir -p "$d"
      ln -sf "$base/log.xes" "$d/log.xes"
      cp "$GEN/benchmark/models/$ds/$miner.pnml" "$d/model.pnml"
      run_cell "$method" "${HEAP[$ds]}" "$d" &
    done
    wait
  done
  echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$base/_DONE_$ds"
done
echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/_ALL_DONE"
