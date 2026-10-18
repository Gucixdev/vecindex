#!/usr/bin/env bash
# run_all.sh DATA_DIR [OUT_DIR] - every measurement of the README's Results section, on this machine, in one go.
#
#   needs: gcc; python3 with numpy, faiss-cpu, turbovec, qdrant-edge-py (pip); DATA_DIR from prepare_glove.py;
#          ~30 GB free in /var/tmp for the 100M run ($VECINDEX_SPILL moves it). 2-3 hours on 2 cores.
#   the README's memory limit: run it as  systemd-run --user --scope -p MemoryMax=5.84G bench/run_all.sh ...
#
# Results go to OUT_DIR (default results-HOSTNAME): glove.jsonl (bench.py, every library), scale_10m.jsonl,
# scale_100m.jsonl, vbench_8m.log, cpu.txt. `python3 bench/tables.py OUT_DIR/glove.jsonl` prints the tables.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
DATA=$(realpath "${1:?usage: run_all.sh DATA_DIR [OUT_DIR]}")
OUT=$(realpath -m "${2:-results-$(hostname)}")
PY=${PYTHON:-python3}
mkdir -p "$OUT"
lscpu > "$OUT/cpu.txt" 2>/dev/null
gcc -std=gnu99 -O3 -march=native -fPIC -shared -I.. vi.c -o libvi.so -lm -lpthread || exit 1
gcc -std=gnu99 -O2 -march=native -I.. scale.c -lm -lpthread -o scale || exit 1
gcc -std=gnu99 -O2 -march=native -I.. ../vbench1m.c -lm -lpthread -o vbench1m 2>/dev/null || exit 1
log() { echo "=== $* $(date +%T)" | tee -a "$OUT/run.log"; }

# GloVe-100: every library through the same harness (turbovec on 1 thread too: its one-query-per-call column)
for job in "vecindex 2" "vecindex-i8 2" "vecindex-2bit 2" "vecindex-2bit-i8 2" "turbovec4 1" "turbovec4 2" \
           "turbovec2 1" "turbovec2 2" "faiss-ivfpqfs 2" "faiss-hnsw 2" "faiss-ivfflat 2" "qdrant 2"; do
    set -- $job
    log "$1 threads=$2"
    "$PY" bench.py "$DATA" "$1" --threads "$2" --lat 1000 >> "$OUT/glove.jsonl" 2>> "$OUT/run.log"
done

# scale: synthetic rows streamed through the store (time and memory at every stage)
log "scale 10M x 128"
TQ_VERBOSE=1 ./scale 10000000 128 --nq 100 --probe 100,200,400,800 > "$OUT/scale_10m.jsonl" 2>&1
log "scale 100M x 96, 2-bit codes + int8 rows"
TQ_VERBOSE=1 ./scale 100000000 96 --nq 100 --bits 2 --int8 --probe 1000,2000,4000,8000 --rerank 1000 > "$OUT/scale_100m.jsonl" 2>&1
log "vecdb 8M notes"
./vbench1m 8000000 > "$OUT/vbench_8m.log" 2>&1
log done
