#!/bin/zsh
cd "${0:A:h}/.."
# wait for the in-flight C-queue to drain before starting
while pgrep -f "main.py --algorithm PerMFL" >/dev/null; do sleep 20; done
print "=== C4 start $(date -u +%FT%TZ) ==="
PAR=3 ./tools/run.sh C4
print "=== C4 done  $(date -u +%FT%TZ) ==="
