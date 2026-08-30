#!/bin/zsh
# Runs every pending batch in sequence, after any in-flight B2 finishes.
cd "${0:A:h}/.."
while pgrep -f "main.py --algorithm PerMFL" >/dev/null; do sleep 20; done
print "B2 clear, starting chain at $(date -u +%FT%TZ)"
for B in B2 B2L B3 B7 B1; do
  print "=== $B start $(date -u +%FT%TZ) ==="
  ./tools/run.sh $B
  print "=== $B done  $(date -u +%FT%TZ) ==="
done
print "ALL BATCHES DONE $(date -u +%FT%TZ)"
