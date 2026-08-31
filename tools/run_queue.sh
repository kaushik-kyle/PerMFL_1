#!/bin/zsh
cd "${0:A:h}/.."
for B in C3 C1 C7; do
  print "=== $B start $(date -u +%FT%TZ) ==="
  PAR=3 ./tools/run.sh $B
  print "=== $B done  $(date -u +%FT%TZ) ==="
done
print "QUEUE DONE $(date -u +%FT%TZ)"
