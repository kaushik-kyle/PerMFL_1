#!/bin/zsh
# Prepared runs. NOTHING executes unless you pass a batch name.
#   tools/run.sh list          show batches
#   tools/run.sh B1            run batch B1
#   tools/run.sh B1 --dry      print the commands only
#
# Logging: every run writes logs/<batch>/<exp>.log capturing stdout+stderr,
# plus logs/<batch>/manifest.tsv recording the full command, git SHA, all
# environment variables, start/end time and exit code for every run.
set -u
cd "${0:A:h}/.."
PY=.venv/bin/python
BATCH="${1:-}"; DRY="${2:-}"
LOGROOT="logs/${BATCH}"

EM_BASE="--algorithm PerMFL --dataset Emnist10 --model_name mclr --lamda 0.5 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_team_iters 10 --local_iters 20 --num_teams 4 --p_teams 4 --num_labels 2 --group_division 0"
CI_BASE="--algorithm PerMFL --dataset Cicids --model_name dnn --lamda 0.05 --gamma 1.5 --beta 0.6 --alpha 0.01 --eta 0.03 --num_global_iters 100 --num_team_iters 10 --local_iters 20 --tot_users 20 --num_teams 5 --numusers 4 --p_teams 5 --num_labels 8 --group_division 3"

emit() {  # $1 exp  $2 log-label  $3... flags
  local exp="$1"; shift; local label="$1"; shift
  local cmd="OMP_NUM_THREADS=1 $PY main.py $* --exp_start $exp --times $((exp+1))"
  if [[ "$DRY" == "--dry" ]]; then print -r -- "$cmd"; return; fi
  mkdir -p "$LOGROOT"
  local log="$LOGROOT/${exp}_${label}.log"
  {
    print -r -- "# command : $cmd"
    print -r -- "# batch   : $BATCH"
    print -r -- "# label   : $label"
    print -r -- "# git     : $(git rev-parse HEAD 2>/dev/null)  $(git status --porcelain | wc -l | tr -d ' ') dirty"
    print -r -- "# host    : $(hostname)  $(sw_vers -productVersion 2>/dev/null)"
    print -r -- "# python  : $($PY -V 2>&1)"
    print -r -- "# env     : $(env | grep -E '^(CICIDS|NSLKDD|TONIOT|PERMFL|OMP|HIDDEN)' | sort | tr '\n' ' ')"
    print -r -- "# started : $(date -u +%FT%TZ)"
    print -r -- "# ---"
  } > "$log"
  local t0=$(date +%s)
  eval "$cmd" >> "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  { print -r -- "# ---"
    print -r -- "# ended   : $(date -u +%FT%TZ)"
    print -r -- "# elapsed : $((t1-t0))s"
    print -r -- "# exit    : $rc"; } >> "$log"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp" "$label" "$rc" "$((t1-t0))" "$(date -u +%FT%TZ)" "$cmd" >> "$LOGROOT/manifest.tsv"
  print "  exp$exp $label exit=$rc $((t1-t0))s"
}

case "$BATCH" in
list)
  cat <<'EOF'
B1   confusion matrices, CICIDS headline, both arms x5 seeds     10 runs  ~40 min
ALL  B2 B2L B3 B7 B1 in sequence
B2   EMNIST-10 paper config, 40 devices T=100, both arms x3 seeds   6 runs  ~27 min (measured 272s/run)
B2L  same at T=400                                                  6 runs  ~110 min (4x B2)
B3  local-steps sweep at 3 seeds                              12 runs  ~30 min
B7  lamda_team sweep on CICIDS, 6 values x3 seeds             18 runs  ~2 h
EOF
  ;;
B2)  # paper's stated setup: 40 devices, four teams of ten, full participation, T=100
     # matches the existing 40-device baseline in results/_archive_T100
  for s in 0 1 2; do
    emit $((2200+s*2)) "permfl_40dev_seed$s"    ${=EM_BASE} --num_global_iters 100 --tot_users 40 --numusers 10 --seed $s
    emit $((2201+s*2)) "finetuned_40dev_seed$s" ${=EM_BASE} --num_global_iters 100 --tot_users 40 --numusers 10 --lamda_team 1.5 --seed $s
  done ;;
B2L) # same at the paper's T=400 horizon
  for s in 0 1 2; do
    emit $((2210+s*2)) "permfl_40dev_T400_seed$s"    ${=EM_BASE} --num_global_iters 400 --tot_users 40 --numusers 10 --seed $s
    emit $((2211+s*2)) "finetuned_40dev_T400_seed$s" ${=EM_BASE} --num_global_iters 400 --tot_users 40 --numusers 10 --lamda_team 1.5 --seed $s
  done ;;
B1)  # headline CICIDS re-run, now persisting confusion matrices
  i=2700
  for s in 0 1 2 3 4; do
    emit $i     "permfl_cm_seed$s"    ${=CI_BASE} --seed $s
    emit $((i+1)) "finetuned_cm_seed$s" ${=CI_BASE} --lamda_team 1.5 --seed $s
    i=$((i+2))
  done ;;
B3)  # local-steps sweep, replicated
  for L in 2 5 10 20; do for s in 0 1 2; do
    emit $((2300+L*10+s)) "L${L}_seed$s" ${=EM_BASE} --num_global_iters 100 --tot_users 20 --numusers 5 --local_iters $L --seed $s
  done; done ;;
B7)  # lamda_team sweep at one heterogeneity setting
  i=2600
  for LT in 0.5 1.0 1.5 3.0 6.0 12.0; do for s in 0 1 2; do
    emit $i "lt${LT}_seed$s" ${=CI_BASE} --lamda_team $LT --seed $s; i=$((i+1))
  done; done ;;
*)
  print "usage: tools/run.sh {list|B1|B2|B2L|B3|B7|ALL} [--dry]"; exit 1 ;;
esac
