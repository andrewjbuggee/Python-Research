#!/usr/bin/env bash
# Submit the Ocean Visions analysis on Casper, right-sized per stage.
#
#   ./submit.sh --project UCSD0001 --region barrow --years 2023 --stages lwph
#   ./submit.sh --project UCSD0001 --region arctic_circle --years 2014-2024 --all
#   ./submit.sh --project UCSD0001 --region barrow --years 2014-2024 --notebook
#   ./submit.sh ... --dry-run                 # print the qsub commands, submit nothing
#
# Any domain (W > E crosses the dateline), and the CSV/netCDF products:
#   ./submit.sh --box 82 -175 65 -110 --years 2014-2024 --all
#   ./submit.sh --box 82 -175 65 -110 --region-name beaufort_wide --csvs all
#   ./submit.sh --region arctic_circle --csvs hour_count,per_cell
#   ./submit.sh --region arctic_circle --csvs per_cell --cell-weighting uniform
#
# --project defaults to $PBS_ACCOUNT_CASPER, then $PBS_ACCOUNT, as in PRISM.
# ERA5 is read in place from GDEX (d633000); nothing is downloaded.
#
# Why staged jobs rather than one: Casper's wall-clock ceiling is 24 h, and a
# pan-Arctic 11-season pass of everything is ~34 CPU-hours. Split by stage, each
# piece fits comfortably and the expensive one (figure 8) gets its own cores.
# --notebook instead runs the real notebook end to end in a single job; use it
# for a domain small enough to finish inside 24 h, or after the staged run has
# filled the chunk cache (which makes a second pass cheap).
#
# Charging is cores x wall-clock; memory is free. So: modest ncpus, generous mem
# (but under 350 GB, which keeps the job in the fast htc queue).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# qsub itself is not documented to read PBS_ACCOUNT, so the code is passed with -A.
PROJECT="${PBS_ACCOUNT_CASPER:-${PBS_ACCOUNT:-}}"; REGION="barrow"; YEARS="2014-2024"; SEASON="10-01:03-31"
STAGES=""; ALL=0; NOTEBOOK=0; DRY=0; OUTDIR=""; EXTRA=""; CHAIN=1; CSVS=""; REGION_NAME=""
CELL_WEIGHTING=""; MEM_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --project|-A) PROJECT="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --box)        REGION="box:$2,$3,$4,$5"; shift 5 ;;   # N W S E
    --region-name) REGION_NAME="$2"; shift 2 ;;
    --csvs)       CSVS="$2"; shift 2 ;;
    --cell-weighting) CELL_WEIGHTING="$2"; shift 2 ;;     # area (default) | uniform
    --mem)        MEM_OVERRIDE="$2"; shift 2 ;;           # e.g. 32GB; replaces each job's default
    --years)      YEARS="$2"; shift 2 ;;
    --season)     SEASON="$2"; shift 2 ;;
    --stages)     shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do STAGES="$STAGES $1"; shift; done ;;
    --all)        ALL=1; shift ;;
    --notebook)   NOTEBOOK=1; shift ;;
    --out)        OUTDIR="$2"; shift 2 ;;
    --extra)      EXTRA="$2"; shift 2 ;;
    --no-chain)   CHAIN=0; shift ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done
[ -n "$PROJECT" ] || { echo "no project code: pass --project <CODE> or export PBS_ACCOUNT=<code>"; exit 1; }

if [ -f "$HOME/.era5_casper_env" ]; then
  set +u; source "$HOME/.era5_casper_env"; set -u
elif [ "$DRY" = 1 ]; then
  RESULTS_ROOT="${RESULTS_ROOT:-<RESULTS_ROOT>}"   # a dry run works anywhere, e.g. the laptop
else
  echo "no ~/.era5_casper_env -- run ./setup_casper.sh on Casper first"; exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
# No commas in the directory name either: it is passed through PBS -v.
YEARS_TAG="${YEARS//,/+}"
REGION_TAG="${REGION_NAME:-$(echo "$REGION" | tr ',:' '__')}"
OUTDIR="${OUTDIR:-$RESULTS_ROOT/${REGION_TAG}_${YEARS_TAG}_${STAMP}}"
# The job scripts create this too; doing it here means qsub's -o log path
# exists before the job starts. Skipped on a dry run so the command can be
# inspected from anywhere, including a laptop with no /glade.
[ "$DRY" = 1 ] || mkdir -p "$OUTDIR"

# stage -> ncpus, mem, walltime.
#
# Charging is ELAPSED wall-clock x cores RESERVED, so cores a stage cannot use
# are pure waste while the walltime request is only a ceiling (finish early,
# pay less). The analysis loops are serial numpy; only the chunk decode threads.
# So: few cores, generous walltime.
#
# Sized from a measured 12.6 us per cell-hour at Barrow (41x61) scaled to
# arctic_circle (136,800 cells, 54.7x) x 11 seasons:
# Measured marginal cost per season (local Barrow archive, per cell-hour, with
# the ~19 s per-prepare dataset open excluded because it does not scale):
#   lwph.prepare 0.68 us    extract_site_table 0.13    prepare_domain 0.35
#   prepare_maps 0.39       prepare_extent 0.48        tfr.prepare 2.81
# The lwph STAGE runs prepare three times (precip, all-sky, sweep) and folds in
# dlr + thumb, so it is ~0.9 of a flux pass, not the ~0.4 an earlier cut of this
# table assumed. Pan-Arctic x 11 seasons that is of order 15-20 CPU-hours for
# lwph and 19-23 for flux, depending on how Casper's Cascade Lake / EPYC cores
# compare with the machine the rates were measured on -- a 3x spread the first
# real run will settle.
#
#   stage   cores that help   charged at this size (elapsed x cores)
#   lwph    4                 ~26-70
#   maps    4                 ~6
#   extent  4                 ~6
#   flux    8                 ~48
#
# WALLTIME IS A CEILING, NOT A CHARGE: a job that finishes early pays only for
# the time it used, and one killed at the ceiling pays in full and produces
# nothing. So every stage gets close to Casper's 24 h maximum. The only cost of
# a generous request is a longer queue wait, since PBS backfills short jobs
# first -- lower these once you have measured a real run from manifest.json.
#
# Memory is free and does not affect charging, so it is set generously -- but
# stays under 350 GB, above which PBS routes to the scarce largemem nodes.
# extent holds 9 bytes per cell-hour (~59 GB pan-Arctic x 11 seasons).
size_for() {
  case "$1" in
    lwph)   echo "4 120GB 23:00:00" ;;
    maps)   echo "4 120GB 12:00:00" ;;
    extent) echo "4 250GB 16:00:00" ;;
    flux)   echo "8 200GB 23:00:00" ;;
    *)      echo "4 120GB 12:00:00" ;;
  esac
}

# PBS -v takes a COMMA-separated list, so a value containing a comma (which a
# year spec like "2020,2022" legitimately is) would be read as two variables.
# Commas travel as '+' and the job scripts turn them back.
pbs_safe() { echo "${1//,/+}"; }      # also covers region specs like box:82,-175,65,-110

submit() {  # submit <jobname> <script> <ncpus> <mem> <walltime> <-v pairs> [depend-jobid]
  local name="$1" script="$2" ncpus="$3" mem="$4" wall="$5" vars="$6" dep="${7:-}"
  # ompthreads: NCAR defaults it to 1, and the first Casper test (select with
  # ncpus=4 and no ompthreads) logged NCPUS=1 and one reader thread while being
  # charged for 4 cores. Docs: pbs/job-scripts, "pure OpenMP job on Casper".
  local cmd=(qsub -N "$name" -A "$PROJECT"
             -l "select=1:ncpus=${ncpus}:ompthreads=${ncpus}:mem=${MEM_OVERRIDE:-$mem}"
             -l "walltime=${wall}"
             -o "${OUTDIR}/${name}.log"
             -v "$vars")
  [ -n "$dep" ] && cmd+=(-W "depend=afterany:${dep}")
  cmd+=("$script")
  if [ "$DRY" = 1 ]; then printf '  %q' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

echo "project $PROJECT | region $REGION | years $YEARS | season $SEASON"
echo "output  $OUTDIR"
echo

# The notebook's per-figure settings default to the TALK's values (figure 8 over
# 2014-2025, figure 6 drawing season 2024). Left alone they contradict a --years
# that does not contain them: figure 8 would silently run different seasons from
# the rest, and figure 6 would raise AFTER every expensive pass. Derive them
# from --years, as run_analysis.py does.
LAST_YEAR="${YEARS##*[-,]}"
# Figure 4's cloud-level wind is a single-cell diagnostic and the most
# expensive input per season (whole-globe pressure-level chunks), so it
# defaults to ONE season -- the last one in --years, which is guaranteed to be
# part of the run. Override with PL_YEARS=... in the environment.
PL_YEARS_DEFAULT="$LAST_YEAR"

if [ "$NOTEBOOK" = 1 ]; then
  IFS=: read -r SS SE <<< "$SEASON"
  # Pan-Arctic the notebook holds every prepared object alive at once, so its
  # memory ask is far larger than any single staged job's. 700 GB still lands on
  # the htc AMD nodes (largemem starts above 733 GB).
  if [ "$REGION" = arctic_circle ] || [ "$REGION" = arctic_70n ]; then
    echo "note: --notebook runs all ten domain passes in ONE 24 h job. Pan-Arctic that is"
    echo "      tight even at 8 cores. Prefer --all, or run --notebook afterwards when the"
    echo "      chunk cache is warm. Submitting anyway."
  fi
  submit "ov_notebook" "$HERE/run_notebook.pbs" 8 700GB 24:00:00 \
    "REGION=$(pbs_safe "$REGION"),YEARS=$(pbs_safe "$YEARS"),SEASON_START=$SS,SEASON_END=$SE,OUTDIR=$OUTDIR,PL_REGION=${PL_REGION:-barrow},PL_YEARS=$(pbs_safe "${PL_YEARS:-$PL_YEARS_DEFAULT}"),FLUX_YEARS=$(pbs_safe "${FLUX_YEARS:-$YEARS}"),MAP_SEASON=${MAP_SEASON:-$LAST_YEAR}"
  exit 0
fi

# stage -> ncpus, mem, walltime for the CSV/netCDF products. Sized like the
# figure stages above, from one lwph.prepare pass at 0.68 us per cell-hour:
# pan-Arctic x 11 seasons (6.6e9 cell-hours) is about 1.2 CPU-hours per counting
# pass, plus chunk decode. precip makes two passes. hourly also reads 23-level
# pressure data for CloudTemp, so it gets more cores. Walltimes are ESTIMATES, not
# measurements on Casper; lower them after a real run.
#
# Memory: 16 GB. MEASURED 2.07 GB (qhist) for hour_count on a 18,009-cell box, one
# season. Block size shrinks as the domain grows, so that holds up to about 75,000
# cells, where block_hours reaches its 24 h floor. CALCULATED for hourly on
# arctic_circle: about 10 GB, from 48 h x 136,800 cells x 23 levels of float64 fields,
# so its margin is thinner. A domain much larger than pan-Arctic needs --mem. Each
# job log prints its peak; check those before relying on these on a new domain.
csv_size_for() {
  case "$1" in
    hour_count|per_cell|cumulative|ice_vs_water) echo "4 16GB 12:00:00" ;;
    precip)              echo "4 16GB 23:00:00" ;;
    hourly)              echo "8 16GB 23:00:00" ;;
    netcdf)              echo "4 16GB 12:00:00" ;;
    *) echo "unknown product $1 (hour_count per_cell precip cumulative ice_vs_water hourly netcdf)" >&2; exit 1 ;;
  esac
}

if [ -n "$CSVS" ]; then
  [ "$CSVS" = all ] && CSVS="hour_count,per_cell,precip,cumulative,ice_vs_water,hourly,netcdf"
  # A domain without the ARM site has no UTQ netCDF to write; the script stops
  # with that message, and afterany lets the rest of the chain continue.
  FIRST_ID=""
  for product in ${CSVS//,/ }; do
    read -r ncpus mem wall <<< "$(csv_size_for "$product")"
    dep=""
    [ "$CHAIN" = 1 ] && dep="$FIRST_ID"
    out="$(submit "ov_${product}" "$HERE/run_csvs.pbs" "$ncpus" "$mem" "$wall" \
      "PRODUCT=$product,REGION=$(pbs_safe "$REGION"),REGION_NAME=$REGION_NAME,YEARS=$(pbs_safe "$YEARS"),CELL_WEIGHTING=$CELL_WEIGHTING,OUTDIR=$OUTDIR" "$dep")"
    echo "$out"
    if [ "$DRY" != 1 ]; then FIRST_ID="$(echo "$out" | tail -1 | tr -d '[:space:]')"; fi
  done
  cat <<MSG

Submitted, not finished. The output folder stays EMPTY until a job ends: each script
writes its files after its last pass, and PBS delivers the .log at the end of the job.
  qstat -u \$USER            # Q = queued, R = running; absent = finished
  qhist -u \$USER -d 1       # finished jobs, with exit status and core-hours
Results: ${OUTDIR}/<domain>_<span>/
MSG
  exit 0
fi

[ "$ALL" = 1 ] && STAGES="lwph maps extent flux"
[ -n "${STAGES// /}" ] || { echo "nothing to do: pass --stages ... , --all, --csvs or --notebook"; exit 1; }

# The stages read heavily overlapping variables, and the decoded-chunk cache is
# what stops each one re-inflating them (~15 CPU-hours per run). Four jobs
# starting together on a cold cache forfeit that and duplicate the work, so the
# rest wait for the first -- afterany, not afterok, since a later stage is still
# worth attempting if an earlier one fails. --no-chain submits them all at once.
FIRST_ID=""
for st in $STAGES; do
  read -r ncpus mem wall <<< "$(size_for "$st")"
  # dlr and thumb reuse the lwph pass in-process, so they ride along with it.
  # '+' not a space: PBS -v takes a comma-separated list and the handling of
  # embedded whitespace varies between versions. run_stage.pbs splits it back.
  stage_arg="$st"; [ "$st" = lwph ] && stage_arg="lwph+dlr+thumb"
  dep=""
  [ "$CHAIN" = 1 ] && dep="$FIRST_ID"
  out="$(submit "era5_${st}" "$HERE/run_stage.pbs" "$ncpus" "$mem" "$wall" \
    "STAGES=$stage_arg,REGION=$(pbs_safe "$REGION"),YEARS=$(pbs_safe "$YEARS"),SEASON=$SEASON,OUTDIR=$OUTDIR,EXTRA=$EXTRA" "$dep")"
  echo "$out"
  if [ "$DRY" != 1 ] && [ -z "$FIRST_ID" ]; then
    FIRST_ID="$(echo "$out" | tail -1 | tr -d '[:space:]')"
  fi
done

cat <<MSG

Submitted. Watch with:
  qstat -u \$USER            # queued / running
  tail -f ${OUTDIR}/*.log
  qhist -u \$USER -d 1       # what finished, and what it cost in core-hours
MSG
