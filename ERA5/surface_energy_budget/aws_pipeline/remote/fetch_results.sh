#!/usr/bin/env bash
# Laptop: copy a results directory back from the instance.
#   ./fetch_results.sh                    # latest run -> aws_pipeline/results/<stamp>
#   ./fetch_results.sh 20260918T210000Z   # a specific run
#   ONLY_FIGURES=1 ./fetch_results.sh     # skip the pickles (they can be large)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r INSTANCE_ID DNS KEY_FILE < "$HERE/.last_instance"
RUN="${1:-latest}"
# 'latest' is a symlink on the instance: resolve it so runs land in their own
# stamped directory here instead of piling into results/latest/.
if [ "$RUN" = latest ]; then
  RUN="$(ssh -i "$KEY_FILE" "ubuntu@$DNS" 'basename "$(readlink -f /data/results/latest)"')"
fi
DEST="$HERE/../results"
mkdir -p "$DEST"
EXCL=()
[ "${ONLY_FIGURES:-0}" = "1" ] && EXCL=(--exclude '*.pkl')
rsync -avz --progress ${EXCL[@]+"${EXCL[@]}"} -e "ssh -i $KEY_FILE" "ubuntu@$DNS:/data/results/$RUN/" "$DEST/$RUN/"
echo "-> $DEST/$RUN"
