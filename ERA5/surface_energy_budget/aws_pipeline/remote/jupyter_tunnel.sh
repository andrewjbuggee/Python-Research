#!/usr/bin/env bash
# Laptop: start JupyterLab on the instance and forward it to http://localhost:8890
#
# Then open the Ocean Visions notebook (it lives in the cloned repo on the
# instance) and set COMMON["storage"] = "aws" and any region/years you like.
# The notebook is otherwise unchanged; figures save under the repo on the
# instance and come back with fetch_results.sh or scp.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r INSTANCE_ID DNS KEY_FILE < "$HERE/.last_instance"
LOCAL_PORT="${LOCAL_PORT:-8890}"
# -t: a pty, so Ctrl-C here ends the remote jupyter too; port_retries=0: fail
# loudly if 8888 is still taken rather than silently starting on 8889 while
# the tunnel points at 8888.
ssh -t -i "$KEY_FILE" -L "$LOCAL_PORT:localhost:8888" "ubuntu@$DNS" \
  'source ~/.era5_env && eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)" && micromamba activate era5-seb && cd $REPO_DIR && jupyter lab --no-browser --port 8888 --ip 127.0.0.1 --ServerApp.port_retries=0'
