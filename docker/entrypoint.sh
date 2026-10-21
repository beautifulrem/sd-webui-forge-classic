#!/bin/bash
set -euo pipefail

# TCMalloc reduces memory fragmentation under large model workloads
TCMALLOC="$(find /usr/lib -name 'libtcmalloc_minimal.so.4' -print -quit 2>/dev/null || true)"
if [[ -n "$TCMALLOC" ]]; then
    export LD_PRELOAD="$TCMALLOC"
fi

# Read COMMANDLINE_ARGS then unset it
RAW_ARGS="${COMMANDLINE_ARGS:-}"
unset COMMANDLINE_ARGS

EXTRA_ARGS=()
if [[ -n "$RAW_ARGS" ]]; then
    read -ra EXTRA_ARGS <<< "$RAW_ARGS"
fi

# Every piece of user data (config.json, styles.csv, user.css, remi-data/, models/,
# output/, extensions/) is in the data folder: one volume keeps it across containers.
mkdir -p "$REMI_DATA_DIR"/models/{Stable-diffusion,VAE,text_encoder,Lora,ControlNet,ESRGAN,embeddings} \
    "$REMI_DATA_DIR"/{output,extensions}

exec python /home/remi/app/launch.py \
    --listen \
    --data-dir "$REMI_DATA_DIR" \
    "${EXTRA_ARGS[@]}" \
    "$@"
