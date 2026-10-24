#!/bin/sh
# Extract an existing local image once; never merge over an earlier runtime.
set -eu
IMAGE=${1:?usage: extract.sh IMAGE NEW_DIRECTORY}
DEST=${2:?usage: extract.sh IMAGE NEW_DIRECTORY}
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
    echo "Refusing to overwrite existing extraction: $DEST" >&2
    exit 2
fi
id=$(docker create --pull=never --entrypoint /run.sh "$IMAGE")
trap 'docker rm "$id" >/dev/null' EXIT
trap 'exit 1' HUP INT TERM
mkdir -p "$DEST"
docker cp "$id":/. "$DEST/"
printf 'Extracted %s to %s\n' "$IMAGE" "$DEST"
