#!/bin/sh
# Package the launcher (the app/ folder) into an .ipk in ./dist using LG's webOS CLI.
set -e
cd "$(dirname "$0")"
mkdir -p dist
npx --yes -p @webos-tools/cli ares-package --no-minify -o dist app
ls -1 dist/*.ipk
