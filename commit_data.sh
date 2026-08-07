#!/usr/bin/env bash
# Commit and push generated data, merging rather than rebasing on a collision.
#
# Usage: commit_data.sh "<commit message>" [paths...]
#
# Two workflow runs can be in flight at once — scheduled runs fire at irregular
# times and sometimes bunch — and both rewrite the same CSVs. `git pull
# --rebase` conflicts on exactly that and fails the run with the snapshot
# already paid for, so on a rejected push this keeps the run's own files, takes
# the remote tree, and merges by meaning via merge_data.py: append-only logs
# are unioned, regenerated snapshots take the later version.
set -euo pipefail

MESSAGE="${1:?commit message required}"
shift || true
# "${@:-data docs}" would collapse the default into one argument containing a
# space, which git then reads as a single missing path.
if [ "$#" -eq 0 ]; then
  PATHS=(data docs)
else
  PATHS=("$@")
fi
BRANCH="${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

# -A over the directories rather than a list of files: a run that died before
# writing one of them would otherwise fail here on a pathspec that matches
# nothing, masking the original error.
git add -A "${PATHS[@]}"
if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi
git commit -m "$MESSAGE"

for attempt in 1 2 3 4; do
  if git push origin "HEAD:${BRANCH}"; then
    exit 0
  fi
  echo "push rejected (attempt $attempt); merging onto the remote"
  ours=$(mktemp -d)
  tar -c "${PATHS[@]}" | tar -x -C "$ours"
  git fetch origin "$BRANCH"
  git reset --hard "origin/${BRANCH}"
  python merge_data.py --ours "$ours" --targets "${PATHS[@]}"
  rm -rf "$ours"
  git add -A "${PATHS[@]}"
  if git diff --cached --quiet; then
    echo "Remote already carries this run's rows."
    exit 0
  fi
  git commit -m "$MESSAGE"
  sleep $((2 ** attempt))
done

echo "could not push after 4 attempts"
exit 1
