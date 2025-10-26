#!/bin/bash
# compile and make
python scripts/update_notes/compile.py

# generate new tags
LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
VERSION_NO_V=$(echo "$LATEST_TAG" | sed 's/^v//')
IFS='.' read -r -a VERSION_ARRAY <<< "$VERSION_NO_V"

MAJOR=${VERSION_ARRAY[0]}
MINOR=${VERSION_ARRAY[1]}
PATCH=${VERSION_ARRAY[2]}
NEXT_PATCH=$((PATCH + 1))
NEW_TAG="v$MAJOR.$MINOR.$NEXT_PATCH"
echo "New tag: $NEW_TAG"

git add .
git commit -m "AutoCommit: Release Updates for $NEW_TAG"
git push

# pushing new tags
git tag "$NEW_TAG"
git push origin "$NEW_TAG"

