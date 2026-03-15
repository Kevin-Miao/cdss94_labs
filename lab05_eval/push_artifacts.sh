#!/bin/bash
# push_artifacts.sh
# Watches for teacher_prep to finish generating artifacts, then pushes to GitHub.
# Usage: bash push_artifacts.sh

ARTIFACTS_DIR="/Users/kevinmiao/Desktop/CDSS94/labs/lab05_eval/lab_artifacts"
REPO_DIR="/tmp/cdss94_labs_repo"
SOURCE_DIR="/Users/kevinmiao/Desktop/CDSS94/labs/lab05_eval"
MANIFEST="$ARTIFACTS_DIR/manifest.pt"

echo "Watching for teacher_prep to finish..."
echo "  Waiting for manifest.pt to be updated..."

# Record current manifest modification time (or 0 if doesn't exist)
if [ -f "$MANIFEST" ]; then
    ORIG_MTIME=$(stat -f %m "$MANIFEST")
else
    ORIG_MTIME=0
fi

# Poll until manifest.pt is newer (means teacher_prep re-saved it)
while true; do
    if [ -f "$MANIFEST" ]; then
        CUR_MTIME=$(stat -f %m "$MANIFEST")
        if [ "$CUR_MTIME" -gt "$ORIG_MTIME" ]; then
            echo "  manifest.pt updated! ($(date))"
            break
        fi
    fi
    sleep 5
done

# Small buffer to let any final file writes complete
sleep 3

echo ""
echo "Copying artifacts to repo..."
cp "$SOURCE_DIR/student_lab.ipynb" "$REPO_DIR/lab_05/"
rm -rf "$REPO_DIR/lab_05/lab_artifacts"
cp -r "$ARTIFACTS_DIR" "$REPO_DIR/lab_05/lab_artifacts"

echo "Staging and pushing..."
cd "$REPO_DIR"
git add lab_05/
git commit -m "Update lab artifacts from teacher_prep run ($(date '+%Y-%m-%d %H:%M'))"
git push

echo ""
echo "Done! Pushed to https://github.com/Kevin-Miao/cdss94_labs"
