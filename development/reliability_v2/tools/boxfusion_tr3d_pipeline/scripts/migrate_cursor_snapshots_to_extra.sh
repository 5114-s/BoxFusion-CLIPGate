#!/usr/bin/env bash
set -euo pipefail

SOURCE="/home/admin1/.cursor-server/data/snapshots"
STAGING="/extra/ZhaoX/.cursor-server-snapshots.migrating"
FINAL="/extra/ZhaoX/cursor-server-snapshots"
LOG_ROOT="/extra/ZhaoX/cursor-server-snapshots-migration-log"
LOCK_FILE="/extra/ZhaoX/.cursor-server-snapshots.migration.lock"

mkdir -p "$LOG_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "Another Cursor snapshot migration is running" >&2; exit 2; }

[[ -d "$SOURCE" && ! -L "$SOURCE" ]] || {
    echo "Source must be the expected regular directory: $SOURCE" >&2
    exit 2
}
[[ ! -e "$FINAL" && ! -L "$FINAL" ]] || {
    echo "Final destination already exists: $FINAL" >&2
    exit 2
}
[[ ! -e "$STAGING" && ! -L "$STAGING" ]] || {
    echo "Staging destination already exists: $STAGING" >&2
    exit 2
}
[[ "$(find "$SOURCE" -type l -print -quit)" == "" ]] || {
    echo "Source unexpectedly contains symlinks" >&2
    exit 2
}
[[ -z "$(find "$SOURCE" ! -type d ! -type f -print -quit)" ]] || {
    echo "Source unexpectedly contains non-regular filesystem entries" >&2
    exit 2
}
open_files="$(lsof +D "$SOURCE" 2>/dev/null || true)"
[[ -z "$open_files" ]] || {
    echo "Source has open files; refusing migration" >&2
    exit 2
}

source_files="$(find "$SOURCE" -type f -printf x | wc -c)"
source_dirs="$(find "$SOURCE" -type d -printf x | wc -c)"
source_bytes="$(find "$SOURCE" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
source_apparent="$(du -sb "$SOURCE" | awk '{print $1}')"

{
    echo "schema=boxfusion.cursor_snapshot_migration.v1"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "source=$SOURCE"
    echo "staging=$STAGING"
    echo "final=$FINAL"
    echo "source_files=$source_files"
    echo "source_dirs=$source_dirs"
    echo "source_regular_file_bytes=$source_bytes"
    echo "source_apparent_bytes=$source_apparent"
} > "$LOG_ROOT/manifest.txt"

mkdir "$STAGING"
echo "[$(date '+%F %T')] Copying $source_files files ($source_bytes bytes)"
rsync -aHAX --numeric-ids --partial --info=progress2 \
    --log-file="$LOG_ROOT/rsync-copy.log" \
    "$SOURCE/" "$STAGING/"

stage_files="$(find "$STAGING" -type f -printf x | wc -c)"
stage_dirs="$(find "$STAGING" -type d -printf x | wc -c)"
stage_bytes="$(find "$STAGING" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
[[ "$stage_files" == "$source_files" && "$stage_dirs" == "$source_dirs" && "$stage_bytes" == "$source_bytes" ]] || {
    echo "Count/size verification failed; source remains untouched" >&2
    exit 1
}

echo "[$(date '+%F %T')] Verifying every file by checksum"
rsync -aHAXcn --numeric-ids --delete --itemize-changes \
    "$SOURCE/" "$STAGING/" > "$LOG_ROOT/rsync-checksum-dry-run.txt"
[[ ! -s "$LOG_ROOT/rsync-checksum-dry-run.txt" ]] || {
    echo "Checksum verification found differences; source remains untouched" >&2
    sed -n '1,50p' "$LOG_ROOT/rsync-checksum-dry-run.txt" >&2
    exit 1
}

mv "$STAGING" "$FINAL"
[[ -d "$FINAL" && ! -L "$FINAL" ]] || {
    echo "Atomic destination rename failed; source remains untouched" >&2
    exit 1
}

echo "[$(date '+%F %T')] Verified destination; removing source files"
find "$SOURCE" -type f -delete
find "$SOURCE" -mindepth 1 -depth -type d -empty -delete
[[ -z "$(find "$SOURCE" -mindepth 1 -print -quit)" ]] || {
    echo "Source contains unexpected residual entries; destination is safe at $FINAL" >&2
    exit 1
}
rmdir "$SOURCE"
ln -s "$FINAL" "$SOURCE"

[[ -L "$SOURCE" && "$(readlink -f "$SOURCE")" == "$FINAL" ]] || {
    echo "Compatibility symlink verification failed" >&2
    exit 1
}

{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "destination_files=$(find "$FINAL" -type f -printf x | wc -c)"
    echo "destination_regular_file_bytes=$(find "$FINAL" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
    echo "source_symlink_target=$(readlink -f "$SOURCE")"
    df -B1 / /extra | sed 's/^/df_after=/'
} >> "$LOG_ROOT/manifest.txt"

echo "[$(date '+%F %T')] Cursor snapshots migrated successfully"
echo "Destination: $FINAL"
echo "Compatibility link: $SOURCE -> $FINAL"
