#!/usr/bin/env bash

# Shared validation for the isolated SGCDet-inspired sparse-refiner protocol.
# This file is sourced by the public scripts; it never launches a process.

sgcdet_sparse_require_file() {
    local path="$1"
    local description="$2"
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "Missing $description: ${path:-<empty>}" >&2
        return 1
    fi
}

sgcdet_sparse_require_directory() {
    local path="$1"
    local description="$2"
    if [[ -z "$path" || ! -d "$path" ]]; then
        echo "Missing $description: ${path:-<empty>}" >&2
        return 1
    fi
}

sgcdet_sparse_scene_count() {
    awk 'NF { count += 1 } END { print count + 0 }' "$1"
}

sgcdet_sparse_assert_train_only() {
    local train_list="$1"
    local validation_list="$2"
    local overlap

    sgcdet_sparse_require_file "$train_list" "ScanNet train-only scene list"
    sgcdet_sparse_require_file "$validation_list" "ScanNet validation scene list used by the leakage guard"

    if [[ "$(basename -- "${train_list,,}")" == *val* ]]; then
        echo "Refusing validation-labelled sparse-refiner training list: $train_list" >&2
        return 1
    fi
    if [[ "$(sgcdet_sparse_scene_count "$train_list")" -eq 0 ]]; then
        echo "Sparse-refiner training scene list is empty: $train_list" >&2
        return 1
    fi

    overlap="$({
        awk '
            NR == FNR {
                gsub(/\r/, "", $0)
                if ($1 != "") forbidden[$1] = 1
                next
            }
            {
                gsub(/\r/, "", $0)
                if ($1 != "" && ($1 in forbidden)) print $1
            }
        ' "$validation_list" "$train_list"
    } | sort -u)"
    if [[ -n "$overlap" ]]; then
        echo "Refusing ScanNet validation leakage in sparse-refiner training list:" >&2
        while IFS= read -r scene_id; do
            [[ -n "$scene_id" ]] && echo "  $scene_id" >&2
        done <<< "$overlap"
        return 1
    fi
}
