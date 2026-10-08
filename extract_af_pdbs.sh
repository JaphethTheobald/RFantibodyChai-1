#!/usr/bin/env bash
# Copy every file with "best_model_0" in its filename from Downloads and all
# nested folders into one flat output folder. Original files are unchanged.

set -euo pipefail

# Edit these two lines only if you want a different source or destination.
SOURCE_ROOT="${HOME}/Downloads"
DESTINATION="${SOURCE_ROOT}/all_best_model_0"
MATCH='*best_model_0*'

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "Downloads folder not found: $SOURCE_ROOT" >&2
    exit 1
fi

mkdir -p "$DESTINATION"
copied=0

while IFS= read -r -d '' source_file; do
    filename=$(basename "$source_file")
    destination_file="$DESTINATION/$filename"

    # Keep files with identical names from different folders rather than
    # overwriting either one.
    if [[ -e "$destination_file" ]]; then
        stem=${filename%.*}
        extension=""
        if [[ "$filename" == *.* ]]; then
            extension=".${filename##*.}"
        fi
        number=2
        while [[ -e "$DESTINATION/${stem}__copy_${number}${extension}" ]]; do
            ((number += 1))
        done
        destination_file="$DESTINATION/${stem}__copy_${number}${extension}"
    fi

    cp -- "$source_file" "$destination_file"
    echo "Copied: $source_file"
    ((copied += 1))
done < <(
    find "$SOURCE_ROOT" \
        -path "$DESTINATION" -prune -o \
        -type f -name "$MATCH" -print0
)

echo
echo "Copied $copied file(s) to: $DESTINATION"
