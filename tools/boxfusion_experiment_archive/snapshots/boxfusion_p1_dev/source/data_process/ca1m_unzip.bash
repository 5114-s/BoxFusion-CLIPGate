#!/bin/bash
# Extract all .tar/.tar.gz/.tar.bz2 files in current directory to folders with same name

for file in *.tar *.tar.gz *.tar.bz2; do
    if [ -f "$file" ]; then
        dir="${file%.*}"  # Remove extension as folder name
        mkdir -p "$dir"   # Create target directory
        
        # Extract based on file extension
        case "$file" in
            *.tar.gz|*.tgz)  tar -xzf "$file" -C "$dir" ;;
            *.tar.bz2|*.tbz2) tar -xjf "$file" -C "$dir" ;;
            *.tar)           tar -xf "$file" -C "$dir"  ;;
        esac
        
        echo "finish unzipping: $file → $dir/"
    fi
done