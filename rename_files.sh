#!/bin/bash

# Script to remove numeric prefixes from filenames
# Usage: ./rename_files.sh <directory_path>
# Example: 202591112940631370249066_Attentia planning medische onderzoeken (2) -> Attentia planning medische onderzoeken (2)

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 /path/to/directory"
    exit 1
fi

# Check if directory exists
if [ ! -d "$1" ]; then
    echo "Error: Directory '$1' does not exist"
    exit 1
fi

# Get the directory path
DIRECTORY="$1"

echo "Processing files in directory: $DIRECTORY"
echo "=========================================="

# Counter for renamed files
renamed_count=0

# Process all files in the directory
for file in "$DIRECTORY"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$file" ]; then
        # Get the filename without path
        filename=$(basename "$file")
        
        # Check if filename starts with numbers followed by underscore
        if [[ $filename =~ ^[0-9]+_ ]]; then
            # Extract the part after the first underscore
            new_filename=$(echo "$filename" | sed 's/^[0-9]*_//')
            
            # Create the new file path
            new_filepath="$DIRECTORY/$new_filename"
            
            # Check if the new filename already exists
            if [ -f "$new_filepath" ]; then
                echo "⚠️  Skipping '$filename' -> '$new_filename' (file already exists)"
            else
                # Rename the file
                if mv "$file" "$new_filepath"; then
                    echo "✅ Renamed: '$filename' -> '$new_filename'"
                    ((renamed_count++))
                else
                    echo "❌ Failed to rename: '$filename'"
                fi
            fi
        else
            echo "⏭️  Skipping '$filename' (no numeric prefix to remove)"
        fi
    fi
done

echo "=========================================="
echo "Renaming complete! $renamed_count files renamed."
