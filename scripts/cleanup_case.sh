#!/bin/bash

# Script to clean up case directories in data/cases
# Cleans eval*, results*, inference* directories and optionally tres_runs

set -e  # Exit on error

CASES_DIR="data/cases"

# Check if cases directory exists
if [ ! -d "$CASES_DIR" ]; then
    echo "Error: $CASES_DIR directory not found!"
    exit 1
fi

# Get list of case directories
CASES=($(find "$CASES_DIR" -maxdepth 1 -type d -not -path "$CASES_DIR" | sort | xargs -n1 basename))

if [ ${#CASES[@]} -eq 0 ]; then
    echo "No case directories found in $CASES_DIR"
    exit 1
fi

# Display available cases
echo "Available cases:"
echo "================"
for i in "${!CASES[@]}"; do
    printf "%d) %s\n" $((i+1)) "${CASES[$i]}"
done
echo ""

# Prompt user for selection
read -p "Enter the number of the case to clean up: " SELECTION

# Validate selection
if ! [[ "$SELECTION" =~ ^[0-9]+$ ]] || [ "$SELECTION" -lt 1 ] || [ "$SELECTION" -gt ${#CASES[@]} ]; then
    echo "Error: Invalid selection!"
    exit 1
fi

# Get selected case
SELECTED_CASE="${CASES[$((SELECTION-1))]}"
CASE_PATH="$CASES_DIR/$SELECTED_CASE"

echo ""
echo "Selected case: $SELECTED_CASE"
echo "Path: $CASE_PATH"
echo ""

# Find and clean directories matching patterns
PATTERNS=("eval*" "results*" "inference*")
PURGED_DIRS=()

for pattern in "${PATTERNS[@]}"; do
    # Find directories matching the pattern
    while IFS= read -r -d '' dir; do
        if [ -d "$dir" ]; then
            echo "Purging: $dir"
            rm -rf "$dir"
            PURGED_DIRS+=("$dir")
        fi
    done < <(find "$CASE_PATH" -maxdepth 1 -type d -name "$pattern" -print0 2>/dev/null || true)
done

# Show summary
echo ""
echo "================"
echo "Directories purged:"
if [ ${#PURGED_DIRS[@]} -eq 0 ]; then
    echo "  (none found)"
else
    for dir in "${PURGED_DIRS[@]}"; do
        echo "  - $dir"
    done
fi
echo ""

# Ask about tres_runs
read -p "Do you want to clean the tres_runs subdirectory as well? [y/N]: " CLEAN_TRES

if [[ "$CLEAN_TRES" =~ ^[Yy]$ ]]; then
    TRES_RUNS_PATH="$CASE_PATH/tres_runs"
    if [ -d "$TRES_RUNS_PATH" ]; then
        echo "Purging: $TRES_RUNS_PATH"
        rm -rf "$TRES_RUNS_PATH"
        echo "  - $TRES_RUNS_PATH (purged)"
    else
        echo "  - $TRES_RUNS_PATH (not found, skipping)"
    fi
else
    echo "Skipping tres_runs cleanup."
fi

echo ""
echo "Cleanup complete!"
