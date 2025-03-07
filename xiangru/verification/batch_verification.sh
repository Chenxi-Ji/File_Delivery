#!/bin/bash
# This script performs the following tasks:
# 1. Iterates over all .npz files in the image_bounds directory.
# 2. For each file, runs:
#    python generate_vnnlib.py <basename> -b image_bounds/<basename>.npz -t 1
# 3. Changes directory two levels up.
# 4. Runs:
#    python abcrown.py --config nerf/verification/nerf_classification_layer12.yaml
#    and logs the output to nerf/verification/logs/<basename>.log
#
# Each step prints a prompt message.

# Save the current directory (assumes the script is run from the directory that contains the image_bounds folder)
BASE_DIR=$(pwd)

# Iterate over all .npz files in the image_bounds folder
for file in "$BASE_DIR"/image_bounds/*.npz; do
    # Extract the filename and the basename (filename without extension)
    filename=$(basename "$file")
    base="${filename%.npz}"

    echo "------------------------------------------------------"
    echo "Processing file: $filename"

    # Step 2: Run generate_vnnlib.py
    echo "Step 2: Running 'python generate_vnnlib.py $base -b $file -t 1'"
    python generate_vnnlib.py "$base" -b "$file" -t 1
    if [ $? -ne 0 ]; then
        echo "Error: generate_vnnlib.py failed. Exiting script."
        exit 1
    fi

    # Step 3: Change directory two levels up
    echo "Step 3: Changing directory two levels up (cd ../../)"
    pushd "$BASE_DIR/../.." > /dev/null

    # Step 4: Run abcrown.py and log the output
    log_dir="nerf/verification/logs"
    mkdir -p "$log_dir"
    echo "Step 4: Running 'python abcrown.py --config nerf/verification/nerf_classification_layer12.yaml' and logging to $log_dir/$base.log"
    python abcrown.py --config nerf/verification/nerf_classification_layer12.yaml 2>&1 | tee "$log_dir/$base.log"
    if [ $? -ne 0 ]; then
        echo "Error: abcrown.py failed."
        popd > /dev/null
        exit 1
    fi

    # Return to the original directory to process the next file
    popd > /dev/null
    echo "Finished processing file $filename."
done

echo "All files have been processed."
