#!/bin/bash

# Name of the conda environment
ENV_NAME="trade-shock-env"

# Output file for the environment
OUTPUT_FILE="environment.yml"

# Check if the environment exists
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "Updating $OUTPUT_FILE for environment $ENV_NAME..."
    
    # Export the environment without the 'prefix' path
    conda env export --name $ENV_NAME | grep -v "^prefix: " > $OUTPUT_FILE

    echo "$OUTPUT_FILE has been updated."
else
    echo "Environment $ENV_NAME does not exist. Please check the environment name."
    exit 1
fi
