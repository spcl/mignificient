#!/bin/bash

# Check if a file name is provided as an argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide an output file name."
    exit 1
fi

output_dir="$1"

# Function to get GPU memory in GB
get_gpu_memory() {
    local gpu_index=$1
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_index | awk '{print $1/1024}'
}

# Function to get GPU compute capability
get_gpu_compute_capability() {
    local gpu_index=$1
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $gpu_index
}

# Collect GPU information
gpu_info=$(nvidia-smi -L)

# Initialize empty JSON array for GPUs
gpus_json="[]"

# Process each GPU
while IFS= read -r line; do
    if [[ $line =~ GPU[[:space:]]([0-9]+):.*(UUID:[[:space:]]GPU-([a-f0-9-]+)) ]]; then
        gpu_index="${BASH_REMATCH[1]}"
        gpu_uuid="${BASH_REMATCH[3]}"
        
        # Get GPU memory and compute capability
        gpu_memory=$(get_gpu_memory $gpu_index)
        gpu_compute_capability=$(get_gpu_compute_capability $gpu_index)
        
        # Initialize empty JSON array for instances
        instances_json="[]"
        
        # Check for MIG instances
        mig_instances=$(echo "$gpu_info" | grep -A10 "GPU $gpu_index:" | grep "MIG.*Device")
        
        if [ -z "$mig_instances" ]; then
            # No MIG instances, treat whole GPU as one instance
            instances_json=$(jq -n --arg uuid "$gpu_uuid" --arg memory "$gpu_memory" \
                '[{uuid: $uuid, memory: ($memory|tonumber), compute_units: 0}]')
        else
            # Process MIG instances
            while IFS= read -r mig_line; do
		if [[ $mig_line =~ MIG[[:space:]]([0-9]+g\.[0-9]+gb).*UUID:[[:space:]]([A-Za-z0-9-]+) ]]; then
                    mig_size="${BASH_REMATCH[1]}"
                    mig_uuid="${BASH_REMATCH[2]}"
                    
                    # Calculate approximate compute units and memory
                    case $mig_size in
                        "1g.5gb") compute_units=14; memory=5 ;;
                        "2g.10gb") compute_units=28; memory=10 ;;
                        "3g.20gb") compute_units=42; memory=20 ;;
                        "4g.20gb") compute_units=56; memory=20 ;;
                        "7g.40gb") compute_units=98; memory=40 ;;
                        *) compute_units=0; memory=0 ;;
                    esac
                    
                    # Add MIG instance to instances array
                    instances_json=$(echo $instances_json | jq --arg uuid "$mig_uuid" \
                        --arg memory "$memory" --arg compute_units "$compute_units" \
                        '. += [{uuid: $uuid, memory: ($memory|tonumber), compute_units: ($compute_units|tonumber)}]')
                fi
            done <<< "$mig_instances"
        fi
        
        # Add GPU to GPUs array
        gpus_json=$(echo $gpus_json | jq --arg uuid "$gpu_uuid" --arg index "$gpu_index" \
            --arg memory "$gpu_memory" --arg compute_capability "$gpu_compute_capability" \
            --argjson instances "$instances_json" \
            '. += [{uuid: $uuid, index: ($index|tonumber), memory: ($memory|tonumber), compute_capability: $compute_capability, instances: $instances}]')
    fi
done <<< "$gpu_info"

# Create final JSON structure with timestamp and GPUs array
final_json=$(jq -n --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --argjson gpus "$gpus_json" \
    '{timestamp: $timestamp, gpus: $gpus}')

# Save to output file
echo $final_json | jq '.' > "${output_dir}/devices.json"

echo "GPU information has been saved to ${output_dir}/devices.json"
