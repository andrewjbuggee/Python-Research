#!/bin/bash
# Script to collect memory efficiency from Alpine seff command
# Extracts memory efficiency % from all jobs in the array 24757758_101 to 24757758_173

output_file="memory_efficiency_data.txt"
> "$output_file"  # Clear file

echo "Collecting memory efficiency data from seff..."
echo "Job_ID,Memory_Efficiency_Percent" > "$output_file"

for i in {201..273}; do
    job_id="25096241_$i"
    # Extract memory efficiency line and parse the percentage
    mem_eff=$(seff $job_id 2>/dev/null | grep "Memory Efficiency" | awk '{print $3}' | sed 's/%//')
    
    if [ ! -z "$mem_eff" ]; then
        echo "$job_id,$mem_eff" >> "$output_file"
        echo "Job $job_id: $mem_eff%"
    else
        echo "Job $job_id: Failed to retrieve data"
    fi
done

echo ""
echo "Data saved to $output_file"
