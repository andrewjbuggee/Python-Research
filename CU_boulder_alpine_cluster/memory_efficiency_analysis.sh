#!/bin/bash
# Script to collect memory efficiency from Alpine seff command
# Extracts memory efficiency % from all jobs in the array 24757758_101 to 24757758_173

# %A: Job ID
# %a: Array Task ID
# ----------------------------------------------------------
#SBATCH --account=ucb762_asc1                   # Ascent Allocation on Alpine
#SBATCH --nodes=1
#SBATCH --time=01:00:00   # Request 1 hour for data collection
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --mem=10G        # Should be closer to 80% efficiency based on previous runs, but giving some buffer for variability
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=mem_collection
#SBATCH --output=mem_collection.out
#SBATCH --error=mem_collection.err
#SBATCH --mail-user=anbu8374@colorado.edu
#SBATCH --mail-type=ALL

output_file="memory_efficiency_data.txt"
> "$output_file"  # Clear file

echo "Collecting memory efficiency data from seff..."
echo "Job_ID,Memory_Efficiency_Percent" > "$output_file"

for i in {401..473}; do
    job_id="25096244_$i"
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
