#!/usr/bin/env python3
"""
Run seff on each job array task and collect memory efficiency stats.

Usage:
    python seff_collector.py <job_array_id> [start_task] [end_task]
    
Example:
    python seff_collector.py 25039800 101 173
    
Note: Requires access to seff command (available on CU Boulder Alpine cluster)
"""

import subprocess
import re
import sys
import statistics
from datetime import timedelta

def parse_seff_output(seff_text):
    """Extract key metrics from seff output."""
    metrics = {}
    
    # Parse state
    m = re.search(r'State:\s+(\S+)', seff_text)
    metrics['state'] = m.group(1) if m else None
    
    # Parse wall time (Elapsed time: HH:MM:SS)
    m = re.search(r'Elapsed time:\s+(\d+):(\d+):(\d+)', seff_text)
    if m:
        hours, minutes, seconds = int(m.group(1)), int(m.group(2)), int(m.group(3))
        metrics['elapsed_sec'] = hours*3600 + minutes*60 + seconds
    
    # Parse CPU efficiency
    m = re.search(r'CPU Efficiency:\s+([\d.]+)%', seff_text)
    metrics['cpu_eff_pct'] = float(m.group(1)) if m else None
    
    # Parse memory - requested and max used
    m = re.search(r'Memory Requested:\s+([\d.]+)\s+(\w+)', seff_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        if unit == 'G':
            metrics['mem_req_mb'] = val * 1024
        elif unit == 'M':
            metrics['mem_req_mb'] = val
        elif unit == 'K':
            metrics['mem_req_mb'] = val / 1024
    
    m = re.search(r'Memory Used:\s+([\d.]+)\s+(\w+)', seff_text)
    if m:
        val, unit = float(m.group(1)), m.group(2)
        if unit == 'G':
            metrics['mem_used_mb'] = val * 1024
        elif unit == 'M':
            metrics['mem_used_mb'] = val
        elif unit == 'K':
            metrics['mem_used_mb'] = val / 1024
    
    # Calculate memory efficiency (seff definition: actual used / requested)
    if 'mem_req_mb' in metrics and 'mem_used_mb' in metrics:
        if metrics['mem_req_mb'] > 0:
            metrics['mem_eff_pct'] = (metrics['mem_used_mb'] / metrics['mem_req_mb']) * 100
    
    return metrics

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    job_array_id = sys.argv[1]
    start_task = int(sys.argv[2]) if len(sys.argv) > 2 else 101
    end_task = int(sys.argv[3]) if len(sys.argv) > 3 else 173
    
    results = {}
    failed_calls = []
    
    print(f"\nCollecting seff data for job array {job_array_id} tasks {start_task}-{end_task}")
    print("=" * 80)
    
    for task_id in range(start_task, end_task + 1):
        job_id = f"{job_array_id}_{task_id}"
        
        try:
            result = subprocess.run(
                ['seff', job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                metrics = parse_seff_output(result.stdout)
                results[task_id] = metrics
                
                state_str = metrics.get('state', '?')
                mem_eff = metrics.get('mem_eff_pct', None)
                wall_sec = metrics.get('elapsed_sec', None)
                cpu_eff = metrics.get('cpu_eff_pct', None)
                
                # Format output line
                parts = [f"Task {task_id:3d}", f"[{state_str:10s}]"]
                if mem_eff is not None:
                    parts.append(f"mem_eff={mem_eff:5.1f}%")
                if wall_sec is not None:
                    parts.append(f"wall={format_time(wall_sec)}")
                if cpu_eff is not None:
                    parts.append(f"cpu_eff={cpu_eff:5.1f}%")
                
                print("  " + "  ".join(parts))
            else:
                failed_calls.append((task_id, result.stderr))
                print(f"  Task {task_id:3d}: ERROR - {result.stderr.strip()[:60]}")
        
        except subprocess.TimeoutExpired:
            failed_calls.append((task_id, "timeout"))
            print(f"  Task {task_id:3d}: TIMEOUT")
        except Exception as e:
            failed_calls.append((task_id, str(e)))
            print(f"  Task {task_id:3d}: {type(e).__name__}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Memory efficiency stats
    mem_effs = [m['mem_eff_pct'] for m in results.values() if 'mem_eff_pct' in m]
    if mem_effs:
        print(f"\nMemory Efficiency (used / requested):")
        print(f"  Count: {len(mem_effs)} tasks")
        print(f"  Min:   {min(mem_effs):7.2f}%")
        print(f"  Max:   {max(mem_effs):7.2f}%")
        print(f"  Mean:  {statistics.mean(mem_effs):7.2f}%")
        if len(mem_effs) > 1:
            print(f"  Stdev: {statistics.stdev(mem_effs):7.2f}%")
    else:
        print("\nMemory Efficiency: NO DATA (cluster may not support seff)")
    
    # Wall time stats
    wall_times = [m['elapsed_sec'] for m in results.values() if 'elapsed_sec' in m]
    if wall_times:
        print(f"\nWall Time:")
        print(f"  Count: {len(wall_times)} tasks")
        print(f"  Min:   {format_time(min(wall_times))}")
        print(f"  Max:   {format_time(max(wall_times))}")
        print(f"  Mean:  {format_time(statistics.mean(wall_times))}")
        if len(wall_times) > 1:
            print(f"  Stdev: {format_time(statistics.stdev(wall_times))}")
    else:
        print("\nWall Time: NO DATA")
    
    # CPU efficiency stats
    cpu_effs = [m['cpu_eff_pct'] for m in results.values() if 'cpu_eff_pct' in m]
    if cpu_effs:
        print(f"\nCPU Efficiency:")
        print(f"  Count: {len(cpu_effs)} tasks")
        print(f"  Min:   {min(cpu_effs):7.2f}%")
        print(f"  Max:   {max(cpu_effs):7.2f}%")
        print(f"  Mean:  {statistics.mean(cpu_effs):7.2f}%")
        if len(cpu_effs) > 1:
            print(f"  Stdev: {statistics.stdev(cpu_effs):7.2f}%")
    else:
        print("\nCPU Efficiency: NO DATA")
    
    print(f"\nSummary: {len(results)} successful seff calls, {len(failed_calls)} failures")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
