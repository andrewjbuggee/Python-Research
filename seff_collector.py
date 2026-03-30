#!/usr/bin/env python3
"""
Run seff on each job array task and collect resource efficiency stats.

Usage:
    python seff_collector.py <job_array_id> [start_task] [end_task]

Example:
    python seff_collector.py 25039800 101 173

Note: Requires access to seff command (available on CU Boulder Alpine cluster)
"""

import subprocess
import re
import sys
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
        hours = int(m.group(1))
        minutes = int(m.group(2))
        seconds = int(m.group(3))
        metrics['elapsed_sec'] = hours * 3600 + minutes * 60 + seconds

    # Parse CPU efficiency
    m = re.search(r'CPU Efficiency:\s+([\d.]+)%', seff_text)
    metrics['cpu_eff_pct'] = float(m.group(1)) if m else None

    # Parse memory - requested
    m = re.search(r'Memory Requested:\s+([\d.]+)\s+(\w+)', seff_text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 'G':
            metrics['mem_req_mb'] = val * 1024
        elif unit == 'M':
            metrics['mem_req_mb'] = val
        elif unit == 'K':
            metrics['mem_req_mb'] = val / 1024

    # Parse memory - used
    m = re.search(r'Memory Used:\s+([\d.]+)\s+(\w+)', seff_text)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit == 'G':
            metrics['mem_used_mb'] = val * 1024
        elif unit == 'M':
            metrics['mem_used_mb'] = val
        elif unit == 'K':
            metrics['mem_used_mb'] = val / 1024

    # Calculate memory efficiency (seff definition: actual used / requested)
    if 'mem_req_mb' in metrics and 'mem_used_mb' in metrics:
        if metrics['mem_req_mb'] > 0:
            eff = (metrics['mem_used_mb'] / metrics['mem_req_mb']) * 100
            metrics['mem_eff_pct'] = eff

    return metrics


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def calc_mean(values):
    """Calculate mean of a list."""
    return sum(values) / len(values) if values else 0


def calc_stdev(values):
    """Calculate sample standard deviation."""
    if len(values) < 2:
        return 0
    mean = calc_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    job_array_id = sys.argv[1]
    start_task = int(sys.argv[2]) if len(sys.argv) > 2 else 101
    end_task = int(sys.argv[3]) if len(sys.argv) > 3 else 173

    results = {}
    failed_calls = []

    print("\nCollecting seff data for job array {} tasks {}-{}".format(
        job_array_id, start_task, end_task))
    print("=" * 80)

    for task_id in range(start_task, end_task + 1):
        job_id = "{}_{}".format(job_array_id, task_id)

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
                parts = ["Task {:3d}".format(task_id), "[{:10s}]".format(state_str)]
                if mem_eff is not None:
                    parts.append("mem_eff={:5.1f}%".format(mem_eff))
                if wall_sec is not None:
                    parts.append("wall={}".format(format_time(wall_sec)))
                if cpu_eff is not None:
                    parts.append("cpu_eff={:5.1f}%".format(cpu_eff))

                print("  " + "  ".join(parts))
            else:
                failed_calls.append((task_id, result.stderr))
                err_msg = result.stderr.strip()[:60]
                print("  Task {:3d}: ERROR - {}".format(task_id, err_msg))

        except subprocess.TimeoutExpired:
            failed_calls.append((task_id, "timeout"))
            print("  Task {:3d}: TIMEOUT".format(task_id))
        except Exception as e:
            failed_calls.append((task_id, str(e)))
            print("  Task {:3d}: {}".format(task_id, type(e).__name__))

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Memory efficiency stats
    mem_effs = [m['mem_eff_pct'] for m in results.values()
                if 'mem_eff_pct' in m]
    if mem_effs:
        print("\nMemory Efficiency (used / requested):")
        print("  Count: {} tasks".format(len(mem_effs)))
        print("  Min:   {:7.2f}%".format(min(mem_effs)))
        print("  Max:   {:7.2f}%".format(max(mem_effs)))
        print("  Mean:  {:7.2f}%".format(calc_mean(mem_effs)))
        if len(mem_effs) > 1:
            print("  Stdev: {:7.2f}%".format(calc_stdev(mem_effs)))
    else:
        print("\nMemory Efficiency: NO DATA (cluster may not support seff)")

    # Wall time stats
    wall_times = [m['elapsed_sec'] for m in results.values()
                  if 'elapsed_sec' in m]
    if wall_times:
        print("\nWall Time:")
        print("  Count: {} tasks".format(len(wall_times)))
        print("  Min:   {}".format(format_time(min(wall_times))))
        print("  Max:   {}".format(format_time(max(wall_times))))
        print("  Mean:  {}".format(format_time(calc_mean(wall_times))))
        if len(wall_times) > 1:
            print("  Stdev: {}".format(format_time(calc_stdev(wall_times))))
    else:
        print("\nWall Time: NO DATA")

    # CPU efficiency stats
    cpu_effs = [m['cpu_eff_pct'] for m in results.values()
                if 'cpu_eff_pct' in m]
    if cpu_effs:
        print("\nCPU Efficiency:")
        print("  Count: {} tasks".format(len(cpu_effs)))
        print("  Min:   {:7.2f}%".format(min(cpu_effs)))
        print("  Max:   {:7.2f}%".format(max(cpu_effs)))
        print("  Mean:  {:7.2f}%".format(calc_mean(cpu_effs)))
        if len(cpu_effs) > 1:
            print("  Stdev: {:7.2f}%".format(calc_stdev(cpu_effs)))
    else:
        print("\nCPU Efficiency: NO DATA")

    summary_msg = "Summary: {} successful seff calls, {} failures".format(
        len(results), len(failed_calls))
    print("\n" + summary_msg)
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
