# Two issues in `identify_radiative_cases.py`

Notes for Rachel Gillespie, from a read-through and direct testing of the case
detection while porting it to ERA5 pressure-level profiles.

Both are narrow. Neither touches the phase-structure logic, which is the hard
part and which reads correctly: the A–F definitions, the purity tests, the
`elif` ordering from most-pure to least-pure, and the rule that a gap is only
bridged between two stretches of the *same* case are all consistent with the
docstring and with each other. What follows is one filter that does nothing and
one assumption about the time axis.

Each was checked by running the code, not just reading it; the reproductions are
at the bottom.

---

## 1. `filter_clear_below` is a tautology — it never rejects anything

**Where:** `detect_column()`

```python
cloud_idx   = np.where(phase_col != CLEAR)[0]
cloud_base  = int(cloud_idx.min())
clear_below = bool(np.all(phase_col[:cloud_base] == CLEAR))
```

`cloud_base` is *defined* as the first index where the phase is not `CLEAR`.
Everything below that index is therefore `CLEAR` by construction, so
`np.all(...)` is always `True`. Tested over 200,000 random phase columns:
`clear_below` was `True` in every one, and was bit-for-bit identical to "this
column contains at least one non-`CLEAR` pixel" (the `False` cases come only
from the `len(cloud_idx) == 0` early return, not from the test itself).

**Why it matters.** Three places present this as a real quality control:

- the module docstring lists "Clear sky below the cloud base" as one of four
  quality filters applied to every case;
- `detect_column()` gates on it — `if not (no_snow and no_drizzle and
  clear_below)` — which reduces to `no_snow and no_drizzle`;
- the console summary prints it as `Clear sky below cloud : {…}%`, where the
  number is actually the fraction of steps containing any cloud at all, under a
  label that says something else.

So a surface-coupled column — cloud sitting in range gate 0, fog or a
surface-based deck — passes a filter named "clear sky below cloud base". If the
intent was to exclude those (and the name suggests it was), they are currently
in the A–F sample.

**Minimal fix**, if the intent was "the cloud does not touch the lowest gate":

```python
clear_below = cloud_base > 0
```

or, for a physical height rather than a gate index, `height[cloud_base] >
MIN_CLOUD_BASE_M` with a stated threshold. Either way the reported percentage
becomes meaningful, and the summary line stops mislabelling a cloud-occurrence
statistic.

**How much would change?** Unknown without rerunning — it depends entirely on
how often the lowest gate is filled, which is an instrument-and-site question.
Worth measuring before deciding whether it is worth a rerun.

---

## 2. The sample cadence and the continuity of the time axis are assumed, not measured

**Where:** `STEP_S = 30.0` (module constant), `run()`, `extract_periods()`

`run()` reads a real time axis (`base_time + time_offset`) and then does all its
duration arithmetic from the constant instead:

```python
gap_steps = int(gap_fill / STEP_S)
min_steps = int(min_dur  / STEP_S)
…
n_total * STEP_S / 3600          # summary hours
np.sum(case_final == cv) * STEP_S / 3600   # plot_summary_table hours
```

Two consequences follow from the same root cause.

### 2a. A different cadence is silently wrong

Nothing compares `np.diff(time_offset)` against `STEP_S`. If any datastream
delivers a cadence other than 30 s, every duration, every gap-bridge window and
every hour count is off by that ratio with no warning. This matters more here
than it would in most scripts because the seasons deliberately span two
datastreams (c0 and c1) that already differ in their variable names — `read_var`
exists precisely to paper over c0/c1 differences, so the possibility of them
differing in other ways is live.

### 2b. Events can span missing days, and the two hour-counts then disagree

`extract_periods()` ends a period only when the qualifying mask goes `False` or
the case changes. A jump in the time axis ends nothing. Because `run()`
concatenates daily files with `all_time.extend(times)` and never checks that
consecutive files are consecutive days, two qualifying stretches separated by
absent files are joined into a single period — and its duration is then computed
from wall-clock timestamps:

```python
dur = (p_end - p_start + step_s) / 60.0     # minutes
```

Reproduced directly: one fully qualifying day, three days of files missing, then
another fully qualifying day.

| | |
|---|---|
| periods returned | **1** (expected 2) |
| `duration_min` in the CSV | 7,200 min = **120 h** |
| `n_30s_steps` in the same row | 5,760 = **48 h** of real samples |
| hours with no data behind the number | **72 h** |

The same run also makes the two hour conventions disagree, because they are
computed differently:

- `plot_summary_table` "Hours in periods ≥ 30 min" counts *samples* ×
  `STEP_S` → **48 h**
- the CSV `duration_min` column sums *wall clock* → **120 h**

The summary table then mixes them within one row: its hours columns come from
sample counts while its "Avg event length (min)" column comes from those
wall-clock durations. There is also a knock-on effect on `apply_min_duration()`,
which counts samples — so an event straddling a gap can clear the 30-minute bar
on samples drawn from two different weeks.

Missing days are not hypothetical for this pipeline; the module docstring's own
"Adding a new season" instructions cover symlinking and copying files in, and
warn that an interrupted copy can leave a truncated file behind.

**Minimal fix.** Derive the step from the data and break periods on any jump:

```python
# in run(), after building all_time
steps = np.diff(all_time)
step_s = float(np.median(steps))
if abs(step_s - STEP_S) > 0.5:
    print(f"warning: cadence is {step_s:.1f} s, not {STEP_S:.1f} s")
```

and in `extract_periods()`, add a discontinuity to the period-break condition:

```python
elif in_p and (not q
               or case_arr[i] != case_arr[i - 1]
               or (time_arr[i] - time_arr[i - 1]) > 1.5 * step_s):
```

That splits an event at a data gap instead of bridging it, which also makes
`duration_min` and `n_30s_steps × STEP_S` agree again by construction.

**How much would change?** Only for seasons with absent or truncated daily
files. Checking is cheap — `np.diff(all_time)` and look for anything much larger
than 30 s — and worth doing before the durations are used for anything.

---

## Reproducing

Both were confirmed by executing the relevant functions copied verbatim out of
the script:

1. `clear_below_as_written()` over 200,000 randomly generated phase columns
   (all eight phase codes, 60 gates), compared against "has any non-`CLEAR`
   pixel" — identical arrays.
2. `extract_periods()` given a two-day qualifying mask on a time axis with a
   three-day hole, printing the returned period count, `duration_min` and
   `n_30s_steps`.

---

## Not in scope of these notes

Two other things came up while porting and are **not** problems with this
script — flagging them only so they are not mistaken for criticism of it:

- **Pixel-count purity does not survive the port.** The 10% rule assumes ~500
  ARM range gates. On ERA5's 23 pressure levels a typical cloud occupies 3–8, so
  10% of 8 rounds to zero and cases B/C/F collapse. That is a property of the
  coarse grid, not of this code; the port switched to mass-weighted fractions.
- **An inverted level axis** in the ERA5 port (`cloud_vertical_cases.py`) makes
  its "liquid-topped" tests read the cloud base instead. That bug is in the port,
  written on this end. It is not in `identify_radiative_cases.py`, whose phase
  columns are indexed bottom-up correctly throughout.
