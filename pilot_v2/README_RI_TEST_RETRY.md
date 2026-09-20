# RI test audit v2: controlled retry and diagnostics

The scientific policy and original model loader are unchanged. This release adds
diagnostics after job 899730 failed on s-003 with a zero RI denominator. The cause
is not yet established. There is no automatic fallback, epsilon adjustment,
precision change, or relaxation of score-replication tolerances.

## Changes

- Request **s-004**, **2 GPUs**, 4 CPUs and 32 GB host RAM; four-hour allocation.
  This may wait longer in the queue because other nodes cannot run it.
- Extract into `pilot_v2/ri_test_audit_v2/`; write a new `runs/ri_test_v2` directory.
  Preserve the old v1 package, manifest and failed run. Do not resume v1 using v2.
- Save host, resolved model path, GPU/driver details, package versions and the
  filesystem mount as seen on the compute node in `loading_diagnostics.json`.
- Before loading, read at most **64 MiB from the beginning of each safetensors
  shard**, timing each read. This is a bounded sample, not a complete bandwidth
  benchmark. It may warm part of the cache; the following load is therefore not
  a cold-cache measurement. No cache eviction or storage copy is performed.
- Time model loading separately, synchronize allocated CUDA devices, and save
  process I/O/memory counters before and after loading where Linux provides them.
- On zero-denominator, scoring-status, nonfinite-probability or replication failure,
  save the exact event, original scores, visible token IDs, cached probabilities,
  centered/clipped values and denominator. Then replay the same small OV batch
  for diagnostic embedding/V/Z/logit/softmax arrays. Replay never replaces scores
  or permits continuing after a failed gate. The initial evidence is saved even
  if diagnostic replay fails.

## Upload (local PowerShell in the project directory)

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts .\pilot_v2\ri_test_audit_v2_update.tar.gz "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/"
```

## Submit (Slurm Bash)

```bash
cd /home/yandex/DLWorkShop2025b/maximg/pilot_v2
tar -xzf ri_test_audit_v2_update.tar.gz
bash ri_test_audit_v2/submit_ri_test_audit.sh
```

The wrapper locates the original `stage1_v4_calibrated` run and local model in
the existing storage root. Runtime setup occurs inside the job. No reinstall,
download or manual runtime setup is necessary. Source and output paths are printed.

```bash
sacct -j JOBID --format=JobID,NodeList,State,Elapsed,ExitCode
tail -n 40 logs/ri-test-JOBID.out
tail -n 40 logs/ri-test-JOBID.err
```

If the gate fails again, inspect the new run's `loading_diagnostics.json`,
`gate_failure.json` and `gate_failure_arrays.npz`. Do not relax thresholds or submit
another identical retry without reading that evidence. The expected output root is
`/home/yandex/DLWorkShop2025b/maximg/storage/runs/ri_test_v2` (the wrapper prints it).

## Retrieve evidence or completed results

After a **failed** job has stopped, archive its output for diagnosis:

```bash
tar -czf ri_test_v2_diagnostics.tar.gz -C /home/yandex/DLWorkShop2025b/maximg/storage/runs ri_test_v2
```

Then locally:

```powershell
scp -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o HostKeyAlgorithms=ssh-ed25519 -o UserKnownHostsFile=tmp/slurm_c002_known_hosts "maximg@132.67.130.126:/home/yandex/DLWorkShop2025b/maximg/pilot_v2/ri_test_v2_diagnostics.tar.gz" .\results\
```

For a **successful** job (COMPLETED, 0:0, final COMPLETE line), use the integrity
checking pack helper instead:

```bash
bash ri_test_audit_v2/pack_ri_test_results.sh
```

It creates `ri_test_v2_results.tar.gz`. Download that name using the same scp command.
Extract locally into `results/` only when no existing `results/ri_test_v2` needs
preserving. The pack helper refuses to overwrite an existing archive.

## Scope and resumption

The job still runs CPU preparation, GPU OV completion, and CPU analysis in one
submission. All 1024 heads and 89 discovery families are retained; QK remains
>2.2, first/last scores are separate, and the full visible-context normalization
is unchanged. Candidates use the previously fixed >=20 events / >=10 families,
top10-positive union rule. No Stage-2 effects or validation families are consulted.
The GPU gate retains absolute tolerance 2e-4 and relative tolerance 2e-3.

Only interrupted **v2** work can be resumed with the same v2 command and unchanged
code/inputs. Completed head files are reused. Never run two jobs against the same
output directory. If a hard kill leaves `.running.lock`, confirm the old job has
stopped before removing only that lock. Detailed protocol remains documented in
the project's `README_RI_TEST_AUDIT.md`; its v1 command examples refer to the older
release, not this retry.

No remote job is submitted by building this archive. Local unit tests cannot
establish that changing compute node fixes the numerical failure.
