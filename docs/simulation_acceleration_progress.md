# Accurate Simulation Speed-Up: Progress Report

Last updated: 28 September 2026. This document records the verifiable progress
of the [implementation task](simulation_acceleration_implementation_prompt.md).
The runtimes below were measured with the listed synthetic benchmarks on the
computer described here. They are not general performance guarantees.

## Environment and Reproducible Build

- Interpreter: `/opt/miniconda3/envs/bloch-simulator/bin/python`, Python 3.12.14.
- NumPy 2.5.1, SciPy 1.18.0, and Apple Accelerate BLAS.
- Computer: Apple M3 with eight logical CPU threads and a 10-core Apple GPU.
- The dynamic OpenMP build was tested with Apple Command Line Tools, Clang, and
  Homebrew `libomp`. `omp_get_max_threads()` reported eight threads.
- The full Xcode licence agreement had not been accepted. It was not changed. The build
  explicitly uses the installed Command Line Tools instead.
- The available Apple GPU has no native Float64 Metal backend. Therefore, this
  computer cannot be used for hardware validation of a Float64 GPU kernel.
- The benchmark JSON files record the import path, modification time, and size
  of the extension that was actually loaded.

The local extensions can be rebuilt with this command:

```bash
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
CC=/Library/Developer/CommandLineTools/usr/bin/clang \
CXX=/Library/Developer/CommandLineTools/usr/bin/clang++ \
/opt/miniconda3/envs/bloch-simulator/bin/python setup.py build_ext --inplace
```

## Phase Status

| Phase | Status | Implementation and evidence | Remaining work |
| --- | --- | --- | --- |
| 0: Baseline and measurements | Implemented and tested | The measurement files include the extension path, software versions, BLAS and OpenMP information, hardware details, warm-up runs, at least five timed repetitions, timing spread, complete error metrics, and peak resident memory. Compilation and plan-building times are reported separately. Profiles are stored for the $B_1$ benchmark. | Fine-grained native ADC and reduction timing without a profiler is not yet separated for every model path. |
| 1: Reference implementations | Implemented and tested | Separate `blochsimulator_cy` and `blochsimulator_strict_cy` extensions are available. The strict build uses `-fno-fast-math` and `-ffp-contract=off`. The public `strict_reference` option and machine-readable numerical contracts are implemented. Shared metrics cover total signal, pools, coils, final states, checkpoints, phase, and time sections. | Large real project files remain optional extra tests when they are available locally. |
| 2: $B_1$ fields | Implemented and tested | Tx sensitivity is classified exactly as unity, constant complex, or spatial. Native constant and spatial RF steps, persistent spatial RF blocks, block-size-stable scalar coefficient preparation, byte-limited simulation-local caches, and a bounded number of spatial RF groups are implemented. | The benefit still depends on the sequence pattern, number of spins, and number of threads. |
| 3: Shared execution plan | Implemented and tested | `dynamic_execution.py` provides an immutable Float64/Complex128 plan. It stores exact boundaries, RF and gradient values, $B_0$ integrals, sources, conversion state, segments, and ordered crusher, ADC, and checkpoint actions. CPU and Metal adapters use the same plan. Its SHA-256 fingerprint does not use quantisation. | Voxel-dependent tables are still created separately for each bounded execution block. |
| 4: Native driver | Implemented and tested | The specialised two-pool driver runs free half-steps, longitudinal conversion, constant or spatial Tx, crushers, ADC sampling, and checkpoints inside bounded native plan blocks. Dynamic $B_0$ uses separate exact integrals for both half-intervals. Explicit source modes cover all existing two-pool inflow and concentration combinations. A separate multi-pool driver handles the existing precursor-to-product conversions, including pyruvate to alanine, without adding overhead to the two-pool loop. Spatial arrival delays use only eight source-value planes for the current time and voxel block. Delayed conversion, multiple coils, subvoxel sampling, boundary actions, and parent-voxel chunking were tested together. | Future kinetic topologies will still need an explicit capability check. In the small ADC-heavy benchmark, the spatial no-inflow case is only about 3% faster and is therefore effectively break-even. |
| 5: Chunking and reduction | Implemented and tested | Parent voxels are split into chunks for the complete sequence, while all subspins of one parent stay together. The sequence and plan are created once. A fixed global parent-voxel order and a streaming binary reduction tree give results that do not depend on the block size or thread count. Final states and checkpoints remain global, previews use only completed object blocks, and memory-budget errors are raised early. A test with 131,074 spins exceeds the previous limit and remains on the native path. | The conservative memory plan should still be measured with more large real projects that use multiple coils and checkpoints. |
| 6: CPU scaling | In progress | Bounded LRU caches, exact cache keys, plan-block tables, separate serial and parallel choices, and reported effective thread counts are implemented. | More profiling and optimisation are needed, especially for spatial $B_1$ plan preparation and small simulations. |
| 7: Affine propagators | Open | No production or experimental backend exists yet. | This should only be investigated after broad native model coverage is complete. |
| 8: Float64 GPU | Blocked by unavailable hardware | The shared plan and interfaces are ready. | No accessible GPU provides a native Float64 compute path. A CPU fallback is not GPU validation. |
| 9: Metal | Experimental and not released | The existing Float32 and high-low experiments and their error measurements are preserved. The Metal path now uses the shared execution plan and rounds to Float32 only in the adapter. | The previous long bSSFP validation exceeded the experimental signal-NRMSE limit. A complete signal and final-state acceptance gate and renewed hardware validation are still required. |
| 10: API and GUI | Implemented and tested | The shared registry in `kernel_capabilities.py` gives the API, GUI, and kernel benchmark the same stable names, availability information, experimental status, numerical contracts, and fallback capabilities. Metadata and exports report the requested and actual kernel, effective threads, execution plan, native coverage, chunking, caches, and fallback reasons. `strict_reference` is available through both Python and the GUI. The completion message clearly shows the actual path, contract, threads, complete-plan model, and fallback. Metal remains clearly marked as experimental and safely returns the exact CPU result when necessary. | The default kernel remains unchanged. |

## Measurement Files

### Constant and Spatial $B_1$ with Coupled Polarised Inflow

The [baseline measurement](simulation_acceleration_benchmark.json) and the
[current transmit-field measurement](simulation_acceleration_b1_benchmark.json)
use 4,096
spins, 776 compiled intervals, 256 ADC values, one warm-up run, and five or
seven timed repetitions. Building the current plan takes about 10 ms in these
runs and is included in the end-to-end runtime.

| Tx case | `optimized` median | Best native median | End-to-end speed-up over `optimized` | Relative L2 signal error | Relative L2 final-pool-state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unity | 259.7 ms | 60.6 ms, 8 threads | 4.28 | $1.80 \times 10^{-15}$ | $3.40 \times 10^{-17}$ |
| Constant 0.9 | 240.6 ms | 62.5 ms, 8 threads | 3.85 | $1.88 \times 10^{-15}$ | $3.20 \times 10^{-17}$ |
| Spatial complex | 235.2 ms | 81.0 ms, 8 threads | 2.90 | $1.87 \times 10^{-15}$ | $2.63 \times 10^{-16}$ |

Compared with the older four-thread path, the current four-thread median fell
from 238.2 ms to 65.8 ms for constant 0.9 Tx and from 222.1 ms to 85.9 ms for
spatial Tx. Eight threads were slightly faster than four in this complete
driver benchmark.

### Multi-Pool Plan with Pyruvate to Lactate and Alanine

The [multi-pool measurement](simulation_acceleration_multipool_benchmark.json)
uses 4,096 voxels, three pools, 544 compiled intervals, 384 ADC values, two
checkpoints, two parent-voxel blocks, spatial complex Tx, dynamic $B_0$, delayed
concentration and polarisation inflow, and five timed repetitions.

| Kernel | Threads | Median | End-to-end speed-up over `optimized` | Relative L2 signal error | Relative L2 final-pool-state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `optimized` | 1 | 1.500 s | 1.00 | $0$ | $0$ |
| `native_serial` | 1 | 0.557 s | 2.70 | $9.13 \times 10^{-17}$ | $3.43 \times 10^{-15}$ |
| `native_parallel` | 4 | 0.453 s | 3.31 | $9.13 \times 10^{-17}$ | $3.43 \times 10^{-15}$ |
| `native_parallel` | 8 | 0.409 s | 3.66 | $9.13 \times 10^{-17}$ | $3.43 \times 10^{-15}$ |

### ADC-Heavy Complete Native Plan Without Inflow

The [native execution-plan measurement](simulation_acceleration_native_plan_benchmark.json)
uses 4,096 spins, 528 intervals, 384 ADC values, two checkpoints, five timed
repetitions, and five bounded native time blocks.

| Tx case | `optimized` median | Best native median | End-to-end speed-up | Relative L2 signal error | Relative L2 final-pool-state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unity | 94.5 ms | 71.7 ms, 8 threads | 1.32 | $1.82 \times 10^{-15}$ | $5.44 \times 10^{-16}$ |
| Constant complex | 82.8 ms | 69.4 ms, 8 threads | 1.19 | $1.78 \times 10^{-15}$ | $7.49 \times 10^{-16}$ |
| Spatial complex | 82.1 ms | 79.6 ms, 8 threads | 1.03 | $1.57 \times 10^{-15}$ | $7.66 \times 10^{-16}$ |

The spatial case is only about 3% faster than `optimized`. It remains a valid
path, but for this small workload it should be treated as break-even rather
than as a general speed improvement.

## Tests

The latest focused test groups include:

```text
tests/test_dynamic_execution_plan.py
tests/test_dynamic_chunking.py
tests/test_dynamic_native_plan.py
tests/test_dynamic_phantom.py
tests/test_sequence_simulation.py
tests/test_b1_fields.py
tests/test_memory_limits.py
tests/test_numerical_contracts.py
tests/test_kernel_capabilities.py
tests/test_gui_smoke.py
tests/test_sequence_gui.py
```

The Qt-safe full test run, `python scripts/run_test_suite.py`, passed on
28 September 2026 with `837 passed, 2 skipped, 12 subtests`. This includes the
execution-plan, chunking, dynamic $B_0$, native-driver, $B_1$, memory, and isolated
GUI tests. Another 10 benchmark tests that do not require external data files
passed separately.

Ten benchmark tests for the bundled real phantoms cannot run locally because
the following ignored files are missing from this working copy:

- `phantoms/Pyruvate Lactate 2 containers static.npz`
- `phantoms/Pyruvate to lactate with inflow 2.npz`

The synthetic benchmark fixtures do not depend on these files.

## Numerical Contracts

- `bitwise`: the established calculation order is unchanged. This statement
  applies only when the result metadata explicitly reports this contract.
- `float64-close`: used for native fusion, complete plan blocks, or fixed
  parent-voxel reduction. The measurement files report all observed
  differences, and the regression tests continue to use the previously defined
  tight limits.
- `experimental-mixed`: used only for explicitly experimental Float32 or Metal
  paths. These paths do not replace a production Float64 path.

The separate ordinary-solver option `strict_reference` reports
`strict-float64`. The existing ordinary `reference` and `optimized` variants
report `legacy-fast-math` because they use the historical build configuration.
