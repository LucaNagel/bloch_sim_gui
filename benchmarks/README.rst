Bloch-simulator benchmarks
==========================

The benchmark runner generates representative Pulseq sequences, loads the
selected phantom files, measures wall-clock simulation time, and saves both
CSV and JSON results. The default aliases point to:

* ``static``: ``phantoms/Pyruvate Lactate 2 containers static.npz``
* ``dynamic``: ``phantoms/Pyruvate to lactate with inflow 2.npz``

What the benchmarks are for
---------------------------

Each benchmark answers a different question:

* ``sequences``: How does runtime change between representative acquisition
  families and between static and dynamic phantom models?
* ``crushers``: How closely does a sampled physical crusher-gradient waveform
  reproduce ideal transverse spoiling, and how many subvoxel spins are needed?
* ``resolution``: At what phantom grid does the received signal and reconstructed
  image stop changing materially for a fixed 2D FLASH acquisition?
* ``kernels``: Which numerical implementation is fastest for the same physical
  model, and does it remain sufficiently close to its reference implementation?

These scripts are intended for reproducible comparisons, regression checks,
and workload sizing. They do not define a universally sufficient image-quality
or accuracy threshold. Choose an acceptable error or similarity threshold for
the scientific question, then select the cheapest configuration that meets it.

Recommended workflow
--------------------

#. Start with ``--profile quick`` and one repeat to verify the selected phantom,
   sequence parameters, kernels, and output directory printed to the console.
#. Use explicit matrices, time step, subvoxel grid, and thread count to match the
   intended production simulation.
#. Use ``--repeats 3`` or more for timing conclusions. The first invocation can
   include one-time library, allocation, or cache effects; compare repeated rows
   rather than relying on one short measurement.
#. Store important runs with ``--output-dir PATH`` and compare the resulting CSV
   files. JSON retains detailed metadata for scripted analysis.
#. Keep hardware load and software settings constant when comparing runtimes.

While it runs, every benchmark prints the selected global configuration,
phantom properties, generated sequence parameters, and the exact simulation
case before entering the simulator. Completion lines include timing and
workload statistics; crusher and resolution runs additionally print their
similarity to the relevant ideal or finest-resolution reference. Output is
flushed immediately, so progress remains visible during long simulations.

SS-bSSFP is always generated with physical scanner **Z as the read axis** and
the right-handed encoding frame ``(+z, +x, +y)`` for read, phase, and
partition.

Select a benchmark
------------------

Run these commands from the repository root::

    python -m benchmarks.run_benchmarks --list
    python -m benchmarks.run_benchmarks --benchmark sequences
    python -m benchmarks.run_benchmarks --benchmark crushers
    python -m benchmarks.run_benchmarks --benchmark resolution
    python -m benchmarks.run_benchmarks --benchmark kernels
    python -m benchmarks.run_benchmarks --benchmark all

The sequence benchmark can be narrowed to individual sequence/phantom cases::

    python -m benchmarks.run_benchmarks \
      --benchmark sequences \
      --sequences ss_bssfp flash csi \
      --phantoms static dynamic

The sub-benchmarks are also directly executable::

    python -m benchmarks.benchmark_sequences --sequences ss_bssfp me_bssfp
    python -m benchmarks.benchmark_crushers --gradient-z-spins 1 5 9
    python -m benchmarks.benchmark_resolution --resolution-scales 0.25 0.5 1 2
    python -m benchmarks.benchmark_kernels
    python -m benchmarks.benchmark_flash_spoiler_train --quick

``benchmark_flash_spoiler_train`` is a separate, deliberately long diagnostic
suite modeled on the 32x32x64, 32 mm FOV FLASH debug project. Its full default
sweeps 32/64/128 acquisition matrices, 0.5/1/2 effective crusher cycles per
phantom voxel, and several regular and deterministic-stratified subvoxel grids.
It compares each physical-gradient result with an ideal-crusher reference and
records signal/image NRMSE, runtime, final transverse magnetization, and the
train-wide coherence error calculated from the actual ADC moment origins. Run
without ``--quick`` when making convergence decisions; the default may take
tens of minutes.

The default ``quick`` profile is intended to complete without editing the
scripts. ``--profile full`` increases the generated acquisition matrices. Use
an explicit ``--output-dir`` for stable paths; otherwise a timestamped
directory is created below ``exports/benchmarks/``.

Common controls
---------------

The following options apply to every benchmark group:

* ``--phantoms`` accepts the ``static`` and ``dynamic`` aliases or explicit
  ``.npz`` paths;
* ``--timestep-us`` sets the maximum integration interval while RF is active;
* ``--threads 0`` uses the simulator default, while a positive value fixes the
  requested CPU thread count;
* ``--sequence-kernel`` and ``--dynamic-kernel`` select the production kernel
  for every benchmark except the dedicated kernel sweep;
* ``--repeats`` records independent timing rows for each case;
* ``--output-dir`` selects the folder containing generated sequences, generated
  phantoms where applicable, CSV files, and JSON files.

Run ``python -m benchmarks.run_benchmarks --help`` or append ``--help`` to an
individual benchmark command to see every available option and an embedded
summary of that benchmark's goal.

How to read the reported metrics
--------------------------------

``simulation_time_s`` wraps the simulator call only. Sequence setup/loading and
resolution-phantom generation have separate fields. ``relative_l2_error`` is
zero for an identical candidate. ``l2_similarity`` is
``1 / (1 + relative_l2_error)`` and is therefore one for an identical candidate.
``complex_correlation`` measures normalized complex alignment and can remain
high even when signal amplitude differs, so it should not be interpreted alone.

For kernel runs, ``speedup_vs_reference`` above one means the candidate was
faster. Check ``actual_kernel``, ``fallback_used``, and ``fallback_reason``
before using a row for performance conclusions. A correct fallback result can
have excellent similarity while not measuring the requested accelerator.

Spectral components versus spectral points
-------------------------------------------

``Pools/metabolites`` reports the number of physical spectral components, not
the number of frequency samples. The bundled static phantom has two component
centres: pyruvate at 0 Hz and lactate at approximately 925 Hz relative to the
scanner reference. Each independent component carries its own centre frequency,
T1, and T2/T2* behavior; the static spectral simulator runs those components
independently and sums their received signals.

The phantom fields ``spectral_reference_ppm``, ``spectral_bandwidth_ppm``, and
``spectral_points`` define its spectrum-preview/display grid. For the bundled
phantoms this is 1,024 points across 30 ppm around a 171 ppm reference. These
1,024 display points do not create 1,024 Bloch states and do not change a FLASH
acquisition's ADC count.

FLASH has spatial read and phase samples but no acquired spectral dimension.
Its component-resolved ``species_signal`` is simulator ground truth, not a
spectrum that FLASH could independently separate. CSI is different: its
sequence ``SpectralPoints`` and ``SpectralBandwidth`` define actual temporal FID
sampling, with nominal frequency spacing ``bandwidth / points``. The console
therefore labels the phantom grid explicitly as ``not FLASH ADC`` and prints CSI
spectral acquisition settings separately with the generated sequence.

Crusher comparison
------------------

The crusher benchmark generates two SS-bSSFP volumes so that the first
end-of-volume crusher can affect the second acquisition. Every physical
gradient-waveform run uses X/Y/Z midpoint-spin counts ``(1, 1, N)``, while the
ideal reference uses ``(1, 1, 1)``. It reports:

* wall-clock time and runtime relative to the ideal crusher;
* relative L2 error (zero is best);
* L2 similarity and complex correlation (one is best);
* the same comparison restricted to ADC samples after the first crusher;
* final transverse-magnetization and pool-resolved signal comparisons.

Crusher strength and Z sampling can be swept explicitly::

    python -m benchmarks.benchmark_crushers \
      --crusher-cycles 0.5 1 2 \
      --gradient-z-spins 1 3 5 9 \
      --phantoms static dynamic

For gradient-waveform mode, higher subvoxel counts cost proportionally more
simulation work. A one-point grid cannot represent intravoxel dephasing and is
included deliberately as an unresolved baseline.

Phantom-resolution comparison
-----------------------------

The resolution benchmark keeps the complete 3D phantom and conservatively
area-averages every Z plane onto increasingly fine X/Y grids. The number of Z
planes, complete 3D field of view, FLASH-selected slice position, slice
thickness, FLASH acquisition matrix, and sequence parameters remain fixed.
Every voxel in the full object remains part of the simulation, while the 2D
FLASH RF pulse selects the requested physical Z slice. Signal uses explicit
voxel-volume weighting, so adding voxels does not artificially multiply the
received signal.

The phantom grid is independent of the fixed FLASH acquisition matrix. By
default, every case uses a ``16 x 16`` FLASH matrix and every Z plane is
resampled in X/Y at scales ``0.25 0.5 1 2``. The final case therefore has twice
as many phantom samples along each in-plane direction as the source and can be
much finer than the acquisition matrix. Absolute X/Y grids can be selected
directly, including grids larger than both the acquisition matrix and the
source phantom::

    python -m benchmarks.benchmark_resolution \
      --flash-matrix 16 16 \
      --phantom-matrices 16x16 32x32 64x64 128x128

``--phantom-matrices`` may also be combined with ``--resolution-scales``.
Every result records the X/Y phantom-to-acquisition ratios and whether the
phantom grid is coarser than, equal to, or finer than the acquisition matrix.
Upsampling beyond the source grid refines the simulator's spatial
discretization of the piecewise-constant source; it does not introduce new
anatomical detail.

The static phantom is acquired once. For a dynamic phantom, FLASH contains
four labelled repetitions one second apart by default. The complete sequence
therefore runs for four seconds while kPL conversion and pyruvate inflow keep
evolving continuously. Change this temporal sampling with::

    python -m benchmarks.benchmark_resolution \
      --dynamic-frames 6 \
      --dynamic-frame-interval-s 0.5

Each generated full-volume resolution phantom is saved beside the generated
sequence.
CSV/JSON output reports runtime, active/total voxels, voxel dimensions,
frame-wise and pool-wise signal norms, temporal signal change, and signal/image
similarity to the finest requested phantom grid.

Kernel comparison
-----------------

The kernel benchmark separates the two independent execution-engine settings:

* the **Sequence Bloch kernel** propagates the ordinary Bloch equation. Static
  spectral metabolites are simulated independently and their signals are
  summed;
* the **Dynamic two-pool kernel** retains coupled pyruvate/lactate states and
  additionally integrates kPL conversion, polarization, inflow, and dynamic
  B0 when present.

By default it compares ``reference`` and ``optimized`` for the ordinary
Sequence Bloch path. For the dynamic phantom it compares ``reference``,
``optimized``, ``native_serial``, ``native_parallel``, and ``metal_hybrid``.
Each candidate uses the same phantom, FLASH acquisition, time step, spin grid,
and ideal-crusher model as its family reference. The console explains every
kernel before the simulation and reports runtime, speed-up, signal similarity,
final-state similarity, the actually used kernel, and explicit fallbacks.

The default comparison resamples every source Z plane to a ``24x24`` X/Y grid,
retaining the complete 3D phantom, and uses a ``16x16`` FLASH matrix. Together
with the common ``1x1x9`` dynamic subvoxel grid, this normally exceeds the
1,024-spin threshold at which ``native_parallel`` enables multiple workers.
Metal is experimental and platform-dependent: when the device is unavailable
or validation fails, the row is retained as a tested exact-CPU fallback but is
marked as invalid for Metal performance conclusions.

Selections and workload can be changed explicitly::

    python -m benchmarks.benchmark_kernels \
      --sequence-kernels reference optimized \
      --dynamic-kernels reference optimized native_serial native_parallel \
      --kernel-phantom-matrix 32x32 \
      --kernel-flash-matrix 16 16 \
      --kernel-subvoxel-spins 1 1 9 \
      --repeats 3

CSV/JSON output includes requested and actual kernels, fallback reasons,
runtime and speed-up relative to ``reference``, signal and state errors, native
block capability flags, effective native thread counts, and Metal validation
status.
