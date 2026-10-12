# RF excitation volumes

The **RF + Slice Selection** source in Sequence Mode simulates one RF pulse
with its simultaneous slice-selection gradient on the full phantom volume.
The final state is taken immediately after the gradient ramp ends. There is
no ADC, spoiler, or subsequent relaxation delay.

## Desktop workflow

1. Create or load a 3D object in the **Phantom** tab and configure its transmit
   field in **B1**.
2. In **Sequence Mode**, choose **Source / mode → RF + Slice Selection**.
3. Set the RF pulse, nominal flip angle, RF phase, slice thickness, slice
   offset, and signed slice normal. Loaded RF waveforms and pulses from the
   RF Pulse Designer use the same controls as other generated sequences.
4. Optionally enable **Add slice rephasing gradient** and set its duration.
5. Generate and run the sequence. **Spatial Magnetization** opens with
   $|M_{xy}|$ selected. Choose **Mz**, **Mx**, **My**, or **Mxy phase** as needed.
6. Inspect linked orthogonal slices or switch the volume viewer to **3D**.

After running a parameter loop, use **Simulation** at the top of **Spatial
Magnetization** to select an individual run. Each entry includes the loop
parameter and value. The viewer uses that run's phantom geometry, final state,
and checkpoints. Map, observation stage, and pool selections are retained when
available across runs. Renaming or deleting a run in **Simulations** updates
this list too. These runs remain available for the current app session.

The offset is measured along the signed normal. For example, an offset of
$3\,\mathrm{mm}$ along $-X$ centres the slice at $x=-3\,\mathrm{mm}$.
The selected nucleus determines conversion to physical gradient units and
the applicable scanner gradient and slew limits. If a thin slice requires
too much gradient amplitude, increase RF duration or slice thickness.

## Observation states

With rephasing disabled, **After slice selection** is the final state. With
rephasing enabled, the state selector offers **After slice selection** and
**After slice rephasing**. The former is retained automatically as a checkpoint;
the latter is the final state. Both show their actual observation time.

The rephasing lobe rewinds the slice-gradient moment accumulated from the RF
centre to the end of slice selection, including gradient ramps. This is half
the gradient area for a symmetric pulse/gradient pair. For pulses with complex
phase or asymmetric excitation, inspect the resulting phase explicitly; this
moment prescription does not guarantee a flat excitation phase for every pulse.

The displayed quantities are local magnetization in object coordinates:

$$
M_{xy}=M_x+iM_y,\qquad |M_{xy}|=\sqrt{M_x^2+M_y^2}.
$$

The Bloch calculation includes the local transmit sensitivity, frequency
offset, relaxation, and initial magnetization. Receive sensitivity is used
for ADC signals and does not weight these magnetization volumes. Spectral
phantoms retain separate pool magnetizations, selectable using **Pool**;
the combined volume follows the simulator's concentration-weighted convention.
The phantom grid sets spatial resolution. With subvoxel sampling enabled in
gradient mode, displayed vectors are the weighted voxel averages, including
phase cancellation inside a voxel.

## Python access

```python
import numpy as np
from blochsimulator import BlochSimulator
from blochsimulator.sequence import make_pulseq_slice_selection, load_pulseq

sequence = make_pulseq_slice_selection(
    flip_angle_deg=30.0,
    rf_duration_s=5e-3,
    slice_thickness_m=3e-3,
    slice_axis="+z",
    rephase=True,
    nucleus="C13",
)
sequence.write("rf_slice_selection.seq", v141_compat=True)
program = load_pulseq("rf_slice_selection.seq")

# phantom is an existing Phantom with its spatial transmit field attached.
result = BlochSimulator().simulate_sequence(program, phantom)
M_after_rephasing = result.final_magnetization
M_after_selection = result.checkpoint_magnetization[0]
Mxy = np.hypot(M_after_selection[..., 0], M_after_selection[..., 1])
Mz = M_after_selection[..., 2]
```

For a `SpectralPhantom`, use `simulate_spectral_sequence` and set the sequence
reference consistently with the desired RF carrier. With rephasing disabled,
read the excitation state directly from `final_magnetization`.

The generated Pulseq sequence, its reproducing notebook, and complete
`.blochproj` projects preserve the excitation settings and observation times.
