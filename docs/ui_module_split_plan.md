# Plan: splitting the two largest GUI modules

`src/blochsimulator/ui/sequence_simulation_widget.py` (about 18,100 lines) and
`src/blochsimulator/ui/main_window.py` (about 10,800 lines) hold one class
each. The size makes changes slow to review, invites merge conflicts between
parallel branches and hides which methods belong together. This plan splits
both files into topic modules without changing behaviour. Line numbers refer
to the state of October 2026 and only locate the groups.

## Principles

- **Moves only.** Every step moves code verbatim. Renames, refactors and
  behaviour changes are separate commits after the move.
- **One topic per commit.** Each commit moves one method group, runs the full
  GUI test suite and is merged before the next starts. Small steps keep
  conflicts with other branches small.
- **Stable import paths.** `blochsimulator.ui.sequence_simulation_widget` and
  `blochsimulator.ui.main_window` stay the modules that define
  `SequenceSimulationWidget` and `BlochSimulatorGUI`, and re-export the names
  that other modules or tests import from them.
- **Mixins first, composition later.** Almost every method reads widgets
  created in `_build_ui`/`init_ui` through `self`. Moving method groups into
  mixin classes therefore needs no interface changes. Groups that turn out to
  be self-contained can later become controller objects with a narrow
  interface (phase 4).

## Mechanics

1. Move the group with an AST-based script that copies method source by name,
   so no line is retyped. The script deletes the methods from the old class and
   writes them, in their original order, into a mixin class:

   ```python
   class ParameterLoopMixin:
       """Parameter loop setup, iteration and saving of SequenceSimulationWidget."""

       def _start_parameter_loop(self): ...
   ```

2. Add the mixin to the class bases, before `QWidget`/`QMainWindow`:
   `class SequenceSimulationWidget(ParameterLoopMixin, ..., QWidget)`.
3. Give the new module the imports its methods use. The project's flake8
   configuration ignores undefined names and unused imports, so check both
   modules explicitly with `flake8 --select F821,F401 <old> <new>`.
4. Run `black`, that check and the full test suite, then commit.

Rules for mixins:

- No `__init__` and no `pyqtSignal` in a mixin. Signals must be class
  attributes of the QObject subclass, so they stay on the widget class.
- Class constants that a mixin uses move with it when only that mixin reads
  them; shared constants stay on the main class.
- A mixin method must not shadow a method of another mixin. The move script
  checks that every moved name is unique across the bases.

`git blame -C -C` follows moved lines across files, so history stays
traceable.

## Test coupling

The GUI tests patch names through the module path. Patches of attributes of
shared objects keep working after a move, e.g.
`sequence_simulation_widget.QMessageBox.warning` or
`sequence_simulation_widget.SequenceProbeThread.start`. Patches that rebind a
module-level name only affect code that looks the name up in that module. They
must follow the code that uses the name:

| Patched name | Used by |
|---|---|
| `main_window.ExportDataDialog` | data export (`export_results`) |
| `main_window.SettingsDialog`, `main_window.QSettings` | settings dialog, settings loaders |
| `main_window.QMessageBox` | several groups; check per step |
| `main_window.SimulationThread` | Free Mode run (`run_simulation`) |
| `sequence_simulation_widget.SequenceExportDialog` | sequence export |

When a step moves a user of one of these names, update the patch targets in
the same commit.

## SequenceSimulationWidget

New package `src/blochsimulator/ui/sequence_simulation/` next to the existing
module. Approximate sizes in lines:

| Module | Content (current location) | Lines |
|---|---|---|
| `threads.py` | `PulseqLoadThread`, `ParameterLoopSaveThread`, `SequenceSimulationThread`, `SequenceProbeThread`, payload dataclasses, animation checkpoint helpers, `_infer_sequence_acquisition` (742–1545) | 800 |
| `plot_items.py` | `_SequenceOverviewPlot`, event/RF-phase step data, timing-marker helpers (345–741) | 400 |
| `form_helpers.py` | `_left_aligned_form`, encoding-order combos, `_add_form_section`, row visibility (255–345) | 90 |
| `presets.py` | sequence preset menu, save/load/import/export (1876–2010) | 130 |
| `rf_controls.py` | RF designer link, shared RF controls, spectral-spatial pulses and schedules, inversion controls (2011–2732, 6741–7372) | 1,300 |
| `forms/<sequence>.py` | one mixin per sequence family: form builder, `_<seq>_changed`, timing settle/resolve/apply/status, labels, FOV, Pulseq parameters, `_load_<seq>` (9180–12900, scattered) | 3,700 |
| `geometry.py` | slice orientation controls, slice packages, offsets, FOV confirmation, geometry viewer refresh (7372–7573, 11585–11990, 14231–14500) | 900 |
| `parameter_loop.py` | loop parameters, preview, start, iteration, saving, finish (7573–8440) | 870 |
| `object_source.py` | object source, default/designed phantom, object summary table, `_build_phantom` (8441–9073, 14600–14690) | 700 |
| `spoiling.py` | FLASH/SS-bSSFP spoilers, spin-grid recommendation, spoiling quality, ideal-crusher reminder (10114–11210) | 1,100 |
| `frequency_reference.py` | nucleus/reference changes, Ernst angle controls (11221–11410, 13398–13410, 13548–13590) | 300 |
| `pulseq_io.py` | generated/loaded Pulseq handling, Python scripts, Pulseq export (12877–13590) | 700 |
| `program_view.py` | waveform units and ranges, program summary, `_show_program`, sequence overviews (13589–14600) | 1,000 |
| `probe.py` | spin probes: setup, run, session list, playback, vector view, export (14715–15845) | 1,100 |
| `receiver.py` | signal weighting, noise calibration, `_apply_noise_to_result` (15844–15992) | 150 |
| `run_control.py` | run checks and confirmations, `_run`, progress, cancel, finish, failure (15992–16320, 16623–16890, 17849–17915) | 900 |
| `live_preview.py` | live preview during runs (16465–16623) | 160 |
| `session_runs.py` | session run registration, snapshots, restore (16886–17210) | 330 |
| `result_views.py` | frame and spectroscopy selectors, split view, image/spectrum/volume results (17211–17849) | 650 |
| `result_export.py` | result export, Bruker defaults (17915–end) | 200 |

`_build_ui` (2733–6686, about 3,950 lines) is the largest single method. It is
split in phase 3 into builder methods per section, each placed in the mixin of
its topic: sequence source, one form builder per sequence family, run box,
and the view tabs (Sequence, Setup, Reconstruction Explorer, probe,
3D Animation). After all steps, `sequence_simulation_widget.py` keeps the class
definition, signals, `__init__`, the remaining `_build_ui` skeleton and the
public setters (`set_*`), about 800 lines.

The per-sequence form mixins have the biggest payoff: adding or changing a
sequence then touches one file instead of eight places in one module.

## BlochSimulatorGUI

New package `src/blochsimulator/ui/main_window_parts/` (the name avoids a
clash with the `main_window` module):

| Module | Content (current location) | Lines |
|---|---|---|
| `workspaces.py` | `_ensure_*_workspace`, phantom/B1/sequence connections, field and nucleus sync (1811–2324) | 510 |
| `tooltips.py` | main tab and control tooltip texts (2324–2766); the texts can become a data dict | 440 |
| `positions.py` | Free Mode position and frequency grids (2796–3087) | 300 |
| `playback.py` | time synchronisation, vector animation, live mode, B1 arrow (3483–3948, 7830–7937) | 570 |
| `free_mode_plots.py` | magnetization and signal plots and heatmaps, selectors, colormaps (3948–4441, 10383–10695) | 800 |
| `spatial_view.py` | spatial plots, profiles, slices, heatmaps (4441–5529) | 1,090 |
| `spectrum_view.py` | spectrum calculation, heatmap, 3D mode, off-resonant spins (8547–9232) | 690 |
| `menus.py` | sequence presets, `create_menu`, tutorials, about (5529–5784, 10695–10788) | 350 |
| `session_runs.py` | project explorer, session runs, session project export (5784–6088) | 300 |
| `tab_groups.py` | Setup/Results/Analysis groups, workspace switch, `set_workspace_mode` (6088–6449) | 360 |
| `app_settings.py` | `_load_*` settings readers, settings dialog, memory settings (6449–6804) | 350 |
| `free_run.py` | status/run bar, `run_simulation`, finish/error/cancel, `update_plots` (6804–7460) | 650 |
| `project_io.py` | project state, save/open project (7460–7830) | 370 |
| `data_export.py` | `export_results`, final-state and full data export, parameter collection (7937–8547) | 600 |
| `animation_export.py` | widget grabbing, image/animation/data export per view (9232–10383) | 1,150 |

`init_ui` (232–1811, about 1,580 lines) is split in phase 3 into builders for
the left panel sections and for each Free Mode tab. `main_window.py` then
keeps the class, `__init__`, the `init_ui` skeleton, `main()` and the platform
style helpers, about 600 lines.

## Phases

1. **Module-level code** (low risk): `threads.py`, `plot_items.py`,
   `form_helpers.py` and the tooltip texts. Re-export moved names from the old
   modules. About one day.
2. **Mixins** in the order of the tables, smallest and most self-contained
   first (presets, receiver, session runs, parameter loop, probe, …). One
   commit per module. About one module per hour including the test run.
3. **Builder split** of `_build_ui` and `init_ui` into per-section builders
   inside their mixins. The section order of the layout must stay the same;
   the GUI smoke tests check it, together with screenshots of both workspaces
   at 1440×900 taken before and after.
4. **Composition where it pays** (optional): parameter loop, spin probe and
   result export have narrow interfaces and can become controller objects
   that own their widgets. Decide per group after phase 2.

## Timing

The split conflicts with every open branch that edits either file. Start it
right after a release, when few branches are open, and merge each step into
`pre-release` before starting the next. Branches opened later rebase onto the
moved code. Git detects renames of whole files only, so a change to a moved
method conflicts and has to be reapplied in the new file.
