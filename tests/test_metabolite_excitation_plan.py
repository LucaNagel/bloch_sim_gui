import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog

from blochsimulator.metabolite_presets import metabolite_presets
from blochsimulator.sequence import (
    ScannerParameters,
    design_metabolite_selective_pulse,
    spectral_spatial_peaks,
)
from blochsimulator.ui.metabolite_excitation_plan import MetaboliteExcitationPlanDialog
from blochsimulator.ui.sequence_simulation_widget import SequenceSimulationWidget

_PLAN = dict(
    nucleus="C13",
    field_strength_t=3.0,
    sequence_reference_ppm=171.0,
    carrier_offset_hz=0.0,
    target_thickness_m=10e-3,
)


def _pyruvate_pathway_peaks(*extra_keys):
    keys = ("pyruvate", "lactate", "alanine", "pyruvatehydrate", *extra_keys)
    presets = {preset.key: preset for preset in metabolite_presets("C13")}
    return spectral_spatial_peaks(
        presets=[presets[key] for key in keys],
        field_strength_t=3.0,
        nucleus="C13",
        sequence_reference_ppm=171.0,
        rf_carrier_offset_hz=0.0,
    )


def _keep_peaks(dialog, names):
    for row in range(dialog.peak_table.rowCount()):
        keep = dialog.peak_table.item(row, 1).text() in names
        dialog.peak_table.item(row, 0).setCheckState(
            Qt.Checked if keep else Qt.Unchecked
        )


def test_metabolite_selective_design_suppresses_the_other_peaks():
    peaks = _pyruvate_pathway_peaks()

    design = design_metabolite_selective_pulse(
        "Lactate", peaks, flip_angle_deg=30.0, subpulse_duration_s=1.2e-3
    )

    assert design.meets_tolerance
    assert design.worst_leak_fraction <= 0.05
    assert design.passband_min_fraction >= 0.9
    assert design.pulse.duration_s <= 0.05
    selective = design.pulse.design_metadata["metabolite_selective"]
    assert selective["target"] == "Lactate"
    assert selective["target_peak"]["name"] == "Lactate"
    assert {peak["name"] for peak in selective["suppressed"]} == {
        "Pyruvate",
        "Alanine",
        "Pyruvate hydrate",
    }
    assert design.peak_flip_angles_deg["Lactate"] == pytest.approx(30.0, abs=1.5)


def test_metabolite_selective_design_rejects_unknown_target():
    with pytest.raises(ValueError, match="exactly one peak"):
        design_metabolite_selective_pulse("Glucose", _pyruvate_pathway_peaks())


def test_metabolite_selective_design_explains_thin_slice_limit():
    # 3 mm needs ~7 ms flyback periods at 32 mT/m: aliases hit the other peaks.
    with pytest.raises(ValueError, match="thicker slice"):
        design_metabolite_selective_pulse(
            "Lactate",
            _pyruvate_pathway_peaks("bicarbonate"),
            slice_thickness_m=3e-3,
            spatial_time_bandwidth_product=3.0,
            subpulse_duration_s=3.065e-3,
            max_gradient_t_per_m=0.032,
            max_slew_t_per_m_s=130.0,
        )


def test_plan_dialog_designs_one_pulse_per_target_and_restores(qt_application):
    dialog = MetaboliteExcitationPlanDialog(
        scanner_parameters=ScannerParameters(), **_PLAN
    )
    _keep_peaks(dialog, {"Pyruvate", "Lactate", "Bicarbonate"})
    dialog._per_metabolite_clicked()
    dialog._add_step()
    dialog.steps_table.cellWidget(3, 0).setCurrentText("Lactate")

    assert [step["target"] for step in dialog.steps()] == [
        "Pyruvate",
        "Lactate",
        "Bicarbonate",
        "Lactate",
    ]
    assert dialog.steps()[0]["flip_angle_deg"] == pytest.approx(10.0)
    assert dialog.steps()[1]["flip_angle_deg"] == pytest.approx(30.0)
    assert dialog.stale_targets() == ["Pyruvate", "Lactate", "Bicarbonate"]

    dialog._accept()
    assert dialog.result() == QDialog.Accepted
    library = dialog.pulse_library()
    schedule = dialog.pulse_schedule()
    assert len(library) == 3
    entries = schedule.entries
    assert [entry.target_metabolite for entry in entries] == [
        "Pyruvate",
        "Lactate",
        "Bicarbonate",
        "Lactate",
    ]
    assert entries[1].pulse_id == entries[3].pulse_id
    assert len({entry.pulse_id for entry in entries}) == 3
    assert entries[0].flip_angle_deg == pytest.approx(10.0)
    assert dialog.selected_pulse_id() == entries[0].pulse_id
    lactate = library.get(entries[1].pulse_id)
    assert lactate.design_metadata["metabolite_selective"]["target"] == "Lactate"
    assert dialog.overview_table.rowCount() == 4

    reopened = MetaboliteExcitationPlanDialog(
        scanner_parameters=ScannerParameters(),
        library=library,
        schedule=schedule,
        **_PLAN,
    )
    assert reopened.peak_source.currentText() == "Peaks of the current plan"
    assert [peak.name for peak in reopened.peaks()] == [
        "Pyruvate",
        "Lactate",
        "Bicarbonate",
    ]
    assert reopened.stale_targets() == []
    reopened._accept()
    assert set(reopened.pulse_library().pulses) == set(library.pulses)
    assert [entry.entry_id for entry in reopened.pulse_schedule().entries] == [
        entry.entry_id for entry in entries
    ]

    retuned = MetaboliteExcitationPlanDialog(
        scanner_parameters=ScannerParameters(),
        library=library,
        schedule=schedule,
        **{**_PLAN, "carrier_offset_hz": 25.0},
    )
    assert retuned.stale_targets() == ["Pyruvate", "Lactate", "Bicarbonate"]

    single = MetaboliteExcitationPlanDialog(
        scanner_parameters=ScannerParameters(),
        library=library,
        schedule=schedule,
        selected_pulse_id=entries[1].pulse_id,
        allow_multiple_steps=False,
        **_PLAN,
    )
    assert [step["target"] for step in single.steps()] == ["Lactate"]
    assert single.step_buttons.isHidden()
    single._accept()
    # A single-pulse sequence keeps the FLASH plan untouched.
    assert len(single.pulse_schedule().entries) == 4
    assert single.selected_pulse_id() == entries[1].pulse_id

    for widget in (dialog, reopened, retuned, single):
        widget.close()
        widget.deleteLater()
    qt_application.processEvents()


def test_flash_adopts_metabolite_plan(qt_application, monkeypatch):
    def accept_reduced_plan(dialog):
        _keep_peaks(dialog, {"Pyruvate", "Lactate"})
        dialog._per_metabolite_clicked()
        dialog._accept()
        return dialog.result()

    monkeypatch.setattr(MetaboliteExcitationPlanDialog, "exec_", accept_reduced_plan)
    widget = SequenceSimulationWidget()
    widget.nucleus.setCurrentText("C13")
    widget.field_strength_t.setValue(3.0)
    widget.sequence_reference_ppm.setValue(171.0)
    widget.flash_slice_thickness_mm.setValue(10.0)

    widget._plan_metabolite_excitation("flash")

    assert widget.flash_rf_pulse_type.currentText() == "Spectral-spatial Designer"
    assert [
        entry.target_metabolite for entry in widget._spectral_spatial_schedule.entries
    ] == ["Pyruvate", "Lactate"]
    assert len(widget._spectral_spatial_pulse_library) == 2
    summary = widget.flash_rf_spsp_summary.text()
    assert summary.startswith("Pyruvate 10° → Lactate 30° (next step per frame)")
    assert not widget.flash_rf_spsp_plan_button.isHidden()

    widget.close()
    widget.deleteLater()
    qt_application.processEvents()


def test_plan_dialog_explains_impossible_thin_slice(qt_application, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "blochsimulator.ui.metabolite_excitation_plan.QMessageBox.warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )
    dialog = MetaboliteExcitationPlanDialog(
        scanner_parameters=ScannerParameters(),
        **{**_PLAN, "target_thickness_m": 3e-3},
    )
    _keep_peaks(dialog, {"Pyruvate", "Lactate", "Bicarbonate"})
    dialog._per_metabolite_clicked()

    dialog._accept()

    assert dialog.result() != QDialog.Accepted
    assert warnings and warnings[0][0] == "Plan incomplete"
    assert "Lactate: no flyback period" in warnings[0][1]
    assert "thicker slice" in warnings[0][1]
    status = dialog.steps_table.item(1, 3)
    assert status.text().startswith("✗ no pulse possible")
    assert "thicker slice" in status.toolTip()

    dialog.close()
    dialog.deleteLater()
    qt_application.processEvents()
