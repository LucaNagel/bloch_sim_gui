import pytest

from blochsimulator.metabolite_presets import (
    METABOLITE_CS_PPM,
    METABOLITE_PRESETS_BY_NUCLEUS,
    metabolite_presets,
)


def test_supplied_c13_chemical_shifts_are_preserved_exactly():
    assert METABOLITE_CS_PPM == {
        "pyruvate": 171.076,
        "lactate": 183.35,
        "alanine": 176.5,
        "pyruvatehydrate": 179.5,
        "fumarate": 175.4,
        "malate1": 181.7,
        "malate4": 180.5,
        "bicarbonate": 161.0,
        "urea": 163.5,
        "co2": 124.5,
        "aspartate1": 176.92,
        "aspartate4": 180.20,
    }
    c13_shifts = {
        preset.key: preset.chemical_shift_ppm for preset in metabolite_presets("C13")
    }
    supplied_c13_shifts = {key: c13_shifts[key] for key in METABOLITE_CS_PPM}
    assert supplied_c13_shifts == pytest.approx(METABOLITE_CS_PPM)


def test_metabolite_presets_are_unique_and_have_positive_relaxation():
    assert set(METABOLITE_PRESETS_BY_NUCLEUS) == {"H1", "C13", "P31"}
    assert {"naag", "glutathione", "glycine", "taurine", "2hg"}.issubset(
        {preset.key for preset in metabolite_presets("H1")}
    )
    assert {
        "alpha_ketoglutarate_c1",
        "2hg_c1",
        "dehydroascorbate_c1",
        "vitamin_c_c1",
    }.issubset({preset.key for preset in metabolite_presets("C13")})
    assert {"nad_plus", "nadh", "udpg", "pi_extracellular"}.issubset(
        {preset.key for preset in metabolite_presets("P31")}
    )
    for nucleus, presets in METABOLITE_PRESETS_BY_NUCLEUS.items():
        assert len({preset.key for preset in presets}) == len(presets)
        assert len({preset.name for preset in presets}) == len(presets)
        assert all(preset.nucleus == nucleus for preset in presets)
        assert all(preset.t1_s > 0 for preset in presets)
        assert all(preset.t2_star_s > 0 for preset in presets)
        assert all(isinstance(preset.relaxation_basis, str) for preset in presets)
        assert all(preset.source_urls for preset in presets)


def test_unsupported_nucleus_has_no_invented_metabolites():
    assert metabolite_presets("F19") == ()
    assert metabolite_presets("Na23") == ()
