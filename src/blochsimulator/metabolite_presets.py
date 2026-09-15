"""Curated, editable spectral-peak presets for the phantom designer.

Relaxation depends strongly on field strength, tissue, temperature, labelling,
and acquisition.  The values below are therefore starting points rather than
universal constants.  The phantom model needs apparent ``T2*``; where a direct
measurement is unavailable, the preset note says that an estimate is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Dict, Tuple


H1_RELAXATION_SOURCE = "https://doi.org/10.4172/2155-9937.S1-002"
H1_EXTENDED_RELAXATION_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5545076/"
H1_COMMON_METABOLITES_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC2843548/"
H1_CHEMICAL_SHIFT_SOURCE = "https://pubmed.ncbi.nlm.nih.gov/10861994/"
C13_T1_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7484340/"
C13_IN_VIVO_T1_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8776582/"
C13_T2_STAR_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10872504/"
C13_UREA_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4011557/"
C13_FUMARATE_SOURCE = "https://doi.org/10.1021/jacs.9b10094"
C13_ASPARTATE_SOURCE = "https://doi.org/10.3389/fphys.2021.792769"
C13_ALPHA_KETOGLUTARATE_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3908661/"
C13_REDOX_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC3219134/"
P31_CHEMICAL_SHIFT_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4438275/"
P31_RELAXATION_SOURCE = "https://doi.org/10.1002/nbm.4169"
P31_NAD_SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4772768/"


# Keep these values exactly as supplied for the hyperpolarized 13C workflow.
METABOLITE_CS_PPM = {
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


def _linewidth_t2_star(linewidth_hz: float) -> float:
    """Return Lorentzian T2* from full-width at half maximum."""

    return 1.0 / (pi * linewidth_hz)


@dataclass(frozen=True)
class MetabolitePreset:
    """One nucleus-specific peak preset shown in the phantom designer."""

    key: str
    nucleus: str
    name: str
    chemical_shift_ppm: float
    t1_s: float
    t2_star_s: float
    relaxation_basis: str
    source_urls: Tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.name} — {self.chemical_shift_ppm:g} ppm"

    @property
    def relaxation_summary(self) -> str:
        return (
            f"T1 {self.t1_s:g} s · T2* {self.t2_star_s * 1000:.3g} ms. "
            f"{self.relaxation_basis}"
        )


def _preset(
    key,
    nucleus,
    name,
    chemical_shift_ppm,
    t1_s,
    t2_star_s,
    relaxation_basis,
    *source_urls,
):
    return MetabolitePreset(
        key=key,
        nucleus=nucleus,
        name=name,
        chemical_shift_ppm=float(chemical_shift_ppm),
        t1_s=float(t1_s),
        t2_star_s=float(t2_star_s),
        relaxation_basis=relaxation_basis,
        source_urls=tuple(source_urls),
    )


_H1_PRESETS = (
    _preset(
        "naa",
        "H1",
        "NAA",
        2.01,
        1.38,
        _linewidth_t2_star(5.28),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        H1_RELAXATION_SOURCE,
    ),
    _preset(
        "creatine",
        "H1",
        "Creatine",
        3.03,
        1.38,
        _linewidth_t2_star(4.06),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        H1_RELAXATION_SOURCE,
    ),
    _preset(
        "choline",
        "H1",
        "Choline",
        3.22,
        1.06,
        _linewidth_t2_star(4.30),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        H1_RELAXATION_SOURCE,
    ),
    _preset(
        "glutamate",
        "H1",
        "Glutamate",
        2.35,
        1.41,
        0.052,
        "Representative human-brain 3 T T1; T2* is a 3 T linewidth estimate.",
        H1_EXTENDED_RELAXATION_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "glutamine",
        "H1",
        "Glutamine",
        2.45,
        1.04,
        0.052,
        "Representative human-brain 3 T T1; T2* is a 3 T linewidth estimate.",
        H1_EXTENDED_RELAXATION_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "myoinositol",
        "H1",
        "myo-Inositol",
        3.56,
        1.20,
        0.052,
        "Representative human-brain 3 T T1; T2* is a 3 T linewidth estimate.",
        H1_RELAXATION_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "gaba",
        "H1",
        "GABA",
        3.01,
        1.30,
        0.052,
        "Common 1H-MRS peak; relaxation values are editable 3 T estimates.",
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "lactate",
        "H1",
        "Lactate",
        1.33,
        1.50,
        0.052,
        "Common 1H-MRS peak; relaxation values are editable 3 T estimates.",
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "naag",
        "H1",
        "NAAG",
        2.04,
        1.38,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "alanine",
        "H1",
        "Alanine",
        1.48,
        1.30,
        0.052,
        "Common short-TE MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
    ),
    _preset(
        "aspartate",
        "H1",
        "Aspartate",
        2.82,
        1.30,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "ascorbate",
        "H1",
        "Ascorbate",
        3.73,
        1.30,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "glucose",
        "H1",
        "Glucose",
        3.43,
        1.20,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "glutathione",
        "H1",
        "Glutathione",
        2.95,
        1.30,
        0.052,
        "Often measured with spectral editing; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "glycine",
        "H1",
        "Glycine",
        3.55,
        1.30,
        0.052,
        "Common singlet overlapping myo-inositol; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "gpc",
        "H1",
        "Glycerophosphocholine",
        3.23,
        1.06,
        0.052,
        "Total-choline component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "phosphocholine",
        "H1",
        "Phosphocholine",
        3.22,
        1.06,
        0.052,
        "Total-choline component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "phosphoethanolamine",
        "H1",
        "Phosphoethanolamine",
        3.98,
        1.30,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "scylloinositol",
        "H1",
        "scyllo-Inositol",
        3.35,
        1.20,
        0.052,
        "Brain-MRS singlet; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "taurine",
        "H1",
        "Taurine",
        3.42,
        1.30,
        0.052,
        "Common short-TE brain-MRS component; relaxation values are editable 3 T estimates.",
        H1_CHEMICAL_SHIFT_SOURCE,
        H1_COMMON_METABOLITES_SOURCE,
    ),
    _preset(
        "2hg",
        "H1",
        "2-Hydroxyglutarate",
        2.25,
        1.30,
        0.052,
        "IDH-mutant glioma marker; relaxation values are editable 3 T estimates.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4704134/",
    ),
)


_C13_PRESETS = (
    _preset(
        "pyruvate",
        "C13",
        "Pyruvate",
        METABOLITE_CS_PPM["pyruvate"],
        30.0,
        0.07556,
        "Human 3 T kinetic-model T1; early-time human-brain T2*.",
        C13_T1_SOURCE,
        C13_T2_STAR_SOURCE,
    ),
    _preset(
        "lactate",
        "C13",
        "Lactate",
        METABOLITE_CS_PPM["lactate"],
        24.0,
        0.030,
        "Human in-vivo 3 T T1; human-brain T2*.",
        C13_IN_VIVO_T1_SOURCE,
        C13_T2_STAR_SOURCE,
    ),
    _preset(
        "alanine",
        "C13",
        "Alanine",
        METABOLITE_CS_PPM["alanine"],
        25.0,
        0.030,
        "Editable hyperpolarized 13C starting estimate.",
        C13_T1_SOURCE,
    ),
    _preset(
        "pyruvatehydrate",
        "C13",
        "Pyruvate hydrate",
        METABOLITE_CS_PPM["pyruvatehydrate"],
        30.0,
        0.030,
        "Editable hyperpolarized 13C starting estimate.",
        C13_T1_SOURCE,
    ),
    _preset(
        "fumarate",
        "C13",
        "Fumarate",
        METABOLITE_CS_PPM["fumarate"],
        28.0,
        0.050,
        "Reported hyperpolarized fumarate T1; T2* is an editable estimate.",
        C13_FUMARATE_SOURCE,
    ),
    _preset(
        "malate1",
        "C13",
        "Malate C1",
        METABOLITE_CS_PPM["malate1"],
        28.0,
        0.050,
        "Editable estimate for a fumarate-to-malate experiment.",
        C13_FUMARATE_SOURCE,
    ),
    _preset(
        "malate4",
        "C13",
        "Malate C4",
        METABOLITE_CS_PPM["malate4"],
        28.0,
        0.050,
        "Editable estimate for a fumarate-to-malate experiment.",
        C13_FUMARATE_SOURCE,
    ),
    _preset(
        "bicarbonate",
        "C13",
        "Bicarbonate",
        METABOLITE_CS_PPM["bicarbonate"],
        25.5,
        0.10817,
        "Human in-vivo 3 T T1 and human-brain T2*.",
        C13_IN_VIVO_T1_SOURCE,
        C13_T2_STAR_SOURCE,
    ),
    _preset(
        "urea",
        "C13",
        "Urea",
        METABOLITE_CS_PPM["urea"],
        46.0,
        0.090,
        "[13C]urea in water at 3 T and 37 °C; reported T2 is used as an upper-bound T2* start.",
        C13_UREA_SOURCE,
    ),
    _preset(
        "co2",
        "C13",
        "CO2",
        METABOLITE_CS_PPM["co2"],
        44.7,
        0.050,
        "Reported solution T1 at 11.7 T; T2* is an editable estimate.",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC2885774/",
    ),
    _preset(
        "aspartate1",
        "C13",
        "Aspartate C1",
        METABOLITE_CS_PPM["aspartate1"],
        25.0,
        0.030,
        "Editable hyperpolarized 13C starting estimate.",
        C13_ASPARTATE_SOURCE,
    ),
    _preset(
        "aspartate4",
        "C13",
        "Aspartate C4",
        METABOLITE_CS_PPM["aspartate4"],
        25.0,
        0.030,
        "Editable hyperpolarized 13C starting estimate.",
        C13_ASPARTATE_SOURCE,
    ),
    _preset(
        "alpha_ketoglutarate_c1",
        "C13",
        "alpha-Ketoglutarate C1",
        172.6,
        52.0,
        0.050,
        "Solution T1 at 3 T and 37 °C; T2* is an editable estimate.",
        C13_ALPHA_KETOGLUTARATE_SOURCE,
    ),
    _preset(
        "alpha_ketoglutarate_hydrate",
        "C13",
        "alpha-Ketoglutarate hydrate",
        180.9,
        54.0,
        0.050,
        "Solution T1 at 3 T and 37 °C; T2* is an editable estimate.",
        C13_ALPHA_KETOGLUTARATE_SOURCE,
    ),
    _preset(
        "2hg_c1",
        "C13",
        "2-Hydroxyglutarate C1",
        183.9,
        26.0,
        0.050,
        "Single-sample solution T1 at 3 T; T2* is an editable estimate.",
        C13_ALPHA_KETOGLUTARATE_SOURCE,
    ),
    _preset(
        "dehydroascorbate_c1",
        "C13",
        "Dehydroascorbate C1",
        174.0,
        56.5,
        0.050,
        "Solution T1 at 3 T and 37 °C; T2* is an editable estimate.",
        C13_REDOX_SOURCE,
    ),
    _preset(
        "vitamin_c_c1",
        "C13",
        "Vitamin C C1",
        177.8,
        29.2,
        0.050,
        "Buffered-solution T1 at 3 T; T2* is an editable estimate.",
        C13_REDOX_SOURCE,
    ),
)


_P31_PRESETS = (
    _preset(
        "pcr",
        "P31",
        "Phosphocreatine",
        0.0,
        2.66,
        _linewidth_t2_star(7.3),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "pi",
        "P31",
        "Inorganic phosphate",
        4.84,
        1.84,
        _linewidth_t2_star(15.5),
        "Intracellular human-brain 3 T T1; T2* is calculated from linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "pe",
        "P31",
        "Phosphoethanolamine",
        6.77,
        3.42,
        _linewidth_t2_star(22.8),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "pc",
        "P31",
        "Phosphocholine",
        6.23,
        3.42,
        _linewidth_t2_star(22.8),
        "PE-based editable 3 T relaxation estimate.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "gpe",
        "P31",
        "Glycerophosphoethanolamine",
        3.49,
        3.44,
        _linewidth_t2_star(22.6),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "gpc",
        "P31",
        "Glycerophosphocholine",
        2.94,
        2.72,
        _linewidth_t2_star(22.6),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "gamma_atp",
        "P31",
        "gamma-ATP",
        -2.53,
        0.80,
        _linewidth_t2_star(19.3),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "alpha_atp",
        "P31",
        "alpha-ATP",
        -7.56,
        0.88,
        _linewidth_t2_star(17.2),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "beta_atp",
        "P31",
        "beta-ATP",
        -16.18,
        0.89,
        _linewidth_t2_star(30.1),
        "Human brain at 3 T; T2* is calculated from the reported linewidth.",
        P31_CHEMICAL_SHIFT_SOURCE,
        P31_RELAXATION_SOURCE,
    ),
    _preset(
        "pi_extracellular",
        "P31",
        "Extracellular inorganic phosphate",
        5.24,
        3.82,
        _linewidth_t2_star(15.5),
        "Human-brain 3 T T1; Pi linewidth is used for T2*.",
        P31_RELAXATION_SOURCE,
        P31_NAD_SOURCE,
    ),
    _preset(
        "nad_plus",
        "P31",
        "NAD+",
        -8.31,
        2.07,
        _linewidth_t2_star(127.2),
        "Pooled NAD relaxation/linewidth at 7 T; represented as one editable peak.",
        P31_NAD_SOURCE,
    ),
    _preset(
        "nadh",
        "P31",
        "NADH",
        -8.13,
        2.07,
        _linewidth_t2_star(127.2),
        "Pooled NAD relaxation/linewidth at 7 T; represented as one editable peak.",
        P31_NAD_SOURCE,
    ),
    _preset(
        "udpg",
        "P31",
        "UDP-glucose",
        -9.72,
        2.95,
        _linewidth_t2_star(101.7),
        "Human-brain measurement at 7 T; represented as one editable peak.",
        P31_NAD_SOURCE,
    ),
    _preset(
        "membrane_phospholipids",
        "P31",
        "Membrane phospholipids",
        2.30,
        2.50,
        0.014,
        "Common broad 31P component; relaxation values are editable estimates.",
        P31_CHEMICAL_SHIFT_SOURCE,
    ),
)


METABOLITE_PRESETS_BY_NUCLEUS: Dict[str, Tuple[MetabolitePreset, ...]] = {
    "H1": _H1_PRESETS,
    "C13": _C13_PRESETS,
    "P31": _P31_PRESETS,
}


def metabolite_presets(nucleus: str) -> Tuple[MetabolitePreset, ...]:
    """Return curated presets for *nucleus*, or an empty tuple."""

    return METABOLITE_PRESETS_BY_NUCLEUS.get(nucleus, ())
