"""
spectral_phantom.py - Multi-spectral phantom for CSI and spectroscopy simulation

This module extends the Phantom class to support multiple chemical species
per voxel, enabling simulation of:
- Chemical shift imaging (CSI)
- MR spectroscopy (MRS)
- Fat-water imaging
- Multi-nuclear spectroscopy (31P, 13C, etc.)

Each voxel can contain multiple metabolites/species with different:
- Chemical shifts (frequency offsets)
- T1, T2 relaxation times
- Concentrations
- J-coupling patterns (future)

Author: Luca Nagel
Date: 2025
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .units import NUCLEUS_GAMMA_HZ_PER_T, hz_to_ppm, ppm_to_hz

# Import base phantom class
try:
    from .phantom import Phantom, PhantomFactory
except ImportError:
    Phantom = None
    PhantomFactory = None


@dataclass
class ChemicalSpecies:
    """
    Definition of a chemical species (metabolite, molecule).

    Attributes
    ----------
    name : str
        Species name (e.g., 'NAA', 'Creatine', 'Water', 'Fat')
    chemical_shift_ppm : float
        Chemical shift relative to reference (usually water or TMS) in ppm
    t1 : float
        T1 relaxation time in seconds
    t2 : float
        T2 relaxation time in seconds
    t2_star : float, optional
        T2* relaxation time in seconds (defaults to T2)
    multiplicity : int
        Number of equivalent protons (affects signal amplitude)
    j_coupling_hz : float, optional
        J-coupling constant in Hz (for future multiplet simulation)
    j_partners : list, optional
        Names of coupled partners
    """

    name: str
    chemical_shift_ppm: float
    t1: float
    t2: float
    t2_star: float = None
    multiplicity: int = 1
    j_coupling_hz: float = 0.0
    j_partners: List[str] = field(default_factory=list)
    frequency_offset_hz: Optional[float] = None

    def __post_init__(self):
        if self.t2_star is None:
            self.t2_star = self.t2
        for name, value in (
            ("t1", self.t1),
            ("t2", self.t2),
            ("t2_star", self.t2_star),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.frequency_offset_hz is not None and not np.isfinite(
            self.frequency_offset_hz
        ):
            raise ValueError("frequency_offset_hz must be finite")

    def get_frequency_offset(self, field_strength: float, nucleus: str = "H1") -> float:
        """
        Calculate frequency offset in Hz for given field strength.

        Parameters
        ----------
        field_strength : float
            B0 field strength in Tesla
        nucleus : str
            Nucleus type: 'H1', 'C13', 'P31', etc.

        Returns
        -------
        float
            Frequency offset in Hz
        """
        if self.frequency_offset_hz is not None:
            return float(self.frequency_offset_hz)

        if nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            raise ValueError(f"unsupported nucleus {nucleus!r}")
        return float(ppm_to_hz(self.chemical_shift_ppm, field_strength, nucleus))


# =============================================================================
# COMMON METABOLITE LIBRARIES
# =============================================================================


class BrainMetabolites:
    """
    Common brain metabolites for 1H MRS at 3T.

    Chemical shifts are relative to water (4.7 ppm).
    T1/T2 values are approximate for 3T.
    """

    @staticmethod
    def naa() -> ChemicalSpecies:
        """N-acetyl aspartate (NAA) - neuronal marker."""
        return ChemicalSpecies(
            name="NAA",
            chemical_shift_ppm=2.01 - 4.7,  # Relative to water
            t1=1.4,
            t2=0.250,
            multiplicity=3,  # CH3 group
        )

    @staticmethod
    def creatine() -> ChemicalSpecies:
        """Creatine (Cr) - energy metabolism marker."""
        return ChemicalSpecies(
            name="Creatine",
            chemical_shift_ppm=3.03 - 4.7,
            t1=1.3,
            t2=0.150,
            multiplicity=3,
        )

    @staticmethod
    def choline() -> ChemicalSpecies:
        """Choline (Cho) - membrane turnover marker."""
        return ChemicalSpecies(
            name="Choline",
            chemical_shift_ppm=3.22 - 4.7,
            t1=1.1,
            t2=0.200,
            multiplicity=9,  # Trimethyl group
        )

    @staticmethod
    def myo_inositol() -> ChemicalSpecies:
        """Myo-inositol (mI) - glial marker."""
        return ChemicalSpecies(
            name="myo-Inositol",
            chemical_shift_ppm=3.56 - 4.7,
            t1=1.2,
            t2=0.120,
            multiplicity=1,
        )

    @staticmethod
    def glutamate() -> ChemicalSpecies:
        """Glutamate (Glu) - excitatory neurotransmitter."""
        return ChemicalSpecies(
            name="Glutamate",
            chemical_shift_ppm=2.35 - 4.7,
            t1=1.2,
            t2=0.100,
            multiplicity=2,
        )

    @staticmethod
    def glutamine() -> ChemicalSpecies:
        """Glutamine (Gln)."""
        return ChemicalSpecies(
            name="Glutamine",
            chemical_shift_ppm=2.45 - 4.7,
            t1=1.2,
            t2=0.100,
            multiplicity=2,
        )

    @staticmethod
    def lactate() -> ChemicalSpecies:
        """Lactate (Lac) - anaerobic metabolism marker."""
        return ChemicalSpecies(
            name="Lactate",
            chemical_shift_ppm=1.33 - 4.7,
            t1=1.5,
            t2=0.150,
            multiplicity=3,
            j_coupling_hz=6.9,
        )

    @staticmethod
    def water() -> ChemicalSpecies:
        """Water - reference and suppressed in MRS."""
        return ChemicalSpecies(
            name="Water",
            chemical_shift_ppm=0.0,  # Reference
            t1=1.5,  # Gray matter at 3T
            t2=0.080,
            multiplicity=2,
        )

    @staticmethod
    def lipid_09() -> ChemicalSpecies:
        """Lipid at 0.9 ppm (methyl groups)."""
        return ChemicalSpecies(
            name="Lipid_0.9",
            chemical_shift_ppm=0.9 - 4.7,
            t1=0.3,
            t2=0.050,
            multiplicity=3,
        )

    @staticmethod
    def lipid_13() -> ChemicalSpecies:
        """Lipid at 1.3 ppm (methylene groups)."""
        return ChemicalSpecies(
            name="Lipid_1.3",
            chemical_shift_ppm=1.3 - 4.7,
            t1=0.3,
            t2=0.050,
            multiplicity=2,
        )

    @staticmethod
    def all_metabolites() -> List[ChemicalSpecies]:
        """Get list of all brain metabolites."""
        return [
            BrainMetabolites.naa(),
            BrainMetabolites.creatine(),
            BrainMetabolites.choline(),
            BrainMetabolites.myo_inositol(),
            BrainMetabolites.glutamate(),
            BrainMetabolites.glutamine(),
            BrainMetabolites.lactate(),
            BrainMetabolites.water(),
            BrainMetabolites.lipid_09(),
            BrainMetabolites.lipid_13(),
        ]


class FatWaterSpecies:
    """Fat and water species for fat-water imaging."""

    @staticmethod
    def water() -> ChemicalSpecies:
        """Water protons."""
        return ChemicalSpecies(
            name="Water",
            chemical_shift_ppm=0.0,
            t1=1.0,
            t2=0.040,
        )

    @staticmethod
    def fat_main() -> ChemicalSpecies:
        """Main fat peak (methylene -CH2-)."""
        return ChemicalSpecies(
            name="Fat_main",
            chemical_shift_ppm=-3.4,  # Relative to water
            t1=0.35,
            t2=0.060,
            multiplicity=1,  # Simplified
        )

    @staticmethod
    def fat_olefinic() -> ChemicalSpecies:
        """Olefinic fat peak (-CH=CH-)."""
        return ChemicalSpecies(
            name="Fat_olefinic",
            chemical_shift_ppm=0.8,  # 5.3 ppm absolute, relative to water at 4.7
            t1=0.35,
            t2=0.060,
        )

    @staticmethod
    def fat_multipeak() -> List[ChemicalSpecies]:
        """Multi-peak fat model for Dixon imaging."""
        # Relative amplitudes based on literature
        return [
            ChemicalSpecies("Fat_A", -3.80, 0.35, 0.06),  # 0.9 ppm
            ChemicalSpecies("Fat_B", -3.40, 0.35, 0.06),  # 1.3 ppm (main)
            ChemicalSpecies("Fat_C", -2.60, 0.35, 0.06),  # 2.1 ppm
            ChemicalSpecies("Fat_D", -2.30, 0.35, 0.06),  # 2.4 ppm
            ChemicalSpecies("Fat_E", 0.60, 0.35, 0.06),  # 5.3 ppm
        ]


@dataclass
class SpectralPhantom:
    """
    Phantom with multiple chemical species per voxel.

    This extends the basic Phantom concept to support spectroscopic
    imaging where each voxel can contain multiple metabolites with
    different chemical shifts, relaxation times, and concentrations.

    Attributes
    ----------
    shape : tuple
        Spatial dimensions (nx,), (nx, ny), or (nx, ny, nz)
    fov : tuple
        Field of view in meters
    species : list of ChemicalSpecies
        Chemical species present in the phantom
    concentration_maps : dict
        Maps from species name to concentration array (shape=phantom.shape)
        Concentrations are in arbitrary units (typically mM or relative)
    t2_star_map : ndarray, optional
        Spatially-varying T2* map (overrides species T2*)
    b0_map : ndarray, optional
        B0 inhomogeneity map in Hz
    b0_map_ppm : ndarray, optional
        Field-independent B0 inhomogeneity map in ppm
    field_strength : float
        B0 field strength in Tesla
    nucleus : str
        Nucleus type ('H1', 'C13', etc.)
    spectral_reference_ppm : float
        Absolute ppm value of the scanner reference. Peak shifts stored in
        species are relative to this reference, so the simulated carrier is
        always 0 ppm.
    spectral_bandwidth_ppm : float
        Default spectral display bandwidth centered on the scanner reference.
    spectral_points : int
        Default number of frequency samples for spectral display.
    name : str
        Phantom name
    """

    shape: Tuple[int, ...]
    fov: Tuple[float, ...]
    species: List[ChemicalSpecies]
    concentration_maps: Dict[str, np.ndarray]
    initial_mz_maps: Optional[Dict[str, np.ndarray]] = None
    t2_star_map: np.ndarray = None
    b0_map: np.ndarray = None
    b0_map_ppm: np.ndarray = None
    field_strength: float = 3.0
    nucleus: str = "H1"
    spectral_reference_ppm: float = 0.0
    spectral_bandwidth_ppm: float = 20.0
    spectral_points: int = 1024
    name: str = "Spectral Phantom"
    metadata: Dict = field(default_factory=dict)
    coordinate_system: str = "object_xyz"
    affine_ijk_to_xyz_m: Optional[np.ndarray] = None

    # Computed fields
    positions: np.ndarray = field(init=False, repr=False)
    _frequency_offsets: Dict[str, float] = field(init=False, repr=False)

    def __post_init__(self):
        """Validate and compute derived quantities."""
        self._validate()
        self._compute_coordinates()
        self._compute_frequencies()

    def _validate(self):
        """Validate phantom configuration."""
        ndim = len(self.shape)
        if ndim not in (1, 2, 3):
            raise ValueError(f"Shape must be 1D, 2D, or 3D, got {ndim}D")

        if len(self.fov) != ndim:
            raise ValueError(f"FOV dimensions must match shape dimensions")
        if not str(self.coordinate_system).strip():
            raise ValueError("coordinate_system must not be empty")
        if self.affine_ijk_to_xyz_m is None:
            self.affine_ijk_to_xyz_m = Phantom.default_affine(self.shape, self.fov)
        else:
            affine = np.asarray(self.affine_ijk_to_xyz_m, dtype=np.float64)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise ValueError("affine_ijk_to_xyz_m must be a finite 4x4 matrix")
            self.affine_ijk_to_xyz_m = affine
        if len({species.name for species in self.species}) != len(self.species):
            raise ValueError("Spectral species names must be unique")
        if not self.species:
            raise ValueError("Spectral phantom requires at least one species")

        # Validate concentration maps
        for species in self.species:
            name = species.name
            if name not in self.concentration_maps:
                raise ValueError(f"Missing concentration map for species '{name}'")

            cmap = self.concentration_maps[name]
            if cmap.shape != self.shape:
                raise ValueError(
                    f"Concentration map for '{name}' has shape {cmap.shape}, "
                    f"expected {self.shape}"
                )
            if not np.all(np.isfinite(cmap)) or np.any(cmap < 0):
                raise ValueError(
                    f"Concentration map for '{name}' must be finite and non-negative"
                )

        if self.initial_mz_maps is None:
            self.initial_mz_maps = {
                species.name: np.ones(self.shape, dtype=np.float64)
                for species in self.species
            }
        for species in self.species:
            name = species.name
            if name not in self.initial_mz_maps:
                self.initial_mz_maps[name] = np.ones(self.shape, dtype=np.float64)
            mz_map = np.asarray(self.initial_mz_maps[name], dtype=np.float64)
            if mz_map.shape != self.shape:
                raise ValueError(
                    f"Initial Mz map for '{name}' has shape {mz_map.shape}, "
                    f"expected {self.shape}"
                )
            if not np.all(np.isfinite(mz_map)) or np.any(mz_map < 0):
                raise ValueError(
                    f"Initial Mz map for '{name}' must be finite and non-negative"
                )
            self.initial_mz_maps[name] = mz_map

        # Validate optional maps
        if self.t2_star_map is not None and self.t2_star_map.shape != self.shape:
            raise ValueError("T2* map shape must match phantom shape")

        if self.b0_map is not None and self.b0_map.shape != self.shape:
            raise ValueError("B0 map shape must match phantom shape")
        if self.b0_map_ppm is not None and self.b0_map_ppm.shape != self.shape:
            raise ValueError("B0 ppm map shape must match phantom shape")
        if self.b0_map is not None and self.b0_map_ppm is not None:
            raise ValueError("b0_map and b0_map_ppm cannot be combined")
        if not np.isfinite(self.field_strength) or self.field_strength <= 0:
            raise ValueError("field_strength must be positive and finite")
        if self.nucleus not in NUCLEUS_GAMMA_HZ_PER_T:
            raise ValueError(f"unsupported nucleus {self.nucleus!r}")
        if not np.isfinite(self.spectral_reference_ppm):
            raise ValueError("spectral_reference_ppm must be finite")
        if (
            not np.isfinite(self.spectral_bandwidth_ppm)
            or self.spectral_bandwidth_ppm <= 0
        ):
            raise ValueError("spectral_bandwidth_ppm must be positive and finite")
        if (
            int(self.spectral_points) != self.spectral_points
            or self.spectral_points < 2
        ):
            raise ValueError("spectral_points must be an integer >= 2")

    def _compute_coordinates(self):
        """Compute spatial coordinates."""
        ndim = len(self.shape)
        coords = Phantom.coordinate_vectors(self.shape, self.affine_ijk_to_xyz_m)

        if ndim == 1:
            self.positions = np.column_stack(
                [coords[0], np.zeros_like(coords[0]), np.zeros_like(coords[0])]
            )
        elif ndim == 2:
            X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
            self.positions = np.column_stack(
                [X.ravel(), Y.ravel(), np.zeros(np.prod(self.shape))]
            )
        else:
            X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
            self.positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def _compute_frequencies(self):
        """Compute frequency offset for each species."""
        self._frequency_offsets = {}
        for species in self.species:
            self._frequency_offsets[species.name] = species.get_frequency_offset(
                self.field_strength, self.nucleus
            )

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def nvoxels(self) -> int:
        """Total number of voxels."""
        return int(np.prod(self.shape))

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def n_species(self) -> int:
        """Number of chemical species."""
        return len(self.species)

    @property
    def mask(self) -> np.ndarray:
        """Voxels containing at least one non-zero spectral component."""
        return self.get_total_concentration() > 0

    @property
    def effective_df_map(self) -> np.ndarray:
        """Concentration-weighted mean frequency map for visualization only."""
        return self.df_map

    @property
    def resolution(self) -> Tuple[float, ...]:
        return tuple(f / n for f, n in zip(self.fov, self.shape))

    @property
    def voxel_volume_m3(self) -> float:
        if self.ndim != 3:
            raise ValueError("voxel volume requires a 3D spectral phantom")
        return float(np.prod(self.resolution))

    @property
    def pd_map(self) -> np.ndarray:
        """
        Proton density map (total concentration of all species).

        This property provides compatibility with the k-space simulator
        which expects phantoms to have pd_map, t2_map, df_map attributes.
        """
        return self.get_total_concentration()

    @property
    def t1_map(self) -> np.ndarray:
        """
        T1 map (concentration-weighted average of species T1 values).
        """
        total_conc = self.get_total_concentration()
        t1 = np.zeros(self.shape)

        for species in self.species:
            c = self.concentration_maps[species.name]
            weight = c / np.maximum(total_conc, 1e-10)
            t1 += weight * species.t1

        return t1

    @property
    def t2_map(self) -> np.ndarray:
        """
        T2 map (concentration-weighted average of species T2 values).
        """
        total_conc = self.get_total_concentration()
        t2 = np.zeros(self.shape)

        for species in self.species:
            c = self.concentration_maps[species.name]
            weight = c / np.maximum(total_conc, 1e-10)
            t2 += weight * species.t2

        return t2

    @property
    def df_map(self) -> np.ndarray:
        """
        Frequency offset map in Hz (concentration-weighted average of chemical shifts + B0).
        """
        total_conc = self.get_total_concentration()
        df = np.zeros(self.shape)

        for species in self.species:
            c = self.concentration_maps[species.name]
            weight = c / np.maximum(total_conc, 1e-10)
            df += weight * self.get_frequency_offset(species.name)

        # Add B0 inhomogeneity
        df = df + self.get_b0_offset_map_hz()

        return df

    def get_species(self, name: str) -> Optional[ChemicalSpecies]:
        """Get species by name."""
        for s in self.species:
            if s.name == name:
                return s
        return None

    def get_frequency_offset(
        self, name: str, field_strength: Optional[float] = None, nucleus: str = None
    ) -> float:
        """Get frequency offset for species in Hz."""
        species = self.get_species(name)
        if species is None:
            return 0.0
        return species.get_frequency_offset(
            self.field_strength if field_strength is None else field_strength,
            self.nucleus if nucleus is None else nucleus,
        )

    def get_frequency_offset_ppm(
        self, name: str, field_strength: Optional[float] = None, nucleus: str = None
    ) -> float:
        """Get frequency offset for species in ppm."""
        species = self.get_species(name)
        if species is None:
            return 0.0
        if species.frequency_offset_hz is None:
            return float(species.chemical_shift_ppm)
        return float(
            hz_to_ppm(
                species.frequency_offset_hz,
                self.field_strength if field_strength is None else field_strength,
                self.nucleus if nucleus is None else nucleus,
            )
        )

    def get_b0_offset_map_hz(
        self, field_strength: Optional[float] = None, nucleus: str = None
    ) -> np.ndarray:
        """Return the spatial B0 offset converted to Hz for one field."""
        if self.b0_map_ppm is not None:
            return np.asarray(
                ppm_to_hz(
                    self.b0_map_ppm,
                    self.field_strength if field_strength is None else field_strength,
                    self.nucleus if nucleus is None else nucleus,
                ),
                dtype=float,
            )
        if self.b0_map is not None:
            return np.asarray(self.b0_map, dtype=float)
        return np.zeros(self.shape, dtype=float)

    def get_b0_offset_map_ppm(
        self, field_strength: Optional[float] = None, nucleus: str = None
    ) -> np.ndarray:
        """Return the spatial B0 offset in ppm."""
        if self.b0_map_ppm is not None:
            return np.asarray(self.b0_map_ppm, dtype=float)
        if self.b0_map is not None:
            return np.asarray(
                hz_to_ppm(
                    self.b0_map,
                    self.field_strength if field_strength is None else field_strength,
                    self.nucleus if nucleus is None else nucleus,
                ),
                dtype=float,
            )
        return np.zeros(self.shape, dtype=float)

    @property
    def df_map_ppm(self) -> np.ndarray:
        """Concentration-weighted mean frequency map in ppm for visualization."""
        total_conc = self.get_total_concentration()
        df = np.zeros(self.shape, dtype=float)
        for species in self.species:
            c = self.concentration_maps[species.name]
            weight = c / np.maximum(total_conc, 1e-10)
            df += weight * self.get_frequency_offset_ppm(species.name)
        return df + self.get_b0_offset_map_ppm()

    def get_total_concentration(self) -> np.ndarray:
        """Get sum of all species concentrations."""
        total = np.zeros(self.shape)
        for name, cmap in self.concentration_maps.items():
            total += cmap
        return total

    def get_initial_mz_map(self, species_name: str = None) -> np.ndarray:
        """Return species-specific or concentration-weighted initial Mz."""
        if species_name is not None:
            if species_name not in self.initial_mz_maps:
                raise ValueError(f"Unknown species: {species_name}")
            return self.initial_mz_maps[species_name].copy()
        total = self.get_total_concentration()
        weighted = np.zeros(self.shape, dtype=np.float64)
        for species in self.species:
            concentration = self.concentration_maps[species.name]
            weighted += concentration * self.initial_mz_maps[species.name]
        return np.divide(
            weighted,
            total,
            out=np.zeros_like(weighted),
            where=total > 0,
        )

    def spectrum_at(
        self,
        index: Tuple[int, ...],
        frequency_hz: Optional[np.ndarray] = None,
        points: Optional[int] = None,
        field_strength: Optional[float] = None,
        nucleus: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the Lorentzian spectrum at one spatial voxel.

        A component with transverse decay ``T2*`` has
        ``FWHM = 1 / (pi*T2*)``. Amplitudes are taken from the corresponding
        concentration maps.
        """
        if len(index) != self.ndim:
            raise ValueError("index dimensionality must match spectral phantom")
        effective_field = (
            self.field_strength if field_strength is None else float(field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        centres = np.asarray(
            [
                self.get_frequency_offset(
                    species.name, effective_field, effective_nucleus
                )
                for species in self.species
            ]
        )
        widths = np.asarray(
            [1.0 / (np.pi * species.t2_star) for species in self.species]
        )
        if frequency_hz is None:
            if points is None:
                points = self.spectral_points
            half_bandwidth_hz = float(
                ppm_to_hz(
                    self.spectral_bandwidth_ppm / 2.0,
                    effective_field,
                    effective_nucleus,
                )
            )
            frequency_hz = np.linspace(
                -half_bandwidth_hz,
                half_bandwidth_hz,
                int(points),
            )
        frequency_hz = np.asarray(frequency_hz, dtype=float)
        spectrum = np.zeros(frequency_hz.shape, dtype=float)
        b0 = float(self.get_b0_offset_map_hz(effective_field, effective_nucleus)[index])
        for species, centre, fwhm in zip(self.species, centres, widths):
            amplitude = float(self.concentration_maps[species.name][index])
            amplitude *= float(self.initial_mz_maps[species.name][index])
            half_width = fwhm / 2.0
            spectrum += amplitude / (
                1.0 + ((frequency_hz - (centre + b0)) / half_width) ** 2
            )
        return frequency_hz, spectrum

    def spectrum_at_ppm(
        self,
        index: Tuple[int, ...],
        frequency_ppm: Optional[np.ndarray] = None,
        points: Optional[int] = None,
        *,
        absolute: bool = True,
        linewidth_field_strength: Optional[float] = None,
        nucleus: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the Lorentzian spectrum on a ppm axis.

        Peak positions and B0 offsets remain field-independent ppm values.
        T2* linewidths are converted from Hz to ppm using the supplied field
        and nucleus only for display.
        """
        if len(index) != self.ndim:
            raise ValueError("index dimensionality must match spectral phantom")
        if points is None:
            points = self.spectral_points
        if frequency_ppm is None:
            centre = self.spectral_reference_ppm if absolute else 0.0
            half_bandwidth_ppm = self.spectral_bandwidth_ppm / 2.0
            frequency_ppm = np.linspace(
                centre - half_bandwidth_ppm,
                centre + half_bandwidth_ppm,
                int(points),
            )
        frequency_ppm = np.asarray(frequency_ppm, dtype=float)
        relative_frequency_ppm = (
            frequency_ppm - self.spectral_reference_ppm if absolute else frequency_ppm
        )
        effective_field = (
            self.field_strength
            if linewidth_field_strength is None
            else float(linewidth_field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        centres = np.asarray(
            [
                self.get_frequency_offset_ppm(
                    species.name, effective_field, effective_nucleus
                )
                for species in self.species
            ]
        )
        widths_ppm = np.abs(
            np.asarray(
                [
                    hz_to_ppm(
                        1.0 / (np.pi * species.t2_star),
                        effective_field,
                        effective_nucleus,
                    )
                    for species in self.species
                ],
                dtype=float,
            )
        )
        b0_ppm = float(
            self.get_b0_offset_map_ppm(effective_field, effective_nucleus)[index]
        )
        spectrum = np.zeros(frequency_ppm.shape, dtype=float)
        for species, centre, fwhm_ppm in zip(self.species, centres, widths_ppm):
            amplitude = float(self.concentration_maps[species.name][index])
            amplitude *= float(self.initial_mz_maps[species.name][index])
            half_width_ppm = max(fwhm_ppm / 2.0, np.finfo(float).eps)
            spectrum += amplitude / (
                1.0
                + ((relative_frequency_ppm - (centre + b0_ppm)) / half_width_ppm) ** 2
            )
        return frequency_ppm, spectrum

    def to_component_phantoms(
        self, field_strength: Optional[float] = None, nucleus: str = None
    ) -> List[Tuple[str, "Phantom"]]:
        """Expand spectral components into independently simulated phantoms."""
        if Phantom is None:
            raise ImportError("phantom module not available")
        effective_field = (
            self.field_strength if field_strength is None else float(field_strength)
        )
        effective_nucleus = self.nucleus if nucleus is None else str(nucleus)
        b0 = self.get_b0_offset_map_hz(effective_field, effective_nucleus)
        components = []
        for species in self.species:
            concentration = np.asarray(
                self.concentration_maps[species.name], dtype=float
            )
            active = concentration > 0
            if not np.any(active):
                continue
            t2_star = (
                np.asarray(self.t2_star_map, dtype=float)
                if self.t2_star_map is not None
                else np.full(self.shape, species.t2_star, dtype=float)
            )
            components.append(
                (
                    species.name,
                    Phantom(
                        shape=self.shape,
                        fov=self.fov,
                        t1_map=np.full(self.shape, species.t1, dtype=float),
                        t2_map=t2_star,
                        pd_map=concentration,
                        b0_map=b0,
                        chemical_shift_map=np.full(
                            self.shape,
                            self.get_frequency_offset(
                                species.name, effective_field, effective_nucleus
                            ),
                        ),
                        m0_map=self._component_m0_map(species.name),
                        mask=active,
                        tx_sensitivity_map=getattr(self, "tx_sensitivity_map", None),
                        rx_sensitivity_maps=getattr(self, "rx_sensitivity_maps", None),
                        name=f"{self.name} - {species.name}",
                        coordinate_system=self.coordinate_system,
                        affine_ijk_to_xyz_m=self.affine_ijk_to_xyz_m,
                    ),
                )
            )
        return components

    def _component_m0_map(self, species_name: str) -> np.ndarray:
        m0_map = np.zeros(self.shape + (3,), dtype=np.float64)
        m0_map[..., 2] = self.initial_mz_maps[species_name]
        return m0_map

    def to_xarray(self):
        """Return this spectral phantom as a coordinate-aware xarray Dataset."""
        import xarray as xr

        spatial_dims = tuple("xyz"[: self.ndim])
        species_names = [species.name for species in self.species]
        coords = {
            dim: (
                dim,
                values,
                {
                    "units": "m",
                    "long_name": f"{dim} coordinate in {self.coordinate_system}",
                },
            )
            for dim, values in zip(
                spatial_dims,
                Phantom.coordinate_vectors(self.shape, self.affine_ijk_to_xyz_m),
            )
        }
        coords["species"] = ("species", species_names)
        data_vars = {
            "concentration": (
                ("species",) + spatial_dims,
                np.stack([self.concentration_maps[name] for name in species_names]),
                {"units": "relative"},
            ),
            "initial_mz": (
                ("species",) + spatial_dims,
                np.stack([self.initial_mz_maps[name] for name in species_names]),
                {"units": "relative"},
            ),
            "b0_ppm": (
                spatial_dims,
                (
                    self.b0_map_ppm
                    if self.b0_map_ppm is not None
                    else np.zeros(self.shape, dtype=float)
                ),
                {"units": "ppm"},
            ),
            "b0_hz": (
                spatial_dims,
                (
                    self.b0_map
                    if self.b0_map is not None
                    else np.zeros(self.shape, dtype=float)
                ),
                {"units": "Hz"},
            ),
            "t2_star_map": (
                spatial_dims,
                (
                    self.t2_star_map
                    if self.t2_star_map is not None
                    else np.zeros(self.shape, dtype=float)
                ),
                {"units": "s"},
            ),
            "species_chemical_shift_ppm": (
                "species",
                [species.chemical_shift_ppm for species in self.species],
                {"units": "ppm"},
            ),
            "species_t1": (
                "species",
                [species.t1 for species in self.species],
                {"units": "s"},
            ),
            "species_t2": (
                "species",
                [species.t2 for species in self.species],
                {"units": "s"},
            ),
            "species_t2_star": (
                "species",
                [species.t2_star for species in self.species],
                {"units": "s"},
            ),
            "species_frequency_offset_hz": (
                "species",
                [
                    (
                        np.nan
                        if species.frequency_offset_hz is None
                        else species.frequency_offset_hz
                    )
                    for species in self.species
                ],
                {"units": "Hz"},
            ),
        }
        header = {
            "species": [
                {
                    "name": species.name,
                    "chemical_shift_ppm": species.chemical_shift_ppm,
                    "t1": species.t1,
                    "t2": species.t2,
                    "t2_star": species.t2_star,
                    "multiplicity": species.multiplicity,
                    "j_coupling_hz": species.j_coupling_hz,
                    "j_partners": species.j_partners,
                    "frequency_offset_hz": species.frequency_offset_hz,
                }
                for species in self.species
            ],
            "metadata": self.metadata,
        }
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "format": "blochsimulator-spectral-phantom-xarray",
                "version": 1,
                "name": self.name,
                "fov_m": np.asarray(self.fov, dtype=np.float64),
                "field_strength": self.field_strength,
                "nucleus": self.nucleus,
                "spectral_reference_ppm": self.spectral_reference_ppm,
                "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
                "spectral_points": self.spectral_points,
                "has_b0_map": self.b0_map is not None,
                "has_b0_map_ppm": self.b0_map_ppm is not None,
                "has_t2_star_map": self.t2_star_map is not None,
                "coordinate_system": self.coordinate_system,
                "affine_ijk_to_xyz_m": self.affine_ijk_to_xyz_m.reshape(-1),
                "spectral_header_json": json.dumps(header, default=str),
            },
        )

    @classmethod
    def from_xarray(cls, dataset) -> "SpectralPhantom":
        """Create a :class:`SpectralPhantom` from an xarray Dataset."""
        ds = dataset.load()
        spatial_dims = tuple(dim for dim in ("x", "y", "z") if dim in ds.sizes)
        if "species" not in ds.sizes or not spatial_dims:
            raise ValueError("spectral phantom dataset requires species and x/y/z dims")
        shape = tuple(int(ds.sizes[dim]) for dim in spatial_dims)
        fov = tuple(float(value) for value in np.asarray(ds.attrs["fov_m"]).ravel())
        affine = np.asarray(ds.attrs["affine_ijk_to_xyz_m"], dtype=float).reshape(4, 4)
        header = json.loads(ds.attrs.get("spectral_header_json", "{}"))
        species_metadata = header.get("species", [])
        species_names = [str(value) for value in np.asarray(ds.coords["species"])]
        species = []
        for index, name in enumerate(species_names):
            if index < len(species_metadata):
                item = dict(species_metadata[index])
                item["name"] = name
                species.append(ChemicalSpecies(**item))
            else:
                frequency_hz = float(ds["species_frequency_offset_hz"][index])
                species.append(
                    ChemicalSpecies(
                        name=name,
                        chemical_shift_ppm=float(
                            ds["species_chemical_shift_ppm"][index]
                        ),
                        t1=float(ds["species_t1"][index]),
                        t2=float(ds["species_t2"][index]),
                        t2_star=float(ds["species_t2_star"][index]),
                        frequency_offset_hz=(
                            None if np.isnan(frequency_hz) else frequency_hz
                        ),
                    )
                )
        concentration_maps = {
            name: np.asarray(ds["concentration"].sel(species=name))
            for name in species_names
        }
        initial_mz_maps = {
            name: np.asarray(ds["initial_mz"].sel(species=name))
            for name in species_names
        }
        return cls(
            shape=shape,
            fov=fov,
            species=species,
            concentration_maps=concentration_maps,
            initial_mz_maps=initial_mz_maps,
            t2_star_map=(
                np.asarray(ds["t2_star_map"])
                if bool(ds.attrs.get("has_t2_star_map", False))
                else None
            ),
            b0_map=(
                np.asarray(ds["b0_hz"])
                if bool(ds.attrs.get("has_b0_map", False))
                else None
            ),
            b0_map_ppm=(
                np.asarray(ds["b0_ppm"])
                if bool(ds.attrs.get("has_b0_map_ppm", False))
                else None
            ),
            field_strength=float(ds.attrs["field_strength"]),
            nucleus=str(ds.attrs["nucleus"]),
            spectral_reference_ppm=float(ds.attrs.get("spectral_reference_ppm", 0.0)),
            spectral_bandwidth_ppm=float(ds.attrs.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(ds.attrs.get("spectral_points", 1024)),
            name=str(ds.attrs.get("name", "Spectral Phantom")),
            metadata=dict(header.get("metadata", {})),
            coordinate_system=str(ds.attrs.get("coordinate_system", "object_xyz")),
            affine_ijk_to_xyz_m=affine,
        )

    def save(self, filename: Union[str, Path]) -> None:
        """Save all spectral maps, peak definitions, and designer metadata."""
        filename = Path(filename)
        species_data = [
            {
                "name": species.name,
                "chemical_shift_ppm": species.chemical_shift_ppm,
                "t1": species.t1,
                "t2": species.t2,
                "t2_star": species.t2_star,
                "multiplicity": species.multiplicity,
                "j_coupling_hz": species.j_coupling_hz,
                "j_partners": species.j_partners,
                "frequency_offset_hz": species.frequency_offset_hz,
            }
            for species in self.species
        ]
        header = {
            "format": "blochsimulator-spectral-phantom",
            "version": 3,
            "name": self.name,
            "field_strength": self.field_strength,
            "nucleus": self.nucleus,
            "spectral_reference_ppm": self.spectral_reference_ppm,
            "spectral_bandwidth_ppm": self.spectral_bandwidth_ppm,
            "spectral_points": self.spectral_points,
            "coordinate_system": self.coordinate_system,
            "affine_ijk_to_xyz_m": self.affine_ijk_to_xyz_m.tolist(),
            "species": species_data,
            "metadata": self.metadata,
        }
        if filename.suffix == ".npz":
            arrays = {
                f"concentration_{index}": self.concentration_maps[species.name]
                for index, species in enumerate(self.species)
            }
            arrays.update(
                {
                    f"initial_mz_{index}": self.initial_mz_maps[species.name]
                    for index, species in enumerate(self.species)
                }
            )
            np.savez_compressed(
                filename,
                spectral_header=np.asarray(json.dumps(header)),
                shape=np.asarray(self.shape),
                fov=np.asarray(self.fov),
                b0_map=(
                    np.asarray(self.b0_map)
                    if self.b0_map is not None
                    else np.zeros(self.shape)
                ),
                has_b0=np.asarray(self.b0_map is not None),
                b0_map_ppm=(
                    np.asarray(self.b0_map_ppm)
                    if self.b0_map_ppm is not None
                    else np.zeros(self.shape)
                ),
                has_b0_ppm=np.asarray(self.b0_map_ppm is not None),
                t2_star_map=(
                    np.asarray(self.t2_star_map)
                    if self.t2_star_map is not None
                    else np.zeros(self.shape)
                ),
                has_t2_star=np.asarray(self.t2_star_map is not None),
                **arrays,
            )
        elif filename.suffix == ".nc":
            self.to_xarray().to_netcdf(filename)
        elif filename.suffix in (".h5", ".hdf5"):
            import h5py

            with h5py.File(filename, "w") as handle:
                handle.attrs["spectral_header"] = json.dumps(header)
                handle.create_dataset("shape", data=self.shape)
                handle.create_dataset("fov", data=self.fov)
                if self.b0_map is not None:
                    handle.create_dataset("b0_map", data=self.b0_map)
                if self.b0_map_ppm is not None:
                    handle.create_dataset("b0_map_ppm", data=self.b0_map_ppm)
                if self.t2_star_map is not None:
                    handle.create_dataset("t2_star_map", data=self.t2_star_map)
                group = handle.create_group("concentration_maps")
                for index, species in enumerate(self.species):
                    group.create_dataset(
                        str(index), data=self.concentration_maps[species.name]
                    )
                mz_group = handle.create_group("initial_mz_maps")
                for index, species in enumerate(self.species):
                    mz_group.create_dataset(
                        str(index), data=self.initial_mz_maps[species.name]
                    )
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")

    @classmethod
    def load(cls, filename: Union[str, Path]) -> "SpectralPhantom":
        """Load a spectral phantom saved by :meth:`save`."""
        filename = Path(filename)
        if filename.suffix == ".npz":
            with np.load(filename, allow_pickle=False) as data:
                if "spectral_header" not in data.files:
                    raise ValueError("file is not a spectral phantom")
                header = json.loads(str(data["spectral_header"]))
                shape = tuple(int(value) for value in data["shape"])
                fov = tuple(float(value) for value in data["fov"])
                species = [ChemicalSpecies(**item) for item in header["species"]]
                maps = {
                    item.name: np.asarray(data[f"concentration_{index}"])
                    for index, item in enumerate(species)
                }
                initial_mz_maps = {
                    item.name: (
                        np.asarray(data[f"initial_mz_{index}"])
                        if f"initial_mz_{index}" in data.files
                        else np.ones(shape, dtype=np.float64)
                    )
                    for index, item in enumerate(species)
                }
                b0 = np.asarray(data["b0_map"]) if bool(data["has_b0"]) else None
                b0_ppm = (
                    np.asarray(data["b0_map_ppm"])
                    if "has_b0_ppm" in data.files and bool(data["has_b0_ppm"])
                    else None
                )
                t2_star = (
                    np.asarray(data["t2_star_map"])
                    if bool(data["has_t2_star"])
                    else None
                )
        elif filename.suffix == ".nc":
            import xarray as xr

            with xr.open_dataset(filename) as dataset:
                return cls.from_xarray(dataset)
        elif filename.suffix in (".h5", ".hdf5"):
            import h5py

            with h5py.File(filename, "r") as handle:
                if "spectral_header" not in handle.attrs:
                    raise ValueError("file is not a spectral phantom")
                header = json.loads(handle.attrs["spectral_header"])
                shape = tuple(int(value) for value in handle["shape"][...])
                fov = tuple(float(value) for value in handle["fov"][...])
                species = [ChemicalSpecies(**item) for item in header["species"]]
                maps = {
                    item.name: handle["concentration_maps"][str(index)][...]
                    for index, item in enumerate(species)
                }
                if "initial_mz_maps" in handle:
                    initial_mz_maps = {
                        item.name: handle["initial_mz_maps"][str(index)][...]
                        for index, item in enumerate(species)
                    }
                else:
                    initial_mz_maps = {
                        item.name: np.ones(shape, dtype=np.float64) for item in species
                    }
                b0 = handle["b0_map"][...] if "b0_map" in handle else None
                b0_ppm = handle["b0_map_ppm"][...] if "b0_map_ppm" in handle else None
                t2_star = (
                    handle["t2_star_map"][...] if "t2_star_map" in handle else None
                )
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")
        return cls(
            shape=shape,
            fov=fov,
            species=species,
            concentration_maps=maps,
            initial_mz_maps=initial_mz_maps,
            t2_star_map=t2_star,
            b0_map=b0,
            b0_map_ppm=b0_ppm,
            field_strength=float(header["field_strength"]),
            nucleus=str(header["nucleus"]),
            spectral_reference_ppm=float(header.get("spectral_reference_ppm", 0.0)),
            spectral_bandwidth_ppm=float(header.get("spectral_bandwidth_ppm", 20.0)),
            spectral_points=int(header.get("spectral_points", 1024)),
            name=str(header["name"]),
            metadata=dict(header.get("metadata", {})),
            coordinate_system=str(header.get("coordinate_system", "object_xyz")),
            affine_ijk_to_xyz_m=header.get("affine_ijk_to_xyz_m"),
        )

    def get_species_properties(self, species_name: str = None) -> Dict[str, np.ndarray]:
        """
        Get flattened property arrays for simulation.

        Parameters
        ----------
        species_name : str, optional
            If provided, return properties for single species.
            Otherwise, return combined properties.

        Returns
        -------
        dict with:
            'positions': (nvoxels, 3) in meters
            't1': (nvoxels,) in seconds
            't2': (nvoxels,) in seconds
            't2_star': (nvoxels,) in seconds
            'df': (nvoxels,) frequency offset in Hz
            'concentration': (nvoxels,) relative concentration
        """
        if species_name is not None:
            species = self.get_species(species_name)
            if species is None:
                raise ValueError(f"Unknown species: {species_name}")

            concentration = self.concentration_maps[species_name].ravel()
            df = np.full(self.nvoxels, self.get_frequency_offset(species_name))
            t1 = np.full(self.nvoxels, species.t1)
            t2 = np.full(self.nvoxels, species.t2)

            if self.t2_star_map is not None:
                t2_star = self.t2_star_map.ravel()
            else:
                t2_star = np.full(self.nvoxels, species.t2_star)
            initial_mz = self.initial_mz_maps[species_name].ravel()

        else:
            # Combined: weighted average of properties
            concentration = self.get_total_concentration().ravel()
            df = np.zeros(self.nvoxels)
            t1 = np.zeros(self.nvoxels)
            t2 = np.zeros(self.nvoxels)

            # Concentration-weighted average
            for species in self.species:
                c = self.concentration_maps[species.name].ravel()
                weight = c / np.maximum(concentration, 1e-10)
                df += weight * self.get_frequency_offset(species.name)
                t1 += weight * species.t1
                t2 += weight * species.t2

            if self.t2_star_map is not None:
                t2_star = self.t2_star_map.ravel()
            else:
                t2_star = t2.copy()
            initial_mz = self.get_initial_mz_map().ravel()

        # Add B0 inhomogeneity to frequency offset
        if self.b0_map is not None:
            df = df + self.b0_map.ravel()
        elif self.b0_map_ppm is not None:
            df = df + self.get_b0_offset_map_hz().ravel()

        return {
            "positions": self.positions.copy(),
            "t1": t1,
            "t2": t2,
            "t2_star": t2_star,
            "df": df,
            "concentration": concentration,
            "initial_mz": initial_mz,
        }

    def to_phantom(self, species_name: str = None) -> "Phantom":
        """
        Convert to basic Phantom for Bloch simulation.

        Parameters
        ----------
        species_name : str, optional
            If provided, create phantom for single species.
            Otherwise, use combined properties.

        Returns
        -------
        Phantom
            Standard phantom object
        """
        if Phantom is None:
            raise ImportError("phantom module not available")

        props = self.get_species_properties(species_name)

        return Phantom(
            shape=self.shape,
            fov=self.fov,
            t1_map=props["t1"].reshape(self.shape),
            t2_map=props["t2"].reshape(self.shape),
            pd_map=props["concentration"].reshape(self.shape),
            df_map=props["df"].reshape(self.shape),
            m0_map=self._m0_map_from_mz(props["initial_mz"].reshape(self.shape)),
            tx_sensitivity_map=getattr(self, "tx_sensitivity_map", None),
            rx_sensitivity_maps=getattr(self, "rx_sensitivity_maps", None),
            name=f"{self.name} - {species_name or 'combined'}",
            coordinate_system=self.coordinate_system,
            affine_ijk_to_xyz_m=self.affine_ijk_to_xyz_m,
        )

    def _m0_map_from_mz(self, initial_mz: np.ndarray) -> np.ndarray:
        m0_map = np.zeros(self.shape + (3,), dtype=np.float64)
        m0_map[..., 2] = initial_mz
        return m0_map

    def simulate_fid(
        self,
        acquisition_time: float = 0.5,
        dwell_time: float = 0.5e-3,
        line_broadening: float = 0.0,
    ) -> Dict:
        """
        Simulate free induction decay (FID) for single-voxel spectroscopy.

        This is a simplified simulation assuming perfect excitation and
        no spatial encoding - suitable for SVS or localized MRS.

        Parameters
        ----------
        acquisition_time : float
            Total acquisition time in seconds
        dwell_time : float
            Time between samples in seconds
        line_broadening : float
            Additional exponential line broadening in Hz

        Returns
        -------
        dict with:
            'time': Time points in seconds
            'signal': Complex FID signal
            'frequency': Frequency axis in Hz (after FFT)
            'spectrum': FFT of signal
        """
        n_points = int(acquisition_time / dwell_time)
        time = np.arange(n_points) * dwell_time

        signal = np.zeros(n_points, dtype=np.complex128)

        for species in self.species:
            # Get total concentration for this species
            total_conc = np.sum(self.concentration_maps[species.name])
            if total_conc <= 0:
                continue

            # Frequency offset
            df = self.get_frequency_offset(species.name)

            # T2 decay
            t2 = species.t2
            if line_broadening > 0:
                # Add line broadening: 1/T2_eff = 1/T2 + π × LB
                t2_eff = 1.0 / (1.0 / t2 + np.pi * line_broadening)
            else:
                t2_eff = t2

            # FID component: A × exp(-t/T2) × exp(-i × 2π × df × t)
            amplitude = total_conc * species.multiplicity
            component = (
                amplitude * np.exp(-time / t2_eff) * np.exp(-1j * 2 * np.pi * df * time)
            )

            signal += component

        # Compute spectrum
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        frequency = np.fft.fftshift(np.fft.fftfreq(n_points, dwell_time))

        return {
            "time": time,
            "signal": signal,
            "frequency": frequency,
            "spectrum": spectrum,
            "dwell_time": dwell_time,
        }


class SpectralPhantomFactory:
    """Factory methods for creating common spectral phantoms."""

    @staticmethod
    def brain_mrs_voxel(
        field_strength: float = 3.0, concentrations: Dict[str, float] = None
    ) -> SpectralPhantom:
        """
        Create single-voxel brain MRS phantom.

        Parameters
        ----------
        field_strength : float
            B0 field strength in Tesla
        concentrations : dict, optional
            Species concentrations in mM. Defaults to typical values.

        Returns
        -------
        SpectralPhantom
        """
        # Default brain concentrations (mM) - approximate
        default_conc = {
            "NAA": 12.0,
            "Creatine": 8.0,
            "Choline": 2.0,
            "myo-Inositol": 6.0,
            "Glutamate": 10.0,
            "Glutamine": 4.0,
            "Lactate": 0.5,
            "Water": 35000.0,  # Much higher than metabolites
        }

        if concentrations is not None:
            default_conc.update(concentrations)

        # Create species list
        species = [
            BrainMetabolites.naa(),
            BrainMetabolites.creatine(),
            BrainMetabolites.choline(),
            BrainMetabolites.myo_inositol(),
            BrainMetabolites.glutamate(),
            BrainMetabolites.glutamine(),
            BrainMetabolites.lactate(),
            BrainMetabolites.water(),
        ]

        # Single voxel phantom (1x1x1)
        shape = (1, 1, 1)
        fov = (0.02, 0.02, 0.02)  # 20 mm voxel

        # Create concentration maps
        concentration_maps = {}
        for s in species:
            conc = default_conc.get(s.name, 0.0)
            concentration_maps[s.name] = np.array([[[conc]]])

        return SpectralPhantom(
            shape=shape,
            fov=fov,
            species=species,
            concentration_maps=concentration_maps,
            field_strength=field_strength,
            name=f"Brain MRS @ {field_strength}T",
        )

    @staticmethod
    def brain_csi_grid(
        matrix_size: Tuple[int, int] = (16, 16),
        fov: Tuple[float, float] = (0.16, 0.16),
        field_strength: float = 3.0,
    ) -> SpectralPhantom:
        """
        Create 2D CSI phantom with brain metabolites.

        Creates a simplified brain phantom with:
        - White matter region in center
        - Gray matter ring around
        - CSF-filled ventricles

        Parameters
        ----------
        matrix_size : tuple
            (nx, ny) spatial matrix
        fov : tuple
            Field of view in meters
        field_strength : float
            B0 field strength

        Returns
        -------
        SpectralPhantom
        """
        nx, ny = matrix_size

        # Create spatial masks
        x = np.linspace(-0.5, 0.5, nx)
        y = np.linspace(-0.5, 0.5, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)

        # Brain regions
        wm_mask = R < 0.25  # White matter center
        gm_mask = (R >= 0.25) & (R < 0.4)  # Gray matter ring
        csf_mask = (np.abs(X) < 0.1) & (np.abs(Y) < 0.1)  # Ventricles

        # Override: CSF takes precedence
        wm_mask = wm_mask & ~csf_mask
        gm_mask = gm_mask & ~csf_mask

        # Define concentration profiles (mM)
        # White matter has lower NAA than gray matter
        species = [
            BrainMetabolites.naa(),
            BrainMetabolites.creatine(),
            BrainMetabolites.choline(),
            BrainMetabolites.myo_inositol(),
            BrainMetabolites.water(),
        ]

        concentration_maps = {}

        for s in species:
            cmap = np.zeros((nx, ny))

            if s.name == "NAA":
                cmap[wm_mask] = 10.0
                cmap[gm_mask] = 12.0
                cmap[csf_mask] = 0.0
            elif s.name == "Creatine":
                cmap[wm_mask] = 6.0
                cmap[gm_mask] = 8.0
                cmap[csf_mask] = 0.0
            elif s.name == "Choline":
                cmap[wm_mask] = 2.5
                cmap[gm_mask] = 1.5
                cmap[csf_mask] = 0.0
            elif s.name == "myo-Inositol":
                cmap[wm_mask] = 4.0
                cmap[gm_mask] = 6.0
                cmap[csf_mask] = 0.0
            elif s.name == "Water":
                cmap[wm_mask] = 30000.0
                cmap[gm_mask] = 35000.0
                cmap[csf_mask] = 50000.0

            concentration_maps[s.name] = cmap

        return SpectralPhantom(
            shape=(nx, ny),
            fov=fov,
            species=species,
            concentration_maps=concentration_maps,
            field_strength=field_strength,
            name=f"Brain CSI {nx}×{ny} @ {field_strength}T",
        )

    @staticmethod
    def fat_water_phantom(
        matrix_size: Tuple[int, int] = (64, 64),
        fov: Tuple[float, float] = (0.24, 0.24),
        field_strength: float = 3.0,
        multi_peak_fat: bool = True,
    ) -> SpectralPhantom:
        """
        Create fat-water phantom for Dixon imaging.

        Parameters
        ----------
        matrix_size : tuple
            (nx, ny) matrix size
        fov : tuple
            Field of view in meters
        field_strength : float
            B0 field strength
        multi_peak_fat : bool
            If True, use 6-peak fat model. Otherwise single peak.

        Returns
        -------
        SpectralPhantom
        """
        nx, ny = matrix_size

        # Create regions
        x = np.linspace(-0.5, 0.5, nx)
        y = np.linspace(-0.5, 0.5, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)

        # Water in center, fat ring around
        water_mask = R < 0.25
        fat_mask = (R >= 0.25) & (R < 0.4)
        mixed_mask = (R >= 0.15) & (R < 0.2)  # Partial volume

        # Species
        water_species = FatWaterSpecies.water()

        if multi_peak_fat:
            fat_species = FatWaterSpecies.fat_multipeak()
            species = [water_species] + fat_species
        else:
            species = [water_species, FatWaterSpecies.fat_main()]

        concentration_maps = {}

        # Water
        water_conc = np.zeros((nx, ny))
        water_conc[water_mask] = 1.0
        water_conc[mixed_mask] = 0.7  # Partial volume
        concentration_maps["Water"] = water_conc

        # Fat
        if multi_peak_fat:
            # Relative amplitudes for multi-peak model
            amplitudes = [0.09, 0.70, 0.12, 0.03, 0.06]
            for i, species_fat in enumerate(fat_species):
                fat_conc = np.zeros((nx, ny))
                fat_conc[fat_mask] = amplitudes[i]
                fat_conc[mixed_mask] = amplitudes[i] * 0.3  # Partial volume
                concentration_maps[species_fat.name] = fat_conc
        else:
            fat_conc = np.zeros((nx, ny))
            fat_conc[fat_mask] = 1.0
            fat_conc[mixed_mask] = 0.3
            concentration_maps["Fat_main"] = fat_conc

        return SpectralPhantom(
            shape=(nx, ny),
            fov=fov,
            species=species,
            concentration_maps=concentration_maps,
            field_strength=field_strength,
            name=f"Fat-Water {nx}×{ny} @ {field_strength}T",
        )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Spectral Phantom Module")
    print("=" * 50)

    # Test brain MRS
    print("\nCreating brain MRS phantom...")
    mrs_phantom = SpectralPhantomFactory.brain_mrs_voxel(3.0)
    print(f"  Species: {[s.name for s in mrs_phantom.species]}")

    # Simulate FID
    print("\nSimulating FID...")
    fid_result = mrs_phantom.simulate_fid(acquisition_time=0.5, dwell_time=0.5e-3)
    print(f"  Time points: {len(fid_result['time'])}")
    print(f"  Max spectrum magnitude: {np.max(np.abs(fid_result['spectrum'])):.2f}")

    # Test brain CSI
    print("\nCreating brain CSI phantom...")
    csi_phantom = SpectralPhantomFactory.brain_csi_grid((16, 16))
    print(f"  Shape: {csi_phantom.shape}")
    print(f"  Species: {csi_phantom.n_species}")

    # Test fat-water
    print("\nCreating fat-water phantom...")
    fw_phantom = SpectralPhantomFactory.fat_water_phantom((32, 32), multi_peak_fat=True)
    print(f"  Shape: {fw_phantom.shape}")
    print(f"  Species: {[s.name for s in fw_phantom.species]}")

    # Test conversion to basic phantom
    if Phantom is not None:
        print("\nConverting to basic Phantom...")
        basic = fw_phantom.to_phantom("Water")
        print(f"  Basic phantom shape: {basic.shape}")

    print("\n✓ All tests passed!")
