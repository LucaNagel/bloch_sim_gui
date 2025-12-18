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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

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
    
    def __post_init__(self):
        if self.t2_star is None:
            self.t2_star = self.t2
    
    def get_frequency_offset(self, field_strength: float, 
                            nucleus: str = 'H1') -> float:
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
        # Larmor frequencies at 1T (MHz)
        larmor_1t = {
            'H1': 42.576,
            'C13': 10.705,
            'P31': 17.235,
            'F19': 40.052,
            'Na23': 11.262,
        }
        
        gamma_mhz = larmor_1t.get(nucleus, 42.576)
        larmor_hz = gamma_mhz * 1e6 * field_strength
        
        return self.chemical_shift_ppm * 1e-6 * larmor_hz


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
            name='NAA',
            chemical_shift_ppm=2.01 - 4.7,  # Relative to water
            t1=1.4,
            t2=0.250,
            multiplicity=3,  # CH3 group
        )
    
    @staticmethod
    def creatine() -> ChemicalSpecies:
        """Creatine (Cr) - energy metabolism marker."""
        return ChemicalSpecies(
            name='Creatine',
            chemical_shift_ppm=3.03 - 4.7,
            t1=1.3,
            t2=0.150,
            multiplicity=3,
        )
    
    @staticmethod
    def choline() -> ChemicalSpecies:
        """Choline (Cho) - membrane turnover marker."""
        return ChemicalSpecies(
            name='Choline',
            chemical_shift_ppm=3.22 - 4.7,
            t1=1.1,
            t2=0.200,
            multiplicity=9,  # Trimethyl group
        )
    
    @staticmethod
    def myo_inositol() -> ChemicalSpecies:
        """Myo-inositol (mI) - glial marker."""
        return ChemicalSpecies(
            name='myo-Inositol',
            chemical_shift_ppm=3.56 - 4.7,
            t1=1.2,
            t2=0.120,
            multiplicity=1,
        )
    
    @staticmethod
    def glutamate() -> ChemicalSpecies:
        """Glutamate (Glu) - excitatory neurotransmitter."""
        return ChemicalSpecies(
            name='Glutamate',
            chemical_shift_ppm=2.35 - 4.7,
            t1=1.2,
            t2=0.100,
            multiplicity=2,
        )
    
    @staticmethod
    def glutamine() -> ChemicalSpecies:
        """Glutamine (Gln)."""
        return ChemicalSpecies(
            name='Glutamine',
            chemical_shift_ppm=2.45 - 4.7,
            t1=1.2,
            t2=0.100,
            multiplicity=2,
        )
    
    @staticmethod
    def lactate() -> ChemicalSpecies:
        """Lactate (Lac) - anaerobic metabolism marker."""
        return ChemicalSpecies(
            name='Lactate',
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
            name='Water',
            chemical_shift_ppm=0.0,  # Reference
            t1=1.5,  # Gray matter at 3T
            t2=0.080,
            multiplicity=2,
        )
    
    @staticmethod
    def lipid_09() -> ChemicalSpecies:
        """Lipid at 0.9 ppm (methyl groups)."""
        return ChemicalSpecies(
            name='Lipid_0.9',
            chemical_shift_ppm=0.9 - 4.7,
            t1=0.3,
            t2=0.050,
            multiplicity=3,
        )
    
    @staticmethod
    def lipid_13() -> ChemicalSpecies:
        """Lipid at 1.3 ppm (methylene groups)."""
        return ChemicalSpecies(
            name='Lipid_1.3',
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
            name='Water',
            chemical_shift_ppm=0.0,
            t1=1.0,
            t2=0.040,
        )
    
    @staticmethod
    def fat_main() -> ChemicalSpecies:
        """Main fat peak (methylene -CH2-)."""
        return ChemicalSpecies(
            name='Fat_main',
            chemical_shift_ppm=-3.4,  # Relative to water
            t1=0.35,
            t2=0.060,
            multiplicity=1,  # Simplified
        )
    
    @staticmethod
    def fat_olefinic() -> ChemicalSpecies:
        """Olefinic fat peak (-CH=CH-)."""
        return ChemicalSpecies(
            name='Fat_olefinic',
            chemical_shift_ppm=0.8,  # 5.3 ppm absolute, relative to water at 4.7
            t1=0.35,
            t2=0.060,
        )
    
    @staticmethod
    def fat_multipeak() -> List[ChemicalSpecies]:
        """Multi-peak fat model for Dixon imaging."""
        # Relative amplitudes based on literature
        return [
            ChemicalSpecies('Fat_A', -3.80, 0.35, 0.06),  # 0.9 ppm
            ChemicalSpecies('Fat_B', -3.40, 0.35, 0.06),  # 1.3 ppm (main)
            ChemicalSpecies('Fat_C', -2.60, 0.35, 0.06),  # 2.1 ppm
            ChemicalSpecies('Fat_D', -2.30, 0.35, 0.06),  # 2.4 ppm
            ChemicalSpecies('Fat_E', 0.60, 0.35, 0.06),   # 5.3 ppm
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
    field_strength : float
        B0 field strength in Tesla
    nucleus : str
        Nucleus type ('H1', 'C13', etc.)
    name : str
        Phantom name
    """
    shape: Tuple[int, ...]
    fov: Tuple[float, ...]
    species: List[ChemicalSpecies]
    concentration_maps: Dict[str, np.ndarray]
    t2_star_map: np.ndarray = None
    b0_map: np.ndarray = None
    field_strength: float = 3.0
    nucleus: str = 'H1'
    name: str = "Spectral Phantom"
    
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
        
        # Validate optional maps
        if self.t2_star_map is not None and self.t2_star_map.shape != self.shape:
            raise ValueError("T2* map shape must match phantom shape")
        
        if self.b0_map is not None and self.b0_map.shape != self.shape:
            raise ValueError("B0 map shape must match phantom shape")
    
    def _compute_coordinates(self):
        """Compute spatial coordinates."""
        ndim = len(self.shape)
        
        coords = []
        for i in range(ndim):
            n = self.shape[i]
            fov = self.fov[i]
            coords.append(np.linspace(-fov/2, fov/2, n, endpoint=True))
        
        if ndim == 1:
            self.positions = np.column_stack([
                coords[0],
                np.zeros_like(coords[0]),
                np.zeros_like(coords[0])
            ])
        elif ndim == 2:
            X, Y = np.meshgrid(coords[0], coords[1], indexing='ij')
            self.positions = np.column_stack([
                X.ravel(), Y.ravel(), np.zeros(np.prod(self.shape))
            ])
        else:
            X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
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
    def n_species(self) -> int:
        """Number of chemical species."""
        return len(self.species)
    
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
        if self.b0_map is not None:
            df = df + self.b0_map
        
        return df
    
    def get_species(self, name: str) -> Optional[ChemicalSpecies]:
        """Get species by name."""
        for s in self.species:
            if s.name == name:
                return s
        return None
    
    def get_frequency_offset(self, name: str) -> float:
        """Get frequency offset for species in Hz."""
        return self._frequency_offsets.get(name, 0.0)
    
    def get_total_concentration(self) -> np.ndarray:
        """Get sum of all species concentrations."""
        total = np.zeros(self.shape)
        for name, cmap in self.concentration_maps.items():
            total += cmap
        return total
    
    def get_species_properties(self, 
                               species_name: str = None) -> Dict[str, np.ndarray]:
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
        
        # Add B0 inhomogeneity to frequency offset
        if self.b0_map is not None:
            df = df + self.b0_map.ravel()
        
        return {
            'positions': self.positions.copy(),
            't1': t1,
            't2': t2,
            't2_star': t2_star,
            'df': df,
            'concentration': concentration,
        }
    
    def to_phantom(self, species_name: str = None) -> 'Phantom':
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
            t1_map=props['t1'].reshape(self.shape),
            t2_map=props['t2'].reshape(self.shape),
            pd_map=props['concentration'].reshape(self.shape),
            df_map=props['df'].reshape(self.shape),
            name=f"{self.name} - {species_name or 'combined'}",
        )
    
    def simulate_fid(self, 
                    acquisition_time: float = 0.5,
                    dwell_time: float = 0.5e-3,
                    line_broadening: float = 0.0) -> Dict:
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
            component = amplitude * np.exp(-time / t2_eff) * np.exp(-1j * 2 * np.pi * df * time)
            
            signal += component
        
        # Compute spectrum
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        frequency = np.fft.fftshift(np.fft.fftfreq(n_points, dwell_time))
        
        return {
            'time': time,
            'signal': signal,
            'frequency': frequency,
            'spectrum': spectrum,
            'dwell_time': dwell_time,
        }


class SpectralPhantomFactory:
    """Factory methods for creating common spectral phantoms."""
    
    @staticmethod
    def brain_mrs_voxel(field_strength: float = 3.0,
                       concentrations: Dict[str, float] = None) -> SpectralPhantom:
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
            'NAA': 12.0,
            'Creatine': 8.0,
            'Choline': 2.0,
            'myo-Inositol': 6.0,
            'Glutamate': 10.0,
            'Glutamine': 4.0,
            'Lactate': 0.5,
            'Water': 35000.0,  # Much higher than metabolites
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
        fov = (0.02, 0.02, 0.02)  # 2 cm voxel
        
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
    def brain_csi_grid(matrix_size: Tuple[int, int] = (16, 16),
                      fov: Tuple[float, float] = (0.16, 0.16),
                      field_strength: float = 3.0) -> SpectralPhantom:
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
        X, Y = np.meshgrid(x, y, indexing='ij')
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
            
            if s.name == 'NAA':
                cmap[wm_mask] = 10.0
                cmap[gm_mask] = 12.0
                cmap[csf_mask] = 0.0
            elif s.name == 'Creatine':
                cmap[wm_mask] = 6.0
                cmap[gm_mask] = 8.0
                cmap[csf_mask] = 0.0
            elif s.name == 'Choline':
                cmap[wm_mask] = 2.5
                cmap[gm_mask] = 1.5
                cmap[csf_mask] = 0.0
            elif s.name == 'myo-Inositol':
                cmap[wm_mask] = 4.0
                cmap[gm_mask] = 6.0
                cmap[csf_mask] = 0.0
            elif s.name == 'Water':
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
    def fat_water_phantom(matrix_size: Tuple[int, int] = (64, 64),
                         fov: Tuple[float, float] = (0.24, 0.24),
                         field_strength: float = 3.0,
                         multi_peak_fat: bool = True) -> SpectralPhantom:
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
        X, Y = np.meshgrid(x, y, indexing='ij')
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
        concentration_maps['Water'] = water_conc
        
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
            concentration_maps['Fat_main'] = fat_conc
        
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
        basic = fw_phantom.to_phantom('Water')
        print(f"  Basic phantom shape: {basic.shape}")
    
    print("\n✓ All tests passed!")
