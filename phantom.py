"""
phantom.py - MRI Phantom classes for heterogeneous tissue simulation

This module provides data structures for representing 2D/3D imaging phantoms
with spatially-varying tissue properties (T1, T2, proton density, frequency offset).

Author: Luca Nagel
Date: 2024/2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union
from pathlib import Path


@dataclass
class Phantom:
    """
    MRI phantom with spatially-varying tissue properties.
    
    A Phantom represents a 1D, 2D, or 3D imaging object where each voxel
    can have different T1, T2, proton density, and frequency offset values.
    This enables realistic simulation of heterogeneous tissue.
    
    Attributes
    ----------
    shape : tuple
        Spatial dimensions: (nx,) for 1D, (nx, ny) for 2D, or (nx, ny, nz) for 3D
    fov : tuple
        Field of view in meters for each dimension
    t1_map : ndarray
        T1 relaxation times in seconds, shape matches `shape`
    t2_map : ndarray
        T2 relaxation times in seconds, shape matches `shape`
    pd_map : ndarray, optional
        Proton density (relative), range [0, 1]. Default: all ones
    df_map : ndarray, optional
        Off-resonance frequency in Hz (for B0 inhomogeneity). Default: all zeros
    m0_map : ndarray, optional
        Initial magnetization, shape (*shape, 3) for [Mx, My, Mz]. Default: [0, 0, 1]
    mask : ndarray, optional
        Boolean mask indicating tissue vs. background. Default: all True
    name : str, optional
        Descriptive name for the phantom
    metadata : dict, optional
        Additional metadata (field strength, creation info, etc.)
    
    Examples
    --------
    >>> # Create a simple 2D phantom
    >>> t1 = np.ones((64, 64)) * 1.0  # 1 second T1
    >>> t2 = np.ones((64, 64)) * 0.1  # 100 ms T2
    >>> phantom = Phantom(
    ...     shape=(64, 64),
    ...     fov=(0.24, 0.24),  # 24 cm FOV
    ...     t1_map=t1,
    ...     t2_map=t2
    ... )
    >>> print(f"Phantom has {phantom.nvoxels} voxels")
    Phantom has 4096 voxels
    """
    shape: Tuple[int, ...]
    fov: Tuple[float, ...]
    t1_map: np.ndarray
    t2_map: np.ndarray
    pd_map: Optional[np.ndarray] = None
    df_map: Optional[np.ndarray] = None
    m0_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    name: str = "Phantom"
    metadata: Dict = field(default_factory=dict)
    
    # Computed fields (populated in __post_init__)
    positions: np.ndarray = field(init=False, repr=False)
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Validate inputs and compute derived quantities."""
        self._validate()
        self._set_defaults()
        self._compute_coordinates()
    
    def _validate(self):
        """Validate phantom configuration."""
        # Check dimensions
        ndim = len(self.shape)
        if ndim not in (1, 2, 3):
            raise ValueError(f"Phantom must be 1D, 2D, or 3D. Got {ndim}D shape: {self.shape}")
        
        if len(self.fov) != ndim:
            raise ValueError(f"FOV dimensions ({len(self.fov)}) must match shape dimensions ({ndim})")
        
        # Check T1/T2 maps
        if self.t1_map.shape != self.shape:
            raise ValueError(f"T1 map shape {self.t1_map.shape} doesn't match phantom shape {self.shape}")
        
        if self.t2_map.shape != self.shape:
            raise ValueError(f"T2 map shape {self.t2_map.shape} doesn't match phantom shape {self.shape}")
        
        # Check optional maps
        if self.pd_map is not None and self.pd_map.shape != self.shape:
            raise ValueError(f"PD map shape {self.pd_map.shape} doesn't match phantom shape {self.shape}")
        
        if self.df_map is not None and self.df_map.shape != self.shape:
            raise ValueError(f"DF map shape {self.df_map.shape} doesn't match phantom shape {self.shape}")
        
        if self.mask is not None and self.mask.shape != self.shape:
            raise ValueError(f"Mask shape {self.mask.shape} doesn't match phantom shape {self.shape}")
        
        if self.m0_map is not None:
            expected_m0_shape = (*self.shape, 3)
            if self.m0_map.shape != expected_m0_shape:
                raise ValueError(f"M0 map shape {self.m0_map.shape} should be {expected_m0_shape}")
    
    def _set_defaults(self):
        """Set default values for optional fields."""
        # Default proton density: all ones
        if self.pd_map is None:
            self.pd_map = np.ones(self.shape, dtype=np.float64)
        
        # Default frequency offset: all zeros
        if self.df_map is None:
            self.df_map = np.zeros(self.shape, dtype=np.float64)
        
        # Default initial magnetization: equilibrium [0, 0, 1]
        if self.m0_map is None:
            self.m0_map = np.zeros((*self.shape, 3), dtype=np.float64)
            self.m0_map[..., 2] = 1.0
        
        # Default mask: all tissue
        if self.mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
    
    def _compute_coordinates(self):
        """Compute position arrays for each voxel."""
        ndim = len(self.shape)
        
        # Create coordinate arrays for each dimension
        coords = []
        for i in range(ndim):
            n = self.shape[i]
            fov = self.fov[i]
            # Center coordinates at 0
            coords.append(np.linspace(-fov/2, fov/2, n, endpoint=True))
        
        if ndim == 1:
            self.x = coords[0]
            self.y = np.zeros(1)
            self.z = np.zeros(1)
            self.positions = np.column_stack([
                self.x,
                np.zeros_like(self.x),
                np.zeros_like(self.x)
            ])
        elif ndim == 2:
            X, Y = np.meshgrid(coords[0], coords[1], indexing='ij')
            self.x = X.ravel()
            self.y = Y.ravel()
            self.z = np.zeros_like(self.x)
            self.positions = np.column_stack([self.x, self.y, self.z])
        else:  # 3D
            X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
            self.x = X.ravel()
            self.y = Y.ravel()
            self.z = Z.ravel()
            self.positions = np.column_stack([self.x, self.y, self.z])
    
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
        """Number of active (non-masked) voxels."""
        return int(self.mask.sum())
    
    @property
    def resolution(self) -> Tuple[float, ...]:
        """Voxel size in meters for each dimension."""
        return tuple(f / n for f, n in zip(self.fov, self.shape))
    
    def get_flat_properties(self) -> Dict[str, np.ndarray]:
        """
        Return flattened arrays for simulation.
        
        All property maps are flattened to 1D arrays suitable for
        passing to the simulation core.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'positions': (nvoxels, 3) array of [x, y, z] coordinates in meters
            - 't1': (nvoxels,) array of T1 values in seconds
            - 't2': (nvoxels,) array of T2 values in seconds
            - 'pd': (nvoxels,) array of proton density values
            - 'df': (nvoxels,) array of frequency offsets in Hz
            - 'm0': (nvoxels, 3) array of initial magnetization [Mx, My, Mz]
            - 'mask': (nvoxels,) boolean array
        """
        return {
            'positions': self.positions.copy(),
            't1': self.t1_map.ravel().copy(),
            't2': self.t2_map.ravel().copy(),
            'pd': self.pd_map.ravel().copy(),
            'df': self.df_map.ravel().copy(),
            'm0': self.m0_map.reshape(-1, 3).copy(),
            'mask': self.mask.ravel().copy(),
        }
    
    def get_active_properties(self) -> Dict[str, np.ndarray]:
        """
        Return flattened arrays for active (non-masked) voxels only.
        
        Returns
        -------
        dict
            Same structure as get_flat_properties(), but only including
            voxels where mask is True.
        """
        flat = self.get_flat_properties()
        active = flat['mask']
        
        return {
            'positions': flat['positions'][active],
            't1': flat['t1'][active],
            't2': flat['t2'][active],
            'pd': flat['pd'][active],
            'df': flat['df'][active],
            'm0': flat['m0'][active],
            'mask': flat['mask'][active],
            'indices': np.where(active)[0],  # Original indices for reconstruction
        }
    
    def reshape_result(self, flat_array: np.ndarray, 
                       has_time: bool = True) -> np.ndarray:
        """
        Reshape flat simulation result back to phantom shape.
        
        Parameters
        ----------
        flat_array : ndarray
            Flattened result from simulation.
            Shape: (ntime, nvoxels) if has_time=True, else (nvoxels,)
        has_time : bool
            Whether the first dimension is time.
            
        Returns
        -------
        ndarray
            Reshaped array with shape (*self.shape,) or (ntime, *self.shape)
        """
        if has_time:
            ntime = flat_array.shape[0]
            return flat_array.reshape((ntime,) + self.shape)
        else:
            return flat_array.reshape(self.shape)
    
    def reconstruct_from_active(self, active_array: np.ndarray,
                                indices: np.ndarray,
                                has_time: bool = True,
                                fill_value: float = 0.0) -> np.ndarray:
        """
        Reconstruct full phantom-shaped array from active-only simulation result.
        
        Parameters
        ----------
        active_array : ndarray
            Result for active voxels only.
            Shape: (ntime, n_active) if has_time=True, else (n_active,)
        indices : ndarray
            Original indices of active voxels (from get_active_properties)
        has_time : bool
            Whether the first dimension is time
        fill_value : float
            Value to use for masked voxels
            
        Returns
        -------
        ndarray
            Full array with shape (ntime, *self.shape) or (*self.shape,)
        """
        if has_time:
            ntime = active_array.shape[0]
            full = np.full((ntime, self.nvoxels), fill_value, dtype=active_array.dtype)
            full[:, indices] = active_array
            return full.reshape((ntime,) + self.shape)
        else:
            full = np.full(self.nvoxels, fill_value, dtype=active_array.dtype)
            full[indices] = active_array
            return full.reshape(self.shape)
    
    def copy(self) -> 'Phantom':
        """Create a deep copy of the phantom."""
        return Phantom(
            shape=self.shape,
            fov=self.fov,
            t1_map=self.t1_map.copy(),
            t2_map=self.t2_map.copy(),
            pd_map=self.pd_map.copy() if self.pd_map is not None else None,
            df_map=self.df_map.copy() if self.df_map is not None else None,
            m0_map=self.m0_map.copy() if self.m0_map is not None else None,
            mask=self.mask.copy() if self.mask is not None else None,
            name=self.name,
            metadata=self.metadata.copy()
        )
    
    def save(self, filename: Union[str, Path]):
        """
        Save phantom to file.
        
        Supports .npz (NumPy) and .h5 (HDF5) formats.
        
        Parameters
        ----------
        filename : str or Path
            Output filename. Extension determines format.
        """
        filename = Path(filename)
        
        data = {
            'shape': np.array(self.shape),
            'fov': np.array(self.fov),
            't1_map': self.t1_map,
            't2_map': self.t2_map,
            'pd_map': self.pd_map,
            'df_map': self.df_map,
            'm0_map': self.m0_map,
            'mask': self.mask,
            'name': np.array(self.name),
        }
        
        if filename.suffix == '.npz':
            np.savez_compressed(filename, **data)
        elif filename.suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(filename, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, str):
                        f.attrs[key] = value
                    else:
                        f.create_dataset(key, data=value)
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")
    
    @classmethod
    def load(cls, filename: Union[str, Path]) -> 'Phantom':
        """
        Load phantom from file.
        
        Parameters
        ----------
        filename : str or Path
            Input filename (.npz or .h5/.hdf5)
            
        Returns
        -------
        Phantom
            Loaded phantom object
        """
        filename = Path(filename)
        
        if filename.suffix == '.npz':
            data = np.load(filename, allow_pickle=True)
            return cls(
                shape=tuple(data['shape']),
                fov=tuple(data['fov']),
                t1_map=data['t1_map'],
                t2_map=data['t2_map'],
                pd_map=data['pd_map'],
                df_map=data['df_map'],
                m0_map=data['m0_map'],
                mask=data['mask'],
                name=str(data['name']),
            )
        elif filename.suffix in ('.h5', '.hdf5'):
            import h5py
            with h5py.File(filename, 'r') as f:
                return cls(
                    shape=tuple(f['shape'][...]),
                    fov=tuple(f['fov'][...]),
                    t1_map=f['t1_map'][...],
                    t2_map=f['t2_map'][...],
                    pd_map=f['pd_map'][...],
                    df_map=f['df_map'][...],
                    m0_map=f['m0_map'][...],
                    mask=f['mask'][...],
                    name=f.attrs.get('name', 'Phantom'),
                )
        else:
            raise ValueError(f"Unsupported file format: {filename.suffix}")
    
    def __repr__(self):
        return (f"Phantom(name='{self.name}', shape={self.shape}, "
                f"fov={self.fov}, active={self.n_active}/{self.nvoxels})")


class PhantomFactory:
    """
    Factory methods for creating common phantom types.
    
    This class provides static methods to create standard phantoms
    like Shepp-Logan, spherical phantoms, and phantoms from segmentations.
    """
    
    # Tissue parameters at different field strengths (T1, T2 in seconds)
    TISSUE_PARAMS = {
        'gray_matter': {
            1.5: (0.95, 0.100, 0.95),   # (T1, T2, PD)
            3.0: (1.33, 0.083, 0.95),
            7.0: (1.92, 0.047, 0.95),
        },
        'white_matter': {
            1.5: (0.65, 0.070, 0.77),
            3.0: (0.83, 0.070, 0.77),
            7.0: (1.22, 0.046, 0.77),
        },
        'csf': {
            1.5: (2.5, 2.0, 1.0),
            3.0: (3.8, 2.0, 1.0),
            7.0: (4.4, 1.5, 1.0),
        },
        'fat': {
            1.5: (0.27, 0.085, 1.0),
            3.0: (0.37, 0.133, 1.0),
            7.0: (0.52, 0.046, 1.0),
        },
        'muscle': {
            1.5: (0.87, 0.047, 1.0),
            3.0: (1.42, 0.032, 1.0),
            7.0: (1.90, 0.022, 1.0),
        },
        'blood': {
            1.5: (1.44, 0.29, 1.0),
            3.0: (1.93, 0.275, 1.0),
            7.0: (2.59, 0.11, 1.0),
        },
    }
    
    @classmethod
    def get_tissue_params(cls, tissue_name: str, field_strength: float = 3.0):
        """
        Get T1, T2, PD for a tissue type at a given field strength.
        
        Parameters
        ----------
        tissue_name : str
            Tissue name (e.g., 'gray_matter', 'white_matter')
        field_strength : float
            B0 field strength in Tesla
            
        Returns
        -------
        tuple
            (T1, T2, PD) - T1 and T2 in seconds, PD relative (0-1)
        """
        if tissue_name not in cls.TISSUE_PARAMS:
            raise ValueError(f"Unknown tissue: {tissue_name}. "
                           f"Available: {list(cls.TISSUE_PARAMS.keys())}")
        
        params = cls.TISSUE_PARAMS[tissue_name]
        
        if field_strength in params:
            return params[field_strength]
        else:
            # Interpolate between known field strengths
            fields = sorted(params.keys())
            if field_strength < fields[0]:
                return params[fields[0]]
            elif field_strength > fields[-1]:
                return params[fields[-1]]
            else:
                # Linear interpolation
                for i in range(len(fields) - 1):
                    if fields[i] <= field_strength <= fields[i + 1]:
                        f1, f2 = fields[i], fields[i + 1]
                        t = (field_strength - f1) / (f2 - f1)
                        p1, p2 = params[f1], params[f2]
                        return tuple(p1[j] + t * (p2[j] - p1[j]) for j in range(3))
    
    @staticmethod
    def uniform(shape: Tuple[int, ...], fov: Tuple[float, ...],
                t1: float, t2: float, pd: float = 1.0,
                name: str = "Uniform") -> Phantom:
        """
        Create a uniform phantom with constant tissue properties.
        
        Parameters
        ----------
        shape : tuple
            Spatial dimensions
        fov : tuple
            Field of view in meters
        t1 : float
            T1 relaxation time in seconds
        t2 : float
            T2 relaxation time in seconds
        pd : float
            Proton density (0-1)
        name : str
            Phantom name
            
        Returns
        -------
        Phantom
        """
        return Phantom(
            shape=shape,
            fov=fov,
            t1_map=np.full(shape, t1, dtype=np.float64),
            t2_map=np.full(shape, t2, dtype=np.float64),
            pd_map=np.full(shape, pd, dtype=np.float64),
            name=name
        )
    
    @staticmethod
    def shepp_logan_2d(n: int = 256, fov: float = 0.24,
                      field_strength: float = 3.0) -> Phantom:
        """
        Create 2D Shepp-Logan phantom with realistic T1/T2 values.
        
        The Shepp-Logan phantom is a standard test phantom used in
        medical imaging. This version assigns realistic brain tissue
        properties based on the phantom regions.
        
        Parameters
        ----------
        n : int
            Matrix size (n x n)
        fov : float
            Field of view in meters
        field_strength : float
            B0 field strength in Tesla (affects T1/T2 values)
            
        Returns
        -------
        Phantom
            Shepp-Logan phantom with tissue-specific T1/T2
        """
        # Generate Shepp-Logan intensity phantom
        intensity = PhantomFactory._shepp_logan_intensities(n)
        
        # Get tissue parameters
        gm = PhantomFactory.get_tissue_params('gray_matter', field_strength)
        wm = PhantomFactory.get_tissue_params('white_matter', field_strength)
        csf = PhantomFactory.get_tissue_params('csf', field_strength)
        
        # Initialize maps
        t1_map = np.zeros((n, n), dtype=np.float64)
        t2_map = np.zeros((n, n), dtype=np.float64)
        pd_map = np.zeros((n, n), dtype=np.float64)
        
        # Map intensities to tissue types
        # Background (air)
        mask = intensity > 0.01
        
        # CSF (highest intensity regions - ventricles)
        csf_mask = intensity > 0.9
        t1_map[csf_mask] = csf[0]
        t2_map[csf_mask] = csf[1]
        pd_map[csf_mask] = csf[2]
        
        # Gray matter (medium-high intensity - cortex)
        gm_mask = (intensity > 0.2) & (intensity <= 0.9)
        t1_map[gm_mask] = gm[0]
        t2_map[gm_mask] = gm[1]
        pd_map[gm_mask] = gm[2]
        
        # White matter (lower intensity - inner brain)
        wm_mask = (intensity > 0.01) & (intensity <= 0.2)
        t1_map[wm_mask] = wm[0]
        t2_map[wm_mask] = wm[1]
        pd_map[wm_mask] = wm[2]
        
        return Phantom(
            shape=(n, n),
            fov=(fov, fov),
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            mask=mask,
            name=f"Shepp-Logan {n}x{n} @ {field_strength}T",
            metadata={'field_strength': field_strength, 'type': 'shepp_logan'}
        )
    
    @staticmethod
    def _shepp_logan_intensities(n: int) -> np.ndarray:
        """
        Generate Shepp-Logan phantom intensity image.
        
        Uses the modified Shepp-Logan parameters for better contrast.
        """
        # Ellipse parameters: (intensity, a, b, x0, y0, phi)
        # a, b are semi-axes, (x0, y0) is center, phi is rotation angle
        ellipses = [
            (1.0, 0.69, 0.92, 0, 0, 0),              # Outer skull
            (-0.80, 0.6624, 0.8740, 0, -0.0184, 0),   # Brain
            (-0.20, 0.1100, 0.3100, 0.22, 0, -18),    # Right ventricle
            (-0.20, 0.1600, 0.4100, -0.22, 0, 18),    # Left ventricle
            (0.10, 0.2100, 0.2500, 0, 0.35, 0),       # Top blob
            (0.10, 0.0460, 0.0460, 0, 0.1, 0),        # Small blob 1
            (0.10, 0.0460, 0.0460, 0, -0.1, 0),       # Small blob 2
            (0.10, 0.0460, 0.0230, -0.08, -0.605, 0), # Small blob 3
            (0.10, 0.0230, 0.0230, 0, -0.606, 0),     # Small blob 4
            (0.10, 0.0230, 0.0460, 0.06, -0.605, 0),  # Small blob 5
        ]
        
        # Create coordinate grid
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        image = np.zeros((n, n), dtype=np.float64)
        
        for intensity, a, b, x0, y0, phi in ellipses:
            phi_rad = np.deg2rad(phi)
            
            # Rotate coordinates
            cos_phi = np.cos(phi_rad)
            sin_phi = np.sin(phi_rad)
            
            X_rot = (X - x0) * cos_phi + (Y - y0) * sin_phi
            Y_rot = -(X - x0) * sin_phi + (Y - y0) * cos_phi
            
            # Check if inside ellipse
            inside = (X_rot / a) ** 2 + (Y_rot / b) ** 2 <= 1
            
            image[inside] += intensity
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, None)
        if image.max() > 0:
            image = image / image.max()
        
        return image
    
    @staticmethod
    def spherical_3d(n: int = 64, fov: float = 0.24,
                    tissue: str = 'gray_matter',
                    field_strength: float = 3.0,
                    radius_fraction: float = 0.4) -> Phantom:
        """
        Create 3D spherical phantom.
        
        Parameters
        ----------
        n : int
            Matrix size (n x n x n)
        fov : float
            Field of view in meters
        tissue : str
            Tissue type name
        field_strength : float
            B0 field strength in Tesla
        radius_fraction : float
            Sphere radius as fraction of FOV (0-0.5)
            
        Returns
        -------
        Phantom
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, n)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        # Sphere mask
        radius = fov * radius_fraction
        mask = R < radius
        
        # Get tissue parameters
        t1, t2, pd = PhantomFactory.get_tissue_params(tissue, field_strength)
        
        t1_map = np.where(mask, t1, 0.0)
        t2_map = np.where(mask, t2, 0.0)
        pd_map = np.where(mask, pd, 0.0)
        
        return Phantom(
            shape=(n, n, n),
            fov=(fov, fov, fov),
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            mask=mask,
            name=f"Spherical {n}^3 {tissue} @ {field_strength}T",
            metadata={'field_strength': field_strength, 'tissue': tissue}
        )
    
    @staticmethod
    def cylindrical_2d(n: int = 64, fov: float = 0.24,
                      tissue: str = 'gray_matter',
                      field_strength: float = 3.0,
                      radius_fraction: float = 0.4) -> Phantom:
        """
        Create 2D circular (cylindrical cross-section) phantom.
        
        Parameters
        ----------
        n : int
            Matrix size (n x n)
        fov : float
            Field of view in meters
        tissue : str
            Tissue type name
        field_strength : float
            B0 field strength in Tesla
        radius_fraction : float
            Circle radius as fraction of FOV
            
        Returns
        -------
        Phantom
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, n)
        X, Y = np.meshgrid(x, x, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        
        # Circle mask
        radius = fov * radius_fraction
        mask = R < radius
        
        # Get tissue parameters
        t1, t2, pd = PhantomFactory.get_tissue_params(tissue, field_strength)
        
        t1_map = np.where(mask, t1, 0.0)
        t2_map = np.where(mask, t2, 0.0)
        pd_map = np.where(mask, pd, 0.0)
        
        return Phantom(
            shape=(n, n),
            fov=(fov, fov),
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            mask=mask,
            name=f"Cylindrical {n}x{n} {tissue} @ {field_strength}T",
            metadata={'field_strength': field_strength, 'tissue': tissue}
        )
    
    @staticmethod
    def multi_tissue_2d(n: int = 64, fov: float = 0.24,
                       field_strength: float = 3.0) -> Phantom:
        """
        Create 2D phantom with multiple tissue types arranged in quadrants.
        
        Useful for testing tissue contrast.
        
        Returns
        -------
        Phantom
            Phantom with gray matter, white matter, CSF, and fat in quadrants
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, n)
        X, Y = np.meshgrid(x, x, indexing='ij')
        
        # Initialize maps
        t1_map = np.zeros((n, n), dtype=np.float64)
        t2_map = np.zeros((n, n), dtype=np.float64)
        pd_map = np.zeros((n, n), dtype=np.float64)
        mask = np.zeros((n, n), dtype=bool)
        
        # Circle in each quadrant
        radius = fov / 6
        quadrants = [
            ('gray_matter', -fov/4, fov/4),
            ('white_matter', fov/4, fov/4),
            ('csf', -fov/4, -fov/4),
            ('fat', fov/4, -fov/4),
        ]
        
        for tissue, cx, cy in quadrants:
            R = np.sqrt((X - cx)**2 + (Y - cy)**2)
            circle_mask = R < radius
            
            t1, t2, pd = PhantomFactory.get_tissue_params(tissue, field_strength)
            
            t1_map[circle_mask] = t1
            t2_map[circle_mask] = t2
            pd_map[circle_mask] = pd
            mask |= circle_mask
        
        return Phantom(
            shape=(n, n),
            fov=(fov, fov),
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            mask=mask,
            name=f"Multi-tissue {n}x{n} @ {field_strength}T",
            metadata={'field_strength': field_strength}
        )
    
    @staticmethod
    def from_segmentation(seg_labels: np.ndarray,
                         label_to_tissue: Dict[int, str],
                         fov: Tuple[float, ...],
                         field_strength: float = 3.0,
                         name: str = "Segmented") -> Phantom:
        """
        Create phantom from segmentation labels.
        
        Parameters
        ----------
        seg_labels : ndarray
            Integer array with tissue labels (0 = background)
        label_to_tissue : dict
            Mapping from label to tissue name
            e.g., {1: 'gray_matter', 2: 'white_matter', 3: 'csf'}
        fov : tuple
            Field of view in meters for each dimension
        field_strength : float
            B0 field strength in Tesla
        name : str
            Phantom name
            
        Returns
        -------
        Phantom
        """
        shape = seg_labels.shape
        
        t1_map = np.zeros(shape, dtype=np.float64)
        t2_map = np.zeros(shape, dtype=np.float64)
        pd_map = np.zeros(shape, dtype=np.float64)
        mask = np.zeros(shape, dtype=bool)
        
        for label, tissue_name in label_to_tissue.items():
            label_mask = seg_labels == label
            if not label_mask.any():
                continue
            
            t1, t2, pd = PhantomFactory.get_tissue_params(tissue_name, field_strength)
            
            t1_map[label_mask] = t1
            t2_map[label_mask] = t2
            pd_map[label_mask] = pd
            mask |= label_mask
        
        return Phantom(
            shape=shape,
            fov=fov,
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            mask=mask,
            name=name,
            metadata={'field_strength': field_strength, 'labels': label_to_tissue}
        )
    
    @staticmethod
    def chemical_shift_phantom(n: int = 64, fov: float = 0.10,
                              field_strength: float = 3.0) -> Phantom:
        """
        Create phantom with water and fat regions for chemical shift imaging.
        
        Parameters
        ----------
        n : int
            Matrix size
        fov : float
            Field of view in meters
        field_strength : float
            B0 field strength in Tesla (affects chemical shift in Hz)
            
        Returns
        -------
        Phantom
            Phantom with separate water and fat regions, including
            frequency offset for fat chemical shift
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, n)
        X, Y = np.meshgrid(x, x, indexing='ij')
        R = np.sqrt(X**2 + Y**2)
        
        # Water region (center circle)
        water_radius = fov / 4
        water_mask = R < water_radius
        
        # Fat region (ring around center)
        fat_inner = fov / 4
        fat_outer = fov / 3
        fat_mask = (R >= fat_inner) & (R < fat_outer)
        
        # Combined mask
        mask = water_mask | fat_mask
        
        # Get tissue parameters
        water_t1, water_t2, water_pd = 3.8, 2.0, 1.0  # CSF-like for water
        fat_t1, fat_t2, fat_pd = PhantomFactory.get_tissue_params('fat', field_strength)
        
        t1_map = np.zeros((n, n), dtype=np.float64)
        t2_map = np.zeros((n, n), dtype=np.float64)
        pd_map = np.zeros((n, n), dtype=np.float64)
        df_map = np.zeros((n, n), dtype=np.float64)
        
        # Water
        t1_map[water_mask] = water_t1
        t2_map[water_mask] = water_t2
        pd_map[water_mask] = water_pd
        
        # Fat
        t1_map[fat_mask] = fat_t1
        t2_map[fat_mask] = fat_t2
        pd_map[fat_mask] = fat_pd
        
        # Chemical shift: fat is ~3.5 ppm from water
        # Proton frequency at different fields:
        # 1.5T: 63.87 MHz, 3.0T: 127.74 MHz, 7.0T: 298.06 MHz
        proton_freq = {1.5: 63.87e6, 3.0: 127.74e6, 7.0: 298.06e6}
        freq_hz = proton_freq.get(field_strength, 42.576e6 * field_strength)
        fat_shift_ppm = -3.5
        fat_shift_hz = fat_shift_ppm * 1e-6 * freq_hz
        
        df_map[fat_mask] = fat_shift_hz
        
        return Phantom(
            shape=(n, n),
            fov=(fov, fov),
            t1_map=t1_map,
            t2_map=t2_map,
            pd_map=pd_map,
            df_map=df_map,
            mask=mask,
            name=f"Water-Fat {n}x{n} @ {field_strength}T",
            metadata={
                'field_strength': field_strength,
                'fat_shift_hz': fat_shift_hz,
                'water_mask': water_mask,
                'fat_mask': fat_mask,
            }
        )


# Utility functions

def display_phantom(phantom: Phantom, slice_idx: Optional[int] = None,
                   show_all_maps: bool = True):
    """
    Display phantom maps using matplotlib.
    
    Parameters
    ----------
    phantom : Phantom
        Phantom to display
    slice_idx : int, optional
        For 3D phantoms, which slice to show. Default: middle slice
    show_all_maps : bool
        If True, show T1, T2, PD, and mask. If False, only T1.
    """
    import matplotlib.pyplot as plt
    
    # Extract slice if 3D
    if phantom.ndim == 3:
        if slice_idx is None:
            slice_idx = phantom.shape[2] // 2
        t1 = phantom.t1_map[:, :, slice_idx]
        t2 = phantom.t2_map[:, :, slice_idx]
        pd = phantom.pd_map[:, :, slice_idx]
        mask = phantom.mask[:, :, slice_idx]
    elif phantom.ndim == 2:
        t1 = phantom.t1_map
        t2 = phantom.t2_map
        pd = phantom.pd_map
        mask = phantom.mask
    else:
        raise ValueError("display_phantom only supports 2D and 3D phantoms")
    
    if show_all_maps:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        im0 = axes[0, 0].imshow(t1.T, origin='lower', cmap='viridis')
        axes[0, 0].set_title('T1 (s)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(t2.T, origin='lower', cmap='plasma')
        axes[0, 1].set_title('T2 (s)')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[1, 0].imshow(pd.T, origin='lower', cmap='gray')
        axes[1, 0].set_title('Proton Density')
        plt.colorbar(im2, ax=axes[1, 0])
        
        im3 = axes[1, 1].imshow(mask.T, origin='lower', cmap='gray')
        axes[1, 1].set_title('Mask')
        plt.colorbar(im3, ax=axes[1, 1])
        
        fig.suptitle(phantom.name)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(t1.T, origin='lower', cmap='viridis')
        ax.set_title(f'{phantom.name} - T1 (s)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Phantom Module Examples")
    print("=" * 50)
    
    # Create Shepp-Logan phantom
    shepp = PhantomFactory.shepp_logan_2d(64, 0.24, 3.0)
    print(f"\nCreated: {shepp}")
    print(f"Resolution: {shepp.resolution}")
    
    # Create multi-tissue phantom
    multi = PhantomFactory.multi_tissue_2d(64, 0.24, 3.0)
    print(f"\nCreated: {multi}")
    
    # Create 3D sphere
    sphere = PhantomFactory.spherical_3d(32, 0.24, 'gray_matter', 3.0)
    print(f"\nCreated: {sphere}")
    
    # Test flat properties
    props = shepp.get_flat_properties()
    print(f"\nFlat properties shapes:")
    for key, arr in props.items():
        print(f"  {key}: {arr.shape}")
    
    # Test active properties (masked)
    active = shepp.get_active_properties()
    print(f"\nActive voxels: {len(active['t1'])} / {shepp.nvoxels}")
