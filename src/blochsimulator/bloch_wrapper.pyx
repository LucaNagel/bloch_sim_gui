# bloch_wrapper.pyx - Cython wrapper for Bloch simulator
# This file provides Python bindings for the C implementation

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt

# Import the C functions
cdef extern from "bloch_core.h":
    int blochsim(double *b1real, double *b1imag,
                 double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                 int ntime, double *e1, double *e2, double df,
                 double dx, double dy, double dz,
                 double *mx, double *my, double *mz, int mode) nogil
    
    int blochsimfz(double *b1r, double *b1i, double *gx, double *gy, double *gz,
                   double *tp, int ntime, double t1, double t2, double *df, int nf,
                   double *dx, double *dy, double *dz, int npos,
                   double *mx, double *my, double *mz, int mode) nogil
    
    void blochsim_batch_optimized(double *b1real, double *b1imag,
                                  double *xgrad, double *ygrad, double *zgrad, double *tsteps,
                                  int ntime, double t1, double t2,
                                  double *df, int nf,
                                  double *dx, double *dy, double *dz, int npos,
                                  double *mx, double *my, double *mz,
                                  int mode, int num_threads) nogil
    
    void calculate_relaxation(double t1, double t2, double dt, double *e1, double *e2) nogil
    void set_equilibrium_magnetization(double *mx, double *my, double *mz, int n) nogil
    
    # Heterogeneous phantom simulation functions
    void blochsim_heterogeneous(
        double *b1real, double *b1imag,
        double *gx, double *gy, double *gz, double *tsteps,
        int ntime,
        double *t1_arr, double *t2_arr, double *df_arr,
        double *dx, double *dy, double *dz,
        double *mx_init, double *my_init, double *mz_init,
        int nvoxels,
        double *mx, double *my, double *mz,
        int mode, int num_threads) nogil
    
    void blochsim_heterogeneous_grouped(
        double *b1real, double *b1imag,
        double *gx, double *gy, double *gz, double *tsteps,
        int ntime,
        int *tissue_labels,
        double *t1_per_label, double *t2_per_label,
        int nlabels,
        double *df_arr,
        double *dx, double *dy, double *dz,
        double *mx_init, double *my_init, double *mz_init,
        int nvoxels,
        double *mx, double *my, double *mz,
        int mode, int num_threads) nogil

# Type definitions
ctypedef np.float64_t DTYPE_t
ctypedef np.complex128_t CTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray[DTYPE_t, ndim=2] _coerce_three_columns(np.ndarray arr, int rows, str name):
    """Ensure a (rows, 3) float64 C-contiguous array, padding/truncating columns as needed."""
    if arr.ndim != 2:
        raise ValueError("%s must be 2D with shape (%d, 3); got ndim=%d" % (name, rows, arr.ndim))
    if arr.shape[0] != rows:
        raise ValueError("%s must have shape (%d, 3); got (%d, %d)" % (name, rows, arr.shape[0], arr.shape[1]))

    cdef np.ndarray[DTYPE_t, ndim=2] arr_c = np.ascontiguousarray(arr, dtype=np.float64)
    if arr_c.shape[1] == 3:
        return arr_c

    cdef np.ndarray[DTYPE_t, ndim=2] padded = np.zeros((rows, 3), dtype=np.float64)
    cdef int copy_cols = 3 if arr_c.shape[1] > 3 else arr_c.shape[1]
    if copy_cols > 0:
        padded[:, :copy_cols] = arr_c[:, :copy_cols]
    return padded

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_bloch(np.ndarray[CTYPE_t, ndim=1] b1_complex,
                   np.ndarray[DTYPE_t, ndim=2] gradients,
                   np.ndarray[DTYPE_t, ndim=1] time_points,
                   double t1, double t2,
                   np.ndarray[DTYPE_t, ndim=1] frequencies,
                   np.ndarray[DTYPE_t, ndim=2] positions,
                   np.ndarray[DTYPE_t, ndim=2] m_init=None,
                   int mode=0,
                   int num_threads=1):
    """
    Simulate Bloch equations for MRI physics.
    
    Parameters
    ----------
    b1_complex : ndarray, shape (ntime,)
        Complex RF pulse (B1 field) in Gauss
    gradients : ndarray, shape (ntime, 3)
        Gradient waveforms [Gx, Gy, Gz] in Gauss/cm
    time_points : ndarray, shape (ntime,)
        Time intervals in seconds
    t1 : float
        Longitudinal relaxation time in seconds
    t2 : float
        Transverse relaxation time in seconds
    frequencies : ndarray, shape (nfreq,)
        Off-resonance frequencies in Hz
    positions : ndarray, shape (npos, 3)
        Spatial positions [x, y, z] in cm
    m_init : ndarray, shape (3, nfreq*npos), optional
        Initial magnetization [Mx, My, Mz]. Default is equilibrium [0, 0, 1]
    mode : int, optional
        Simulation mode:
        - 0: Transient from initial condition to endpoint (default)
        - 1: Steady-state simulation
        - 2: Transient with all time points output
        - 3: Steady-state with all time points output
    num_threads : int, optional
        Number of threads for parallel computation (default: 1)
    
    Returns
    -------
    mx, my, mz : ndarray
        Magnetization components
        Shape depends on mode:
        - mode 0,1: (npos, nfreq)
        - mode 2,3: (ntime, npos, nfreq)
    
    Notes
    -----
    This function wraps the C implementation of the Bloch simulator,
    originally developed by Brian Hargreaves at Stanford University.
    
    The simulator uses rotation matrices for RF and gradient effects,
    followed by relaxation/recovery steps at each time point.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Simple FID simulation
    >>> ntime = 1000
    >>> dt = 1e-5  # 10 us
    >>> time_points = np.ones(ntime) * dt
    >>> 
    >>> # 90-degree pulse
    >>> b1 = np.zeros(ntime, dtype=complex)
    >>> b1[0] = 0.1 + 0j  # 0.1 G for 10 us
    >>> 
    >>> # No gradients
    >>> gradients = np.zeros((ntime, 3))
    >>> 
    >>> # Single position, single frequency
    >>> positions = np.array([[0, 0, 0]])
    >>> frequencies = np.array([0])
    >>> 
    >>> # Simulate
    >>> mx, my, mz = simulate_bloch(b1, gradients, time_points,
    ...                            t1=1.0, t2=0.1,
    ...                            frequencies=frequencies,
    ...                            positions=positions)
    """
    
    cdef int ntime = len(b1_complex)
    cdef int nfreq = len(frequencies)
    cdef int npos = len(positions)
    cdef int ntout = ntime if (mode & 2) else 1
    cdef int nfnpos = nfreq * npos
    cdef int ntnfnpos = ntout * nfnpos
    cdef np.ndarray[DTYPE_t, ndim=2] m_init_arr
    
    cdef np.ndarray[CTYPE_t, ndim=1] b1_c = np.ascontiguousarray(b1_complex, dtype=np.complex128)
    cdef np.ndarray[DTYPE_t, ndim=1] time_c = np.ascontiguousarray(time_points, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] freq_c = np.ascontiguousarray(frequencies, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] grad_c = _coerce_three_columns(gradients, ntime, "gradients")
    cdef np.ndarray[DTYPE_t, ndim=2] pos_c = _coerce_three_columns(positions, npos, "positions")
    cdef Py_ssize_t time_len = time_c.shape[0]
    cdef Py_ssize_t freq_len = freq_c.shape[0] if freq_c.ndim == 1 else -1
    if time_len != ntime:
        raise ValueError("time_points must have length %d; got %d" % (ntime, time_len))
    if freq_len != nfreq:
        raise ValueError("frequencies must have length %d; got %d" % (nfreq, freq_len))

    # Real/imag views from a complex array are strided; copy directly into contiguous buffers
    cdef np.ndarray[DTYPE_t, ndim=1] b1_real = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_imag = np.empty(ntime, dtype=np.float64)
    np.copyto(b1_real, b1_c.real)
    np.copyto(b1_imag, b1_c.imag)

    # Extract gradient components without extra temporary arrays
    cdef np.ndarray[DTYPE_t, ndim=1] gx = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] gy = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] gz = np.empty(ntime, dtype=np.float64)
    np.copyto(gx, grad_c[:, 0])
    np.copyto(gy, grad_c[:, 1])
    np.copyto(gz, grad_c[:, 2])

    # Extract position components
    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.empty(npos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dy = np.empty(npos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dz = np.empty(npos, dtype=np.float64)
    np.copyto(dx, pos_c[:, 0])
    np.copyto(dy, pos_c[:, 1])
    np.copyto(dz, pos_c[:, 2])

    # Allocate output arrays
    cdef np.ndarray[DTYPE_t, ndim=1] mx_buf = np.empty(ntnfnpos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] my_buf = np.empty(ntnfnpos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mz_buf = np.empty(ntnfnpos, dtype=np.float64)

    # Set initial magnetization
    cdef slice start_slice = slice(0, ntnfnpos, ntout)
    if m_init is not None:
        m_init_arr = np.ascontiguousarray(m_init, dtype=np.float64)
        if m_init_arr.shape[0] != 3 or m_init_arr.shape[1] != nfnpos:
            raise ValueError("m_init must have shape (3, nfreq * npos)")
        mx_buf[start_slice] = m_init_arr[0]
        my_buf[start_slice] = m_init_arr[1]
        mz_buf[start_slice] = m_init_arr[2]
    else:
        mx_buf[start_slice] = 0.0
        my_buf[start_slice] = 0.0
        mz_buf[start_slice] = 1.0

    # Call the C function
    blochsimfz(<double*>b1_real.data, <double*>b1_imag.data,
               <double*>gx.data, <double*>gy.data, <double*>gz.data,
               <double*>time_c.data, ntime, t1, t2,
               <double*>freq_c.data, nfreq,
               <double*>dx.data, <double*>dy.data, <double*>dz.data, npos,
               <double*>mx_buf.data, <double*>my_buf.data, <double*>mz_buf.data,
               mode)
    
    # Reshape output based on mode. The C core stores blocks in order:
    #   for freq in nfreq:
    #       for pos in npos:
    #           time samples...
    # So the flat buffer layout is (freq, pos, time). We need (time, pos, freq).
    if ntout > 1:
        mx_out = mx_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
        my_out = my_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
        mz_out = mz_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
    else:
        mx_out = mx_buf.reshape((nfreq, npos)).T
        my_out = my_buf.reshape((nfreq, npos)).T
        mz_out = mz_buf.reshape((nfreq, npos)).T
    
    return mx_out, my_out, mz_out


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_bloch_parallel(np.ndarray[CTYPE_t, ndim=1] b1_complex,
                           np.ndarray[DTYPE_t, ndim=2] gradients,
                           np.ndarray[DTYPE_t, ndim=1] time_points,
                           double t1, double t2,
                           np.ndarray[DTYPE_t, ndim=1] frequencies,
                           np.ndarray[DTYPE_t, ndim=2] positions,
                           np.ndarray[DTYPE_t, ndim=2] m_init=None,
                           int mode=0,
                           int num_threads=4):
    """
    Parallel version of Bloch simulation using OpenMP.
    
    This function provides the same interface as simulate_bloch but
    uses parallel processing to speed up computation for multiple
    frequencies and positions.
    
    See simulate_bloch for full documentation.
    """
    
    cdef int ntime = len(b1_complex)
    cdef int nfreq = len(frequencies)
    cdef int npos = len(positions)
    cdef int ntout = ntime if (mode & 2) else 1
    cdef int nfnpos = nfreq * npos
    cdef int ntnfnpos = ntout * nfnpos
    cdef np.ndarray[DTYPE_t, ndim=2] m_init_arr

    # Prepare input arrays
    cdef np.ndarray[CTYPE_t, ndim=1] b1_c = np.ascontiguousarray(b1_complex, dtype=np.complex128)
    cdef np.ndarray[DTYPE_t, ndim=1] time_c = np.ascontiguousarray(time_points, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] freq_c = np.ascontiguousarray(frequencies, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] grad_c = _coerce_three_columns(gradients, ntime, "gradients")
    cdef np.ndarray[DTYPE_t, ndim=2] pos_c = _coerce_three_columns(positions, npos, "positions")
    cdef Py_ssize_t time_len = time_c.shape[0]
    cdef Py_ssize_t freq_len = freq_c.shape[0] if freq_c.ndim == 1 else -1
    if time_len != ntime:
        raise ValueError("time_points must have length %d; got %d" % (ntime, time_len))
    if freq_len != nfreq:
        raise ValueError("frequencies must have length %d; got %d" % (nfreq, freq_len))

    cdef np.ndarray[DTYPE_t, ndim=1] b1_real = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_imag = np.empty(ntime, dtype=np.float64)
    np.copyto(b1_real, b1_c.real)
    np.copyto(b1_imag, b1_c.imag)

    cdef np.ndarray[DTYPE_t, ndim=1] gx = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] gy = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] gz = np.empty(ntime, dtype=np.float64)
    np.copyto(gx, grad_c[:, 0])
    np.copyto(gy, grad_c[:, 1])
    np.copyto(gz, grad_c[:, 2])

    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.empty(npos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dy = np.empty(npos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] dz = np.empty(npos, dtype=np.float64)
    np.copyto(dx, pos_c[:, 0])
    np.copyto(dy, pos_c[:, 1])
    np.copyto(dz, pos_c[:, 2])

    # Flat output buffers to match blochsim layout (freq-major, then position, then time)
    cdef np.ndarray[DTYPE_t, ndim=1] mx_buf = np.empty(ntnfnpos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] my_buf = np.empty(ntnfnpos, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mz_buf = np.empty(ntnfnpos, dtype=np.float64)

    cdef slice start_slice = slice(0, ntnfnpos, ntout)
    if m_init is not None:
        m_init_arr = np.ascontiguousarray(m_init, dtype=np.float64)
        if m_init_arr.shape[0] != 3 or m_init_arr.shape[1] != nfnpos:
            raise ValueError("m_init must have shape (3, nfreq * npos)")
        mx_buf[start_slice] = m_init_arr[0]
        my_buf[start_slice] = m_init_arr[1]
        mz_buf[start_slice] = m_init_arr[2]
    else:
        mx_buf[start_slice] = 0.0
        my_buf[start_slice] = 0.0
        mz_buf[start_slice] = 1.0

    with nogil:
        blochsim_batch_optimized(
            <double*>b1_real.data, <double*>b1_imag.data,
            <double*>gx.data, <double*>gy.data, <double*>gz.data,
            <double*>time_c.data, ntime, t1, t2,
            <double*>freq_c.data, nfreq,
            <double*>dx.data, <double*>dy.data, <double*>dz.data, npos,
            <double*>mx_buf.data, <double*>my_buf.data, <double*>mz_buf.data,
            mode, num_threads)

    if ntout > 1:
        mx_out = mx_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
        my_out = my_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
        mz_out = mz_buf.reshape((nfreq, npos, ntout)).transpose(2, 1, 0)
    else:
        mx_out = mx_buf.reshape((nfreq, npos)).T
        my_out = my_buf.reshape((nfreq, npos)).T
        mz_out = mz_buf.reshape((nfreq, npos)).T

    return mx_out, my_out, mz_out


def calculate_signal(mx, my, mz, receiver_phase=0.0):
    """
    Calculate complex signal from magnetization components.
    
    Parameters
    ----------
    mx, my, mz : ndarray
        Magnetization components
    receiver_phase : float, optional
        Receiver phase in radians (default: 0)
    
    Returns
    -------
    signal : ndarray, complex
        Complex MRI signal
    """
    phase_factor = np.exp(-1j * receiver_phase)
    return (mx + 1j * my) * phase_factor


def design_rf_pulse(pulse_type='rect', duration=1e-3, flip_angle=90, 
                    time_bw_product=4, npoints=100):
    """
    Design common RF pulse shapes.
    
    Parameters
    ----------
    pulse_type : str
        Type of pulse: 'rect', 'sinc', 'gaussian', 'hermite'
    duration : float
        Pulse duration in seconds
    flip_angle : float
        Flip angle in degrees
    time_bw_product : float
        Time-bandwidth product for sinc/gaussian pulses
    npoints : int
        Number of time points
    
    Returns
    -------
    b1 : ndarray, complex
        Complex B1 field in Gauss
    time : ndarray
        Time points in seconds
    """
    time = np.linspace(0, duration, npoints, endpoint=False)
    dt = duration / npoints
    gamma = 4257.0  # Hz/Gauss for protons
    flip_rad = np.deg2rad(flip_angle)
    target_area = flip_rad / (gamma * 2 * np.pi)  # integral of B1 over time
    if pulse_type == 'rect':
        b1 = np.ones(npoints) * (target_area / duration)
        
    elif pulse_type == 'sinc':
        t_centered = time - duration/2
        bw = time_bw_product / duration
        envelope = np.sinc(bw * t_centered)
        area = np.trapezoid(envelope, time)
        b1 = envelope * (target_area / area)
        
    elif pulse_type == 'gaussian':
        t_centered = time - duration/2
        sigma = duration / (2 * np.sqrt(2 * np.log(2)) * time_bw_product)
        envelope = np.exp(-t_centered**2 / (2 * sigma**2))
        area = np.trapezoid(envelope, time)
        b1 = envelope * (target_area / area)
        
    else:
        raise ValueError(f"Unknown pulse type: {pulse_type}")
    
    return b1.astype(complex), time


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_phantom(np.ndarray[CTYPE_t, ndim=1] b1_complex,
                    np.ndarray[DTYPE_t, ndim=2] gradients,
                    np.ndarray[DTYPE_t, ndim=1] time_points,
                    np.ndarray[DTYPE_t, ndim=1] t1_array,
                    np.ndarray[DTYPE_t, ndim=1] t2_array,
                    np.ndarray[DTYPE_t, ndim=1] df_array,
                    np.ndarray[DTYPE_t, ndim=2] positions,
                    np.ndarray[DTYPE_t, ndim=2] m_init=None,
                    int mode=0,
                    int num_threads=4):
    """
    Simulate Bloch equations for a phantom with heterogeneous tissue properties.
    
    This is the core function for simulating imaging phantoms where each voxel
    can have different T1, T2, and frequency offset values.
    
    Parameters
    ----------
    b1_complex : ndarray, shape (ntime,)
        Complex RF pulse (B1 field) in Gauss
    gradients : ndarray, shape (ntime, 3)
        Gradient waveforms [Gx, Gy, Gz] in Gauss/cm
    time_points : ndarray, shape (ntime,)
        Time intervals in seconds
    t1_array : ndarray, shape (nvoxels,)
        T1 relaxation time per voxel in seconds
    t2_array : ndarray, shape (nvoxels,)
        T2 relaxation time per voxel in seconds
    df_array : ndarray, shape (nvoxels,)
        Frequency offset per voxel in Hz
    positions : ndarray, shape (nvoxels, 3)
        Spatial positions [x, y, z] in cm
    m_init : ndarray, shape (nvoxels, 3), optional
        Initial magnetization [Mx, My, Mz] per voxel. Default: [0, 0, 1]
    mode : int, optional
        Simulation mode:
        - 0: Endpoint only (default)
        - 2: Time-resolved (returns all time points)
    num_threads : int, optional
        Number of OpenMP threads (default: 4)
    
    Returns
    -------
    mx, my, mz : ndarray
        Magnetization components.
        Shape: (nvoxels,) for mode=0, or (ntime, nvoxels) for mode=2
    
    Notes
    -----
    - Voxels with T1 <= 0 or T2 <= 0 are treated as background and return zero
    - Positions should be in cm (consistent with gradient units Gauss/cm)
    - This function uses OpenMP parallelization across voxels
    
    Examples
    --------
    >>> # Simulate a 64x64 phantom
    >>> nvoxels = 64 * 64
    >>> t1 = np.ones(nvoxels) * 1.0  # 1 second T1
    >>> t2 = np.ones(nvoxels) * 0.1  # 100 ms T2
    >>> df = np.zeros(nvoxels)  # On-resonance
    >>> positions = ...  # (nvoxels, 3) array of positions in cm
    >>> mx, my, mz = simulate_phantom(b1, grads, time, t1, t2, df, positions)
    """
    cdef int ntime = len(b1_complex)
    cdef int nvoxels = len(t1_array)
    cdef int ntout = ntime if (mode & 2) else 1
    
    # Validate array lengths
    if len(t2_array) != nvoxels:
        raise ValueError(f"t2_array length ({len(t2_array)}) must match t1_array ({nvoxels})")
    if len(df_array) != nvoxels:
        raise ValueError(f"df_array length ({len(df_array)}) must match t1_array ({nvoxels})")
    if positions.shape[0] != nvoxels:
        raise ValueError(f"positions rows ({positions.shape[0]}) must match nvoxels ({nvoxels})")
    
    # Prepare B1 arrays (split complex into real/imag)
    cdef np.ndarray[CTYPE_t, ndim=1] b1_c = np.ascontiguousarray(b1_complex, dtype=np.complex128)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_real = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_imag = np.empty(ntime, dtype=np.float64)
    np.copyto(b1_real, b1_c.real)
    np.copyto(b1_imag, b1_c.imag)
    
    # Prepare gradient arrays
    cdef np.ndarray[DTYPE_t, ndim=2] grad_c = _coerce_three_columns(gradients, ntime, "gradients")
    cdef np.ndarray[DTYPE_t, ndim=1] gx = np.ascontiguousarray(grad_c[:, 0])
    cdef np.ndarray[DTYPE_t, ndim=1] gy = np.ascontiguousarray(grad_c[:, 1])
    cdef np.ndarray[DTYPE_t, ndim=1] gz = np.ascontiguousarray(grad_c[:, 2])
    
    # Time array
    cdef np.ndarray[DTYPE_t, ndim=1] time_c = np.ascontiguousarray(time_points, dtype=np.float64)
    
    # Tissue parameter arrays
    cdef np.ndarray[DTYPE_t, ndim=1] t1_c = np.ascontiguousarray(t1_array, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] t2_c = np.ascontiguousarray(t2_array, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] df_c = np.ascontiguousarray(df_array, dtype=np.float64)
    
    # Position arrays
    cdef np.ndarray[DTYPE_t, ndim=2] pos_c = _coerce_three_columns(positions, nvoxels, "positions")
    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.ascontiguousarray(pos_c[:, 0])
    cdef np.ndarray[DTYPE_t, ndim=1] dy = np.ascontiguousarray(pos_c[:, 1])
    cdef np.ndarray[DTYPE_t, ndim=1] dz = np.ascontiguousarray(pos_c[:, 2])
    
    # Initial magnetization
    cdef np.ndarray[DTYPE_t, ndim=1] mx_init_c = None
    cdef np.ndarray[DTYPE_t, ndim=1] my_init_c = None
    cdef np.ndarray[DTYPE_t, ndim=1] mz_init_c = None
    cdef double *mx_init_ptr = NULL
    cdef double *my_init_ptr = NULL
    cdef double *mz_init_ptr = NULL
    cdef np.ndarray[DTYPE_t, ndim=2] m_init_c
    
    if m_init is not None:
        if m_init.shape[0] != nvoxels or m_init.shape[1] != 3:
            raise ValueError(f"m_init must have shape ({nvoxels}, 3); got ({m_init.shape[0]}, {m_init.shape[1]})")
        m_init_c = np.ascontiguousarray(m_init, dtype=np.float64)
        mx_init_c = np.ascontiguousarray(m_init_c[:, 0])
        my_init_c = np.ascontiguousarray(m_init_c[:, 1])
        mz_init_c = np.ascontiguousarray(m_init_c[:, 2])
        mx_init_ptr = <double*>mx_init_c.data
        my_init_ptr = <double*>my_init_c.data
        mz_init_ptr = <double*>mz_init_c.data
    
    # Output arrays - flat layout: voxel-major, then time
    cdef np.ndarray[DTYPE_t, ndim=1] mx_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] my_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mz_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    
    # Run heterogeneous simulation
    with nogil:
        blochsim_heterogeneous(
            <double*>b1_real.data, <double*>b1_imag.data,
            <double*>gx.data, <double*>gy.data, <double*>gz.data,
            <double*>time_c.data, ntime,
            <double*>t1_c.data, <double*>t2_c.data, <double*>df_c.data,
            <double*>dx.data, <double*>dy.data, <double*>dz.data,
            mx_init_ptr, my_init_ptr, mz_init_ptr,
            nvoxels,
            <double*>mx_buf.data, <double*>my_buf.data, <double*>mz_buf.data,
            mode, num_threads)
    
    # Reshape output
    # C layout is (nvoxels, ntout) -> we want (ntout, nvoxels) for time-resolved
    if ntout > 1:
        mx_out = mx_buf.reshape((nvoxels, ntout)).T.copy()  # (ntime, nvoxels)
        my_out = my_buf.reshape((nvoxels, ntout)).T.copy()
        mz_out = mz_buf.reshape((nvoxels, ntout)).T.copy()
    else:
        mx_out = mx_buf  # (nvoxels,)
        my_out = my_buf
        mz_out = mz_buf
    
    return mx_out, my_out, mz_out


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_phantom_grouped(np.ndarray[CTYPE_t, ndim=1] b1_complex,
                            np.ndarray[DTYPE_t, ndim=2] gradients,
                            np.ndarray[DTYPE_t, ndim=1] time_points,
                            np.ndarray[np.int32_t, ndim=1] tissue_labels,
                            np.ndarray[DTYPE_t, ndim=1] t1_per_label,
                            np.ndarray[DTYPE_t, ndim=1] t2_per_label,
                            np.ndarray[DTYPE_t, ndim=1] df_array,
                            np.ndarray[DTYPE_t, ndim=2] positions,
                            np.ndarray[DTYPE_t, ndim=2] m_init=None,
                            int mode=0,
                            int num_threads=4):
    """
    Optimized phantom simulation for phantoms with grouped tissue types.
    
    This version is more efficient when many voxels share the same T1/T2 values
    (e.g., segmented phantoms with a small number of tissue types).
    
    Parameters
    ----------
    b1_complex : ndarray, shape (ntime,)
        Complex RF pulse in Gauss
    gradients : ndarray, shape (ntime, 3)
        Gradient waveforms in Gauss/cm
    time_points : ndarray, shape (ntime,)
        Time intervals in seconds
    tissue_labels : ndarray, shape (nvoxels,), dtype=int32
        Tissue label per voxel (0 to nlabels-1). Use -1 for background.
    t1_per_label : ndarray, shape (nlabels,)
        T1 value for each tissue label
    t2_per_label : ndarray, shape (nlabels,)
        T2 value for each tissue label
    df_array : ndarray, shape (nvoxels,)
        Frequency offset per voxel in Hz
    positions : ndarray, shape (nvoxels, 3)
        Spatial positions in cm
    m_init : ndarray, shape (nvoxels, 3), optional
        Initial magnetization per voxel
    mode : int
        0=endpoint, 2=time-resolved
    num_threads : int
        Number of OpenMP threads
    
    Returns
    -------
    mx, my, mz : ndarray
        Magnetization components
    """
    cdef int ntime = len(b1_complex)
    cdef int nvoxels = len(tissue_labels)
    cdef int nlabels = len(t1_per_label)
    cdef int ntout = ntime if (mode & 2) else 1
    
    # Validate
    if len(t2_per_label) != nlabels:
        raise ValueError(f"t2_per_label length must match t1_per_label ({nlabels})")
    
    # Prepare arrays (similar to simulate_phantom)
    cdef np.ndarray[CTYPE_t, ndim=1] b1_c = np.ascontiguousarray(b1_complex, dtype=np.complex128)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_real = np.empty(ntime, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] b1_imag = np.empty(ntime, dtype=np.float64)
    np.copyto(b1_real, b1_c.real)
    np.copyto(b1_imag, b1_c.imag)
    
    cdef np.ndarray[DTYPE_t, ndim=2] grad_c = _coerce_three_columns(gradients, ntime, "gradients")
    cdef np.ndarray[DTYPE_t, ndim=1] gx = np.ascontiguousarray(grad_c[:, 0])
    cdef np.ndarray[DTYPE_t, ndim=1] gy = np.ascontiguousarray(grad_c[:, 1])
    cdef np.ndarray[DTYPE_t, ndim=1] gz = np.ascontiguousarray(grad_c[:, 2])
    
    cdef np.ndarray[DTYPE_t, ndim=1] time_c = np.ascontiguousarray(time_points, dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] labels_c = np.ascontiguousarray(tissue_labels, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] t1_c = np.ascontiguousarray(t1_per_label, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] t2_c = np.ascontiguousarray(t2_per_label, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] df_c = np.ascontiguousarray(df_array, dtype=np.float64)
    
    cdef np.ndarray[DTYPE_t, ndim=2] pos_c = _coerce_three_columns(positions, nvoxels, "positions")
    cdef np.ndarray[DTYPE_t, ndim=1] dx = np.ascontiguousarray(pos_c[:, 0])
    cdef np.ndarray[DTYPE_t, ndim=1] dy = np.ascontiguousarray(pos_c[:, 1])
    cdef np.ndarray[DTYPE_t, ndim=1] dz = np.ascontiguousarray(pos_c[:, 2])
    
    cdef np.ndarray[DTYPE_t, ndim=1] mx_init_c = None
    cdef np.ndarray[DTYPE_t, ndim=1] my_init_c = None
    cdef np.ndarray[DTYPE_t, ndim=1] mz_init_c = None
    cdef double *mx_init_ptr = NULL
    cdef double *my_init_ptr = NULL
    cdef double *mz_init_ptr = NULL
    cdef np.ndarray[DTYPE_t, ndim=2] m_init_cont
    
    if m_init is not None:
        m_init_cont = np.ascontiguousarray(m_init, dtype=np.float64)
        mx_init_c = np.ascontiguousarray(m_init_cont[:, 0])
        my_init_c = np.ascontiguousarray(m_init_cont[:, 1])
        mz_init_c = np.ascontiguousarray(m_init_cont[:, 2])
        mx_init_ptr = <double*>mx_init_c.data
        my_init_ptr = <double*>my_init_c.data
        mz_init_ptr = <double*>mz_init_c.data
    
    cdef np.ndarray[DTYPE_t, ndim=1] mx_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] my_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mz_buf = np.zeros(nvoxels * ntout, dtype=np.float64)
    
    with nogil:
        blochsim_heterogeneous_grouped(
            <double*>b1_real.data, <double*>b1_imag.data,
            <double*>gx.data, <double*>gy.data, <double*>gz.data,
            <double*>time_c.data, ntime,
            <int*>labels_c.data,
            <double*>t1_c.data, <double*>t2_c.data,
            nlabels,
            <double*>df_c.data,
            <double*>dx.data, <double*>dy.data, <double*>dz.data,
            mx_init_ptr, my_init_ptr, mz_init_ptr,
            nvoxels,
            <double*>mx_buf.data, <double*>my_buf.data, <double*>mz_buf.data,
            mode, num_threads)
    
    if ntout > 1:
        mx_out = mx_buf.reshape((nvoxels, ntout)).T.copy()
        my_out = my_buf.reshape((nvoxels, ntout)).T.copy()
        mz_out = mz_buf.reshape((nvoxels, ntout)).T.copy()
    else:
        mx_out = mx_buf
        my_out = my_buf
        mz_out = mz_buf
    
    return mx_out, my_out, mz_out
