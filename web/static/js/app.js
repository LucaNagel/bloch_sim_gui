// --- CONFIGURATION ---
const WHEEL_FILENAME = "WHEEL_FILE_PLACEHOLDER";

// --- GLOBAL STATE ---
let pyodide = null;
let isPyodideReady = false;
let isSimulationPending = false;
let updateTimer = null;

// --- ROUTER ---
function router(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));

    // Show target view
    const target = document.getElementById(viewName + '-view');
    // Handle mismatch naming for rf-pulse (rf-explorer.html uses rf-pulse-view but router arg might be 'rf-pulse' or 'rf_explorer'?)
    // Partial says id="rf-pulse-view".
    // Partial for slice says id="slice-explorer-view".

    // Normalize view names
    let targetId = viewName + '-view';
    if (viewName === 'home') targetId = 'home-view';
    if (viewName === 'rf-pulse') targetId = 'rf-pulse-view';
    if (viewName === 'slice-explorer') targetId = 'slice-explorer-view';

    const viewEl = document.getElementById(targetId);
    if (viewEl) {
        viewEl.classList.add('active');
    }

    if (!isPyodideReady) return;

    // View-specific initialization
    if (viewName === 'rf-pulse') {
        // Switch to RF plot layout
        try {
             pyodide.globals.get("init_plot")();
             triggerSimulation(null, true);
        } catch(e) { console.error(e); }

        // Move canvas if possible (Matplotlib Pyodide usually targets document.body or specific div if configured)
        // We will just let it be for now, assuming it finds the visible container or we might need a specialized move
        moveCanvasTo('plot');
    }

    if (viewName === 'slice-explorer') {
        try {
            pyodide.globals.get("init_slice_plot")();
            triggerSliceSimulation();
        } catch(e) { console.error(e); }
        moveCanvasTo('slice-plot-container');
    }
}

function moveCanvasTo(containerId) {
    const container = document.getElementById(containerId);
    // Find canvas - usually it's a div with class 'mpld3-figure' or just the last canvas/div appended by pyodide
    // Matplotlib-pyodide typically creates a div with id 'figure1' or similar, or appends to body.
    // Let's search for a div that looks like a plot.
    const figures = document.querySelectorAll('.matplotlib-figure, .mpld3-figure, div[id^="figure"]');
    if (figures.length > 0 && container) {
        // Move the most recent one
        container.appendChild(figures[figures.length - 1]);
    }
}

function toggleLog() {
    const log = document.getElementById('error-log');
    if (log) {
        log.style.display = log.style.display === 'none' ? 'block' : 'none';
    }
}

function logMessage(msg, level = 'info') {
    console.log(`[Log] ${level}: ${msg}`);
    const errorLog = document.getElementById('error-log');
    const errorContent = document.getElementById('error-log-content');
    if (errorLog && errorContent) {
        // Remove 'No errors reported' if it's there
        if (errorContent.textContent.includes('No errors reported')) {
            errorContent.textContent = '';
        }

        const timestamp = new Date().toLocaleTimeString();
        const prefix = level.toUpperCase();
        const line = document.createElement('div');
        line.style.marginBottom = '2px';
        line.style.borderBottom = '1px solid #eee';
        line.style.padding = '2px 0';
        line.innerText = `[${timestamp}] ${prefix}: ${msg}`;

        if (level === 'error') line.style.color = '#d32f2f';
        else if (level === 'debug') line.style.color = '#666';
        else line.style.color = '#333';

        errorContent.appendChild(line);
        // Auto-scroll to bottom
        errorLog.scrollTop = errorLog.scrollHeight;
    }
}
window.logMessage = logMessage;

// --- INITIALIZATION ---
async function init() {
    const status = document.getElementById("status-text");

    try {
        // 1. Load Pyodide
        if (typeof loadPyodide === 'undefined') {
            throw new Error("Pyodide script not loaded.");
        }
        pyodide = await loadPyodide();
        status.innerText = "Loading libraries...";

        // 2. Install dependencies
        await pyodide.loadPackage(["numpy", "matplotlib", "micropip", "h5py", "scipy"]);

        // 3. Install Bloch Simulator Wheel
        status.innerText = `Installing simulator...`;
        const micropip = pyodide.pyimport("micropip");
        const wheelUrl = new URL(WHEEL_FILENAME, window.location.href).href + "?v=" + Date.now();

        try {
            await micropip.install(wheelUrl);
            logMessage("Bloch Simulator package installed successfully.");
        } catch (e) {
            console.warn("Wheel install failed (expected in dev without build):", e);
            status.innerText = "Dev mode: Wheel missing (Simulation mock)";
            logMessage("Running in dev mode (Mocking simulation backend)", "debug");
        }

        // 4. Setup Python Environment
        status.innerText = "Starting engine...";
        logMessage("Initializing Python environment...");
        await pyodide.runPythonAsync(`
import sys
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_pyodide.wasm_backend")
import matplotlib.pyplot as plt
from js import document, logMessage

# Redirect stdout/stderr to our JS log window
class WebLogger:
    def __init__(self, level="info"):
        self.level = level
    def write(self, text):
        if text and text.strip():
            logMessage(text.strip(), self.level)
    def flush(self):
        pass

sys.stdout = WebLogger("info")
sys.stderr = WebLogger("error")

print("Python redirection active.")

def _compute_integration_factor_from_wave(b1_wave, t_wave):
    """Compute integration factor |∫shape dt| / duration for a given complex waveform."""
    try:
        b1_wave = np.asarray(b1_wave, dtype=complex)
        t_wave = np.asarray(t_wave, dtype=float)
        if b1_wave.size < 2 or t_wave.size < 2:
            return 1.0
        duration = float(t_wave[-1] - t_wave[0])
        dt = float(np.median(np.diff(t_wave)))
        peak = np.max(np.abs(b1_wave)) if np.any(np.abs(b1_wave)) else 1.0
        shape = b1_wave / peak if peak != 0 else b1_wave
        area = np.trapz(shape, dx=dt)
        aligned = np.real(area * np.exp(-1j * np.angle(area)))
        if not np.isfinite(aligned) or abs(aligned) < 1e-12:
            return 1.0
        return abs(aligned) / max(duration, 1e-12)
    except Exception as e:
        print(f"Integration factor error: {e}")
        return 1.0

# Try importing, otherwise mock for UI testing if wheel is missing
try:
    from blochsimulator import BlochSimulator, TissueParameters, design_rf_pulse
    HAS_BACKEND = True
    sim = BlochSimulator(use_parallel=False)
    print("Bloch Simulator backend loaded.")
except ImportError as e:
    HAS_BACKEND = False
    print(f"Warning: Backend not found ({e}). Using mocks.")

# Global State
fig, axs = None, None
lines = {}
last_result = None
last_params = {}
is_3d_mode = False

def init_plot():
    global fig, axs, lines, is_3d_mode
    plt.clf()
    # Disable constrained_layout to prevent jumping
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=False)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle("RF Pulse Simulations", fontsize=16, fontweight='bold')

    # Fix margins manually
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.88, wspace=0.25)

    # Create axes manually to support dynamic switching
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    axs = [ax0, ax1, ax2]
    is_3d_mode = False

    # 1. RF Pulse (Static)
    axs[0].set_title("RF Pulse Shape")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude (uT)")
    axs[0].axhline(0, color='black', linewidth=0.8, alpha=0.3)
    lines['rf_real'], = axs[0].plot([], [], label='Real', color='#0056b3')
    lines['rf_imag'], = axs[0].plot([], [], label='Imag', color='#ff9900', alpha=0.7)
    lines['rf_abs'], = axs[0].plot([], [], label='Abs', color='gray', alpha=0.7)
    lines['time_line'], = axs[0].plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Magnetization (Dynamic 2D/3D)
    # Default is 2D
    _setup_2d_mag_axes()

    # 3. Frequency Profile (Static)
    axs[2].set_title("Excitation Profile")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylim(-1.1, 1.1)
    axs[2].axhline(0, color='black', linewidth=0.8, alpha=0.3)
    axs[2].axvline(0, color='black', linewidth=0.8, alpha=0.3)
    lines['mxy'], = axs[2].plot([], [], label='Mxy', color='purple')
    lines['mz_prof'], = axs[2].plot([], [], label='Mz', color='gray', linestyle='--')
    lines['freq_line'], = axs[2].plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)
    axs[2].legend(loc='upper right', fontsize='small')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.show()

def _setup_2d_mag_axes():
    global axs, lines
    axs[1].set_title("Magnetization")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylim(-1.1, 1.1)
    axs[1].axhline(0, color='black', linewidth=0.8, alpha=0.3)
    lines['mx'], = axs[1].plot([], [], label='Mx', color='r', alpha=0.6)
    lines['my'], = axs[1].plot([], [], label='My', color='g', alpha=0.6)
    lines['mz'], = axs[1].plot([], [], label='Mz', color='b')
    lines['time_line_2'], = axs[1].plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)
    axs[1].legend(loc='upper right', fontsize='small')
    axs[1].grid(True, linestyle='--', alpha=0.5)

def update_plot_mode(want_3d):
    global is_3d_mode, axs, fig, lines
    if want_3d == is_3d_mode:
        return

    # Remove old axes
    fig.delaxes(axs[1])

    if want_3d:
        # Create 3D axes
        axs[1] = fig.add_subplot(1, 3, 2, projection='3d')
        axs[1].set_title("Magnetization Path")
        # No permanent lines in 3D, we redraw
    else:
        # Create 2D axes
        axs[1] = fig.add_subplot(1, 3, 2)
        _setup_2d_mag_axes()

    is_3d_mode = want_3d

is_slice_3d_mode = False

def init_slice_plot():
    global fig, axs, lines, is_slice_3d_mode
    plt.clf()
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=False)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle("Slice Selection Profile", fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.88, wspace=0.25)

    # Ax0: RF Pulse
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.set_title("RF Pulse & Gradient")
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Amplitude")
    lines['rf_amp'], = ax0.plot([], [], label='|B1| (G)', color='blue')
    lines['grad'], = ax0.plot([], [], label='Gz (G/cm)', color='red', alpha=0.5)
    ax0.legend(loc='upper right', fontsize='small')
    ax0.grid(True, linestyle='--', alpha=0.5)

    # Ax1: Profile (Default 2D)
    ax1 = fig.add_subplot(1, 2, 2)
    _setup_slice_profile_2d(ax1)

    axs = [ax0, ax1]
    is_slice_3d_mode = False
    plt.show()

def _setup_slice_profile_2d(ax):
    ax.set_title("Excitation Profile")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Magnetization")
    ax.set_ylim(-1.1, 1.1)
    lines['mz_slice'], = ax.plot([], [], label='Mz', color='green')
    lines['mxy_slice'], = ax.plot([], [], label='|Mxy|', color='orange')
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)

def run_slice_simulation(flip, duration_ms, tbw, apod, thick_mm, rephase, range_cm, n_points, is3d):
    global fig, axs, lines, is_slice_3d_mode

    if fig is None or len(axs) != 2:
        init_slice_plot()

    # Handle 3D mode switch
    if is3d != is_slice_3d_mode:
        fig.delaxes(axs[1])
        if is3d:
            axs[1] = fig.add_subplot(1, 2, 2, projection='3d')
            axs[1].set_title("Magnetization Helix")
            # 3D setup done in plotting phase
        else:
            axs[1] = fig.add_subplot(1, 2, 2)
            _setup_slice_profile_2d(axs[1])
        is_slice_3d_mode = is3d

    dur_s = duration_ms * 1e-3
    dt = 1e-5

    # Design RF
    b1, time = design_rf_pulse('sinc', duration=dur_s, flip_angle=flip, time_bw_product=tbw, npoints=int(dur_s/dt))

    if apod != "None":
        n = len(b1)
        if apod == "Hamming": win = np.hamming(n)
        elif apod == "Hanning": win = np.hanning(n)
        elif apod == "Blackman": win = np.blackman(n)
        else: win = np.ones(n)
        b1 = b1 * win
        target_area = np.deg2rad(flip) / (4258.0 * 2 * np.pi)
        curr_area = np.trapz(np.abs(b1), dx=dt)
        if curr_area > 0: b1 *= (target_area / curr_area)

    bw_hz = tbw / dur_s
    gamma = 4258.0
    gz = bw_hz / (gamma * (thick_mm/10.0))

    grads = np.zeros((len(b1), 3))
    grads[:, 2] = gz

    if rephase > 0:
        rephase_dur = 1e-3
        n_rephase = int(rephase_dur/dt)
        rephase_amp = -(gz * dur_s * 0.5) / rephase_dur
        b1 = np.concatenate([b1, np.zeros(n_rephase)])
        grad_rephase = np.zeros((n_rephase, 3))
        grad_rephase[:, 2] = rephase_amp
        grads = np.concatenate([grads, grad_rephase])

    time = np.arange(len(b1)) * dt
    pos = np.zeros((int(n_points), 3))
    half_range = range_cm / 2.0
    pos[:, 2] = np.linspace(-half_range/100.0, half_range/100.0, int(n_points))

    tissue = TissueParameters("Water", 2.0, 2.0)

    mx, my, mz = np.zeros(int(n_points)), np.zeros(int(n_points)), np.ones(int(n_points))

    if HAS_BACKEND:
        res = sim.simulate((b1, grads, time), tissue, positions=pos, mode=0)
        mx, my, mz = res['mx'], res['my'], res['mz']
    else:
        # Mock
        z = pos[:, 2]
        mz = np.cos(z * 10)
        mx = np.sin(z * 10)
        my = np.zeros_like(z)

    # Plot Update
    t_ms = time * 1000
    lines['rf_amp'].set_data(t_ms, np.abs(b1))
    lines['grad'].set_data(t_ms, grads[:, 2])
    axs[0].relim()
    axs[0].autoscale_view()

    pos_cm = pos[:, 2] * 100

    if is3d:
        axs[1].cla()
        axs[1].set_title("Magnetization Helix (Slice Profile)")
        axs[1].set_xlabel("Mx")
        axs[1].set_ylabel("My")
        axs[1].set_zlabel("Position (cm)")
        axs[1].set_xlim(-1, 1)
        axs[1].set_ylim(-1, 1)
        axs[1].set_zlim(pos_cm.min(), pos_cm.max())

        # Plot helix
        axs[1].plot(mx, my, zs=pos_cm, color='purple', linewidth=1.5, alpha=0.8)

        # Add reference line
        axs[1].plot(np.zeros_like(pos_cm), np.zeros_like(pos_cm), zs=pos_cm, color='gray', linestyle=':', alpha=0.5)

    else:
        mxy = np.sqrt(mx**2 + my**2)
        lines['mz_slice'].set_data(pos_cm, mz)
        lines['mxy_slice'].set_data(pos_cm, mxy)
        axs[1].set_xlim(pos_cm.min(), pos_cm.max())

    fig.canvas.draw()
    fig.canvas.flush_events()

def run_simulation(t1_ms, t2_ms, duration_ms, freq_offset_hz, pulse_type, flip_angle, freq_range_val, freq_points, tbw):    global last_result, last_params

    # Store params to check against later if needed, though JS handles triggering
    last_params = locals()

    if fig is None:
        init_plot()

    t1_s = t1_ms * 1e-3
    t2_s = t2_ms * 1e-3
    duration_s = duration_ms * 1e-3

    # Calculate frequency range
    f_limit = max(100, float(freq_range_val))
    n_freq = max(10, int(freq_points))
    freq_range = np.linspace(-f_limit, f_limit, n_freq)

    positions = np.array([[0, 0, 0]])

    if HAS_BACKEND:
        tissue = TissueParameters(name="Generic", t1=t1_s, t2=t2_s)
        npoints = 400

        # Ensure TBW is sane
        tbw_val = float(tbw) if tbw > 0 else 4.0

        print(f"Simulation: Pulse={pulse_type}, Flip={flip_angle}, Dur={duration_ms}ms, TBW={tbw_val:.2f}")

        b1, time_s = design_rf_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            flip_angle=flip_angle,
            time_bw_product=tbw_val,
            npoints=npoints,
            freq_offset=freq_offset_hz
        )

        # Calculate integration factor (matching GUI logic)
        integration_factor = _compute_integration_factor_from_wave(b1_wave=b1, t_wave=time_s)
        try:
            if integration_factor > 0:
                eff_tbw = 1.0 / integration_factor
                print(f"Pulse analysis: Pulse={pulse_type}, Integration Factor={integration_factor:.4f}, Effective TBW={eff_tbw:.4f}")
            else:
                print(f"Pulse analysis: Pulse={pulse_type}, Integration Factor={integration_factor:.4f}")
        except Exception as e:
            print(f"Analysis error: {e}")

        time_ms = time_s * 1e3
        gradients = np.zeros((len(time_s), 3))

        sim_res = sim.simulate(
            sequence=(b1, gradients, time_s),
            tissue=tissue,
            frequencies=freq_range,
            positions=positions,
            mode=2
        )

        last_result = {
            "mx": sim_res['mx'],
            "my": sim_res['my'],
            "mz": sim_res['mz'],
            "time_ms": time_ms,
            "freq_range": freq_range,
            "rf_real": np.real(b1),
            "rf_imag": np.imag(b1),
            "rf_abs": np.abs(b1)
        }

    else:
        # Mock logic
        time_ms = np.linspace(0, duration_ms, 400)
        rf_real = np.zeros_like(time_ms)
        last_result = {
            "mx": np.zeros((400, 1, len(freq_range))),
            "my": np.zeros((400, 1, len(freq_range))),
            "mz": np.ones((400, 1, len(freq_range))),
            "time_ms": time_ms,
            "freq_range": freq_range,
            "rf_real": rf_real,
            "rf_imag": rf_real,
            "rf_abs": np.abs(rf_real)
        }

def extract_view(view_freq_hz, view_time_ms, want_3d):
    global last_result, fig, axs, lines

    if last_result is None or fig is None:
        return

    # Update mode (swaps axes if needed)
    update_plot_mode(want_3d)

    time_ms = last_result["time_ms"]
    freq_range = last_result["freq_range"]

    # Indices
    freq_idx = int(np.argmin(np.abs(freq_range - view_freq_hz)))
    time_idx = int(np.argmin(np.abs(time_ms - view_time_ms)))

    # Extract Magnetization (Time Evolution)
    mx_all = last_result['mx']
    my_all = last_result['my']
    mz_all = last_result['mz']

    # Robust shape handling
    if mx_all.ndim == 3:
        # (time, pos, freq)
        mx = mx_all[:, 0, freq_idx]
        my = my_all[:, 0, freq_idx]
        mz = mz_all[:, 0, freq_idx]

        # Profile at specific time
        mx_t = mx_all[time_idx, 0, :]
        my_t = my_all[time_idx, 0, :]
        mz_t = mz_all[time_idx, 0, :]

    elif mx_all.ndim == 2:
        # (time, freq)
        mx = mx_all[:, freq_idx]
        my = my_all[:, freq_idx]
        mz = mz_all[:, freq_idx]

        mx_t = mx_all[time_idx, :]
        my_t = my_all[time_idx, :]
        mz_t = mz_all[time_idx, :]
    else:
        mx = mx_all.ravel()
        my = my_all.ravel()
        mz = mz_all.ravel()
        mx_t, my_t, mz_t = np.zeros_like(freq_range), np.zeros_like(freq_range), np.zeros_like(freq_range)

    # Profile
    mxy_prof = np.sqrt(mx_t**2 + my_t**2)
    mz_prof = mz_t

    # Update Labels in JS
    document.getElementById("view_time_val").innerText = str(round(view_time_ms, 2))
    document.getElementById("view_freq_val").innerText = str(int(view_freq_hz))

    # 1. Update RF Plot
    lines['rf_real'].set_data(time_ms, last_result['rf_real'])
    lines['rf_imag'].set_data(time_ms, last_result['rf_imag'])
    lines['rf_abs'].set_data(time_ms, last_result['rf_abs'])

    # Manually calculate Y limits to avoid the 'zooming out' bug caused by relim() seeing the indicator line
    rf_max = np.max(last_result['rf_abs'])
    if rf_max <= 0: rf_max = 1.0
    axs[0].set_ylim(-rf_max * 1.1, rf_max * 1.1)
    axs[0].set_xlim(0, np.max(time_ms))

    # Set indicator after limits are finalized
    lines['time_line'].set_data([view_time_ms, view_time_ms], axs[0].get_ylim())

    # 2. Update Magnetization Plot
    if want_3d:
        # 3D Plot logic
        axs[1].cla()
        axs[1].set_title(f"Trajectory @ {int(view_freq_hz)} Hz")

        # Fixed limits for Bloch Sphere
        axs[1].set_xlim(-1, 1)
        axs[1].set_ylim(-1, 1)
        axs[1].set_zlim(-1, 1)
        axs[1].set_xlabel('Mx')
        axs[1].set_ylabel('My')
        axs[1].set_zlabel('Mz')

        # Draw Cartesian Axes
        axs[1].plot([-1, 1], [0, 0], zs=[0, 0], color='black', linewidth=0.8, alpha=0.3)
        axs[1].plot([0, 0], [-1, 1], zs=[0, 0], color='black', linewidth=0.8, alpha=0.3)
        axs[1].plot([0, 0], [0, 0], zs=[-1, 1], color='black', linewidth=0.8, alpha=0.3)

        # Draw unit circles for context (Bloch sphere visual aid)
        t_circ = np.linspace(0, 2*np.pi, 60)
        z_circ = np.zeros_like(t_circ)
        axs[1].plot(np.cos(t_circ), np.sin(t_circ), zs=z_circ, color='gray', linestyle=':', alpha=0.2, linewidth=0.5)
        axs[1].plot(np.cos(t_circ), z_circ, zs=np.sin(t_circ), color='gray', linestyle=':', alpha=0.2, linewidth=0.5)

        # Plot the full magnetization path
        axs[1].plot(mx, my, zs=mz, color='#0056b3', linewidth=1.5, alpha=0.8)

        # Draw vector from origin to current point
        m_curr = [mx[time_idx], my[time_idx], mz[time_idx]]
        axs[1].plot([0, m_curr[0]], [0, m_curr[1]], zs=[0, m_curr[2]], color='red', linewidth=2, alpha=0.6)

        # Plot indicator dot (current point)
        axs[1].scatter([m_curr[0]], [m_curr[1]], [m_curr[2]], color='red', s=25, zorder=20)
    else:
        # 2D Plot logic (Standard)
        lines['mx'].set_data(time_ms, mx)
        lines['my'].set_data(time_ms, my)
        lines['mz'].set_data(time_ms, mz)
        axs[1].set_xlim(0, np.max(time_ms))
        axs[1].set_ylim(-1.1, 1.1)
        lines['time_line_2'].set_data([view_time_ms, view_time_ms], axs[1].get_ylim())

    # 3. Update Profile Plot
    lines['mxy'].set_data(freq_range, mxy_prof)
    lines['mz_prof'].set_data(freq_range, mz_prof)
    axs[2].set_xlim(np.min(freq_range), np.max(freq_range))
    axs[2].set_ylim(-1.1, 1.1)
    lines['freq_line'].set_data([view_freq_hz, view_freq_hz], axs[2].get_ylim())

    fig.canvas.draw()
    fig.canvas.flush_events()
`);

        isPyodideReady = true;
        status.innerText = "Ready";

        // Initialize inputs
        const dInput = document.getElementById("duration");
        const vInput = document.getElementById("view_time");
        if (dInput && vInput) vInput.value = dInput.value;

        // Run initial simulation
        if (document.getElementById('rf-pulse-view').classList.contains('active')) {
            triggerSimulation(null, true); // force run
        }

    } catch (err) {
        status.innerText = "Error: " + err.message;
        console.error(err);
    }
}

// --- INTERACTION ---
function triggerSliceSimulation() {
    if (!isPyodideReady) return;

    // Debounce
    if (updateTimer) clearTimeout(updateTimer);

    const statusEl = document.getElementById('status-text');
    statusEl.innerHTML = '<span class="spinner"></span>Calculating...';

    updateTimer = setTimeout(async () => {
        try {
            const vals = {
                flip: parseFloat(document.getElementById("slice_flip_angle").value),
                dur: parseFloat(document.getElementById("slice_duration").value),
                tbw: parseFloat(document.getElementById("slice_tbw").value),
                apod: document.getElementById("slice_apodization").value,
                thick: parseFloat(document.getElementById("slice_thickness").value),
                                rephase: parseInt(document.getElementById("slice_rephase").value),
                                range: parseFloat(document.getElementById("slice_range").value),
                                points: parseInt(document.getElementById("slice_points").value),
                                is3d: document.getElementById("slice_3d").checked
                            };

                            const runSlice = pyodide.globals.get("run_slice_simulation");
                            if (runSlice) {
                                runSlice(vals.flip, vals.dur, vals.tbw, vals.apod, vals.thick, vals.rephase, vals.range, vals.points, vals.is3d);
                            }

                            statusEl.innerText = "Ready";        } catch (e) {
            console.error("Slice Sim Error", e);
            statusEl.innerText = "Error";
        }
    }, 50);
}

function triggerSimulation(event, forceRun = false) {
    if (!isPyodideReady) return;

    const sourceId = event ? event.target.id : null;

    // Sync view_time max
    const durationInput = document.getElementById("duration");
    const viewTimeInput = document.getElementById("view_time");
    if (durationInput && viewTimeInput) {
        const dur = parseFloat(durationInput.value);
        if (!isNaN(dur)) {
            viewTimeInput.max = dur;
            // If duration changed, snap view to end
            if (sourceId === 'duration') {
                viewTimeInput.value = dur;
            }
            if (parseFloat(viewTimeInput.value) > dur) {
                viewTimeInput.value = dur;
            }
        }
    }

    // Determine if we need a full simulation or just a view update
    // Full sim needed if physics params change
    const physicsParams = [
        "t1", "t2", "duration", "freq_offset", "pulse_type",
        "flip_angle", "freq_range", "freq_points", "tbw"
    ];

    let needFullSim = forceRun;
    if (sourceId && physicsParams.includes(sourceId)) {
        needFullSim = true;
    }

    // If no sourceId (e.g. init), assume full sim
    if (!sourceId && !forceRun) needFullSim = true;

    // Debounce
    if (updateTimer) clearTimeout(updateTimer);

    const statusEl = document.getElementById('status-text');
    statusEl.innerHTML = needFullSim ? '<span class="spinner"></span>Calculating...' : 'Updating view...';

    updateTimer = setTimeout(async () => {
        try {
            const vals = {
                t1: parseFloat(document.getElementById("t1").value),
                t2: parseFloat(document.getElementById("t2").value),
                duration: parseFloat(document.getElementById("duration").value),
                freq: parseFloat(document.getElementById("freq_offset").value),
                type: document.getElementById("pulse_type").value,
                flip: parseFloat(document.getElementById("flip_angle").value),
                fRange: parseFloat(document.getElementById("freq_range").value),
                fPoints: parseFloat(document.getElementById("freq_points").value),
                tbw: parseFloat(document.getElementById("tbw").value),
                viewFreq: parseFloat(document.getElementById("view_freq").value),
                viewTime: parseFloat(document.getElementById("view_time").value),
                is3d: document.getElementById("toggle_3d").checked
            };

            const runSim = pyodide.globals.get("run_simulation");
            const extractView = pyodide.globals.get("extract_view");

            if (needFullSim && runSim) {
                runSim(vals.t1, vals.t2, vals.duration, vals.freq, vals.type,
                       vals.flip, vals.fRange, vals.fPoints, vals.tbw);
            }

            if (extractView) {
                extractView(vals.viewFreq, vals.viewTime, vals.is3d);
            }

            statusEl.innerText = "Ready";
            const errorLog = document.getElementById('error-log');
            // Don't auto-hide on success if the user has it open, just clear potential error status
        } catch (e) {
            console.error("Sim Error", e);
            let msg = e.message || e.toString();
            if (msg.includes("PythonError:")) {
                msg = msg.split("PythonError:")[1];
            }
            statusEl.innerText = "Error (Details in Log)";
            logMessage(msg, "error");
        }
    }, 50);
}

// Attach listeners
document.querySelectorAll('.sim-input').forEach(input => {
    input.addEventListener('input', triggerSimulation);
});
document.querySelectorAll('.slice-input').forEach(input => {
    input.addEventListener('input', triggerSliceSimulation);
});

// Start
init();
