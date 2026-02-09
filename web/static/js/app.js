// --- CONFIGURATION ---
const WHEEL_FILENAME = "WHEEL_FILE_PLACEHOLDER";

// --- GLOBAL STATE ---
let pyodide = null;
let isPyodideReady = false;
let isSimulationPending = false;
let updateTimer = null;
let playbackInterval = null;
let isPlaying = false;

// --- ROUTER ---
function cleanupFigures() {
    // 1. Remove by known selectors
    const figures = document.querySelectorAll('.matplotlib-figure, .mpld3-figure, div[id^="figure"]');
    figures.forEach(fig => fig.remove());

    // 2. Remove orphaned elements on document.body (common behavior of Matplotlib Pyodide backend)
    // We expect ONLY: <nav>, <div class="container">, <div id="status-bar">, <script>, <style>, <link>
    const whitelistTags = ['NAV', 'SCRIPT', 'STYLE', 'LINK'];
    const whitelistIds = ['status-bar'];
    const whitelistClasses = ['container'];

    Array.from(document.body.children).forEach(child => {
        // Skip whitelisted elements
        if (whitelistTags.includes(child.tagName)) return;
        if (child.id && whitelistIds.includes(child.id)) return;
        if (whitelistClasses.some(cls => child.classList.contains(cls))) return;

        // Inspect suspect element
        // If it's a div and looks like a plot (has canvas or figure ID), remove it.
        const isPlotCandidate =
            child.tagName === 'DIV' && (
                child.querySelector('canvas') ||
                (child.id && child.id.startsWith('figure')) ||
                child.className.includes('matplotlib')
            );

        if (isPlotCandidate) {
            // console.log("Cleaning up orphaned plot element:", child);
            child.remove();
        }
    });
}

function router(viewName) {
    // Hide all views
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));

    // Clear ALL plot containers to prevent ghost figures
    const plotContainers = ['plot', 'slice-plot-container'];
    plotContainers.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });

    // Cleanup lingering global figures (e.g. attached to body)
    cleanupFigures();

    // Show target view
    const target = document.getElementById(viewName + '-view');
    // Handle mismatch naming for rf-pulse
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

        // Remove any others that might have been created but not used (safety cleanup)
        for (let i = 0; i < figures.length - 1; i++) {
            figures[i].remove();
        }
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
        area = np.trapezoid(shape, dx=dt)
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
is_3d_mode = True  # Default to 3D

def init_plot():
    global fig, axs, lines, is_3d_mode
    plt.close('all')
    plt.clf()
    # Disable constrained_layout to prevent jumping
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=False)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle("RF Pulse Simulations", fontsize=16, fontweight='bold')

    # Fix margins manually
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.12, top=0.88, wspace=0.25)

    # Create axes manually to support dynamic switching
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, projection='3d') # Default 3D
    ax2 = fig.add_subplot(1, 3, 3)
    axs = [ax0, ax1, ax2]
    is_3d_mode = True

    # 1. RF Pulse (Static)
    axs[0].set_title("RF Pulse Shape")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude (G)")
    axs[0].axhline(0, color='black', linewidth=0.8, alpha=0.3)
    lines['rf_real'], = axs[0].plot([], [], label='Real', color='#0056b3')
    lines['rf_imag'], = axs[0].plot([], [], label='Imag', color='#ff9900', alpha=0.7)
    lines['rf_abs'], = axs[0].plot([], [], label='Abs', color='gray', alpha=0.7)
    lines['time_line'], = axs[0].plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Magnetization (Dynamic 2D/3D) - Init in 3D
    # (Plot setup happens in extract_view)

    # 3. Frequency Profile (Static)
    axs[2].set_title("Excitation Profile")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylim(-1.1, 1.1)
    axs[2].axhline(0, color='black', linewidth=0.8, alpha=0.3)
    axs[2].axvline(0, color='black', linewidth=0.8, alpha=0.3)
    lines['mxy'], = axs[2].plot([], [], label='Mxy', color='purple')
    lines['mz_prof'], = axs[2].plot([], [], label='Mz', color='gray', linestyle='--')
    lines['mx_prof'], = axs[2].plot([], [], label='Mx', color='red', alpha=0.5, linestyle=':')
    lines['my_prof'], = axs[2].plot([], [], label='My', color='blue', alpha=0.5, linestyle=':')
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
last_slice_result = None

def init_slice_plot():
    global fig, axs, lines, is_slice_3d_mode
    plt.close('all')
    plt.clf()
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=False)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle("Slice Selection Profile", fontsize=16, fontweight='bold')
    # Right margin increased to accommodate secondary Y axis
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.12, top=0.88, wspace=0.30)

    # Ax0: RF Pulse
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.set_title("RF Pulse & Gradient")
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Amplitude (G)")
    lines['rf_amp'], = ax0.plot([], [], label='|B1|', color='blue')

    # Dual Axis for Gradient
    ax0b = ax0.twinx()
    ax0b.set_ylabel("Gradient (G/cm)", color='magenta')
    ax0b.tick_params(axis='y', labelcolor='magenta')
    lines['grad'], = ax0b.plot([], [], label='Gz', color='magenta', alpha=0.5)

    # Legend combined? Matplotlib legends are per-axis.
    # Ax0 legend
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    ax0.legend(h1+h2, l1+l2, loc='upper right', fontsize='small')
    ax0.grid(True, linestyle='--', alpha=0.5)

    lines['slice_time_line'], = ax0.plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)

    # Ax1: Profile (Default 2D)
    ax1 = fig.add_subplot(1, 2, 2)
    _setup_slice_profile_2d(ax1)

    axs = [ax0, ax0b, ax1]
    is_slice_3d_mode = False
    plt.show()

def _setup_slice_profile_2d(ax):
    ax.set_title("Excitation Profile")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Magnetization")
    ax.set_ylim(-1.1, 1.1)
    lines['mz_slice'], = ax.plot([], [], label='Mz', color='green')
    lines['mxy_slice'], = ax.plot([], [], label='|Mxy|', color='orange')
    lines['mx_slice'], = ax.plot([], [], label='Mx', color='red', alpha=0.6, linestyle='--')
    lines['my_slice'], = ax.plot([], [], label='My', color='blue', alpha=0.6, linestyle='--')
    lines['slice_pos_line'], = ax.plot([], [], color='#e74c3c', linestyle='-', alpha=1.0, linewidth=2.0, zorder=10)

    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.5)

def update_slice_plot_mode(want_3d):
    global is_slice_3d_mode, axs, fig, lines
    if want_3d == is_slice_3d_mode:
        return

    # Remove old axes (Ax1 is at index 2 in axs list: [ax0, ax0b, ax1])
    fig.delaxes(axs[2])

    if want_3d:
        # Create 3D axes
        axs[2] = fig.add_subplot(1, 2, 2, projection='3d')
        axs[2].set_title("Magnetization Evolution")
        axs[2].set_xlim(-1, 1)
        axs[2].set_ylim(-1, 1)
        axs[2].set_zlim(-1, 1)
        axs[2].set_xlabel('Mx')
        axs[2].set_ylabel('My')
        axs[2].set_zlabel('Mz')
    else:
        # Create 2D axes
        axs[2] = fig.add_subplot(1, 2, 2)
        _setup_slice_profile_2d(axs[2])

    is_slice_3d_mode = want_3d

def run_slice_simulation(flip, duration_ms, extra_time_ms, tbw, apod, thick_mm, rephase_pct, range_cm, n_points, is3d, b1_amp_gauss, pulse_type, show_mx, show_my):
    global fig, axs, lines, is_slice_3d_mode, last_slice_result

    if fig is None or len(axs) != 3:
        init_slice_plot()

    # NOTE: View updates happen in extract_slice_view. Here we run simulation and store result.

    dur_s = duration_ms * 1e-3
    extra_s = extra_time_ms * 1e-3
    dt = 1e-5

    # Auto TBW logic (Fixed by Type)
    calc_tbw = tbw
    if calc_tbw <= 0:
        if pulse_type == "sinc":
            calc_tbw = 4.0
        elif pulse_type == "rect":
            calc_tbw = 1.0 # approx
        elif pulse_type == "gaussian":
            calc_tbw = 2.5 # approx

    # 1. Determine Flip Angle from B1 Override if provided
    calc_flip = flip
    if b1_amp_gauss is not None and b1_amp_gauss > 0:
        # Design a probe pulse (1 deg) to find the B1 per Flip ratio
        probe_flip = 1.0
        b1_probe, _ = design_rf_pulse(pulse_type, duration=dur_s, flip_angle=probe_flip, time_bw_product=calc_tbw, npoints=int(dur_s/dt))

        # Apply apodization to probe to measure *actual* peak
        if apod != "None":
            n_p = len(b1_probe)
            if apod == "Hamming": win = np.hamming(n_p)
            elif apod == "Hanning": win = np.hanning(n_p)
            elif apod == "Blackman": win = np.blackman(n_p)
            else: win = np.ones(n_p)
            b1_probe = b1_probe * win

        peak_probe = np.max(np.abs(b1_probe))
        if peak_probe > 0:
            calc_flip = probe_flip * (b1_amp_gauss / peak_probe)

    # 2. Design Final Pulse
    b1, time = design_rf_pulse(pulse_type, duration=dur_s, flip_angle=calc_flip, time_bw_product=calc_tbw, npoints=int(dur_s/dt))

    if apod != "None":
        n = len(b1)
        if apod == "Hamming": win = np.hamming(n)
        elif apod == "Hanning": win = np.hanning(n)
        elif apod == "Blackman": win = np.blackman(n)
        else: win = np.ones(n)
        b1 = b1 * win

        # Only rescale if we didn't calculate flip from B1
        if b1_amp_gauss is None or b1_amp_gauss <= 0:
            target_area = np.deg2rad(calc_flip) / (4258.0 * 2 * np.pi)
            curr_area = np.trapezoid(np.abs(b1), dx=dt)
            if curr_area > 0: b1 *= (target_area / curr_area)

    bw_hz = calc_tbw / dur_s
    gamma = 4258.0
    gz = bw_hz / (gamma * (thick_mm/10.0))

    grads = np.zeros((len(b1), 3))
    grads[:, 2] = gz

    # Rephase Logic (Percentage)
    if rephase_pct != 0:
        rephase_dur = 0.0005
        n_rephase = int(rephase_dur/dt)
        slice_area = gz * dur_s
        rewind_area = -slice_area * (rephase_pct / 100.0)
        rephase_amp = rewind_area / rephase_dur

        b1 = np.concatenate([b1, np.zeros(n_rephase)])
        grad_rephase = np.zeros((n_rephase, 3))
        grad_rephase[:, 2] = rephase_amp
        grads = np.concatenate([grads, grad_rephase])

    # Extra Time Padding
    if extra_s > 0:
        n_extra = int(extra_s / dt)
        if n_extra > 0:
            b1 = np.concatenate([b1, np.zeros(n_extra)])
            grads = np.concatenate([grads, np.zeros((n_extra, 3))])

    time = np.arange(len(b1)) * dt
    pos = np.zeros((int(n_points), 3))
    half_range = range_cm / 2.0
    pos[:, 2] = np.linspace(-half_range/100.0, half_range/100.0, int(n_points))

    tissue = TissueParameters("Water", 2.0, 2.0)

    if HAS_BACKEND:
        # Mode 2 for time-resolved
        res = sim.simulate((b1, grads, time), tissue, positions=pos, mode=2)
        mx, my, mz = res['mx'], res['my'], res['mz']
    else:
        # Mock
        z = pos[:, 2]
        mz = np.tile(np.cos(z * 10), (len(time), 1))
        mx = np.tile(np.sin(z * 10), (len(time), 1))
        my = np.zeros_like(mx)

    last_slice_result = {
        "mx": mx, "my": my, "mz": mz,
        "time_ms": time * 1000,
        "pos_cm": pos[:, 2] * 100,
        "b1_abs": np.abs(b1),
        "grads_z": grads[:, 2],
        "show_mx": show_mx,
        "show_my": show_my
    }

    # Return 3 values now: ActFlip, ActB1, ActTBW
    return float(calc_flip), float(np.max(np.abs(b1))), float(calc_tbw)

def extract_slice_view(view_time_ms, view_pos_cm, is3d):
    global last_slice_result, fig, axs, lines

    if last_slice_result is None or fig is None:
        return

    update_slice_plot_mode(is3d)

    time_ms = last_slice_result["time_ms"]
    pos_cm = last_slice_result["pos_cm"]

    # Indices
    t_idx = int(np.argmin(np.abs(time_ms - view_time_ms)))
    p_idx = int(np.argmin(np.abs(pos_cm - view_pos_cm)))

    # Update Ax0 (RF/Grad)
    lines['rf_amp'].set_data(time_ms, last_slice_result['b1_abs'])
    lines['grad'].set_data(time_ms, last_slice_result['grads_z'])

    # Scaling
    max_b1 = np.max(last_slice_result['b1_abs'])
    if max_b1 <= 0: max_b1 = 1.0
    axs[0].set_ylim(-0.1, max_b1 * 1.1)

    max_gz = np.max(np.abs(last_slice_result['grads_z']))
    if max_gz <= 0: max_gz = 1.0
    axs[1].set_ylim(-max_gz * 1.1, max_gz * 1.1) # axs[1] is ax0b (twinx)

    axs[0].set_xlim(0, time_ms[-1])
    lines['slice_time_line'].set_data([view_time_ms, view_time_ms], axs[0].get_ylim())

    # Data
    mx_all = last_slice_result['mx']
    my_all = last_slice_result['my']
    mz_all = last_slice_result['mz']
    show_mx = last_slice_result.get('show_mx', False)
    show_my = last_slice_result.get('show_my', False)

    if is3d:
        # 3D: Trajectory of single spin at p_idx (EVOLUTION)
        # mx_all shape: (Time, Pos, Freq=1) -> squeeze to (Time, Pos)
        # We need the time evolution at the selected POSITION p_idx

        # Check shape to be safe
        if mx_all.ndim == 3:
             mx_traj = mx_all[:, p_idx, 0] # (Time)
             my_traj = my_all[:, p_idx, 0]
             mz_traj = mz_all[:, p_idx, 0]
        else:
             mx_traj = mx_all[:, p_idx]
             my_traj = my_all[:, p_idx]
             mz_traj = mz_all[:, p_idx]

        ax3d = axs[2]
        ax3d.cla()
        ax3d.set_title(f"Trajectory @ {view_pos_cm:.2f} cm")
        ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
        ax3d.set_xlabel('Mx'); ax3d.set_ylabel('My'); ax3d.set_zlabel('Mz')

        # Axes
        ax3d.plot([-1, 1], [0, 0], zs=[0, 0], color='black', alpha=0.3)
        ax3d.plot([0, 0], [-1, 1], zs=[0, 0], color='black', alpha=0.3)
        ax3d.plot([0, 0], [0, 0], zs=[-1, 1], color='black', alpha=0.3)

        # Plot Trajectory
        ax3d.plot(mx_traj, my_traj, zs=mz_traj, color='#0056b3', linewidth=1.5, alpha=0.8)

        # Current Point
        cur_mx, cur_my, cur_mz = mx_traj[t_idx], my_traj[t_idx], mz_traj[t_idx]
        ax3d.plot([0, cur_mx], [0, cur_my], zs=[0, cur_mz], color='red', linewidth=2)
        ax3d.scatter([cur_mx], [cur_my], [cur_mz], color='red', s=25)

    else:
        # 2D: Profile at time t_idx (PROFILE)
        if mx_all.ndim == 3:
             mx_prof = mx_all[t_idx, :, 0]
             my_prof = my_all[t_idx, :, 0]
             mz_prof = mz_all[t_idx, :, 0]
        else:
             mx_prof = mx_all[t_idx, :]
             my_prof = my_all[t_idx, :]
             mz_prof = mz_all[t_idx, :]

        mxy_prof = np.sqrt(mx_prof**2 + my_prof**2)

        lines['mz_slice'].set_data(pos_cm, mz_prof)
        lines['mxy_slice'].set_data(pos_cm, mxy_prof)

        if show_mx:
            lines['mx_slice'].set_data(pos_cm, mx_prof)
            lines['mx_slice'].set_visible(True)
        else:
            lines['mx_slice'].set_visible(False)

        if show_my:
            lines['my_slice'].set_data(pos_cm, my_prof)
            lines['my_slice'].set_visible(True)
        else:
            lines['my_slice'].set_visible(False)

        lines['slice_pos_line'].set_data([view_pos_cm, view_pos_cm], [-1.1, 1.1])
        axs[2].set_xlim(pos_cm.min(), pos_cm.max())

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Update Text
    document.getElementById("slice_view_time_val").innerText = str(round(view_time_ms, 2))
    document.getElementById("slice_view_pos_val").innerText = str(round(view_pos_cm, 2))

def run_simulation(t1_ms, t2_ms, duration_ms, extra_time_ms, freq_offset_hz, pulse_type, flip_angle, freq_range_val, freq_points, tbw, b1_amp, show_mx, show_my):
    global last_result, last_params
    last_params = locals()

    if fig is None:
        init_plot()

    t1_s = t1_ms * 1e-3
    t2_s = t2_ms * 1e-3
    duration_s = duration_ms * 1e-3
    extra_s = extra_time_ms * 1e-3

    # Calculate frequency range
    f_limit = max(100, float(freq_range_val))
    n_freq = max(10, int(freq_points))
    freq_range = np.linspace(-f_limit, f_limit, n_freq)

    positions = np.array([[0, 0, 0]])
    dt = 1e-5
    npoints = int(duration_s / dt)
    if npoints < 10: npoints = 10

    # Ensure TBW is sane
    tbw_val = float(tbw) if tbw > 0 else 4.0

    if HAS_BACKEND:
        tissue = TissueParameters(name="Generic", t1=t1_s, t2=t2_s)

        # B1 Override Logic (b1_amp is in Gauss)
        calc_flip = flip_angle
        if b1_amp > 0:
             # Design a probe pulse (1 deg) to find the B1 per Flip ratio
            probe_flip = 1.0
            b1_probe, _ = design_rf_pulse(
                pulse_type=pulse_type,
                duration=duration_s,
                flip_angle=probe_flip,
                time_bw_product=tbw_val,
                npoints=npoints,
                freq_offset=freq_offset_hz
            )
            peak_probe = np.max(np.abs(b1_probe))
            if peak_probe > 0:
                calc_flip = probe_flip * (b1_amp / peak_probe)

        print(f"Simulation: Pulse={pulse_type}, Flip={calc_flip:.2f}, Dur={duration_ms}ms, TBW={tbw_val:.2f}")

        b1, time_s = design_rf_pulse(
            pulse_type=pulse_type,
            duration=duration_s,
            flip_angle=calc_flip,
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

        # Extra Time Padding
        dt = 1e-5
        if extra_s > 0:
            n_extra = int(extra_s / dt)
            if n_extra > 0:
                b1 = np.concatenate([b1, np.zeros(n_extra)])

        time_s = np.arange(len(b1)) * dt
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
            "rf_abs": np.abs(b1),
            "show_mx": show_mx, # Store flags
            "show_my": show_my
        }

        return float(calc_flip), float(np.max(np.abs(b1)))

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
            "rf_abs": np.abs(rf_real),
            "show_mx": show_mx,
            "show_my": show_my
        }
        return float(flip_angle), 0.0

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

    actual_freq = freq_range[freq_idx]

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
    document.getElementById("view_freq_val").innerText = f"{actual_freq:.1f}"

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

    # Check Show Mx/My flags
    show_mx = last_result.get('show_mx', False)
    show_my = last_result.get('show_my', False)

    if show_mx:
        lines['mx_prof'].set_data(freq_range, mx_t)
        lines['mx_prof'].set_visible(True)
    else:
        lines['mx_prof'].set_visible(False)

    if show_my:
        lines['my_prof'].set_data(freq_range, my_t)
        lines['my_prof'].set_visible(True)
    else:
        lines['my_prof'].set_visible(False)

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
async function performSliceSimulation(sourceId) {
    const statusEl = document.getElementById('status-text');
    try {
        // View Update Only?
        const viewParams = ["slice_view_time", "slice_view_pos"];
        const isViewOnly = sourceId && viewParams.includes(sourceId);

        // Gather Values
        const vals = {
            flip: parseFloat(document.getElementById("slice_flip_angle").value),
            dur: parseFloat(document.getElementById("slice_duration").value),
            extra: parseFloat(document.getElementById("slice_extra_time").value) || 0.0,
            tbw: parseFloat(document.getElementById("slice_tbw").value),
            apod: document.getElementById("slice_apodization").value,
            thick: parseFloat(document.getElementById("slice_thickness").value),
            rephase: parseFloat(document.getElementById("slice_rephase_pct").value),
            range: parseFloat(document.getElementById("slice_range").value),
            points: parseInt(document.getElementById("slice_points").value),
            is3d: document.getElementById("slice_3d").checked,
            b1: parseFloat(document.getElementById("slice_b1").value),
            type: document.getElementById("slice_pulse_type").value,
            showMx: document.getElementById("slice_show_mx").checked,
            showMy: document.getElementById("slice_show_my").checked,
            viewTime: parseFloat(document.getElementById("slice_view_time").value),
            viewPos: parseFloat(document.getElementById("slice_view_pos").value)
        };

        if (isNaN(vals.b1)) vals.b1 = -1;

        // Last Changed Logic (Only if not view update)
        if (!isViewOnly) {
            // If B1 Changed -> Keep B1 (vals.b1 is valid)
            // If Flip/Dur/TBW/Type Changed -> Reset B1
            const physicsParamsThatResetB1 = ["slice_flip_angle", "slice_duration", "slice_tbw", "slice_thickness", "slice_pulse_type", "slice_rephase_pct"];
            if (physicsParamsThatResetB1.includes(sourceId)) {
                document.getElementById("slice_b1").value = "";
                vals.b1 = -1;
            }
        }

        if (!isViewOnly) {
            const runSlice = pyodide.globals.get("run_slice_simulation");
            if (runSlice) {
                const result = runSlice(vals.flip, vals.dur, vals.extra, vals.tbw, vals.apod, vals.thick, vals.rephase, vals.range, vals.points, vals.is3d, vals.b1, vals.type, vals.showMx, vals.showMy);

                // Sync UI
                if (result && result.length === 3) {
                    const actFlip = result.get(0);
                    const actB1 = result.get(1);
                    const actTBW = result.get(2);

                    if (vals.b1 > 0) {
                        // B1 drove simulation -> update Flip
                        if (document.activeElement.id !== "slice_flip_angle") {
                            document.getElementById("slice_flip_angle").value = actFlip.toFixed(2);
                        }
                    } else {
                        // Flip drove simulation -> update B1
                        if (document.activeElement.id !== "slice_b1") {
                            document.getElementById("slice_b1").value = actB1.toFixed(4);
                        }
                    }
                    // Update TBW (always calculated now)
                    document.getElementById("slice_tbw").value = actTBW.toFixed(1);

                    result.destroy();
                }
            }
        }

        // Always update view
        const extractView = pyodide.globals.get("extract_slice_view");
        if (extractView) {
            extractView(vals.viewTime, vals.viewPos, vals.is3d);
        }

        statusEl.innerText = "Ready";
    } catch (e) {
        console.error("Slice Sim Error", e);
        statusEl.innerText = "Error";
        logMessage(e.message || e.toString(), "error");
    }
}

function triggerSliceSimulation(event) {
    if (!isPyodideReady) return;

    const sourceId = event ? event.target.id : null;
    const isPlayback = event ? event.isPlayback : false;

    // View Update Only?
    const viewParams = ["slice_view_time", "slice_view_pos"];
    const isViewOnly = sourceId && viewParams.includes(sourceId);

    // Sync View Time Max
    const durationInput = document.getElementById("slice_duration");
    const extraInput = document.getElementById("slice_extra_time");
    const rephaseInput = document.getElementById("slice_rephase_pct");
    const viewTimeInput = document.getElementById("slice_view_time");

    if (durationInput && viewTimeInput) {
        const dur = parseFloat(durationInput.value) || 0;
        const extra = parseFloat(extraInput ? extraInput.value : 0) || 0;
        const rephase = parseFloat(rephaseInput ? rephaseInput.value : 0);
        // Rephase duration is fixed at 0.5ms in simulation if pct != 0
        const rephaseDur = (rephase !== 0) ? 0.5 : 0.0;
        const totalDur = dur + rephaseDur + extra;

        viewTimeInput.max = totalDur;

        // Only clamp if not playback (playback handles its own limits/wrapping)
        if (!isPlayback) {
            let val = parseFloat(viewTimeInput.value);
            if (val > totalDur) viewTimeInput.value = totalDur;
        }
    }

    // Sync View Pos Max (Range)
    const rangeInput = document.getElementById("slice_range");
    const viewPosInput = document.getElementById("slice_view_pos");
    if (rangeInput && viewPosInput) {
        const rng = parseFloat(rangeInput.value);
        if (!isNaN(rng)) {
            const half = rng/2;
            viewPosInput.min = -half;
            viewPosInput.max = half;
            let val = parseFloat(viewPosInput.value);
            if (val < -half) viewPosInput.value = -half;
            if (val > half) viewPosInput.value = half;
        }
    }

    if (updateTimer) clearTimeout(updateTimer);

    const statusEl = document.getElementById('status-text');
    if (!isPlayback) {
         statusEl.innerHTML = isViewOnly ? 'Updating view...' : '<span class="spinner"></span>Calculating...';
    }

    if (isPlayback) {
        // Immediate execution for playback
        performSliceSimulation(sourceId);
    } else {
        // Debounced execution for user input
        updateTimer = setTimeout(() => performSliceSimulation(sourceId), 50);
    }
}

async function performRfSimulation(sourceId, forceRun) {
    const statusEl = document.getElementById('status-text');
    try {
        // Determine if we need a full simulation or just a view update
        const physicsParams = [
            "t1", "t2", "duration", "freq_offset", "pulse_type",
            "flip_angle", "freq_range", "freq_points", "tbw",
            "b1_amp", "show_mx", "show_my"
        ];

        let needFullSim = forceRun;
        if (sourceId && physicsParams.includes(sourceId)) {
            needFullSim = true;
        }

        // If no sourceId (e.g. init), assume full sim
        if (!sourceId && !forceRun) needFullSim = true;

        // Last Changed Logic
        // If sourceId is Flip/Dur/TBW/Type -> Clear B1
        const resetB1Params = ["flip_angle", "duration", "tbw", "pulse_type"];
        if (resetB1Params.includes(sourceId)) {
            document.getElementById("b1_amp").value = "";
        }

        let b1val = parseFloat(document.getElementById("b1_amp").value);
        if (isNaN(b1val)) b1val = -1;

        // Update View Freq Slider Range/Step to match simulation grid
        const rawFRange = parseFloat(document.getElementById("freq_range").value);
        const rawFPoints = parseFloat(document.getElementById("freq_points").value);
        // Match Python logic for grid generation: max(100, range), max(10, points)
        const fRangeEff = Math.max(100, isNaN(rawFRange) ? 1000 : rawFRange);
        const fPointsEff = Math.max(10, isNaN(rawFPoints) ? 100 : Math.floor(rawFPoints));

        const viewFreqInput = document.getElementById("view_freq");
        if (viewFreqInput) {
            const step = (2 * fRangeEff) / (fPointsEff - 1);
            viewFreqInput.min = -fRangeEff;
            viewFreqInput.max = fRangeEff;
            viewFreqInput.step = step;
        }

        const vals = {
            t1: parseFloat(document.getElementById("t1").value),
            t2: parseFloat(document.getElementById("t2").value),
            duration: parseFloat(document.getElementById("duration").value),
            extra: parseFloat(document.getElementById("extra_time").value) || 0.0,
            freq: parseFloat(document.getElementById("freq_offset").value),
            type: document.getElementById("pulse_type").value,
            flip: parseFloat(document.getElementById("flip_angle").value),
            fRange: parseFloat(document.getElementById("freq_range").value),
            fPoints: parseFloat(document.getElementById("freq_points").value),
            tbw: parseFloat(document.getElementById("tbw").value),
            viewFreq: parseFloat(viewFreqInput ? viewFreqInput.value : document.getElementById("view_freq").value),
            viewTime: parseFloat(document.getElementById("view_time").value),
            is3d: document.getElementById("toggle_3d").checked,
            b1: b1val,
            showMx: document.getElementById("show_mx").checked,
            showMy: document.getElementById("show_my").checked
        };

        const runSim = pyodide.globals.get("run_simulation");
        const extractView = pyodide.globals.get("extract_view");

        if (needFullSim && runSim) {
            const result = runSim(vals.t1, vals.t2, vals.duration, vals.extra, vals.freq, vals.type,
                   vals.flip, vals.fRange, vals.fPoints, vals.tbw, vals.b1, vals.showMx, vals.showMy);

            // Sync UI
            if (result && result.length === 2) {
                const actFlip = result.get(0);
                const actB1 = result.get(1);

                if (vals.b1 > 0) {
                    if (document.activeElement.id !== "flip_angle") {
                        document.getElementById("flip_angle").value = actFlip.toFixed(2);
                    }
                } else {
                    if (document.activeElement.id !== "b1_amp") {
                        document.getElementById("b1_amp").value = actB1.toFixed(4); // G
                    }
                }
                result.destroy();
            }
        }

        if (extractView) {
            extractView(vals.viewFreq, vals.viewTime, vals.is3d);
        }

        statusEl.innerText = "Ready";
    } catch (e) {
        console.error("Sim Error", e);
        let msg = e.message || e.toString();
        if (msg.includes("PythonError:")) {
            msg = msg.split("PythonError:")[1];
        }
        statusEl.innerText = "Error (Details in Log)";
        logMessage(msg, "error");
    }
}

function triggerSimulation(event, forceRun = false) {
    if (!isPyodideReady) return;

    const sourceId = event ? event.target.id : null;
    const isPlayback = event ? event.isPlayback : false;

    // Sync view_time max
    const durationInput = document.getElementById("duration");
    const extraInput = document.getElementById("extra_time");
    const viewTimeInput = document.getElementById("view_time");

    if (durationInput && viewTimeInput) {
        const dur = parseFloat(durationInput.value) || 0;
        const extra = parseFloat(extraInput ? extraInput.value : 0) || 0;
        const totalDur = dur + extra;

        viewTimeInput.max = totalDur;

        // Only clamp if not playback
        if (!isPlayback) {
            if (parseFloat(viewTimeInput.value) > totalDur) {
                viewTimeInput.value = totalDur;
            }
        }
    }

    if (updateTimer) clearTimeout(updateTimer);

    const statusEl = document.getElementById('status-text');
    // For playback, don't show "Calculating..." constantly flickering
    if (!isPlayback) {
         statusEl.innerHTML = (sourceId || forceRun) ? '<span class="spinner"></span>Calculating...' : 'Updating view...';
    }

    if (isPlayback) {
        performRfSimulation(sourceId, forceRun);
    } else {
        updateTimer = setTimeout(() => performRfSimulation(sourceId, forceRun), 50);
    }
}

// Attach listeners
document.querySelectorAll('.sim-input').forEach(input => {
    input.addEventListener('input', triggerSimulation);
});
document.querySelectorAll('.slice-input').forEach(input => {
    input.addEventListener('input', triggerSliceSimulation);
});

// --- PLAYBACK ---
function togglePlayback(type) {
    const btnId = type === 'rf' ? 'play_btn_rf' : 'play_btn_slice';
    const sliderId = type === 'rf' ? 'view_time' : 'slice_view_time';
    const triggerFunc = type === 'rf' ? triggerSimulation : triggerSliceSimulation;

    const btn = document.getElementById(btnId);
    const slider = document.getElementById(sliderId);

    if (!btn || !slider) return;

    if (isPlaying) {
        // STOP
        clearInterval(playbackInterval);
        playbackInterval = null;
        isPlaying = false;
        btn.innerHTML = "&#9658;"; // Play symbol
    } else {
        // START
        isPlaying = true;
        btn.innerHTML = "&#10074;&#10074;"; // Pause symbol

        // If at end, restart
        if (parseFloat(slider.value) >= parseFloat(slider.max)) {
            slider.value = 0;
            triggerFunc({ target: { id: sliderId } });
        }

        playbackInterval = setInterval(() => {
            let val = parseFloat(slider.value);
            let max = parseFloat(slider.max);
            let step = parseFloat(slider.step) || 0.01;

            // Increment faster than single step for smoothness/speed balance
            // A 2ms pulse with 0.01 step has 200 steps. 50ms interval -> 10s duration.
            // Let's take 2x step.
            let nextVal = val + (step * 2);

            if (nextVal >= max) {
                nextVal = max;
                clearInterval(playbackInterval);
                playbackInterval = null;
                isPlaying = false;
                btn.innerHTML = "&#9658;";
            }

            slider.value = nextVal;
            // Mock an event object for the trigger function
            triggerFunc({ target: { id: sliderId }, isPlayback: true });

        }, 50); // 20fps
    }
}

// Attach Playback Listeners
document.getElementById('play_btn_rf').addEventListener('click', () => togglePlayback('rf'));
document.getElementById('play_btn_slice').addEventListener('click', () => togglePlayback('slice'));

// Stop playback on manual interaction
['view_time', 'slice_view_time'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
        el.addEventListener('mousedown', () => {
            if (isPlaying) togglePlayback(id === 'view_time' ? 'rf' : 'slice');
        });
        el.addEventListener('touchstart', () => { // Mobile support
            if (isPlaying) togglePlayback(id === 'view_time' ? 'rf' : 'slice');
        });
    }
});

// Start
init();
