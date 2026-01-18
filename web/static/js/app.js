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
    if (target) {
        target.classList.add('active');
    }

    // Trigger sim if entering sim view and ready
    if (viewName === 'rf-pulse' && isPyodideReady) {
        triggerSimulation();
    }
}

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
        } catch (e) {
            console.warn("Wheel install failed (expected in dev without build):", e);
            status.innerText = "Dev mode: Wheel missing (Simulation mock)";
            // In real deployment, this throws. In dev, we might want to fail gracefully or show a message.
        }

        // 4. Setup Python Environment
        status.innerText = "Starting engine...";
        await pyodide.runPythonAsync(`
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_pyodide.wasm_backend")
import matplotlib.pyplot as plt
from js import document

# Try importing, otherwise mock for UI testing if wheel is missing
try:
    from blochsimulator import BlochSimulator, TissueParameters, design_rf_pulse
    HAS_BACKEND = True
    sim = BlochSimulator(use_parallel=False)
except ImportError:
    HAS_BACKEND = False
    print("Warning: Backend not found. Using mocks.")

# Global Plot Objects
fig, axs = None, None
lines = {} # Store line objects for updates

def init_plot():
    global fig, axs, lines
    plt.clf()
    # Create 3 subplots with shared x/y where appropriate
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    # Customize layout
    fig.patch.set_facecolor('#ffffff')

    # 1. RF Pulse
    axs[0].set_title("RF Pulse Shape")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude (uT)")
    lines['rf_real'], = axs[0].plot([], [], label='Real', color='#0056b3')
    lines['rf_imag'], = axs[0].plot([], [], label='Imag', color='#ff9900', alpha=0.7)
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Magnetization Evolution
    axs[1].set_title("Magnetization (Center)")
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylim(-1.1, 1.1)
    lines['mx'], = axs[1].plot([], [], label='Mx', color='r', alpha=0.6)
    lines['my'], = axs[1].plot([], [], label='My', color='g', alpha=0.6)
    lines['mz'], = axs[1].plot([], [], label='Mz', color='b')
    axs[1].legend(loc='upper right', fontsize='small')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Frequency Profile
    axs[2].set_title("Excitation Profile")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylim(-0.1, 1.1)
    lines['mxy'], = axs[2].plot([], [], label='Mxy', color='purple')
    lines['mz_prof'], = axs[2].plot([], [], label='Mz', color='gray', linestyle='--')
    axs[2].legend(loc='upper right', fontsize='small')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Initial draw to attach to DOM
    plt.show()

def update_simulation(t1_ms, t2_ms, duration_ms, freq_offset_hz, pulse_type):
    global fig, axs, lines

    if not HAS_BACKEND:
        return

    # Initialize plot if needed
    if fig is None:
        init_plot()

    # --- PHYSICS ---
    t1_s = t1_ms * 1e-3
    t2_s = t2_ms * 1e-3
    duration_s = duration_ms * 1e-3

    tissue = TissueParameters(name="Gray Matter", t1=t1_s, t2=t2_s)

    # Pulse Design
    npoints = 400
    b1, time_s = design_rf_pulse(
        pulse_type=pulse_type,
        duration=duration_s,
        flip_angle=90,
        time_bw_product=4,
        npoints=npoints,
        freq_offset=freq_offset_hz
    )
    time_ms = time_s * 1e3

    # Simulation
    freq_range = np.linspace(-1000, 1000, 80) # Reduced points for speed
    positions = np.array([[0.0, 0.0, 0.0]])

    gradients = np.zeros((len(time_s), 3))

    result = sim.simulate(
        sequence=(b1, gradients, time_s),
        tissue=tissue,
        frequencies=freq_range,
        positions=positions,
        mode=2
    )

    # --- UPDATE PLOTS ---
    # 1. RF Pulse
    lines['rf_real'].set_data(time_ms, np.real(b1))
    lines['rf_imag'].set_data(time_ms, np.imag(b1))
    axs[0].relim()
    axs[0].autoscale_view()

    # 2. Magnetization (Center Freq)
    center_idx = len(freq_range) // 2
    mx = result['mx'][:, 0, center_idx]
    my = result['my'][:, 0, center_idx]
    mz = result['mz'][:, 0, center_idx]

    lines['mx'].set_data(time_ms, mx)
    lines['my'].set_data(time_ms, my)
    lines['mz'].set_data(time_ms, mz)
    axs[1].set_xlim(0, max(time_ms))

    # 3. Profile
    mx_end = result['mx'][-1, 0, :]
    my_end = result['my'][-1, 0, :]
    mz_end = result['mz'][-1, 0, :]
    mxy_end = np.sqrt(mx_end**2 + my_end**2)

    lines['mxy'].set_data(freq_range, mxy_end)
    lines['mz_prof'].set_data(freq_range, mz_end)

    # Redraw
    fig.canvas.draw()
    fig.canvas.flush_events()
        `);

        isPyodideReady = true;
        status.innerText = "Ready";

        // If we are already on the sim page, run it
        if (document.getElementById('rf-pulse-view').classList.contains('active')) {
            triggerSimulation();
        }

    } catch (err) {
        status.innerText = "Error: " + err.message;
        console.error(err);
    }
}

// --- INTERACTION ---
function triggerSimulation() {
    if (!isPyodideReady) return;

    // Debounce logic
    if (updateTimer) clearTimeout(updateTimer);

    document.getElementById('status-text').innerHTML = '<span class="spinner"></span>Calculating...';

    updateTimer = setTimeout(async () => {
        const vals = {
            t1: parseFloat(document.getElementById("t1").value),
            t2: parseFloat(document.getElementById("t2").value),
            duration: parseFloat(document.getElementById("duration").value),
            freq: parseFloat(document.getElementById("freq_offset").value),
            type: document.getElementById("pulse_type").value
        };

        try {
            const pyFunc = pyodide.globals.get("update_simulation");
            if (pyFunc) {
                pyFunc(vals.t1, vals.t2, vals.duration, vals.freq, vals.type);
            }
            document.getElementById('status-text').innerText = "Ready";
        } catch (e) {
            console.error("Sim Error", e);
            document.getElementById('status-text').innerText = "Simulation Error";
        }
    }, 50); // 50ms delay
}

// Attach listeners
document.querySelectorAll('.sim-input').forEach(input => {
    input.addEventListener('input', triggerSimulation);
});

// Start
init();
