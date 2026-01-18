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
    # increased height slightly, used constrained_layout for tighter packing
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)

    # Customize layout
    fig.patch.set_facecolor('#ffffff')
    # 1. RF Pulse
    axs[0].set_title("RF Pulse Shape")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude (uT)")
    lines['rf_real'], = axs[0].plot([], [], label='Real', color='#0056b3')
    lines['rf_imag'], = axs[0].plot([], [], label='Imag', color='#ff9900', alpha=0.7)
    lines['time_line'] = axs[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right', fontsize='small')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Magnetization Evolution
    axs[1].set_title("Magnetization")
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
    axs[2].set_ylim(-2.1, 1.1)
    lines['mxy'], = axs[2].plot([], [], label='Mxy', color='purple')
    lines['mz_prof'], = axs[2].plot([], [], label='Mz', color='gray', linestyle='--')
    lines['freq_line'], = axs[2].plot([], [], label='Freq Offset', color='black', linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right', fontsize='small')
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Initial draw to attach to DOM
    plt.show()

def update_simulation(t1_ms, t2_ms, duration_ms, freq_offset_hz, pulse_type, view_freq_hz, view_time_ms):

    global fig, axs, lines



    # Initialize plot if needed

    if fig is None:

        init_plot()



    # --- PHYSICS or MOCK ---

    t1_s = t1_ms * 1e-3

    t2_s = t2_ms * 1e-3

    duration_s = duration_ms * 1e-3

    time_ms = np.linspace(0, duration_ms, 400)

    freq_range = np.linspace(-1000, 1000, 80)

    positions = np.array([[0, 0, 0]])  # Single position



    if HAS_BACKEND:

        tissue = TissueParameters(name="Gray Matter", t1=t1_s, t2=t2_s)

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

        gradients = np.zeros((len(time_s), 3))

        result = sim.simulate(

            sequence=(b1, gradients, time_s),

            tissue=tissue,

            frequencies=freq_range,

            positions=positions,

            mode=2

        )



        # Extract data

        freq_idx = int(np.argmin(np.abs(freq_range - view_freq_hz)))

        time_idx = int(np.argmin(np.abs(time_ms - view_time_ms)))

        mx, my, mz = result['mx'][:, time_idx, freq_idx], result['my'][:, time_idx, freq_idx], result['mz'][:, time_idx, freq_idx]

        mxy_prof = np.sqrt(result['mx'][time_idx, 0, :]**2 + result['my'][time_idx, 0, :]**2)

        mz_prof = result['mz'][time_idx, 0, :]

        rf_real, rf_imag = np.real(b1), np.imag(b1)

    else:

        # --- MOCK DATA FOR UI TESTING ---

        # Generate a shape based on pulse_type

        if pulse_type == 'sinc':

            rf_real = np.sinc((time_ms - duration_ms/2) / (duration_ms/4))

        elif pulse_type == 'gaussian':

            rf_real = np.exp(-(time_ms - duration_ms/2)**2 / (2 * (duration_ms/6)**2))

        else: # rect

            rf_real = np.ones_like(time_ms)



        # Add a "phase" based on freq_offset

        phase = 2 * np.pi * freq_offset_hz * (time_ms/1000)

        rf_complex = rf_real * np.exp(1j * phase)

        rf_real, rf_imag = np.real(rf_complex), np.imag(rf_complex)



        # Mock magnetization (simple decay/rotation)

        mx = np.sin(time_ms / duration_ms * np.pi/2) * np.exp(-time_ms/t2_ms)

        my = np.zeros_like(time_ms)

        mz = np.cos(time_ms / duration_ms * np.pi/2)



        # Mock Profile (Gaussian peak)

        mxy_prof = np.exp(-(freq_range - freq_offset_hz)**2 / (2 * 200**2))

        mz_prof = 1 - mxy_prof



    # --- UPDATE PLOTS ---

    # 1. RF Pulse

    lines['rf_real'].set_data(time_ms, rf_real)

    lines['rf_imag'].set_data(time_ms, rf_imag)

    lines['time_line'].set_data([view_time_ms, view_time_ms], [min(rf_real)-0.1, max(rf_real)+0.1])

    axs[0].relim()

    axs[0].autoscale_view()



    # 2. Magnetization

    lines['mx'].set_data(time_ms, mx)

    lines['my'].set_data(time_ms, my)

    lines['mz'].set_data(time_ms, mz)

    axs[1].set_xlim(0, max(time_ms))

    axs[1].set_ylim(-1.1, 1.1)



    # 3. Profile

    lines['mxy'].set_data(freq_range, mxy_prof)

    lines['mz_prof'].set_data(freq_range, mz_prof)

    lines['freq_line'].set_data([view_freq_hz, view_freq_hz], [-1, 1])

    axs[2].set_xlim(min(freq_range), max(freq_range))



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

    // Sync view_time max with duration
    const durationInput = document.getElementById("duration");
    const viewTimeInput = document.getElementById("view_time");
    if (durationInput && viewTimeInput) {
        const dur = parseFloat(durationInput.value);
        if (!isNaN(dur)) {
            viewTimeInput.max = dur;
            if (parseFloat(viewTimeInput.value) > dur) {
                viewTimeInput.value = dur;
                // Update label immediately for responsiveness
                const timeLabel = document.getElementById("time_ms_val");
                if (timeLabel) timeLabel.innerText = dur;
            }
        }
    }

    // Debounce logic
    if (updateTimer) clearTimeout(updateTimer);

    document.getElementById('status-text').innerHTML = '<span class="spinner"></span>Calculating...';

    updateTimer = setTimeout(async () => {
        const vals = {
            t1: parseFloat(document.getElementById("t1").value),
            t2: parseFloat(document.getElementById("t2").value),
            duration: parseFloat(document.getElementById("duration").value),
            freq: parseFloat(document.getElementById("freq_offset").value),
            type: document.getElementById("pulse_type").value,
            viewFreq: parseFloat(document.getElementById("view_freq").value),
            viewTime: parseFloat(document.getElementById("view_time").value)
        };

        // Update freq label
        document.getElementById("freq_hz_val").innerText = vals.viewFreq;
        document.getElementById("time_ms_val").innerText = vals.viewTime;

        try {
            const pyFunc = pyodide.globals.get("update_simulation");
            if (pyFunc) {
                pyFunc(vals.t1, vals.t2, vals.duration, vals.freq, vals.type, vals.viewFreq, vals.viewTime);
            }
            document.getElementById('status-text').innerText = "Ready";
        } catch (e) {
            console.error("Sim Error", e);
            // Show the actual error message (truncated if too long)
            let msg = e.message || e.toString();
            if (msg.includes("PythonError:")) {
                msg = msg.split("PythonError:")[1].split("\n")[0];
            }
            document.getElementById('status-text').innerText = "Error: " + msg.substring(0, 60);
            document.getElementById('status-text').title = msg; // Hover to see full error
        }
    }, 50); // 50ms delay
}

// Attach listeners
document.querySelectorAll('.sim-input').forEach(input => {
    input.addEventListener('input', triggerSimulation);
});

// Start
init();
