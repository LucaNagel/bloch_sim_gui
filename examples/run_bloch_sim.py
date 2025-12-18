"""
Command-line helper to run the Bloch simulator without the GUI.

Examples
--------
python run_bloch_sim.py --flip 90 --duration_ms 1.0 --nfreq 3 --freq_span 100
python run_bloch_sim.py --flip 30 --duration_ms 0.5 --freqs -50,0,50 --npos 5 --pos_span_cm 10
"""

import argparse
import numpy as np

from bloch_simulator import BlochSimulator, TissueParameters, design_rf_pulse


def parse_args():
    p = argparse.ArgumentParser(description="Run Bloch simulation from the command line.")
    p.add_argument("--flip", type=float, default=90.0, help="Flip angle in degrees (default: 90)")
    p.add_argument("--duration_ms", type=float, default=1.0, help="RF pulse duration in ms (default: 1.0)")
    p.add_argument("--nfreq", type=int, default=1, help="Number of off-resonance frequencies (default: 1)")
    p.add_argument(
        "--freq_span",
        type=float,
        default=0.0,
        help="Symmetric frequency span in Hz (centered at 0). Ignored if --freqs is provided.",
    )
    p.add_argument(
        "--freqs",
        type=str,
        default=None,
        help="Comma-separated list of frequencies in Hz (overrides nfreq/freq_span). Example: -50,0,50",
    )
    p.add_argument("--npos", type=int, default=1, help="Number of positions along x (default: 1)")
    p.add_argument(
        "--pos_span_cm",
        type=float,
        default=0.0,
        help="Symmetric spatial span in cm along x (default: 0 => all at 0)",
    )
    p.add_argument("--t1", type=float, default=1.0, help="T1 in seconds (default: 1.0)")
    p.add_argument("--t2", type=float, default=0.1, help="T2 in seconds (default: 0.1)")
    p.add_argument(
        "--mode",
        type=str,
        choices=["endpoint", "time-resolved"],
        default="time-resolved",
        help="Simulation mode (default: time-resolved)",
    )
    p.add_argument("--use_parallel", action="store_true", help="Enable parallel simulation when many spins are present.")
    return p.parse_args()


def build_frequencies(args: argparse.Namespace) -> np.ndarray:
    if args.freqs:
        freqs = np.array([float(f) for f in args.freqs.split(",")])
        return freqs
    if args.nfreq <= 1:
        return np.array([0.0])
    span = args.freq_span if args.freq_span > 0 else max(1.0, args.nfreq - 1)
    return np.linspace(-span / 2.0, span / 2.0, args.nfreq)


def build_positions(args: argparse.Namespace) -> np.ndarray:
    if args.npos <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    span_m = args.pos_span_cm / 100.0
    xs = np.linspace(-span_m / 2.0, span_m / 2.0, args.npos)
    pos = np.zeros((args.npos, 3))
    pos[:, 0] = xs
    return pos


def main():
    args = parse_args()

    # RF pulse
    duration_s = args.duration_ms / 1000.0
    npoints = max(8, int(np.ceil(duration_s / 5e-6)))
    b1, time = design_rf_pulse("rect", duration=duration_s, flip_angle=args.flip, npoints=npoints)
    gradients = np.zeros((len(time), 3))

    # Tissue and sim
    tissue = TissueParameters(name="CLI", t1=args.t1, t2=args.t2)
    sim = BlochSimulator(use_parallel=args.use_parallel)

    frequencies = build_frequencies(args)
    positions = build_positions(args)
    mode = 2 if args.mode == "time-resolved" else 0

    result = sim.simulate((b1, gradients, time), tissue, positions=positions, frequencies=frequencies, mode=mode)

    print("== Simulation complete ==")
    print(f"Mode: {args.mode}")
    print(f"RF points: {len(b1)}, dt ~ {np.mean(np.diff(time))*1e6:.3f} us, duration: {duration_s*1e3:.3f} ms")
    print(f"Positions: {positions.shape}, Frequencies: {frequencies}")
    print(f"mx shape: {np.shape(result['mx'])}, my shape: {np.shape(result['my'])}, mz shape: {np.shape(result['mz'])}")
    print(f"signal shape: {np.shape(result['signal'])}, time shape: {np.shape(result['time'])}")

    # Show final magnetization per position/freq
    if mode == 0:
        mx = result["mx"]
        my = result["my"]
        mz = result["mz"]
    else:
        mx = result["mx"][-1]
        my = result["my"][-1]
        mz = result["mz"][-1]
    print("Final magnetization (per position x frequency):")
    for pi in range(mx.shape[0]):
        for fi in range(mx.shape[1]):
            mxy = np.sqrt(mx[pi, fi] ** 2 + my[pi, fi] ** 2)
            print(f"  pos {pi:02d}, freq {fi:02d}: mx={mx[pi,fi]: .4f}, my={my[pi,fi]: .4f}, mz={mz[pi,fi]: .4f}, |Mxy|={mxy: .4f}")


if __name__ == "__main__":
    main()
