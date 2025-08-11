from PyLTSpice import SimRunner, AscEditor, RawRead
from spicelib.editor.base_schematic import TextTypeEnum
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import shutil
import time

# TODO: run sims in parallel, use callbacks


def print_directives(netlist: AscEditor):
    for x in netlist.directives:
        if x.type == TextTypeEnum.DIRECTIVE:
            print(x.text)


def copy_libs(ltc: SimRunner):
    output_folder = Path(ltc.output_folder)

    for path in Path(".").glob("*.lib"):
        shutil.copy(path, output_folder)
    for path in Path(".").glob("*.asy"):
        shutil.copy(path, output_folder)


# Returns F_0db (Hz), phase margin (deg)
def compute_margins(freq: np.ndarray, gain: np.ndarray):
    freq = freq.real
    mag = np.abs(gain)
    phase = np.angle(gain)

    # Phase margin: phase at 0 dB gain
    idx = np.where(mag > 1)[0][-1]
    f_0db = np.interp(1, mag[idx : idx + 2], freq[idx : idx + 2])
    phase_margin = np.interp(f_0db, freq, phase)

    # TODO: Gain margin

    return f_0db, np.rad2deg(phase_margin)


def fra_sim(
    netlist: AscEditor,
    ltc: SimRunner,
    name: str = "fra",
    use_cached_results: bool = False,
):
    netlist.add_instruction(".fra")

    if use_cached_results:
        raw_file = f"tmp/{name}.raw"
        log_file = f"tmp/{name}.log"
    else:
        raw_file, log_file = ltc.run_now(netlist, run_filename=f"{name}.asc")

    if raw_file is None:
        with open(log_file) as f:
            print(f.read())

    fra_file = str(raw_file).replace(".raw", ".fra_1.raw")
    fra = RawRead(fra_file)

    freq = fra.get_trace("frequency").data
    gain = fra.get_trace("probe_full").data

    netlist.remove_instruction(".fra")
    
    # TODO: bode plot

    return compute_margins(freq, gain)


def ripple_sim(
    netlist: AscEditor,
    ltc: SimRunner,
    name: str = "ripple",
    use_cached_results: bool = False,
):
    netlist.add_instruction(".tran 25m")

    if use_cached_results:
        raw_file = f"tmp/{name}.raw"
        log_file = f"tmp/{name}.log"
    else:
        raw_file, log_file = ltc.run_now(netlist, run_filename=f"{name}.asc")

    if raw_file is None:
        with open(log_file) as f:
            print(f.read())

    # Speed up loading by only reading needed traces
    raw = RawRead(raw_file, traces_to_read=["time", "V(out)", "V(filt)"])
    t = raw.get_trace("time")
    out = raw.get_trace("V(out)")
    filt = raw.get_trace("V(filt)")

    i_last_ms = np.where(t > t[-1] - 1e-3)[0][0]

    filt_last_ms = filt[i_last_ms:]

    v_ppk = np.abs(np.max(filt_last_ms) - np.min(filt_last_ms))

    plt.figure(figsize=(8, 6), dpi=200)

    plt.subplot(2, 1, 1)
    plt.plot(t, out)
    plt.plot(t, filt)

    plt.subplot(2, 1, 2)
    plt.plot(t[i_last_ms:], filt_last_ms)

    plt.tight_layout()
    plt.savefig("ripple.png")

    netlist.remove_instruction(".tran 25m")

    # TODO: dissipation

    return v_ppk


def main():
    cached = False

    ltc = SimRunner(output_folder="tmp")

    netlist = AscEditor("power_supply_linear.asc")

    # Remove any existing simulation commands
    netlist.remove_Xinstruction(r"\.tran.*")
    netlist.remove_Xinstruction(r"\.fra.*")

    copy_libs(ltc)

    f0db, phase_margin = fra_sim(netlist, ltc, use_cached_results=cached)
    print(f"Crossover frequency: {f0db / 1e3:.4g} kHz")
    print(f"Phase margin: {phase_margin:.4g} deg")

    v_ppk = ripple_sim(netlist, ltc, use_cached_results=cached)
    print(f"Vppk = {v_ppk * 1e3:0.4g} mV")


if __name__ == "__main__":
    main()
