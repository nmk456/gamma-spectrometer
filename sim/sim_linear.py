import matplotlib

matplotlib.use("Agg")

from PyLTSpice import SimRunner, AscEditor, RawRead
from spicelib.editor.base_schematic import TextTypeEnum
from pathlib import Path
from matplotlib import pyplot as plt
from threading import Lock
import numpy as np
import shutil
import time
import re
import json

ANALYSIS_REGEX = [r"\.tran.*", r"\.fra.*"]


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


# Copied from spicelib, modified to not print error message if instruction not found
def remove_Xinstruction_quiet(netlist: AscEditor, pattern: str):
    regex = re.compile(pattern, re.IGNORECASE)
    instr_removed = False
    i = 0
    while i < len(netlist.directives):
        if netlist.directives[i].type == TextTypeEnum.COMMENT:
            i += 1
            continue  # this is a comment
        instruction = netlist.directives[i].text
        if regex.match(instruction) is not None:
            instr_removed = True
            del netlist.directives[i]
            super(AscEditor, netlist).remove_instruction(instruction)
        else:
            i += 1
    if instr_removed:
        netlist.updated = True
        return True
    else:
        return False


class Sim:
    def __init__(
        self, source: str, parallel_sims: int, output_folder: str, cached: bool = False
    ):
        self.source = Path(source)
        self.cached = cached
        self.ltc = SimRunner(
            parallel_sims=parallel_sims, timeout=180, output_folder=output_folder
        )
        copy_libs(self.ltc)

        self.results = {}
        self.results_log = []

        self.plot_lock = Lock()

    def get_netlist(self):
        netlist = AscEditor(self.source)

        # Remove existing sim commands
        for r in ANALYSIS_REGEX:
            # netlist.remove_Xinstruction(r)
            remove_Xinstruction_quiet(netlist, r)

        return netlist

    def run_sim(self, netlist: AscEditor, name: str, cmd: str):
        if self.cached:
            raw_file = self.ltc.output_folder / f"{name}.raw"
            log_file = self.ltc.output_folder / f"{name}.log"

            print(f"Using cached files for {name}")
        else:
            netlist.add_instruction(cmd)
            start = time.time()
            raw_file, log_file = self.ltc.run_now(netlist, run_filename=f"{name}.asc")
            end = time.time()

            print(f"Ran {name} sim, took {end - start:0.2f}s")

        if raw_file is None:
            with open(log_file) as f:
                print(f.read())

                return None

        return raw_file, log_file

    def start_sim_callback(self, netlist: AscEditor, name: str, cmd: str, callback):
        args = {"name": name}

        netlist.add_instruction(cmd)
        self.ltc.run(
            netlist, callback=callback, callback_args=args, run_filename=f"{name}.asc"
        )
        netlist.remove_instruction(cmd)

        self.results[name] = {
            "cmd": cmd,
            "netlist": netlist,
        }

    def start_fra_sim(self, netlist: AscEditor, name: str = "fra"):
        self.start_sim_callback(netlist, name, ".fra startup", self.fra_sim_callback)

    def fra_sim_callback(self, raw_file: Path, log_file: Path, name: str):
        fra_file = str(raw_file).replace(".raw", ".fra_1.raw")
        fra = RawRead(fra_file)

        freq = fra.get_trace("frequency").data.real
        gain = fra.get_trace("probe_full").data

        mag = np.abs(gain)
        phase = np.unwrap(np.angle(gain))

        # Phase margin
        idx_cg = np.where(mag < 1)[0][0] - 1
        gain_crossover = np.interp(
            1, mag[idx_cg : idx_cg + 2][::-1], freq[idx_cg : idx_cg + 2][::-1]
        )
        phi_margin = np.interp(gain_crossover, freq, phase)

        self.results[name]["gain_crossover"] = gain_crossover
        self.results[name]["phi_margin"] = np.rad2deg(phi_margin)

        # Gain margin
        idx_cp = np.where(phase > 0)[0][-1]
        phase_crossover = np.interp(
            0, phase[idx_cp : idx_cp + 2], freq[idx_cp : idx_cp + 2]
        )
        g_margin = np.interp(phase_crossover, freq, mag)

        self.results[name]["phase_crossover"] = phase_crossover
        self.results[name]["g_margin"] = g_margin

        with self.plot_lock:
            plt.figure(figsize=(8, 6), dpi=200)

            plt.subplot(2, 1, 1)
            plt.semilogx(freq.real, 20 * np.log10(mag))
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.semilogx(freq.real, np.rad2deg(phase))
            plt.grid()

            plt.tight_layout()
            plt.savefig(f"{name}.png")

        self.results[name]["graph"] = [f"{name}.png"]

        log = f"# {name}\n\n"
        log += f"Crossover frequency: {self.results[name]['gain_crossover'] / 1e3:.4g} kHz\n"
        log += f"Phase margin: {self.results[name]['phi_margin']:.4g} deg\n\n"

        self.results_log.append(log)

    def start_ripple_sim(self, netlist: AscEditor, name: str = "ripple"):
        self.start_sim_callback(
            netlist, name, ".tran 40m startup", self.ripple_sim_callback
        )

    def ripple_sim_callback(self, raw_file: Path, log_file: Path, name: str):
        # Speed up loading by only reading needed traces
        raw = RawRead(raw_file, traces_to_read=["time", "V(out_ac)", "V(filt_ac)"])
        t = raw.get_trace("time").get_wave()  # Workaround for compression issues
        out = raw.get_trace("V(out_ac)")
        filt = raw.get_trace("V(filt_ac)")

        i_last_ms = np.where(t > t[-1] - 1e-3)[0][0]

        t_last_ms = t[i_last_ms:]
        filt_last_ms = filt[i_last_ms:]
        out_last_ms = out[i_last_ms:]

        self.results[name]["Vppk(OUT)"] = float(
            np.abs(np.max(out_last_ms) - np.min(out_last_ms))
        )
        self.results[name]["Vppk(FILT)"] = float(
            np.abs(np.max(filt_last_ms) - np.min(filt_last_ms))
        )

        with self.plot_lock:
            plt.figure(figsize=(8, 8), dpi=200)

            plt.subplot(3, 1, 1)
            plt.plot(t, out, label="OUT")
            plt.plot(t, filt, label="FILT")
            plt.title("Full Span")
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(t_last_ms, out_last_ms)
            plt.title("OUT (last 1ms)")

            plt.subplot(3, 1, 3)
            plt.plot(t_last_ms, filt_last_ms)
            plt.title("FILT (last 1ms)")

            plt.tight_layout()
            plt.savefig(f"{name}.png")

        self.results[name]["graph"] = [f"{name}.png"]

        log = f"# {name}\n\n"
        log += f"Vppk(OUT): {self.results[name]['Vppk(OUT)'] * 1e3:0.4g} mV\n"
        log += f"Vppk(FILT): {self.results[name]['Vppk(FILT)'] * 1e3:0.4g} mV\n\n"

        self.results_log.append(log)

    def log_results(self, file: str):
        with open(file, "w") as f:
            for log in self.results_log:
                f.write(log)

    def log_json(self, file: str):
        with open(file, "w") as f:
            f.write(
                json.dumps(
                    self.results,
                    default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
                )
            )

    def sensitivity_analysis(self, components: list, rel_step=0.01):
        for component in components:
            netlist = self.get_netlist()

            value = netlist.get_component_floatvalue(component)
            value += value * rel_step
            netlist.set_component_value(component, value)

            self.start_fra_sim(netlist, f"fra_{component}")
            self.start_ripple_sim(netlist, f"ripple_{component}")

        self.ltc.wait_completion()
        netlist = self.get_netlist()

        # Partial derivatives
        # results = {}
        log = "# Sensitivity Analysis\n\n"

        log += f"Increased each component's value by {rel_step:.3%}\n\n"

        for component in components:
            try:
                log += f"{component} ({netlist.get_component_value(component)}):\n"

                results = {}

                params = [
                    ("fra", "gain_crossover"),
                    ("fra", "phi_margin"),
                    ("ripple", "Vppk(OUT)"),
                    ("ripple", "Vppk(FILT)"),
                ]

                for sim_type, param in params:
                    baseline = self.results[f"{sim_type}_baseline"][param]
                    changed = self.results[f"{sim_type}_{component}"][param]

                    results[f"{param}_delta"] = changed - baseline
                    results[f"{param}_rel"] = changed / baseline - 1

                log += f"* Gain crossover frequency: {results['gain_crossover_delta']:0.4g} Hz ({results['gain_crossover_rel']:.5%})\n"
                log += f"* Phase margin: {results['phi_margin_delta']:0.4g} deg ({results['phi_margin_rel']:.5%})\n"
                log += f"* OUT ripple: {results['Vppk(OUT)_delta'] * 1e3:0.4g} mV ({results['Vppk(OUT)_rel']:.5%})\n"
                log += f"* FILT ripple: {results['Vppk(FILT)_delta'] * 1e3:0.4g} mV ({results['Vppk(FILT)_rel']:.5%})\n"

                log += "\n"
            except KeyError:
                pass

        self.results_log.append(log)


def main():
    cached = True

    start = time.time()
    sim = Sim("power_supply_linear.asc", 4, "tmp", cached)

    sim.start_fra_sim(sim.get_netlist(), "fra_baseline")
    sim.start_ripple_sim(sim.get_netlist(), "ripple_baseline")

    # components = ["C1", "C4"]
    components = ["C1", "C4", "C5", "R2", "R3", "R14", "R15", "C13", "R4"]
    sim.sensitivity_analysis(components)

    sim.ltc.wait_completion()

    sim.log_results("log.md")
    sim.log_json("results.json")

    end = time.time()
    print(f"Total time: {end - start:0.2f}s")


if __name__ == "__main__":
    main()
