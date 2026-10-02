"""Quantum tunneling backend: NumPy split-step Fourier physics served over HTTP.

Run:  python server.py   (opens http://127.0.0.1:8000 in your browser)
"""
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ----------------------------------------------------------------------
# Grid and constants
# ----------------------------------------------------------------------
L = 360.0                  # wide box so waves can travel far past the barrier
N = 12288                  # keeps dx identical to the original (L=120, N=4096)
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
dx = x[1] - x[0]
k_grid = 2 * np.pi * np.fft.fftfreq(N, d=dx)

K_NYQUIST = np.pi / dx
K0_SAFE_MAX = 0.75 * K_NYQUIST

DT = 0.004
SUBSTEPS_PER_FRAME = 20
MAX_FRAMES = 900
MAX_OUT = 2400             # max points per packet sent to the browser per frame
MAX_PACKETS = 5
X0 = -30.0                 # start of packet 1; packet j starts PACKET_SPACING further left
PACKET_SPACING = 25.0
SIGMA = 2.5

# Absorbing layers at both ends. The FFT is periodic, so without them a wave
# leaving on the right would re-enter on the left. The layer fades the wave
# out smoothly (quadratic ramp) so it vanishes instead of reflecting.
EDGE_WIDTH = 40.0
EDGE_START = L / 2 - EDGE_WIDTH
ETA_MAX = 4.0
_dist = np.abs(x) - EDGE_START
_eta = np.zeros(N)
_m = _dist > 0
_eta[_m] = ETA_MAX * (_dist[_m] / EDGE_WIDTH) ** 2
ABSORBER = np.exp(-_eta * DT)
KIN = np.exp(-1j * (k_grid ** 2) / 2 * DT)
DEFAULT_VIEW = (-60.0, 60.0)


# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
def make_packet(k0, x0):
    psi = np.exp(-(x - x0) ** 2 / (4 * SIGMA ** 2)) * np.exp(1j * k0 * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi


def analytic_transmission(V0, width, k0):
    """Exact rectangular-barrier transmission at the packet's central energy."""
    E = k0 ** 2 / 2
    if E < V0:
        kappa = np.sqrt(2 * (V0 - E))
        denom = 1 + (V0 ** 2 * np.sinh(kappa * width) ** 2) / (4 * E * (V0 - E) + 1e-300)
    elif E > V0:
        kap = np.sqrt(2 * (E - V0))
        denom = 1 + (V0 ** 2 * np.sin(kap * width) ** 2) / (4 * E * (E - V0) + 1e-300)
    else:
        denom = 1 + V0 * width ** 2 / 2
    return float(1.0 / denom)


class Simulation:
    """Several non-interacting wave packets in the same barrier.

    Each packet is an independent solution of the Schrodinger equation (the
    equation is linear), so they are simulated side by side and drawn separately.
    """

    def __init__(self):
        self.reset(10.0, 2.0, 4.0, 1, 0.0, *DEFAULT_VIEW)

    def reset(self, V0, width, k0, count, dk, xmin, xmax):
        V = np.where(np.abs(x) < width / 2, V0, 0.0)
        self.half_pot = np.exp(-1j * V * DT / 2)
        # Consecutive half-steps of the potential merge into one full step;
        # the absorber sits between them (all three are pointwise, so they commute).
        self.full_abs = np.exp(-1j * V * DT) * ABSORBER
        self.half_abs = self.half_pot * ABSORBER
        self.V0, self.width, self.count = V0, width, count
        self.k0s = [k0 + j * dk for j in range(count)]
        self.x0s = [X0 - PACKET_SPACING * j for j in range(count)]
        self.psi = np.array([make_packet(k, x0) for k, x0 in zip(self.k0s, self.x0s)])   # (count, N)
        self.frame = 0
        self.entered = [False] * count
        self.measured = [False] * count
        self.meas = [(None, None)] * count
        ymax = float(np.max(np.abs(self.psi) ** 2) * 1.3)
        return {**self._meta(), "ymax": ymax, **self._observe(xmin, xmax, update=False)}

    def _meta(self):
        packets = []
        for k0, x0 in zip(self.k0s, self.x0s):
            E = k0 ** 2 / 2
            packets.append({"k0": k0, "x0": x0, "energy": E, "forbidden": bool(E < self.V0),
                            "t_analytic": analytic_transmission(self.V0, self.width, k0)})
        return {"V0": self.V0, "width": self.width, "specs": packets}

    def step(self, xmin, xmax, advance=True):
        if advance and self.frame < MAX_FRAMES:
            psi = self.half_pot * self.psi
            for s in range(SUBSTEPS_PER_FRAME):
                psi = np.fft.ifft(KIN * np.fft.fft(psi, axis=1), axis=1)
                psi *= self.full_abs if s < SUBSTEPS_PER_FRAME - 1 else self.half_abs
            self.psi = psi
            self.frame += 1
        return self._observe(xmin, xmax, update=advance)

    def _observe(self, xmin, xmax, update):
        density = np.abs(self.psi) ** 2                      # (count, N)
        h = self.width / 2
        P_barrier = density[:, np.abs(x) <= h].sum(axis=1) * dx
        P_left = density[:, x < -h].sum(axis=1) * dx
        P_right = density[:, x > h].sum(axis=1) * dx

        # Freeze each packet's transmission the first time it has cleared the
        # barrier, before the absorber removes a meaningful share of its norm.
        if update:
            norm = P_left + P_barrier + P_right
            for j in range(self.count):
                if not self.entered[j] and P_barrier[j] > 0.01:
                    self.entered[j] = True
                if self.entered[j] and not self.measured[j] and P_barrier[j] < 1e-3 and norm[j] > 0.9:
                    self.measured[j] = True
                    self.meas[j] = (float(P_left[j]), float(P_right[j]))

        # Only the part the browser is looking at, thinned to <= MAX_OUT points.
        i0 = int(np.clip(np.floor((xmin + L / 2) / dx), 0, N - 2))
        i1 = min(N, max(i0 + 2, int(np.ceil((xmax + L / 2) / dx)) + 1))
        stride = max(1, -(-(i1 - i0) // MAX_OUT))
        packets = [{
            "density": np.round(density[j, i0:i1:stride], 6).tolist(),
            "p_left": float(P_left[j]), "p_right": float(P_right[j]),
            "measured": self.measured[j],
            "p_left_meas": self.meas[j][0], "p_right_meas": self.meas[j][1],
        } for j in range(self.count)]
        return {"x0": float(x[i0]), "dx_out": float(dx * stride), "packets": packets,
                "frame": self.frame, "finished": self.frame >= MAX_FRAMES}


# ----------------------------------------------------------------------
# HTTP API
# ----------------------------------------------------------------------
# When packaged by PyInstaller, bundled files are unpacked to sys._MEIPASS
BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
sim = Simulation()
lock = threading.Lock()


def parse_view(data):
    try:
        a, b = float(data.get("xmin", DEFAULT_VIEW[0])), float(data.get("xmax", DEFAULT_VIEW[1]))
    except (TypeError, ValueError):
        return DEFAULT_VIEW
    return (a, b) if np.isfinite(a) and np.isfinite(b) and b > a else DEFAULT_VIEW


def validate(data):
    try:
        V0, width, k0, dk = (float(data[k]) for k in ("V0", "width", "k0", "dk"))
        count = int(data["count"])
    except (KeyError, TypeError, ValueError):
        return None, "Enter a valid number in every field."
    if not all(np.isfinite([V0, width, k0, dk])):
        return None, "Enter a valid number in every field."
    if width <= 0:
        return None, "Barrier width must be positive."
    if not 1 <= count <= MAX_PACKETS:
        return None, f"Choose between 1 and {MAX_PACKETS} wave packets."
    for j in range(count):
        kj = k0 + j * dk
        if kj <= 0:
            return None, f"Packet {j + 1} would have k0 = {kj:g}. Momentum must stay positive."
        if kj > K0_SAFE_MAX:
            return None, (f"Packet {j + 1} would have k0 = {kj:g}, too high for this grid "
                          f"(max safe k0 = {K0_SAFE_MAX:.1f}). Above it the packet aliases and "
                          f"appears to move the wrong way. Increase N in server.py to raise the limit.")
    return (V0, width, k0, count, dk), None


@app.get("/")
def index():
    for folder in (app.static_folder, BASE):
        if os.path.exists(os.path.join(folder, "index.html")):
            return send_from_directory(folder, "index.html")
    return "index.html not found next to server.py or in static/", 404


@app.get("/api/config")
def config():
    return jsonify(L=L, edge_start=EDGE_START, k0_max=float(K0_SAFE_MAX),
                   max_frames=MAX_FRAMES, max_packets=MAX_PACKETS)


@app.post("/api/reset")
def reset():
    data = request.get_json(silent=True) or {}
    params, error = validate(data)
    if error:
        return jsonify(error=error), 400
    with lock:
        return jsonify(sim.reset(*params, *parse_view(data)))


@app.post("/api/step")
def step():
    data = request.get_json(silent=True) or {}
    with lock:
        return jsonify(sim.step(*parse_view(data), advance=data.get("advance", True) is not False))


# ----------------------------------------------------------------------
# Launcher: open the browser, and quit when the browser tab is closed
# ----------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 8000
URL = f"http://{HOST}:{PORT}"
_seen = {"last": time.time(), "bye": None}


@app.post("/api/ping")
def ping():                      # the page calls this every few seconds
    _seen["last"] = time.time()
    _seen["bye"] = None          # a refresh pings again, which cancels a pending quit
    return jsonify(ok=True)


@app.post("/api/bye")
def bye():                       # the page calls this as it closes
    _seen["bye"] = time.time() + 5
    return jsonify(ok=True)


def _watchdog():
    while True:
        time.sleep(1)
        now = time.time()
        if (_seen["bye"] and now > _seen["bye"]) or now - _seen["last"] > 120:
            os._exit(0)


def _already_running():
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((HOST, PORT)) == 0


def main():
    if _already_running():       # double-clicked twice: just reopen the page
        webbrowser.open(URL)
        return
    for name in ("stdout", "stderr"):          # pythonw has no console streams
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(target=_watchdog, daemon=True).start()
    threading.Timer(1.0, webbrowser.open, args=[URL]).start()
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    print(f"Quantum tunneling server running at {URL}  (Ctrl+C to stop)")
    main()
