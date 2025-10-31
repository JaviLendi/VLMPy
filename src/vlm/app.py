import os
# Limit BLAS/OMP threads early to avoid oversubscription on multi-core systems
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import logging
from uuid import uuid4
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports kept together; optional/possibly heavy imports guarded for clearer errors
try:
    import numpy as np
except Exception as e:
    raise ImportError("numpy is required for this application") from e

try:
    import psutil
except Exception:
    # psutil is useful but optional for resource tuning; fall back gracefully
    psutil = None

try:
    from flask import (
        Flask, render_template, request, jsonify, session,
        redirect, url_for, send_file
    )
except Exception as e:
    raise ImportError("Flask is required for this application") from e

# Compression and UI are optional; provide informative fallbacks
try:
    from flask_compress import Compress
except Exception:
    Compress = None

try:
    from webui import WebUI
except Exception:
    WebUI = None

# Plotly imports; raise clear error if missing
try:
    import plotly.graph_objs as go
    import plotly.utils
    import plotly.io as pio
except Exception as e:
    raise ImportError("plotly is required for plotting endpoints") from e

import re

# Plotly default format
pio.kaleido.scope.default_format = "svg"

# Local imports (asegúrate que el path '../' es correcto)
import sys
sys.path.append('../')
from lib.vlm import (
    VLM, plot_distribution_all, plot_wing_heatmap,
    plot_wing_heatmap_2d, plot_coefficient_vs_alpha,
    plot_distribution, plot_CL_CD, plot_CLCD_vs_alpha
)
from lib.naca import plot_naca_airfoil, naca_csv
from lib.geometry import (
    plot_wing_geometry, plot_wing_geometry_2d,
    plot_wing_discretization_2d, plot_wing_discretization_3d
)
from lib.config_manager import config_manager

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App
app = Flask(__name__)
Compress(app)
app.secret_key = os.urandom(24)
ui = WebUI(app, debug=True)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAVED_AIRFOILS = DATA_DIR / "saved_airfoils"
SAVED_STATES = DATA_DIR / "saved_states"
SAVED_AIRFOILS.mkdir(exist_ok=True)
SAVED_STATES.mkdir(exist_ok=True)

# Resource optimization (robust, non‑intrusive)
def optimize_resources(reserve_cores: int = 1, omp_threads: Optional[int] = None):
    """
    Try to set sane defaults for threading and CPU usage:
    - set OMP/BLAS env vars (if not already set)
    - set CPU affinity to use physical cores minus reserve_cores (if supported)
    - increase niceness on Unix to be less aggressive (requires no special permission)
    The function is conservative and will never raise.
    """
    try:
        # prefer explicit caller value, otherwise keep single-threaded BLAS by default
        threads = 1 if omp_threads is None else max(1, int(omp_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    except Exception:
        logger.debug("Failed to set BLAS/OMP environment variables")

    if psutil is None:
        logger.debug("psutil not available, skipping affinity/nice tuning")
        return

    try:
        proc = psutil.Process()
    except Exception as e:
        logger.debug(f"psutil.Process() failed: {e}")
        return

    try:
        phys = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        # Reserve at least `reserve_cores` cores for the system; never drop below 1
        usable = max(1, phys - max(0, int(reserve_cores)))
        # Use the first `usable` logical cores (safe mapping)
        logical = psutil.cpu_count(logical=True) or phys
        cpus = [i for i in range(min(usable, logical))]
        try:
            proc.cpu_affinity(cpus)
            logger.info(f"Set CPU affinity to cores: {cpus}")
        except Exception:
            logger.debug("Could not set cpu_affinity (unsupported or insufficient permissions)")

        # On Unix, increase niceness (make process less aggressive) — positive value lowers priority
        if os.name != "nt":
            try:
                current = proc.nice()
                target = max(current, 5)  # at least nice 5, don't decrease priority
                proc.nice(target)
                logger.info(f"Set process niceness to {target}")
            except Exception:
                logger.debug("Could not change niceness (insufficient permissions)")
    except Exception as e:
        logger.warning(f"Resource optimization failed: {e}")

# Try to optimize resources at import time but fail gracefully (no hard side effects)
try:
    optimize_resources()
except Exception:
    logger.debug("optimize_resources() failed at import time; continuing without crash")

# Determine a sensible thread pool size (follow concurrent.futures default heuristic,
# but cap to avoid huge thread counts on very large machines)
def _determine_worker_count(reserve_cores: int = 1) -> int:
    try:
        # Prefer psutil if available (gives physical/logical distinction)
        if psutil is not None:
            logical = psutil.cpu_count(logical=True)
            physical = psutil.cpu_count(logical=False)
        else:
            logical = os.cpu_count()
            physical = None
        # Choose logical cores when available, else physical, else os.cpu_count fallback
        cores = logical or physical or os.cpu_count() or 1
        cores = max(1, int(cores) - max(0, int(reserve_cores)))
    except Exception:
        cores = 1
    # concurrent.futures default: min(32, os.cpu_count() + 4)
    # Use that heuristic but ensure at least 2 workers for responsiveness
    return max(2, min(32, cores + 4))

CPU_COUNT = _determine_worker_count(reserve_cores=1)
thread_pool = ThreadPoolExecutor(max_workers=CPU_COUNT)

# Global state (with locks)
vlm_sessions: Dict[str, VLM] = {}
vlm_sessions_lock = Lock()

default_cache: Dict[str, Any] = {}
sessions_lock = Lock()  # for short duration disk writes / critical sections

# Helpers for parsing and validation (fast, safe, small overhead)
def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

# Load default data synchronously at startup (small files, avoid async complexity here)
def load_default_data_sync(filename: str, cache_key: str) -> dict:
    if cache_key in default_cache:
        return default_cache[cache_key]
    path = BASE_DIR / filename
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        default_cache[cache_key] = data
        return data
    except FileNotFoundError:
        logger.warning(f"{filename} no encontrado en {path}")
        default_cache[cache_key] = {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON inválido en {filename}: {e}", exc_info=True)
        default_cache[cache_key] = {}
    return {}

default_parameters = load_default_data_sync("data/default_parameters.txt", "parameters")
default_naca = load_default_data_sync("data/default_naca.txt", "naca")
default_plane = load_default_data_sync("data/default_plane.txt", "plane") or {
    "wing_sections": [],
    "horizontal_stabilizer": {},
    "vertical_stabilizer": {}
}

# Decorator to ensure session exists
def session_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get("session_id")
        if not session_id:
            logger.warning("Session ID missing, redirecting to index.")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function

@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid4())
        logger.info(f"Nueva sesión creada: {session['session_id']}")
    if "appearance_mode" not in session:
        session["appearance_mode"] = "System"
    if "effective_mode" not in session:
        session["effective_mode"] = "Light"

# Appearance management
@app.route("/set-appearance", methods=["POST"])
def set_appearance():
    try:
        data = request.get_json() or {}
        appearance_mode = data.get("appearance_mode", "System")
        effective_mode = data.get("effective_mode", "Light")
        if appearance_mode not in ("Light", "Dark", "System"):
            appearance_mode = "System"
        if effective_mode not in ("Light", "Dark"):
            effective_mode = "Light"

        session["appearance_mode"] = appearance_mode
        session["effective_mode"] = effective_mode

        template = "plotly_dark" if effective_mode == "Dark" else "plotly_white"
        pio.templates.default = template
        logger.info(f"Template de Plotly establecido a {template} ({appearance_mode}/{effective_mode})")
        return jsonify({"status": "success", "template": template})
    except Exception as e:
        logger.error("Error al establecer apariencia", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    template = "plotly_dark" if session.get("effective_mode", "Light") == "Dark" else "plotly_white"
    fig.update_layout(template=template)
    return fig

# VLM: helper to create, save, and store in global dict
def _vlm_create_and_save(plane: dict, u: float, rho: float, alpha: float, beta: float, n: int, m: int, session_id: str) -> dict:
    try:
        vlm = VLM(plane, u, rho, alpha, beta, n, m)
        # operación que depende de VLM
        vlm.save_and_load_plane_variables(filename=str(BASE_DIR / 'data' / 'cache_plane_variables.txt'), option='save_and_load')
        state_path = SAVED_STATES / f"{session_id}.pkl"
        vlm.save_state(str(state_path))
        with vlm_sessions_lock:
            vlm_sessions[session_id] = vlm
        return {"status": "success", "message": "Wing design computed."}
    except Exception as e:
        logger.exception("VLM computation error")
        return {"status": "error", "message": f"VLM computation failed: {e}"}

def compute_vlm_async(plane, u, rho, alpha, beta, n, m, session_id, timeout: Optional[float] = None):
    # Envía el trabajo a thread_pool y espera el resultado (manteniendo la API existente)
    future = thread_pool.submit(_vlm_create_and_save, plane, u, rho, alpha, beta, n, m, session_id)
    try:
        return future.result(timeout=timeout)
    except Exception as e:
        logger.exception("Error waiting VLM future")
        return {"status": "error", "message": f"VLM thread failed: {e}"}

# Routes
@app.route("/")
def index():
    return render_template("welcome.html")

@app.route("/airfoil", methods=["GET", "POST"])
def airfoil():
    return render_template("airfoil.html", default_naca=default_naca)

def _parse_wing_form(form) -> dict:
    # Extrae secciones del formulario de manera robusta
    chord_roots = form.getlist("chord_root[]")
    num_sections = len(chord_roots)
    sections = []
    for i in range(num_sections):
        try:
            chord_root = safe_float(form.getlist("chord_root[]")[i], 0.0)
            chord_tip = safe_float(form.getlist("chord_tip[]")[i], chord_root)
            span_fraction = safe_float(form.getlist("span_fraction[]")[i], 0.0)
            sweep = np.radians(safe_float(form.getlist("sweep[]")[i], 0.0))
            alpha = np.radians(safe_float(form.get("alpha", 0.0), 0.0))
            dihedral = np.radians(safe_float(form.getlist("dihedral[]")[i], 0.0))
            naca_root = form.getlist("naca_root[]")[i] if i < len(form.getlist("naca_root[]")) else ""
            naca_tip = form.getlist("naca_tip[]")[i] if i < len(form.getlist("naca_tip[]")) else ""

            section = {
                "chord_root": chord_root,
                "chord_tip": chord_tip,
                "span_fraction": span_fraction,
                "sweep": sweep,
                "alpha": alpha,
                "dihedral": dihedral,
                "NACA_root": naca_root,
                "NACA_tip": naca_tip
            }

            flap_toggled = form.getlist("flap_toggled[]")
            if i < len(flap_toggled) and flap_toggled[i] == "1":
                section.update({
                    "flap_start": safe_float(form.getlist("flap_start[]")[i], 0.0),
                    "flap_end": safe_float(form.getlist("flap_end[]")[i], 0.0),
                    "flap_hinge_chord": safe_float(form.getlist("flap_hinge_chord[]")[i], 0.0),
                    "deflection_angle": np.radians(safe_float(form.getlist("deflection_angle[]")[i], 0.0)),
                    "deflection_type": form.getlist("deflection_type[]")[i] if i < len(form.getlist("deflection_type[]")) else ""
                })
            sections.append(section)
        except Exception:
            logger.exception("Error parsing wing section")
            continue
    plane = {"wing_sections": sections}

    # HTP & VTP
    if form.getlist("horizontal_toggled[]") == ["1"]:
        plane["horizontal_stabilizer"] = {
            "x_translate": safe_float(form.get("x_translate", 0.0)),
            "z_translate": safe_float(form.get("z_translate", 0.0)),
            "NACA_root": form.get("NACA_root", ""),
            "NACA_tip": form.get("NACA_tip", ""),
            "chord_root": safe_float(form.get("chord_root", 0.0)),
            "chord_tip": safe_float(form.get("chord_tip", 0.0)),
            "span_fraction": safe_float(form.get("span_fraction", 0.0)),
            "sweep": np.radians(safe_float(form.get("sweep", 0.0))),
            "alpha": np.radians(safe_float(form.get("htp_alpha", 0.0))),
            "dihedral": np.radians(safe_float(form.get("htp_dihedral", 0.0))),
        }

    if form.getlist("vertical_toggled[]") == ["1"]:
        plane["vertical_stabilizer"] = {
            "x_translate": safe_float(form.get("x_translate_v", 0.0)),
            "z_translate": safe_float(form.get("z_translate_v", 0.0)),
            "NACA_root": form.get("NACA_root_v", ""),
            "NACA_tip": form.get("NACA_tip_v", ""),
            "chord_root": safe_float(form.get("chord_root_v", 0.0)),
            "chord_tip": safe_float(form.get("chord_tip_v", 0.0)),
            "span_fraction": safe_float(form.get("span_fraction_v", 0.0)),
            "sweep": np.radians(safe_float(form.get("sweep_v", 0.0))),
            "alpha": np.radians(safe_float(form.get("alpha_v", 0.0))),
            "dihedral": np.radians(90),
        }

    return plane

@app.route("/wing", methods=["GET", "POST"])
def plane_route():
    if request.method == "POST":
        try:
            form = request.form
            session_id = session.get("session_id")
            if not session_id:
                logger.error("Session ID missing in /wing POST")
                return jsonify({"status": "error", "message": "Session ID missing"}), 400

            plane = _parse_wing_form(form)
            u = safe_float(form.get("u", 50.0))
            rho = safe_float(form.get("rho", 1.225))
            alpha = np.radians(safe_float(form.get("alpha", 0.0)))
            beta = np.radians(safe_float(form.get("beta", 0.0)))
            n = safe_int(form.get("n", 10), 10)
            m = safe_int(form.get("m", 10), 10)

            # Check if the plane has at least one wing section
            if not plane.get("wing_sections"):
                return jsonify({"status": "error", "message": "At least one wing section is required."}), 400
            

            logger.info(f"Submitting VLM computation for session: {session_id}")

            # Execute VLM computation asynchronously but wait for result here
            result = compute_vlm_async(plane, u, rho, alpha, beta, n, m, session_id)
            logger.info(f"VLM computation result: {result.keys() if isinstance(result, dict) else result}, vlm_sessions keys: {list(vlm_sessions.keys())}")

            return jsonify(result), 200 if result.get("status") == "success" else 500

        except ValueError as e:
            logger.error("Invalid input in /wing", exc_info=True)
            return jsonify({"status": "error", "message": f"Invalid input: {e}"}), 400
        except Exception as e:
            logger.exception("Error in /wing")
            return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

    return render_template("wing.html", default_plane=default_plane, default_parameters=default_parameters)

@app.route("/angles", methods=["POST"])
@session_required
def angles():
    try:
        data = request.form
        session_id = session.get("session_id")

        angles_deg_str = data.get("angles_deg", "")
        angles_deg = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', angles_deg_str)]
        if not angles_deg:
            return jsonify({"status": "error", "message": "No angles provided"}), 400

        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            return jsonify({"status": "error", "message": "VLM not initialized. Please design a wing first."}), 400

        vlm.compute_coefficients_vs_alpha(angles_deg)

        if vlm.results is None or 'CL' not in vlm.results or 'CD' not in vlm.results:
            return jsonify({"status": "error", "message": "Failed to compute coefficients"}), 500

        return jsonify({"status": "success", "message": "Coefficients computed successfully"}), 200

    except Exception as e:
        logger.exception("Error in /angles")
        return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

@app.route("/results", methods=["GET", "POST"])
@session_required
def results():
    session_id = session.get("session_id")
    logger.info(f"Accessing /results for session: {session_id}")

    def compute(vlm: VLM):
        if vlm is None:
            raise ValueError("VLM object is not initialized")
        if vlm.wing_geometry is None:
            logger.info("Calculating geometry...")
            vlm.calculate_geometry()
        if vlm.panel_data is None:
            logger.info("Calculating discretization...")
            vlm.calculate_discretization()
        if not hasattr(vlm, 'lift'):
            logger.info("Calculating wing lift...")
            vlm.calculate_wing_lift()
        return vlm

    try:
        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            logger.error("VLM not initialized despite session check")
            return redirect(url_for("plane_route"))

        if vlm.results is None:
            future = thread_pool.submit(compute, vlm)
            vlm = future.result()

        wing_lift_total = np.sum(vlm.lift_sum.get('lift_wing', [])) if hasattr(vlm, 'lift_sum') else None
        hs_lift_total = np.sum(vlm.lift_sum.get('lift_hs', [])) if hasattr(vlm, 'lift_sum') else None
        vs_lift_total = np.sum(vlm.lift_sum.get('lift_vs', [])) if hasattr(vlm, 'lift_sum') else None

        cl = f"{vlm.CL:.4f}" if getattr(vlm, 'CL', None) is not None else "--"
        cd = f"{vlm.CD:.4f}" if getattr(vlm, 'CD', None) is not None else "--"
        total_lift = f"{vlm.lift:.2f}" if getattr(vlm, 'lift', None) is not None else "--"
        total_drag = f"{vlm.drag:.2f}" if getattr(vlm, 'drag', None) is not None else "--"
        wing_lift = f"{wing_lift_total:.2f}" if wing_lift_total is not None else "--"
        hs_lift = f"{hs_lift_total:.2f}" if hs_lift_total is not None else "--"
        vs_lift = f"{vs_lift_total:.2f}" if vs_lift_total is not None else "--"

        return render_template(
            "results.html",
            cl=cl, cd=cd, total_lift=total_lift, total_drag=total_drag,
            wing_lift=wing_lift, hs_lift=hs_lift, vs_lift=vs_lift,
            results=vlm.results
        )
    except Exception as e:
        logger.exception("Error in /results")
        return jsonify({"status": "error", "message": f"Exception: {e}"}), 500

@app.route('/plot/airfoil', methods=['POST'])
@session_required
def plot_airfoil():
    try:
        data = request.get_json() or {}
        naca = data.get('naca', '1310')
        chord = safe_float(data.get('chord', 1.0))
        alpha = safe_float(data.get('alpha', 0.0))

        fig = go.Figure()
        plot_naca_airfoil(fig, naca, chord, alpha)
        fig_result = apply_plotly_theme(fig)
        plot_json = json.dumps(fig_result, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"plot": plot_json})
    except Exception as e:
        logger.exception("Error en /plot/airfoil")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_airfoil', methods=['POST'])
@session_required
def save_airfoil():
    try:
        data = request.get_json() or {}
        naca = data.get('naca', '1310')
        chord = safe_float(data.get('chord', 1.0))
        alpha = safe_float(data.get('alpha', 0.0))

        airfoil_data = {"naca": naca, "chord": chord, "alpha": alpha}
        json_path = SAVED_AIRFOILS / f"{naca}.json"

        with sessions_lock:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(airfoil_data, f)

        csv_path = SAVED_AIRFOILS / f"{naca}.csv"
        naca_csv(naca, chord, alpha, filename=str(csv_path))

        return send_file(str(csv_path), mimetype='text/csv', as_attachment=True, download_name=f"naca_{naca}_airfoil.csv")
    except Exception as e:
        logger.exception("Error in /save_airfoil")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/load_airfoil', methods=['POST'])
@session_required
def load_airfoil():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        expected_fields = ['naca', 'chord', 'alpha']
        for field in expected_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400

        airfoil_data = {
            'naca': data['naca'],
            'chord': safe_float(data['chord']),
            'alpha': safe_float(data['alpha']),
            'u': safe_float(data.get('u', default_naca.get('u', 50.0))),
            'n': safe_int(data.get('n', default_naca.get('n', 100)))
        }

        if airfoil_data['u'] < 0.01:
            return jsonify({"status": "error", "message": "Free stream velocity must be at least 0.01"}), 400
        if not (-90 <= airfoil_data['alpha'] <= 90):
            return jsonify({"status": "error", "message": "Angle of attack must be between -90 and 90"}), 400
        if not (-30 <= airfoil_data['chord'] <= 30):
            return jsonify({"status": "error", "message": "Chord length must be between -30 and 30"}), 400

        return jsonify({"status": "success", "airfoil": airfoil_data}), 200
    except Exception as e:
        logger.exception("Error in /load_airfoil")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/load_vlm_state', methods=['POST'])
@session_required
def load_vlm_state():
    try:
        session_id = session.get("session_id")
        state_file = SAVED_STATES / f"{session_id}.pkl"
        if not state_file.exists():
            return jsonify({"status": "error", "message": "No saved state found"}), 404

        with vlm_sessions_lock:
            if session_id not in vlm_sessions:
                return jsonify({"status": "error", "message": "No active VLM session"}), 400
            vlm_sessions[session_id].load_state(str(state_file))
        return jsonify({"status": "success", "message": "VLM state loaded successfully"}), 200
    except Exception as e:
        logger.exception("Error in /load_vlm_state")
        return jsonify({"status": "error", "message": f"Failed to load state: {e}"}), 500

# Optimized plot dispatch endpoint
@app.route('/plot/<plot_type>', methods=['GET', 'POST'])
@session_required
def plot_data(plot_type):
    try:
        session_id = session.get("session_id")
        data = request.get_json(silent=True) or {}

        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            return jsonify({"status": "error", "message": "VLM no inicializado"}), 400

        # Ensure geometry/discretization
        def ensure_vlm_ready(vlm_obj):
            if vlm_obj.wing_geometry is None:
                vlm_obj.calculate_geometry()
            if vlm_obj.panel_data is None:
                vlm_obj.calculate_discretization()
            return vlm_obj

        vlm = thread_pool.submit(ensure_vlm_ready, vlm).result()
        fig = go.Figure()

        # Plot dispatch table for clarity and speed
        plot_map = {
            'geometry': lambda: (
                plot_wing_geometry(fig, vlm.wing_geometry, legend="Wing Geometry"),
                getattr(vlm, 'hs_geometry', None) and plot_wing_geometry(fig, vlm.hs_geometry, legend="Horizontal Stabilizer"),
                getattr(vlm, 'vs_geometry', None) and plot_wing_geometry(fig, vlm.vs_geometry, legend="Vertical Stabilizer")
            )[0],
            'geometry_2d': lambda: (
                plot_wing_geometry_2d(fig, vlm.wing_geometry, legend="Wing Geometry"),
                getattr(vlm, 'hs_geometry', None) and plot_wing_geometry_2d(fig, vlm.hs_geometry, legend="Horizontal Stabilizer")
            )[0],
            'discretization2D': lambda: plot_wing_discretization_2d(fig, vlm.panel_data),
            'discretization3D': lambda: plot_wing_discretization_3d(fig, vlm.panel_data),
            'Lift': lambda: plot_distribution_all(fig, vlm, quantity='LIFT'),
            'Drag': lambda: plot_distribution_all(fig, vlm, quantity='DRAG'),
            'CL': lambda: plot_distribution_all(fig, vlm, quantity='CL'),
            'CD': lambda: plot_distribution_all(fig, vlm, quantity='CD'),
            'wi': lambda: plot_wing_heatmap(fig, vlm.panel_data, vlm.w_i, title='Induced Velocity Distribution', legend='Induced Velocity [m/s]'),
            'gammas': lambda: plot_wing_heatmap(fig, vlm.panel_data, vlm.gammas, title='Gamma Distribution', legend='Gamma [m²/s]'),
            'curvature': lambda: plot_wing_heatmap(fig, vlm.panel_data, vlm.dz_c, title='Curvature Distribution', legend='Curvature [1/m]'),
            'wi_2d': lambda: plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.w_i, title='Induced Velocity Distribution', legend='Induced Velocity [m/s]', u_vec=getattr(vlm, 'u_', None)),
            'gammas_2d': lambda: plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.gammas, title='Gamma Distribution', legend='Gamma [m²/s]', u_vec=getattr(vlm, 'u_', None)),
            'curvature_2d': lambda: plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.dz_c, title='Curvature Distribution', legend='Curvature [1/m]', u_vec=getattr(vlm, 'u_', None)),
            'CL_CD': lambda: plot_CL_CD(fig, vlm, title='CL vs CD of the Wing'),
            'CL-alpha': lambda: plot_coefficient_vs_alpha(fig, vlm, coefficient='CL') if vlm.results and 'CL' in vlm.results else None,
            'CD-alpha': lambda: plot_coefficient_vs_alpha(fig, vlm, coefficient='CD') if vlm.results and 'CD' in vlm.results else None,
            'CL-CD-alpha': lambda: plot_CLCD_vs_alpha(fig, vlm, title=None) if vlm.results and 'CL' in vlm.results and 'CD' in vlm.results else None,
            'LiftSection': lambda: plot_distribution(vlm, safe_int(data.get('n_section', None)), quantity='lift'),
            'DragSection': lambda: plot_distribution(vlm, safe_int(data.get('n_section', None)), quantity='drag')
        }

        plot_func = plot_map.get(plot_type)
        if not plot_func:
            return jsonify({"status": "error", "message": "Tipo de gráfico inválido"}), 400

        fig_result = plot_func()
        if fig_result is None:
            # Specific error for missing coefficients
            if plot_type in ('CL-alpha', 'CD-alpha', 'CL-CD-alpha'):
                return jsonify({"status": "error", "message": "Please calculate coefficients first"}), 400
            return jsonify({"status": "error", "message": "Plot data unavailable"}), 400

        fig = apply_plotly_theme(fig_result)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"plot": plot_json})
    except Exception as e:
        logger.exception(f"Error en /plot/{plot_type}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Config management API (se mantienen las funciones del manager)
@app.route('/api/configs', methods=['GET'])
@session_required
def list_configs():
    try:
        configs = config_manager.list_configs()
        return jsonify({"status": "success", "configs": configs, "total_count": len(configs)})
    except Exception as e:
        logger.exception("Error listing configs")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/save_plane_config', methods=['POST'])
@session_required
def save_plane_config():
    try:
        session_id = session.get('session_id')
        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            return jsonify({"status": "error", "message": "No hay una sesión VLM activa"}), 400

        data = request.get_json() or {}
        filename = data.get('filename', f'plane_{session_id}')

        config = getattr(vlm, 'plane', {}).copy()
        if data.get('include_flight_params', False):
            config['flight_parameters'] = {
                'u': getattr(vlm, 'u', None),
                'rho': getattr(vlm, 'rho', None),
                'alpha': getattr(vlm, 'alpha', None),
                'beta': getattr(vlm, 'beta', None),
                'n': getattr(vm, 'n', None) if False else getattr(vlm, 'n', None),  # fallback safe access
                'm': getattr(vlm, 'm', None)
            }
        config['name'] = data.get('name', f'Configuración {session_id}')
        config['description'] = data.get('description', '')

        result = config_manager.save_config(config, filename)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.exception("Error in save_plane_config")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/load_plane_config', methods=['POST'])
@session_required
def load_plane_config():
    try:
        data = request.get_json() or {}
        filename = data.get('filename')
        if not filename:
            return jsonify({"status": "error", "message": "Nombre de archivo requerido"}), 400

        result = config_manager.load_config(filename)
        if result.get('status') != 'success':
            return jsonify(result), 400

        config = result.get('config', {})
        session_id = session.get('session_id')
        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
            if not vlm:
                return jsonify({"status": "error", "message": "No hay una sesión VLM activa"}), 400
            vlm.plane = config

            # Cargar parámetros de vuelo si están incluidos
            if 'flightparameters' in config and data.get('loadflightparams', False):
                fp = config['flightparameters']
                vlm.u = fp.get('u', vlm.u)
                vlm.rho = fp.get('rho', vlm.rho)
                vlm.alpha = fp.get('alpha', vlm.alpha)
                vlm.beta = fp.get('beta', vlm.beta)
                vlm.n = fp.get('n', vlm.n)
                vlm.m = fp.get('m', vlm.m)

        return jsonify(status="success", message=result.get('message'), config=config, checksum_valid=result.get('checksum_valid'), version=result.get('version'))
    except Exception as e:
        logger.exception("Error in load_plane_config")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/delete_config', methods=['DELETE'])
@session_required
def delete_config():
    try:
        data = request.get_json() or {}
        filename = data.get('filename')
        if not filename:
            return jsonify({"status": "error", "message": "Nombre de archivo requerido"}), 400
        result = config_manager.delete_config(filename)
        if result.get('status') == 'success':
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.exception("Error in delete_config")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/validate_config', methods=['POST'])
def validate_config():
    try:
        config = request.get_json()
        if not config:
            return jsonify({"status": "error", "message": "No se proporcionó configuración"}), 400
        config_type = "wing" if "wingsections" in config else "parameters"
        errors = config_manager.validate_config(config, config_type)
        if errors:
            return jsonify({"status": "error", "message": "Configuración inválida", "validation_errors": errors, "is_valid": False})
        return jsonify({"status": "success", "message": "Configuración válida", "is_valid": True, "checksum": config_manager.calculate_checksum(config)})
    except Exception as e:
        logger.exception("Error in validate_config")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/airfoil-configs', methods=['GET'])
def list_airfoil_configs():
    """Lista todas las configuraciones de airfoil guardadas"""
    try:
        # Busca archivos que empiecen con 'airfoil_' o terminen con '_airfoil.json'
        configs = config_manager.list_configs("*airfoil*.json")
        
        return jsonify({
            "status": "success",
            "configs": configs,
            "count": len(configs)
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error listing airfoil configs: {str(e)}"
        }), 500

@app.route('/api/saveairfoilconfig', methods=['POST'])
def save_airfoil_config():
    """Guarda una configuración de airfoil"""
    try:
        data = request.get_json()
        filename = data.get('filename', '').strip()
        airfoil_data = data.get('data', {})
        
        if not filename:
            return jsonify({
                "status": "error",
                "message": "Filename is required"
            }), 400
            
        # Validación básica de datos de airfoil
        required_fields = ['naca', 'u', 'alpha', 'chord', 'n']
        missing_fields = [field for field in required_fields if field not in airfoil_data]
        
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Prefija el nombre del archivo para identificarlo como configuración de airfoil
        if not filename.startswith('airfoil_'):
            filename = f"airfoil_{filename}"
        
        # Estructura de datos para airfoil
        config_data = {
            "type": "airfoil",
            "naca": airfoil_data['naca'],
            "u": float(airfoil_data['u']),
            "alpha": float(airfoil_data['alpha']),
            "chord": float(airfoil_data['chord']),
            "n": int(airfoil_data['n']),
            "description": airfoil_data.get('description', ''),
            "parameters": {
                "velocity": float(airfoil_data['u']),
                "angle_of_attack": float(airfoil_data['alpha']),
                "chord_length": float(airfoil_data['chord']),
                "points": int(airfoil_data['n'])
            }
        }
        
        result = config_manager.save_config(config_data, filename)
        
        if result['status'] == 'success':
            return jsonify({
                "status": "success",
                "message": f"Airfoil configuration '{filename}' saved successfully!",
                "filename": result['filename'],
                "backup_created": result.get('backup_created', False)
            }), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error saving airfoil config: {str(e)}"
        }), 500

@app.route('/api/loadairfoilconfig', methods=['POST'])
def load_airfoil_config():
    """Carga una configuración de airfoil"""
    try:
        data = request.get_json()
        filename = data.get('filename', '').strip()
        
        if not filename:
            return jsonify({
                "status": "error",
                "message": "Filename is required"
            }), 400
        
        result = config_manager.load_config(filename)
        
        if result['status'] == 'success':
            config = result['config']
            
            # Valida que sea una configuración de airfoil
            if config.get('type') != 'airfoil':
                return jsonify({
                    "status": "error",
                    "message": "Selected file is not an airfoil configuration"
                }), 400
            
            # Extrae los datos del airfoil para el frontend
            airfoil_data = {
                "naca": config.get('naca', '0012'),
                "u": config.get('u', 10.0),
                "alpha": config.get('alpha', 5.0),
                "chord": config.get('chord', 1.0),
                "n": config.get('n', 100),
                "description": config.get('description', '')
            }
            
            return jsonify({
                "status": "success",
                "message": f"Airfoil configuration loaded successfully!",
                "airfoil": airfoil_data,
                "filename": filename,
                "version": result.get('version', 'unknown'),
                "checksum_valid": result.get('checksum_valid', True)
            }), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error loading airfoil config: {str(e)}"
        }), 500

@app.route('/api/deleteairfoilconfig', methods=['POST'])
def delete_airfoil_config():
    """Elimina una configuración de airfoil"""
    try:
        data = request.get_json()
        filename = data.get('filename', '').strip()
        
        if not filename:
            return jsonify({
                "status": "error",
                "message": "Filename is required"
            }), 400
        
        result = config_manager.delete_config(filename)
        
        if result['status'] == 'success':
            return jsonify({
                "status": "success",
                "message": f"Airfoil configuration '{filename}' deleted successfully!",
                "backup_created": result.get('backup_created', False)
            }), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error deleting airfoil config: {str(e)}"
        }), 500

@app.route('/api/validateairfoilconfig', methods=['POST'])
def validate_airfoil_config():
    """Valida una configuración de airfoil"""
    try:
        data = request.get_json()
        airfoil_data = data.get('data', {})
        
        errors = []
        
        # Validaciones específicas para airfoil
        validations = {
            'naca': {
                'required': True,
                'type': str,
                'pattern': r'^[0-9]{4}$',
                'message': 'NACA profile must be 4 digits (e.g., 0012)'
            },
            'u': {
                'required': True,
                'type': (int, float),
                'range': (0.1, 1000),
                'message': 'Velocity must be between 0.1 and 1000 m/s'
            },
            'alpha': {
                'required': True,
                'type': (int, float),
                'range': (-90, 90),
                'message': 'Angle of attack must be between -90 and 90 degrees'
            },
            'chord': {
                'required': True,
                'type': (int, float),
                'range': (0.01, 100),
                'message': 'Chord length must be between 0.01 and 100 meters'
            },
            'n': {
                'required': True,
                'type': int,
                'range': (10, 5000),
                'message': 'Number of points must be between 10 and 5000'
            }
        }
        
        for field, rules in validations.items():
            value = airfoil_data.get(field)
            
            if rules['required'] and (value is None or value == ''):
                errors.append(f"{field}: Required field missing")
                continue
            
            if value is not None:
                # Type check
                if not isinstance(value, rules['type']):
                    try:
                        if rules['type'] == int:
                            value = int(float(value))
                        elif rules['type'] == (int, float):
                            value = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{field}: {rules['message']}")
                        continue
                
                # Range check
                if 'range' in rules:
                    min_val, max_val = rules['range']
                    if not (min_val <= value <= max_val):
                        errors.append(f"{field}: {rules['message']}")
                
                # Pattern check (for NACA)
                if 'pattern' in rules:
                    import re
                    if not re.match(rules['pattern'], str(value)):
                        errors.append(f"{field}: {rules['message']}")
        
        return jsonify({
            "status": "success",
            "valid": len(errors) == 0,
            "errors": errors,
            "message": "Validation passed" if not errors else f"Found {len(errors)} validation errors"
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error validating airfoil config: {str(e)}"
        }), 500

if __name__ == "__main__":
    optimize_resources()
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
        #ui.run()
    finally:
        thread_pool.shutdown(wait=True)                                                                                                                                                                 