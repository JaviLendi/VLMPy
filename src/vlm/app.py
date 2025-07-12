import os
import json
import pickle
import logging
from uuid import uuid4 
import multiprocessing
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import asyncio
import aiofiles

import numpy as np
import psutil
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_compress import Compress
from webui import WebUI
import plotly.graph_objs as go
import plotly.utils
import plotly.io as pio
import re

# Configure Plotly default format
pio.kaleido.scope.default_format = "svg"

# Ensure local module import
import sys
sys.path.append('../')
from lib.vlm import VLM, plot_distribution_all, plot_wing_heatmap, plot_wing_heatmap_2d, plot_coefficient_vs_alpha, plot_distribution, plot_CL_CD, plot_CLCD_vs_alpha
from lib.naca import plot_naca_airfoil, naca_csv
from lib.geometry import plot_wing_geometry, plot_wing_geometry_2d, plot_wing_discretization_2d, plot_wing_discretization_3d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
Compress(app)  # Enable compression for Flask app
app.secret_key = os.urandom(24)  # Secret key for session management
ui = WebUI(app, debug=True)  # Initialize WebUI

# Optimize CPU & memory usage
def optimize_resources():
    try:
        p = psutil.Process()
        p.nice(-5 if os.name != 'nt' else psutil.HIGH_PRIORITY_CLASS)
        p.cpu_affinity(range(multiprocessing.cpu_count()))
    except Exception as e:
        logger.warning(f"Resource optimization failed: {e}")

optimize_resources()

# Thread pool & session store
thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
sessions = {}
sessions_lock = Lock()

# Cache for default data
default_cache = {}

# Load default data asynchronously with caching
async def load_default_data(file_name, cache_key):
    if cache_key in default_cache:
        return default_cache[cache_key]
    try:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        async with aiofiles.open(file_path, 'r') as f:
            data = json.loads(await f.read())
        default_cache[cache_key] = data
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load {file_name}: {e}", exc_info=True)
        return {}

# Synchronous wrappers for Flask compatibility
def load_default_parameters():
    return asyncio.run(load_default_data("default_parameters.txt", "parameters"))

def load_default_naca():
    return asyncio.run(load_default_data("default_naca.txt", "naca"))

def load_default_plane():
    return asyncio.run(load_default_data("default_plane.txt", "plane")) or {
        "wing_sections": [],
        "horizontal_stabilizer": {},
        "vertical_stabilizer": {}
    }

# Load defaults at startup
default_parameters = load_default_parameters()
default_naca = load_default_naca()
default_plane = load_default_plane()

vlm_sessions = {}
vlm_sessions_lock = Lock()

# Thread-safe VLM computation
def compute_vlm(plane, u, rho, alpha, beta, n, m, session_id):
    try:
        vlm = VLM(plane, u, rho, alpha, beta, n, m)
        vlm.save_and_load_plane_variables(filename='cache_plane_variables.txt', option='save_and_load')
        with vlm_sessions_lock:
            vlm_sessions[session_id] = vlm
            saved_states = os.path.join(os.path.dirname(__file__), 'saved_states')
            os.makedirs(saved_states, exist_ok=True)
            state_path = os.path.join(saved_states, f"{session_id}.pkl")
            vlm.save_state(state_path)
        return {"status": "success", "message": "Wing design computed."}
    except Exception as e:
        return {"status": "error", "message": f"VLM computation failed: {e}"}

@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid4())
        logger.info(f"Nueva sesión creada: {session['session_id']}")
    if "appearance_mode" not in session:
        session["appearance_mode"] = "System"  # Modo por defecto
        session["effective_mode"] = "Light"   # Tema efectivo inicial

@app.route("/set-appearance", methods=["POST"])
def set_appearance():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No se proporcionaron datos JSON"}), 400
        
        # Obtener modos del cliente
        appearance_mode = data.get("appearance_mode", "System")
        effective_mode = data.get("effective_mode", "Light")
        
        # Validar modos
        valid_appearance_modes = ["Light", "Dark", "System"]
        valid_effective_modes = ["Light", "Dark"]
        if appearance_mode not in valid_appearance_modes:
            appearance_mode = "System"
        if effective_mode not in valid_effective_modes:
            effective_mode = "Light"
        
        # Almacenar en la sesión
        session["appearance_mode"] = appearance_mode
        session["effective_mode"] = effective_mode
        
        # Configurar el template de Plotly
        template = "plotly_dark" if effective_mode == "Dark" else "plotly_white"
        pio.templates.default = template
        logger.info(f"Template de Plotly establecido a {template} para modo {appearance_mode}/{effective_mode}")
        
        return jsonify({
            "status": "success",
            "message": "Modo de apariencia actualizado",
            "template": template
        }), 200
    except Exception as e:
        logger.error(f"Error al establecer modo de apariencia: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# Función para aplicar el tema a los gráficos
def apply_plotly_theme(fig):
    template = "plotly_dark" if session.get("effective_mode", "Light") == "Dark" else "plotly_white"
    fig.update_layout(template=template)
    return fig

def session_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get("session_id")
        if not session_id:
            logger.warning("Session ID missing, redirecting to index.")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    return render_template("welcome.html")

@app.route("/airfoil", methods=["GET", "POST"])
def airfoil():
    return render_template("airfoil.html", default_naca=default_naca)

@app.route("/wing", methods=["GET", "POST"])
def plane():
    if request.method == "POST":
        try:
            data = request.form
            session_id = session.get("session_id")
            if not session_id:
                logger.error("Session ID missing in /wing POST", exc_info=True)
                return jsonify({"status": "error", "message": "Session ID missing"}), 400

            sections = []
            num_sections = len(data.getlist("chord_root[]"))
            for i in range(num_sections):
                section = {
                    "chord_root": float(data.getlist("chord_root[]")[i]),
                    "chord_tip": float(data.getlist("chord_tip[]")[i]),
                    "span_fraction": float(data.getlist("span_fraction[]")[i]),
                    "sweep": np.radians(float(data.getlist("sweep[]")[i])),
                    "alpha": np.radians(float(data.get("alpha", 0.0))),
                    "dihedral": np.radians(float(data.getlist("dihedral[]")[i])),
                    "NACA_root": data.getlist("naca_root[]")[i],
                    "NACA_tip": data.getlist("naca_tip[]")[i]
                }
                if data.getlist("flap_toggled[]")[i] == "1":
                    section.update({
                        "flap_start": float(data.getlist("flap_start[]")[i]),
                        "flap_end": float(data.getlist("flap_end[]")[i]),
                        "flap_hinge_chord": float(data.getlist("flap_hinge_chord[]")[i]),
                        "deflection_angle": np.radians(float(data.getlist("deflection_angle[]")[i])),
                        "deflection_type": data.getlist("deflection_type[]")[i]  # Assuming this is a string
                    })
                sections.append(section)

            plane = {"wing_sections": sections}
            if data.getlist("horizontal_toggled[]") == ["1"]:
                plane["horizontal_stabilizer"] = {
                    "x_translate": float(data.get("x_translate", 0.0)),
                    "z_translate": float(data.get("z_translate", 0.0)),  
                    "NACA_root": data.get("NACA_root", ""),
                    "NACA_tip": data.get("NACA_tip", ""),
                    "chord_root": float(data.get("chord_root", 0.0)),
                    "chord_tip": float(data.get("chord_tip", 0.0)),
                    "span_fraction": float(data.get("span_fraction", 0.0)),
                    "sweep": np.radians(float(data.get("sweep", 0.0))),
                    "alpha": np.radians(float(data.get("htp_alpha", 0.0))),
                    "dihedral": np.radians(float(data.get("htp_dihedral", 0.0))),
                }
            if data.getlist("vertical_toggled[]") == ["1"]:
                plane["vertical_stabilizer"] = {
                    "x_translate": float(data.get("x_translate_v", 0.0)),
                    "z_translate": float(data.get("z_translate_v", 0.0)),
                    "NACA_root": data.get("NACA_root_v", ""),
                    "NACA_tip": data.get("NACA_tip_v", ""),
                    "chord_root": float(data.get("chord_root_v", 0.0)),
                    "chord_tip": float(data.get("chord_tip_v", 0.0)),
                    "span_fraction": float(data.get("span_fraction_v", 0.0)),
                    "sweep": np.radians(float(data.get("sweep_v", 0.0))),
                    "alpha": np.radians(float(data.get("alpha_v", 0.0))),
                    "dihedral": np.radians(90),
                }

            u = float(data.get("u", 50.0))
            rho = float(data.get("rho", 1.225))
            alpha = np.radians(float(data.get("alpha", 0.0)))
            beta = np.radians(float(data.get("beta", 0.0)))
            n = int(data.get("n", 10))
            m = int(data.get("m", 10))

            logger.info(f"Submitting VLM computation for session: {session_id}")

            result = compute_vlm(plane, u, rho, alpha, beta, n, m, session_id)
            logger.info(f"VLM computation result: {result}, vlm_sessions keys: {list(vlm_sessions.keys())}")
            
            return jsonify(result), 200 if result["status"] == "success" else 500
        except ValueError as e:
            logger.error(f"Invalid input in /wing: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Invalid input: {e}"}), 400
        except Exception as e:
            logger.error(f"Error in /wing: {e}", exc_info=True)
            return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

    return render_template("wing.html", default_plane=default_plane, default_parameters=default_parameters)

@app.route("/angles", methods=["POST"])
@session_required
def angles():
    try:
        data = request.form
        session_id = session.get("session_id")
        if not session_id:
            logger.error("Session ID missing in /angles POST", exc_info=True)
            return jsonify({"status": "error", "message": "Session ID missing"}), 400

        # Obtener los ángulos de ataque como una lista desde el campo angles_deg
        angles_deg_str = data.get("angles_deg", "")
        # Use regex to correctly split and parse negative numbers
        angles_deg = [float(match) for match in re.findall(r'-?\d+(?:\.\d+)?', angles_deg_str)]

        if not angles_deg:
            return jsonify({"status": "error", "message": "No angles provided"}), 400

        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            logger.error("VLM not initialized", exc_info=True)
            return jsonify({"status": "error", "message": "VLM not initialized. Please design a wing first."}), 400

        # Calcular los coeficientes para los ángulos dados
        vlm.compute_coefficients_vs_alpha(angles_deg)

        # Confirmar que los resultados se han calculado y almacenado
        if vlm.results is None or 'CL' not in vlm.results or 'CD' not in vlm.results:
            return jsonify({"status": "error", "message": "Failed to compute coefficients"}), 500

        return jsonify({"status": "success", "message": "Coefficients computed successfully"}), 200

    except ValueError as e:
        logger.error(f"Invalid input in /angles: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Invalid input: {e}"}), 400
    except Exception as e:
        logger.error(f"Error in /angles: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Server error: {e}"}), 500

@app.route("/results", methods=["GET", "POST"])
@session_required
def results():
    session_id = session.get("session_id")
    def compute(vlm):
        logger.info("Starting compute function")
        if not vlm:
            logger.error("VLM object is None", exc_info=True)
            raise ValueError("VLM object is not initialized")
        
        if vlm.wing_geometry is None:
            logger.info("Calling calculate_geometry")
            vlm.calculate_geometry()
            logger.info("calculate_geometry completed")
        else:
            logger.info("wing_geometry already computed")
        
        if vlm.panel_data is None:
            logger.info("Calling calculate_discretization")
            vlm.calculate_discretization()
            logger.info("calculate_discretization completed")
        else:
            logger.info("panel_data already computed")
        
        logger.info("Calling calculate_wing_lift")
        try:
            vlm.calculate_wing_lift()
            logger.info("calculate_wing_lift completed")
        except Exception as e:
            logger.error(f"Error in calculate_wing_lift: {str(e)}", exc_info=True)
            raise
        return vlm
    
    try:
        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            logger.error("VLM not initialized despite session check", exc_info=True)
            return redirect(url_for("plane"))

        future = thread_pool.submit(compute, vlm)
        vlm = future.result()
        
        # Compute total lifts for each component
        wing_lift_total = np.sum(vlm.lift_sum['lift_wing']) if 'lift_wing' in vlm.lift_sum else None
        hs_lift_total = np.sum(vlm.lift_sum['lift_hs']) if 'lift_hs' in vlm.lift_sum else None
        vs_lift_total = np.sum(vlm.lift_sum['lift_vs']) if 'lift_vs' in vlm.lift_sum else None

        # Format values with appropriate precision
        cl = f"{vlm.CL:.4f}" if vlm.CL is not None else "--"
        cd = f"{vlm.CD:.4f}" if vlm.CD is not None else "--"
        total_lift = f"{vlm.lift:.2f}" if vlm.lift is not None else "--"
        total_drag = f"{vlm.drag:.2f}" if vlm.drag is not None else "--"
        wing_lift = f"{wing_lift_total:.2f}" if wing_lift_total is not None else "--"
        hs_lift = f"{hs_lift_total:.2f}" if hs_lift_total is not None else "--"
        vs_lift = f"{vs_lift_total:.2f}" if vs_lift_total is not None else "--"

        logger.info("Compute completed successfully")
        # Pass formatted values to the template
        return render_template(
            "results.html",
            cl=cl,
            cd=cd,
            total_lift=total_lift,
            total_drag=total_drag,
            wing_lift=wing_lift,
            hs_lift=hs_lift,
            vs_lift=vs_lift,
            results=results,
        )
    
    except Exception as e:
        logger.error(f"Error in /results: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": f"Exception: {str(e)}"}), 500

@app.route('/plot/airfoil', methods=['POST'])
@session_required
def plot_airfoil():
    try:
        data = request.get_json()
        naca = data.get('naca', '1310')
        cuerda = float(data.get('chord', 1.0))
        alpha = float(data.get('alpha', 0))

        fig = go.Figure()
        plot_naca_airfoil(fig, naca, cuerda, alpha)
        fig = apply_plotly_theme(fig)  # Aplicar tema
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"plot": plot_json})
    
    except Exception as e:
        logger.error(f"Error en /plot/airfoil: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/save_airfoil', methods=['POST'])
@session_required
def save_airfoil():
    try:
        data = request.get_json()
        naca = data.get('naca', '1310')
        chord = float(data.get('chord', 1.0))  # Use 'chord' consistently
        alpha = float(data.get('alpha', 0))

        # Save the airfoil data to a JSON file
        airfoil_data = {
            "naca": naca,
            "chord": chord,  # Consistent naming
            "alpha": alpha
        }
        saved_dir = os.path.join(os.path.dirname(__file__), 'saved_airfoils')
        os.makedirs(saved_dir, exist_ok=True)
        json_path = os.path.join(saved_dir, f"{naca}.json")
        
        # Thread-safe file writing
        with sessions_lock:  # Use existing lock for thread safety
            with open(json_path, 'w') as f:
                json.dump(airfoil_data, f)

        # Generate CSV file
        csv_path = os.path.join(saved_dir, f"{naca}.csv")
        naca_csv(naca, chord, alpha, filename=csv_path)

        # Return the CSV file for download
        return send_file(
            csv_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"naca_{naca}_airfoil.csv"
        )

    except Exception as e:
        logger.error(f"Error in /save_airfoil: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/load_airfoil', methods=['POST'])
@session_required
def load_airfoil():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Validate expected fields
        expected_fields = ['naca', 'chord', 'alpha']
        airfoil_data = {}
        for field in expected_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
            airfoil_data[field] = data[field]
        
        # Optional fields with defaults
        airfoil_data['u'] = data.get('u', default_naca.get('u', 50.0))
        airfoil_data['n'] = data.get('n', default_naca.get('n', 100))
        
        # Basic validation
        try:
            airfoil_data['chord'] = float(airfoil_data['chord'])
            airfoil_data['alpha'] = float(airfoil_data['alpha'])
            airfoil_data['u'] = float(airfoil_data['u'])
            airfoil_data['n'] = int(airfoil_data['n'])
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Invalid numeric values in file"}), 400
        
        # Additional checks
        if airfoil_data['u'] < 0.01:
            return jsonify({"status": "error", "message": "Free stream velocity must be at least 0.01"}), 400
        if airfoil_data['alpha'] < -90 or airfoil_data['alpha'] > 90:
            return jsonify({"status": "error", "message": "Angle of attack must be between -90 and 90"}), 400
        if airfoil_data['chord'] < -30 or airfoil_data['chord'] > 30:
            return jsonify({"status": "error", "message": "Chord length must be between -30 and 30"}), 400
        
        return jsonify({"status": "success", "airfoil": airfoil_data}), 200

    except Exception as e:
        logger.error(f"Error in /load_airfoil: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_plane', methods=['POST'])
@session_required
def save_plane():
    session_id = session.get("session_id")
    vlm = vlm_sessions[session_id]
    
    saved_dir = os.path.join(os.path.dirname(__file__), 'saved_planes')
    os.makedirs(saved_dir, exist_ok=True)
    filename = f"plane_{session_id}.json"
    json_path = os.path.join(saved_dir, filename)
    
    vlm.save_plane(json_path)
    return send_file(
        json_path,
        mimetype='application/json',
        as_attachment=True,
        download_name=filename
    )


@app.route('/load_plane', methods=['POST'])
@session_required
def load_plane():
    session_id = session.get("session_id")
    vlm = vlm_sessions.get(session_id)
    if not vlm:
        return jsonify({"status":"error","message":"Design a wing first"}), 400

    data = request.get_json()

    # Case 1: browser just posted the plane dict
    if data and isinstance(data, dict) and data.get('wing_sections') is not None:
        vlm.plane = data
        return jsonify({"status":"success", "plane": vlm.plane}), 200

    # Case 2: browser asked us to load a previously saved JSON by filename
    filename = data.get('filename')
    if not filename:
        return jsonify({"status":"error","message":"No filename provided"}), 400
    file_path = os.path.join(os.path.dirname(__file__), 'saved_planes', filename)
    if not os.path.exists(file_path):
        return jsonify({"status":"error","message":"File not found"}), 404

    vlm.load_plane(file_path)
    return jsonify({"status":"success", "plane": vlm.plane}), 200

@app.route('/load_vlm_state', methods=['POST'])
@session_required
def load_vlm_state():
    session_id = session.get("session_id")
    vlm = vlm_sessions.get(session_id)
    if not vlm:
        return jsonify({"status":"error","message":"No session"}), 400

    # Ignore payload; pick up the state file for the current session
    session_id = session.get("session_id")
    if session_id not in vlm_sessions:
        return jsonify({"status":"error","message":"No active VLM session"}), 400
    state_dir = os.path.join(os.path.dirname(__file__), 'saved_states')
    filename = f"{session_id}.pkl"
    path = os.path.join(state_dir, filename)
    if not os.path.exists(path):
        return jsonify({"status":"error","message":"No saved state found"}), 404

    try:
        vlm_sessions[session_id].load_state(path)
        return jsonify({"status":"success","message":"VLM state loaded successfully"}), 200
    except Exception as e:
        return jsonify({"status":"error","message":f"Failed to load state: {e}"}), 500

@app.route('/plot/<plot_type>', methods=['GET','POST'])
@session_required
def plot_data(plot_type):
    session_id = session.get("session_id")

    # Get the JSON data from the request
    data = request.get_json(silent=True) or {}
    
    def compute(vlm):
        if vlm.wing_geometry is None:
            vlm.calculate_geometry()
        if vlm.panel_data is None:
            vlm.calculate_discretization()
        if plot_type in ['Lift', 'Drag'] and vlm.lift_wing is None:
            vlm.calculate_wing_lift()
        return vlm

    try:
        with vlm_sessions_lock:
            vlm = vlm_sessions.get(session_id)
        if not vlm:
            return jsonify({"status": "error", "message": "VLM no inicializado"}), 400

        future = thread_pool.submit(compute, vlm)
        vlm = future.result()

        fig = go.Figure()
        if plot_type == 'geometry':
            fig = plot_wing_geometry(fig, vlm.wing_geometry, legend="Wing Geometry")
            if vlm.hs_geometry:
                fig = plot_wing_geometry(fig, vlm.hs_geometry, legend="Horizontal Stabilizer")
            if vlm.vs_geometry:
                fig = plot_wing_geometry(fig, vlm.vs_geometry, legend="Vertical Stabilizer")
        elif plot_type == 'geometry_2d':
            fig = plot_wing_geometry_2d(fig, vlm.wing_geometry, legend="Wing Geometry")
            if vlm.hs_geometry:
                fig = plot_wing_geometry_2d(fig, vlm.hs_geometry, legend="Horizontal Stabilizer")
        elif plot_type == 'discretization2D':
            fig = plot_wing_discretization_2d(fig, vlm.panel_data)
        elif plot_type == 'discretization3D':
            fig = plot_wing_discretization_3d(fig, vlm.panel_data)
        elif plot_type == 'Lift':
            fig = plot_distribution_all(fig, vlm, quantity='LIFT')
        elif plot_type == 'Drag':
            fig = plot_distribution_all(fig, vlm, quantity='DRAG')
        elif plot_type == 'CL':
            fig = plot_distribution_all(fig, vlm, quantity='CL')
        elif plot_type == 'CD':
            fig = plot_distribution_all(fig, vlm, quantity='CD')
        elif plot_type == 'wi':
            fig = plot_wing_heatmap(fig, vlm.panel_data, vlm.w_i, title='Induced Velocity Distribution', legend='Induced Velocity [m/s]')
        elif plot_type == 'gammas':
            fig = plot_wing_heatmap(fig, vlm.panel_data, vlm.gammas, title='Gamma Distribution', legend='Gamma [m²/s]')
        elif plot_type == 'curvature':
            fig = plot_wing_heatmap(fig, vlm.panel_data, vlm.dz_c, title='Curvature Distribution', legend='Curvature [1/m]')
        elif plot_type == 'wi_2d':
            fig = plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.w_i, title='Induced Velocity Distribution', legend='Induced Velocity [m/s]', u_vec=vlm.u_)
        elif plot_type == 'gammas_2d':
            fig = plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.gammas, title='Gamma Distribution', legend='Gamma [m²/s]', u_vec=vlm.u_)
        elif plot_type == 'curvature_2d':
            fig = plot_wing_heatmap_2d(fig, vlm.panel_data, vlm.dz_c, title='Curvature Distribution', legend='Curvature [1/m]', u_vec=vlm.u_)
        elif plot_type == 'CL_CD':
            fig = plot_CL_CD(fig, vlm, title='CL vs CD of the Wing')
        elif plot_type == 'CL-alpha':
            if vlm.results is None or 'CL' not in vlm.results:
                return jsonify({"status": "error", "message": "Please calculate coefficients first"}), 400
            fig = plot_coefficient_vs_alpha(fig, vlm, coefficient='CL')
        elif plot_type == 'CD-alpha':
            if vlm.results is None or 'CD' not in vlm.results:
                return jsonify({"status": "error", "message": "Please calculate coefficients first"}), 400
            fig = plot_coefficient_vs_alpha(fig, vlm, coefficient='CD')
        elif plot_type == 'CL-CD-alpha':
            if vlm.results is None or 'CD' not in vlm.results or 'CL' not in vlm.results:
                return jsonify({"status": "error", "message": "Please calculate coefficients first"}), 400
            fig = plot_CLCD_vs_alpha(fig, vlm, title=None)
        elif plot_type == 'LiftSection':
            if 'n_section' in data:
                try:
                    n_section = int(data['n_section'])  # Convert to integer
                except (ValueError, TypeError):
                    return jsonify({"status": "error", "message": "Invalid n_section value, must be a number"}), 400
            fig = plot_distribution(vlm, n_section, quantity='lift')
        elif plot_type == 'DragSection':
            if 'n_section' in data:
                try:
                    n_section = int(data['n_section'])  # Convert to integer
                except (ValueError, TypeError):
                    return jsonify({"status": "error", "message": "Invalid n_section value, must be a number"}), 400    
            fig = plot_distribution(vlm, n_section, quantity='drag')   
        else:
            return jsonify({"status": "error", "message": "Tipo de gráfico inválido"}), 400
        
        fig = apply_plotly_theme(fig)  # Aplicar tema
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"plot": plot_json})

    except Exception as e:
        logger.error(f"Error en /plot/{plot_type}: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    optimize_resources()
    try:
        #app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
        ui.run()
    finally:
        thread_pool.shutdown(wait=True)