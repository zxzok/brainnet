"""
BrainNet Web Application for MRI Data Management
This module provides a Flask-based web interface for managing MRI brain images,
patient data, and computed features.
"""

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_file,
)
import os
import json
import shutil
from datetime import datetime
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    from brainnet.analysis_service import (
        analyze_and_store_image,
        generate_patient_report as generate_patient_report_file,
        get_analysis_deps_error,
    )
    from brainnet.data_management import DatasetManager
    from brainnet import openneuro_client
    from brainnet.runtime import connect_db, get_database_path, get_reports_dir, get_upload_dir
except ImportError:
    from analysis_service import (
        analyze_and_store_image,
        generate_patient_report as generate_patient_report_file,
        get_analysis_deps_error,
    )
    from data_management import DatasetManager
    import openneuro_client
    from runtime import connect_db, get_database_path, get_reports_dir, get_upload_dir


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(get_upload_dir())
app.config['DATABASE'] = str(get_database_path())
app.config['REPORT_FOLDER'] = str(get_reports_dir())
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Background executor for long-running analysis tasks
executor = ThreadPoolExecutor(max_workers=2)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
def init_db():
    """Initialize the SQLite database with required tables."""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            diagnosis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create mri_images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mri_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_path TEXT NOT NULL,
            image_type TEXT,
            acquisition_date TIMESTAMP,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
    ''')
    
    # Create features table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER,
            feature_name TEXT NOT NULL,
            feature_value REAL,
            feature_type TEXT,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES mri_images (id)
        )
    ''')

    # Track downloaded OpenNeuro datasets
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS openneuro_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE NOT NULL,
            name TEXT,
            description TEXT,
            modalities TEXT,
            tasks TEXT,
            sessions INTEGER,
            subjects INTEGER,
            size INTEGER,
            total_files INTEGER,
            path TEXT NOT NULL,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    # Track download attempts
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS download_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
        '''
    )

    # Ensure newer columns exist when upgrading from older schema
    cursor.execute('PRAGMA table_info(openneuro_datasets)')
    existing = {row[1] for row in cursor.fetchall()}
    required = {
        'name': 'TEXT',
        'description': 'TEXT',
        'modalities': 'TEXT',
        'tasks': 'TEXT',
        'sessions': 'INTEGER',
        'subjects': 'INTEGER',
        'size': 'INTEGER',
        'total_files': 'INTEGER',
        'path': 'TEXT',
        'status': 'TEXT',
    }
    for col, col_type in required.items():
        if col not in existing:
            cursor.execute(f'ALTER TABLE openneuro_datasets ADD COLUMN {col} {col_type}')

    conn.commit()
    conn.close()

# Initialize database
init_db()

def process_image(image_id: int, filepath: str) -> None:
    """Run preprocessing and analysis pipelines for an uploaded image.

    Results are stored in the ``features`` table via the shared analysis
    service so the Web app and CLI use the same execution logic.
    """
    analyze_and_store_image(image_id, filepath)


def _download_openneuro_dataset(dataset_id: str) -> None:
    """Download dataset from OpenNeuro and record its path in the database."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE download_tasks SET status = 'downloading' WHERE dataset_id = ? AND status = 'pending'",
        (dataset_id,),
    )
    conn.commit()
    conn.close()

    try:
        manager = DatasetManager.fetch_from_openneuro(dataset_id)
        metadata = openneuro_client.get_dataset_metadata(dataset_id)
        summary = metadata.get("summary", {})
        sessions = summary.get("sessions")
        subjects = summary.get("subjects")
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(
            '''
            INSERT OR REPLACE INTO openneuro_datasets (
                dataset_id, name, description, modalities, tasks, sessions,
                subjects, size, total_files, path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')
            ''',
            (
                dataset_id,
                metadata.get("name"),
                metadata.get("description"),
                json.dumps(summary.get("modalities") or []),
                json.dumps(summary.get("tasks") or []),
                len(sessions) if isinstance(sessions, list) else sessions,
                len(subjects) if isinstance(subjects, list) else subjects,
                summary.get("size"),
                summary.get("totalFiles"),
                manager.root,
            ),
        )
        cur.execute(
            "UPDATE download_tasks SET status = 'completed', completed_at = CURRENT_TIMESTAMP "
            "WHERE dataset_id = ? AND status = 'downloading'",
            (dataset_id,),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(
            "UPDATE download_tasks SET status = 'failed', error_message = ?, completed_at = CURRENT_TIMESTAMP "
            "WHERE dataset_id = ? AND status = 'downloading'",
            (str(exc), dataset_id),
        )
        cur.execute(
            "UPDATE openneuro_datasets SET status = 'failed' WHERE dataset_id = ?",
            (dataset_id,),
        )
        conn.commit()
        conn.close()


def _process_openneuro_for_patient(
    patient_id: int,
    dataset_id: str,
    dataset_path: str,
    selected_subjects: list | None = None,
    selected_runs: list | None = None,
) -> None:
    """Attach dataset images to a patient and run analysis.

    Parameters
    ----------
    selected_subjects : list | None
        Subject labels to process. If *None*, falls back to the first subject.
    selected_runs : list | None
        Specific file paths to process. Takes priority over *selected_subjects*.
    """

    manager = DatasetManager(dataset_path)
    all_subjects = getattr(manager.index, "_subjects", [])
    if not all_subjects:
        return

    runs_to_process = []

    if selected_runs:
        # Process specific file paths chosen by the user
        for subj in all_subjects:
            for run in manager.index.get_functional_runs(subj):
                if run.path in selected_runs:
                    runs_to_process.append(run)
    elif selected_subjects:
        for subj in selected_subjects:
            if subj in all_subjects:
                runs_to_process.extend(manager.index.get_functional_runs(subj))
    else:
        # Legacy fallback: first subject
        runs_to_process = manager.index.get_functional_runs(all_subjects[0])

    for run in runs_to_process:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO mri_images (patient_id, image_path, image_type, description) VALUES (?, ?, ?, ?)',
            (patient_id, run.path, 'func', f'OpenNeuro {dataset_id}'),
        )
        image_id = cur.lastrowid
        conn.commit()
        conn.close()
        process_image(image_id, run.path)

# Routes
@app.route('/')
def index():
    """Main page showing all patients."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, name, age, sex, diagnosis, created_at 
        FROM patients 
        ORDER BY created_at DESC
    ''')
    patients = cursor.fetchall()
    conn.close()
    
    return render_template('index.html', patients=patients)

@app.route('/patients')
def patients():
    """List all patients."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, name, age, sex, diagnosis, created_at 
        FROM patients 
        ORDER BY created_at DESC
    ''')
    patients = cursor.fetchall()
    conn.close()
    
    return render_template('patients.html', patients=patients)


@app.route('/openneuro')
def openneuro():
    """Display available OpenNeuro datasets."""
    search = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    listing = openneuro_client.list_datasets(search=search, page=page, per_page=per_page)
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT dataset_id, status FROM openneuro_datasets')
    rows = cur.fetchall()
    downloaded = {row[0] for row in rows if row[1] == 'ready'}
    downloading = {row[0] for row in rows if row[1] == 'downloading'}
    conn.close()
    return render_template(
        'openneuro.html',
        datasets=listing['datasets'],
        search=search,
        page=page,
        has_next=listing['has_next'],
        downloaded=downloaded,
        downloading=downloading,
    )


@app.route('/openneuro/download', methods=['POST'])
def download_openneuro():
    dataset_id = request.form['dataset_id']
    conn = connect_db()
    cur = conn.cursor()
    # Record the download task
    cur.execute(
        "INSERT INTO download_tasks (dataset_id, status) VALUES (?, 'pending')",
        (dataset_id,),
    )
    # Pre-create openneuro_datasets row with downloading status
    cur.execute(
        "INSERT OR IGNORE INTO openneuro_datasets (dataset_id, path, status) VALUES (?, '', 'downloading')",
        (dataset_id,),
    )
    cur.execute(
        "UPDATE openneuro_datasets SET status = 'downloading' WHERE dataset_id = ?",
        (dataset_id,),
    )
    conn.commit()
    conn.close()
    executor.submit(_download_openneuro_dataset, dataset_id)
    return redirect(url_for('openneuro_detail', dataset_id=dataset_id))

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    """Show details of a specific patient."""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get patient info
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    patient = cursor.fetchone()
    
    # Get associated MRI images
    cursor.execute('''
        SELECT id, image_path, image_type, acquisition_date, description, created_at
        FROM mri_images
        WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    images = cursor.fetchall()
    cursor.execute('SELECT dataset_id, name FROM openneuro_datasets ORDER BY downloaded_at DESC')
    on_datasets = cursor.fetchall()

    conn.close()

    if patient:
        return render_template(
            'patient_detail.html',
            patient=patient,
            images=images,
            openneuro_datasets=on_datasets,
        )
    else:
        return "Patient not found", 404


@app.route('/patient/<int:patient_id>/report')
def patient_report(patient_id):
    """Generate and return an HTML report for the patient."""
    try:
        report_path = generate_patient_report_file(
            patient_id,
            output_dir=app.config['REPORT_FOLDER'],
        )
    except ValueError as exc:
        message = str(exc)
        if message in {'Patient not found', 'No images for patient'}:
            return message, 404
        return message, 400
    return send_file(report_path)

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    """Add a new patient."""
    if request.method == 'POST':
        patient_id = request.form['patient_id']
        name = request.form['name']
        age = request.form.get('age', type=int)
        sex = request.form['sex']
        diagnosis = request.form['diagnosis']
        
        conn = connect_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO patients (patient_id, name, age, sex, diagnosis)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, name, age, sex, diagnosis))
            conn.commit()
            conn.close()
            return redirect(url_for('patients'))
        except sqlite3.IntegrityError:
            conn.close()
            return "Patient ID already exists", 400
    
    return render_template('add_patient.html')

@app.route('/edit_patient/<int:patient_id>', methods=['GET', 'POST'])
def edit_patient(patient_id):
    """Edit an existing patient."""
    conn = connect_db()
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form['name']
        age = request.form.get('age', type=int)
        sex = request.form['sex']
        diagnosis = request.form['diagnosis']
        
        cursor.execute('''
            UPDATE patients 
            SET name = ?, age = ?, sex = ?, diagnosis = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (name, age, sex, diagnosis, patient_id))
        conn.commit()
        conn.close()
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    patient = cursor.fetchone()
    conn.close()
    
    if patient:
        return render_template('edit_patient.html', patient=patient)
    else:
        return "Patient not found", 404

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    """Delete a patient."""
    conn = connect_db()
    cursor = conn.cursor()
    
    # First delete associated images
    cursor.execute('SELECT id FROM mri_images WHERE patient_id = ?', (patient_id,))
    images = cursor.fetchall()
    
    for img in images:
        # Delete the image file from disk
        cursor.execute('SELECT image_path FROM mri_images WHERE id = ?', (img[0],))
        row = cursor.fetchone()
        if row:
            image_path = row[0]
            try:
                os.remove(image_path)
            except OSError:
                pass  # File might not exist
    
    # Delete from database
    cursor.execute('DELETE FROM mri_images WHERE patient_id = ?', (patient_id,))
    cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
    conn.commit()
    conn.close()
    
    return redirect(url_for('patients'))

@app.route('/upload_image/<int:patient_id>', methods=['POST'])
def upload_image(patient_id):
    """Upload an MRI image for a patient."""
    if 'image' not in request.files:
        return "No file selected", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    if file:
        # Save the file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)

            image_type = request.form.get('image_type', 'unknown')
            description = request.form.get('description', '')

            conn = connect_db()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO mri_images (patient_id, image_path, image_type, description)
                VALUES (?, ?, ?, ?)
            ''', (patient_id, filepath, image_type, description))
            image_id = cursor.lastrowid
            conn.commit()
            conn.close()

            # kick off analysis in the background to avoid blocking
            executor.submit(process_image, image_id, filepath)

            return redirect(url_for('patient_detail', patient_id=patient_id))
        except Exception as exc:
            return f"Upload failed: {exc}", 500

    return "Upload failed", 500


@app.route('/patient/<int:patient_id>/use_openneuro', methods=['POST'])
def use_openneuro(patient_id):
    """Process a downloaded OpenNeuro dataset for this patient."""

    dataset_id = request.form['dataset_id']
    selected_subjects = request.form.getlist('subjects')
    selected_runs = request.form.getlist('runs')
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT path FROM openneuro_datasets WHERE dataset_id = ?', (dataset_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        executor.submit(
            _process_openneuro_for_patient,
            patient_id,
            dataset_id,
            row[0],
            selected_subjects=selected_subjects or None,
            selected_runs=selected_runs or None,
        )
    return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Delete an MRI image."""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get the file path
    cursor.execute('SELECT image_path FROM mri_images WHERE id = ?', (image_id,))
    row = cursor.fetchone()
    
    if row:
        image_path = row[0]
        try:
            os.remove(image_path)
        except OSError:
            pass  # File might not exist
    
    # Delete from database
    cursor.execute('DELETE FROM mri_images WHERE id = ?', (image_id,))
    conn.commit()
    conn.close()
    
    return "Image deleted successfully"

@app.route('/features/<int:image_id>')
def view_features(image_id):
    """View features for a specific image with visualizations."""
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT mri_images.id, mri_images.image_path, mri_images.image_type,
               mri_images.description, patients.name, patients.id
        FROM mri_images
        JOIN patients ON mri_images.patient_id = patients.id
        WHERE mri_images.id = ?
    ''', (image_id,))
    image_info = cursor.fetchone()

    cursor.execute('''
        SELECT id, feature_name, feature_value, feature_type, calculated_at
        FROM features
        WHERE image_id = ?
        ORDER BY feature_type, feature_name
    ''', (image_id,))
    features = cursor.fetchall()

    conn.close()

    if not image_info:
        return "Image not found", 404

    # Separate features by type and extract visualization data
    static_features = []
    static_node_features = []
    dynamic_features = []
    errors = []
    conn_matrix = None
    conn_labels = None
    state_sequence = None

    for f in features:
        fid, fname, fval, ftype, fcalc = f
        if fname.startswith('_connectivity_matrix'):
            try:
                conn_matrix = json.loads(ftype)
            except (json.JSONDecodeError, TypeError):
                pass
        elif fname.startswith('_connectivity_labels'):
            try:
                conn_labels = json.loads(ftype)
            except (json.JSONDecodeError, TypeError):
                pass
        elif fname.startswith('_state_sequence'):
            try:
                state_sequence = json.loads(ftype)
            except (json.JSONDecodeError, TypeError):
                pass
        elif fname == 'error':
            errors.append({'message': ftype, 'time': fcalc})
        elif ftype == 'static':
            static_features.append({'id': fid, 'name': fname, 'value': fval, 'time': fcalc})
        elif ftype == 'static_node':
            static_node_features.append({'id': fid, 'name': fname, 'value': fval, 'time': fcalc})
        elif ftype == 'dynamic':
            dynamic_features.append({'id': fid, 'name': fname, 'value': fval, 'time': fcalc})

    return render_template(
        'features_detail.html',
        image=image_info,
        static_features=static_features,
        static_node_features=static_node_features,
        dynamic_features=dynamic_features,
        errors=errors,
        conn_matrix=json.dumps(conn_matrix) if conn_matrix else 'null',
        conn_labels=json.dumps(conn_labels) if conn_labels else '[]',
        state_sequence=json.dumps(state_sequence) if state_sequence else 'null',
    )


@app.route('/features/<int:image_id>/delete', methods=['POST'])
def delete_features(image_id):
    """Delete all computed features for an image."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT patient_id FROM mri_images WHERE id = ?', (image_id,))
    row = cur.fetchone()
    cur.execute('DELETE FROM features WHERE image_id = ?', (image_id,))
    conn.commit()
    conn.close()
    if row:
        return redirect(url_for('patient_detail', patient_id=row[0]))
    return redirect(url_for('index'))


@app.route('/features/<int:image_id>/recompute', methods=['POST'])
def recompute_features(image_id):
    """Delete existing features and re-run analysis."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT image_path FROM mri_images WHERE id = ?', (image_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return "Image not found", 404
    cur.execute('DELETE FROM features WHERE image_id = ?', (image_id,))
    conn.commit()
    conn.close()
    executor.submit(process_image, image_id, row[0])
    return redirect(url_for('view_features', image_id=image_id))


@app.route('/api/features/<int:image_id>/export')
def export_features(image_id):
    """Export features as JSON."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        'SELECT feature_name, feature_value, feature_type FROM features WHERE image_id = ?',
        (image_id,),
    )
    rows = cur.fetchall()
    conn.close()
    result = {}
    for name, value, ftype in rows:
        if name.startswith('_'):
            continue
        if ftype not in result:
            result[ftype] = {}
        result[ftype][name] = value
    return jsonify({'image_id': image_id, 'features': result})


@app.route('/system/status')
def system_status():
    """Show system dependency status."""
    deps = {}
    for pkg in ['numpy', 'scipy', 'pandas', 'nibabel', 'nilearn',
                 'sklearn', 'networkx', 'plotly', 'hmmlearn']:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, '__version__', 'installed')
        except ImportError:
            deps[pkg] = None
    return render_template('system_status.html', deps=deps, analysis_error=get_analysis_deps_error())

@app.route('/api/patients')
def api_patients():
    """API endpoint to get all patients."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, name, age, sex, diagnosis, created_at 
        FROM patients 
        ORDER BY created_at DESC
    ''')
    patients = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    result = []
    for patient in patients:
        result.append({
            'id': patient[0],
            'patient_id': patient[1],
            'name': patient[2],
            'age': patient[3],
            'sex': patient[4],
            'diagnosis': patient[5],
            'created_at': patient[6]
        })
    
    return jsonify(result)

@app.route('/api/patients/<int:patient_id>')
def api_patient(patient_id):
    """API endpoint to get a specific patient."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    patient = cursor.fetchone()
    conn.close()
    
    if patient:
        return jsonify({
            'id': patient[0],
            'patient_id': patient[1],
            'name': patient[2],
            'age': patient[3],
            'sex': patient[4],
            'diagnosis': patient[5],
            'created_at': patient[6],
            'updated_at': patient[7]
        })
    else:
        return jsonify({'error': 'Patient not found'}), 404

@app.route('/api/search')
def api_search():
    """API endpoint for searching patients."""
    query = request.args.get('q', '')
    conn = connect_db()
    cursor = conn.cursor()
    
    if query:
        cursor.execute('''
            SELECT id, patient_id, name, age, sex, diagnosis, created_at 
            FROM patients 
            WHERE patient_id LIKE ? OR name LIKE ? OR diagnosis LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
    else:
        cursor.execute('''
            SELECT id, patient_id, name, age, sex, diagnosis, created_at 
            FROM patients 
            ORDER BY created_at DESC
        ''')
    
    patients = cursor.fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    result = []
    for patient in patients:
        result.append({
            'id': patient[0],
            'patient_id': patient[1],
            'name': patient[2],
            'age': patient[3],
            'sex': patient[4],
            'diagnosis': patient[5],
            'created_at': patient[6]
        })
    
    return jsonify(result)

# MRI visualization page
@app.route('/mri')
def mri_visualization():
    return render_template('mri_visualization.html')

@app.route('/features')
def features_visualization():
    return render_template('features.html')

# Network visualization page
@app.route('/network_visualization', methods=['GET'])
def network_visualization():
    return render_template('network_visualization.html')

# API route for network data
@app.route('/api/network_data', methods=['GET'])
def api_network_data():
    import json
    network_file = 'network.json'
    if os.path.exists(network_file):
        with open(network_file, 'r') as f:
            data = json.load(f)
    else:
        # Dummy network data
        data = {
            'elements': [
                { 'data': { 'id': 'a', 'label': 'Node A' } },
                { 'data': { 'id': 'b', 'label': 'Node B' } },
                { 'data': { 'id': 'c', 'label': 'Node C' } },
                { 'data': { 'source': 'a', 'target': 'b' } },
                { 'data': { 'source': 'a', 'target': 'c' } }
            ]
        }
    return jsonify(data)

@app.route('/data')
def data_hub():
    """Data Hub landing page showing downloaded datasets and active downloads."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT dataset_id, name, description, modalities, tasks, sessions, subjects, size, total_files "
        "FROM openneuro_datasets WHERE status = 'ready' ORDER BY downloaded_at DESC"
    )
    ready = cur.fetchall()
    cur.execute(
        "SELECT dataset_id, name FROM openneuro_datasets WHERE status = 'downloading'"
    )
    active = cur.fetchall()
    conn.close()

    datasets = []
    for row in ready:
        modalities = []
        tasks = []
        try:
            modalities = json.loads(row[3]) if row[3] else []
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            tasks = json.loads(row[4]) if row[4] else []
        except (json.JSONDecodeError, TypeError):
            pass
        datasets.append({
            'dataset_id': row[0],
            'name': row[1] or row[0],
            'description': row[2],
            'modalities': modalities,
            'tasks': tasks,
            'sessions': row[5],
            'subjects': row[6],
            'size': row[7],
            'total_files': row[8],
        })

    downloads = [{'dataset_id': r[0], 'name': r[1] or r[0]} for r in active]
    return render_template('data_hub.html', datasets=datasets, downloads=downloads)


@app.route('/openneuro/<dataset_id>')
def openneuro_detail(dataset_id):
    """Detail page for an OpenNeuro dataset."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        'SELECT dataset_id, name, description, modalities, tasks, sessions, subjects, size, total_files, status '
        'FROM openneuro_datasets WHERE dataset_id = ?',
        (dataset_id,),
    )
    local = cur.fetchone()
    conn.close()

    local_status = local[9] if local else None

    # Always fetch fresh metadata from OpenNeuro API
    try:
        metadata = openneuro_client.get_dataset_metadata(dataset_id)
        summary = metadata.get('summary', {})
        sessions = summary.get('sessions')
        subjects = summary.get('subjects')
        ds_info = {
            'id': dataset_id,
            'name': metadata.get('name') or (local[1] if local else dataset_id),
            'description': metadata.get('description') or (local[2] if local else ''),
            'modalities': summary.get('modalities', []),
            'tasks': summary.get('tasks', []),
            'sessions': len(sessions) if isinstance(sessions, list) else sessions,
            'subjects': len(subjects) if isinstance(subjects, list) else subjects,
            'size': summary.get('size'),
            'total_files': summary.get('totalFiles'),
        }
    except Exception:
        # Fallback to local data if API fails
        if local:
            modalities = []
            tasks = []
            try:
                modalities = json.loads(local[3]) if local[3] else []
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                tasks = json.loads(local[4]) if local[4] else []
            except (json.JSONDecodeError, TypeError):
                pass
            ds_info = {
                'id': dataset_id,
                'name': local[1] or dataset_id,
                'description': local[2] or '',
                'modalities': modalities,
                'tasks': tasks,
                'sessions': local[5],
                'subjects': local[6],
                'size': local[7],
                'total_files': local[8],
            }
        else:
            ds_info = {'id': dataset_id, 'name': dataset_id, 'description': '',
                       'modalities': [], 'tasks': [], 'sessions': None,
                       'subjects': None, 'size': None, 'total_files': None}

    return render_template(
        'openneuro_detail.html', dataset=ds_info, status=local_status,
    )


@app.route('/api/download_status/<dataset_id>')
def api_download_status(dataset_id):
    """Return current download status as JSON for polling."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT status, error_message FROM download_tasks WHERE dataset_id = ? ORDER BY id DESC LIMIT 1",
        (dataset_id,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return jsonify({'status': row[0], 'error': row[1]})
    return jsonify({'status': 'unknown', 'error': None})


@app.route('/openneuro/<dataset_id>/browse')
def dataset_browse(dataset_id):
    """Browse the contents of a downloaded dataset."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        'SELECT path, name FROM openneuro_datasets WHERE dataset_id = ? AND status = ?',
        (dataset_id, 'ready'),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return "Dataset not found or not yet downloaded", 404

    dataset_path, dataset_name = row[0], row[1]

    # Get patients for the analysis form
    cur.execute('SELECT id, patient_id, name FROM patients ORDER BY name')
    patients = cur.fetchall()
    conn.close()

    from data_management import DatasetIndex
    try:
        index = DatasetIndex(dataset_path, datatypes=['func', 'anat', 'dwi'])
    except Exception:
        index = DatasetIndex(dataset_path)

    subjects_data = []
    for subj in index.list_subjects():
        sessions = index.list_sessions(subj)
        sess_data = []
        for ses in sessions:
            runs = []
            for dtype in index.datatypes:
                try:
                    files = index.get_files(dtype, subj, session=ses)
                except KeyError:
                    files = []
                for f in files:
                    runs.append({
                        'path': f.path,
                        'task': f.task,
                        'run': f.run,
                        'suffix': f.suffix,
                        'datatype': f.datatype,
                    })
            sess_data.append({'session': ses, 'runs': runs})
        subjects_data.append({'subject': subj, 'sessions': sess_data})

    ds_summary = index.summary()

    return render_template(
        'dataset_browse.html',
        dataset_id=dataset_id,
        dataset_name=dataset_name or dataset_id,
        subjects=subjects_data,
        summary=ds_summary,
        patients=patients,
    )


@app.route('/openneuro/<dataset_id>/delete', methods=['POST'])
def delete_dataset(dataset_id):
    """Delete a downloaded dataset from disk and database."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT path FROM openneuro_datasets WHERE dataset_id = ?', (dataset_id,))
    row = cur.fetchone()
    if row and row[0]:
        try:
            shutil.rmtree(row[0])
        except OSError:
            pass
    cur.execute('DELETE FROM openneuro_datasets WHERE dataset_id = ?', (dataset_id,))
    cur.execute('DELETE FROM download_tasks WHERE dataset_id = ?', (dataset_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('data_hub'))


@app.route('/openneuro/<dataset_id>/analyze', methods=['POST'])
def analyze_dataset(dataset_id):
    """Run analysis on selected runs from a dataset for a patient."""
    patient_id = request.form.get('patient_id', type=int)
    selected_runs = request.form.getlist('runs')

    if not patient_id:
        return "Patient is required", 400

    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT path FROM openneuro_datasets WHERE dataset_id = ?', (dataset_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Dataset not found", 404

    executor.submit(
        _process_openneuro_for_patient,
        patient_id,
        dataset_id,
        row[0],
        selected_runs=selected_runs or None,
    )
    return redirect(url_for('patient_detail', patient_id=patient_id))

# Analysis results page
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# MRI image gallery page
@app.route('/gallery')
def gallery():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, image_path, image_type, acquisition_date, description, created_at 
        FROM mri_images 
        ORDER BY created_at DESC
    ''')
    images = cursor.fetchall()
    conn.close()
    return render_template('gallery.html', images=images)

# Register chat blueprint
try:
    try:
        from brainnet.web_chat import chat_bp
    except ImportError:
        from web_chat import chat_bp
    app.register_blueprint(chat_bp)
except Exception:
    pass  # Chat feature unavailable — web app still works

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


def main() -> None:
    """Run the Flask development server."""

    app.run(debug=True, host='0.0.0.0', port=6525)


if __name__ == '__main__':
    main()
