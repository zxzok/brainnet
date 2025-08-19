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
from datetime import datetime
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from preprocessing_full import PreprocessPipeline, PreprocessPipelineConfig
from dynamic import DynamicAnalyzer, DynamicConfig
from static_analysis import StaticAnalyzer

from visualization import ReportConfig, ReportGenerator

from data_management import DatasetManager
import openneuro_client


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Background executor for long-running analysis tasks
executor = ThreadPoolExecutor(max_workers=2)

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
def init_db():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect('brainnet.db')
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS openneuro_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()


def process_image(image_id: int, filepath: str) -> None:
    """Run preprocessing and analysis pipelines for an uploaded image.

    Results are stored in the ``features`` table.  Any exceptions are
    caught and logged as an ``error`` feature to aid debugging.
    """
    try:
        pipeline = PreprocessPipeline(PreprocessPipelineConfig())
        preproc = pipeline.run(filepath)
        roi_ts = preproc.get('roi_timeseries')
        labels = preproc.get('roi_labels') or []

        if roi_ts is None or not len(labels):
            raise ValueError('Preprocessing produced no ROI time series')

        static_analyzer = StaticAnalyzer()
        conn_matrix = static_analyzer.compute_connectivity(roi_ts, labels)
        graph_metrics = static_analyzer.compute_graph_metrics(conn_matrix)

        dyn_cfg = DynamicConfig(window_length=10, step=5, n_states=2)
        dyn_analyzer = DynamicAnalyzer(dyn_cfg)
        dyn_result = dyn_analyzer.analyse(roi_ts)

        db = sqlite3.connect('brainnet.db')
        cur = db.cursor()
        for name, value in graph_metrics.global_metrics.items():
            cur.execute(
                'INSERT INTO features (image_id, feature_name, feature_value, feature_type) VALUES (?, ?, ?, ?)',
                (image_id, name, float(value), 'static'),
            )
        for idx, occ in enumerate(dyn_result.metrics.occupancy):
            cur.execute(
                'INSERT INTO features (image_id, feature_name, feature_value, feature_type) VALUES (?, ?, ?, ?)',
                (image_id, f"state_{idx}_occupancy", float(occ), 'dynamic'),
            )
        db.commit()
        db.close()
    except Exception as exc:  # pragma: no cover - best effort logging
        db = sqlite3.connect('brainnet.db')
        cur = db.cursor()
        cur.execute(
            'INSERT INTO features (image_id, feature_name, feature_value, feature_type) VALUES (?, ?, ?, ?)',
            (image_id, 'error', 0.0, str(exc)),
        )
        db.commit()
        db.close()


def _download_openneuro_dataset(dataset_id: str) -> None:
    """Download dataset from OpenNeuro and record its path in the database."""

    manager = DatasetManager.fetch_from_openneuro(dataset_id)
    conn = sqlite3.connect('brainnet.db')
    cur = conn.cursor()
    cur.execute(
        'INSERT OR IGNORE INTO openneuro_datasets (dataset_id, path) VALUES (?, ?)',
        (dataset_id, manager.root),
    )
    conn.commit()
    conn.close()


def _process_openneuro_for_patient(
    patient_id: int, dataset_id: str, dataset_path: str
) -> None:
    """Attach dataset images to a patient and run analysis."""

    manager = DatasetManager(dataset_path)
    subjects = getattr(manager.index, "_subjects", [])
    if not subjects:
        return
    subject = subjects[0]
    runs = manager.index.get_functional_runs(subject)
    for run in runs:
        conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
    cur = conn.cursor()
    cur.execute('SELECT dataset_id FROM openneuro_datasets')
    downloaded = {row[0] for row in cur.fetchall()}
    conn.close()
    return render_template(
        'openneuro.html',

        datasets=listing['datasets'],
        search=search,
        page=page,
        has_next=listing['has_next'],

        downloaded=downloaded,
    )


@app.route('/openneuro/download', methods=['POST'])
def download_openneuro():
    dataset_id = request.form['dataset_id']
    executor.submit(_download_openneuro_dataset, dataset_id)
    return redirect(url_for('openneuro'))

@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    """Show details of a specific patient."""
    conn = sqlite3.connect('brainnet.db')
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
    cursor.execute('SELECT dataset_id FROM openneuro_datasets ORDER BY downloaded_at DESC')
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
    conn = sqlite3.connect('brainnet.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    patient = cursor.fetchone()
    if not patient:
        conn.close()
        return "Patient not found", 404
    cursor.execute('SELECT id, image_path FROM mri_images WHERE patient_id = ?', (patient_id,))
    images = cursor.fetchall()
    conn.close()
    if not images:
        return "No images for patient", 404

    # Ensure analysis features exist, run if missing
    db = sqlite3.connect('brainnet.db')
    cur = db.cursor()
    for img_id, path in images:
        cur.execute('SELECT COUNT(*) FROM features WHERE image_id = ?', (img_id,))
        if cur.fetchone()[0] == 0:
            process_image(img_id, path)
    db.close()

    # Re-run analysis for the first image to obtain objects for report
    first_id, first_path = images[0]
    pipeline = PreprocessPipeline(PreprocessPipelineConfig())
    preproc = pipeline.run(first_path)
    roi_ts = preproc.get('roi_timeseries')
    labels = preproc.get('roi_labels') or []
    static_analyzer = StaticAnalyzer()
    conn_matrix = static_analyzer.compute_connectivity(roi_ts, labels)
    graph_metrics = static_analyzer.compute_graph_metrics(conn_matrix)
    dyn_cfg = DynamicConfig(window_length=10, step=5, n_states=2)
    dyn_analyzer = DynamicAnalyzer(dyn_cfg)
    dyn_model = dyn_analyzer.analyse(roi_ts)

    rep_cfg = ReportConfig(output_dir='reports')
    rep_gen = ReportGenerator(rep_cfg)
    patient_info = {"Name": patient[2], "Sex": patient[4] or '', "Age": patient[3] or ''}
    report_path = rep_gen.generate(
        subject_id=patient[1],
        conn_matrix=conn_matrix,
        graph_metrics=graph_metrics,
        dyn_model=dyn_model,
        roi_labels=labels,
        qc_metrics=preproc.get('qc_metrics', {}),
        patient_info=patient_info,
    )

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
        
        conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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

            conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
    cur = conn.cursor()
    cur.execute('SELECT path FROM openneuro_datasets WHERE dataset_id = ?', (dataset_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        executor.submit(_process_openneuro_for_patient, patient_id, dataset_id, row[0])
    return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Delete an MRI image."""
    conn = sqlite3.connect('brainnet.db')
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
    """View features for a specific image."""
    conn = sqlite3.connect('brainnet.db')
    cursor = conn.cursor()
    
    # Get image info
    cursor.execute('''
        SELECT mri_images.image_path, patients.name 
        FROM mri_images 
        JOIN patients ON mri_images.patient_id = patients.id
        WHERE mri_images.id = ?
    ''', (image_id,))
    image_info = cursor.fetchone()
    
    # Get features
    cursor.execute('''
        SELECT feature_name, feature_value, feature_type, calculated_at 
        FROM features 
        WHERE image_id = ? 
        ORDER BY calculated_at DESC
    ''', (image_id,))
    features = cursor.fetchall()
    
    conn.close()
    
    if image_info:
        return render_template('features.html', image_info=image_info, features=features)
    else:
        return "Image not found", 404

@app.route('/api/patients')
def api_patients():
    """API endpoint to get all patients."""
    conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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
    conn = sqlite3.connect('brainnet.db')
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6525)

# Analysis results page
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# MRI image gallery page
@app.route('/gallery')
def gallery():
    conn = sqlite3.connect('brainnet.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, image_path, image_type, acquisition_date, description, created_at 
        FROM mri_images 
        ORDER BY created_at DESC
    ''')
    images = cursor.fetchall()
    conn.close()
    return render_template('gallery.html', images=images)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500
