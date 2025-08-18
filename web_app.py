"""
BrainNet Web Application for MRI Data Management
This module provides a Flask-based web interface for managing MRI brain images,
patient data, and computed features.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    UserMixin,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import json
from datetime import datetime
import sqlite3
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

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

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')

    # Create default admin user if none exist
    cursor.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        cursor.execute(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            ('admin', generate_password_hash('admin'), 'admin'),
        )
    
    conn.commit()
    conn.close()

# Initialize database
init_db()


class User(UserMixin):
    """Simple User model for authentication."""

    def __init__(self, id, username, password_hash, role):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect('brainnet.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, password_hash, role FROM users WHERE id = ?',
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(*row)
        return None

    @staticmethod
    def get_by_username(username):
        conn = sqlite3.connect('brainnet.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, username, password_hash, role FROM users WHERE username = ?',
            (username,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(*row)
        return None


@login_manager.user_loader
def load_user(user_id):
    return User.get(int(user_id))


def admin_required(f):
    """Decorator to restrict routes to admin users."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return "Forbidden", 403
        return f(*args, **kwargs)

    return decorated_function

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_by_username(username)
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            error = 'Invalid credentials'
    return render_template('login.html', error=error)


@app.route('/logout')
@login_required
def logout():
    """Log out the current user."""
    logout_user()
    return redirect(url_for('login'))


# Routes
@app.route('/')
@login_required
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
@login_required
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

@app.route('/patient/<int:patient_id>')
@login_required
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
    
    conn.close()
    
    if patient:
        return render_template('patient_detail.html', patient=patient, images=images)
    else:
        return "Patient not found", 404

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
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
@login_required
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
@login_required
@admin_required
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
@login_required
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
        file.save(filepath)
        
        image_type = request.form.get('image_type', 'unknown')
        description = request.form.get('description', '')
        
        conn = sqlite3.connect('brainnet.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO mri_images (patient_id, image_path, image_type, description)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, filepath, image_type, description))
        conn.commit()
        conn.close()
        
        return redirect(url_for('patient_detail', patient_id=patient_id))
    
    return "Upload failed", 500

@app.route('/delete_image/<int:image_id>', methods=['POST'])
@login_required
@admin_required
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
@login_required
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
@login_required
def mri_visualization():
    return render_template('mri_visualization.html')

@app.route('/features')
@login_required
def features_visualization():
    return render_template('features.html')

# Network visualization page
@app.route('/network_visualization', methods=['GET'])
@login_required
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
@login_required
def analysis():
    return render_template('analysis.html')

# MRI image gallery page
@app.route('/gallery')
@login_required
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
