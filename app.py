from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lung_cancer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

# Model configuration
IMG_SIZE = 224
# IMPORTANT: Class names must match your training folder names exactly!
# Update these if your folder names are different
class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']
model = None

# Database Models
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='patient', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    prediction_label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    recommendations = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database and load model
def init_app():
    global model
    with app.app_context():
        db.create_all()
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        try:
            model = tf.keras.models.load_model("lung_cancer_full_model.h5")
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

def get_display_name(prediction_label):
    """Convert 'Bengin cases' to 'Benign', 'Malignant cases' to 'Malignant', etc."""
    label_map = {
        'Bengin cases': 'Benign',
        'Malignant cases': 'Malignant',
        'Normal cases': 'Normal'
    }
    return label_map.get(prediction_label, prediction_label)

def get_label_class(prediction_label):
    """Get CSS class for prediction label"""
    if 'Malignant' in prediction_label:
        return 'malignant'
    elif 'Bengin' in prediction_label:
        return 'benign'
    else:
        return 'normal'

# Make helper functions available in templates
@app.context_processor
def utility_processor():
    return dict(get_display_name=get_display_name, get_label_class=get_label_class)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = np.max(preds) * 100
    
    return class_names[class_id], confidence

def get_recommendations(prediction_label, confidence):
    recommendations = {
        'Malignant cases': [
            "Immediate consultation with an oncologist is strongly recommended.",
            "Consider getting a biopsy for confirmation.",
            "Discuss treatment options including chemotherapy, radiation, or surgery.",
            "Join a support group for emotional and mental health support.",
            "Maintain a healthy diet rich in antioxidants.",
            "Avoid smoking and exposure to secondhand smoke.",
            "Regular follow-up scans are essential."
        ],
        'Bengin cases': [
            "Schedule a follow-up appointment with your pulmonologist.",
            "Continue regular monitoring with periodic CT scans.",
            "Maintain a healthy lifestyle with regular exercise.",
            "Avoid smoking and limit alcohol consumption.",
            "Monitor for any changes in symptoms like cough or breathing difficulty.",
            "Stay updated with your healthcare provider.",
            "Consider stress management techniques."
        ],
        'Normal cases': [
            "Your lungs appear healthy - continue maintaining good habits.",
            "Schedule annual health check-ups as preventive care.",
            "Avoid smoking and exposure to pollutants.",
            "Maintain regular physical activity.",
            "Eat a balanced diet with plenty of fruits and vegetables.",
            "Stay hydrated and practice deep breathing exercises.",
            "Monitor any respiratory symptoms and consult if needed."
        ]
    }
    
    base_recommendations = recommendations.get(prediction_label, [])
    
    if confidence < 70:
        base_recommendations.insert(0, "Note: Prediction confidence is below 70%. Please consult a medical professional for confirmation.")
    
    return '\n'.join([f"{i+1}. {rec}" for i, rec in enumerate(base_recommendations)])

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        phone = request.form.get('phone')
        
        if Patient.query.filter_by(email=email).first():
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_patient = Patient(
            name=name,
            email=email,
            password=hashed_password,
            age=age,
            gender=gender,
            phone=phone
        )
        
        db.session.add(new_patient)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        patient = Patient.query.filter_by(email=email).first()
        
        if patient and check_password_hash(patient.password, password):
            session['patient_id'] = patient.id
            session['patient_name'] = patient.name
            flash('Login successful!', 'success')
            return redirect(url_for('patient_home'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html')

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Hardcoded credentials
        if username == 'doctor' and password == 'doctor123':
            session['doctor'] = True
            flash('Doctor login successful!', 'success')
            return redirect(url_for('doctor_home'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('doctor_login.html')

@app.route('/patient/home')
def patient_home():
    if 'patient_id' not in session:
        return redirect(url_for('login'))
    
    patient = Patient.query.get(session['patient_id'])
    recent_predictions = Prediction.query.filter_by(patient_id=session['patient_id']).order_by(Prediction.created_at.desc()).limit(5).all()
    
    return render_template('patient_home.html', patient=patient, recent_predictions=recent_predictions)

@app.route('/patient/predict', methods=['GET', 'POST'])
def predict():
    if 'patient_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file uploaded!', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{session['patient_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            prediction_label, confidence = predict_image(filepath)
            recommendations = get_recommendations(prediction_label, confidence)
            
            # Save to database
            new_prediction = Prediction(
                patient_id=session['patient_id'],
                image_path=filename,
                prediction_label=prediction_label,
                confidence=confidence,
                recommendations=recommendations
            )
            
            db.session.add(new_prediction)
            db.session.commit()
            
            return redirect(url_for('result', prediction_id=new_prediction.id))
        else:
            flash('Invalid file type! Please upload an image.', 'error')
            return redirect(request.url)
    
    return render_template('predict.html')

@app.route('/patient/result/<int:prediction_id>')
def result(prediction_id):
    if 'patient_id' not in session:
        return redirect(url_for('login'))
    
    prediction = Prediction.query.get_or_404(prediction_id)
    
    if prediction.patient_id != session['patient_id']:
        flash('Unauthorized access!', 'error')
        return redirect(url_for('patient_home'))
    
    return render_template('result.html', prediction=prediction)

@app.route('/patient/history')
def history():
    if 'patient_id' not in session:
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(patient_id=session['patient_id']).order_by(Prediction.created_at.desc()).all()
    
    return render_template('history.html', predictions=predictions)

@app.route('/download/<filename>')
def download_file(filename):
    if 'patient_id' not in session and 'doctor' not in session:
        return redirect(url_for('login'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

@app.route('/doctor/home')
def doctor_home():
    if 'doctor' not in session:
        return redirect(url_for('doctor_login'))
    
    patients = Patient.query.all()
    total_predictions = Prediction.query.count()
    
    malignant_count = Prediction.query.filter_by(prediction_label='Malignant cases').count()
    benign_count = Prediction.query.filter_by(prediction_label='Bengin cases').count()
    normal_count = Prediction.query.filter_by(prediction_label='Normal cases').count()
    
    return render_template('doctor_home.html', 
                         patients=patients, 
                         total_predictions=total_predictions,
                         malignant_count=malignant_count,
                         benign_count=benign_count,
                         normal_count=normal_count)

@app.route('/doctor/patient/<int:patient_id>')
def patient_details(patient_id):
    if 'doctor' not in session:
        return redirect(url_for('doctor_login'))
    
    patient = Patient.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(patient_id=patient_id).order_by(Prediction.created_at.desc()).all()
    
    return render_template('patient_details.html', patient=patient, predictions=predictions)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('landing'))

if __name__ == '__main__':
    init_app()
    app.run(debug=True)