🫁 Lung Cancer Detection System
Flask + CNN + ResNet50 + TensorFlow

📌 Project Overview
This project is an AI-powered web application designed to detect lung cancer from medical images such as CT scans and X-rays.
The system uses a CNN + ResNet50 deep learning model integrated with a Flask web application, enabling patients and doctors to upload scans and receive predictions with recommendations.

🚀 Features
👤 Patient & Doctor Authentication
🖼️ Medical Image Upload (CT Scan / X-ray)
🤖 AI-Based Lung Cancer Prediction
📊 Classification:
Normal
Benign
Malignant
🧠 Recommendation Engine (basic medical guidance)
📁 Prediction History & Reports
🗄️ SQLite Database Integration
🌐 Web Interface using HTML, CSS, Bootstrap

🏗️ Tech Stack
Backend
Python
Flask
AI / ML
TensorFlow
Keras
CNN + ResNet50
Frontend
HTML
CSS
Bootstrap
Jinja2 Templates
Database
SQLite
Tools & Libraries
NumPy
Pandas
OpenCV / PIL
Werkzeug

📂 Project Structure
Lung_Cancer_Detection/
│
├── app.py                      # Main Flask application
├── cnn_res_final.h5            # Trained deep learning model
├── instance/
│   └── lung_cancer.db         # SQLite database
│
├── static/
│   ├── uploads/               # Uploaded medical images
│   └── assets/                # CSS, JS, images
│
├── templates/                 # HTML templates
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── result.html
│   └── history.html
│
└── README.md


⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone <your-repo-link>
cd Lung_Cancer_Detection


2️⃣ Create Virtual Environment
python -m venv venv

Activate it:
Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate


3️⃣ Install Dependencies
pip install flask tensorflow numpy pandas pillow opencv-python

(Optional: create a requirements.txt later)

4️⃣ Fix Code Issue (Important)
In app.py, replace:
if not os.path.exists(app.config['UPLOAD_FOLDER'w]):

with:
if not os.path.exists(app.config['UPLOAD_FOLDER']):


5️⃣ Add Model File
Ensure this file exists in root directory:
cnn_res_final.h5


6️⃣ Run the Application
python app.py


7️⃣ Open in Browser
http://127.0.0.1:5000


🧪 How It Works
User logs in (Patient / Doctor)
Patient uploads medical image
Image is preprocessed:
Resized
Normalized
Converted to tensor
Model (cnn_res_final.h5) predicts:
Normal / Benign / Malignant
Result displayed with recommendations
Data stored in SQLite database
Doctor can review reports

🔐 Security Considerations
Basic authentication implemented
Input validation for file uploads
Can be extended with:
JWT authentication
Role-based access control
Cloud security integrations

⚠️ Limitations
Not a certified medical tool
Basic recommendation logic
SQLite not suitable for production scale
No advanced model explainability

🌱 Future Enhancements
Deploy on AWS (EC2 + S3 + RDS)
Add REST APIs
Implement Docker & CI/CD
Improve model accuracy
Add explainable AI (Grad-CAM)
Integrate real-time monitoring

📜 License
This project is for educational purposes only.

👨‍💻 Authors
Nikhil Chowdary
Sampath Reddy
Surendra Reddy
