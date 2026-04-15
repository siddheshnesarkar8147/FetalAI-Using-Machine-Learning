# 🚀 Quick Start Guide - Run in 3 Steps

## ⚡ Step 1: Install Dependencies (First Time Only)

Open PowerShell in the project directory:

```powershell
cd d:\fetal-health-classification-main

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

---

## ⚡ Step 2: Run the Web Application

```powershell
# Make sure you're in the project directory
cd d:\fetal-health-classification-main

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start the Flask application
python app.py
```

You should see output like:
```
============================================================
🏥 Fetal Health Classification Web Application
============================================================

✅ Server starting...
📍 Navigate to http://127.0.0.1:5000 in your browser
🔍 Model Information: http://127.0.0.1:5000/inspect

Press CTRL+C to stop the server
```

---

## ⚡ Step 3: Open in Browser

1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Go to: **http://127.0.0.1:5000**
3. Fill in the fetal health measurements
4. Click **🔍 Predict**
5. View your results!

---

## 📍 Available URLs

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:5000 | Main prediction form |
| http://127.0.0.1:5000/inspect | Model information |
| http://127.0.0.1:5000/health | Health check API |
| http://127.0.0.1:5000/api/predict | JSON prediction API |

---

## ⛔ To Stop the Server

Press **CTRL + C** in the terminal.

---

## 🆘 Common Issues

### Issue: "Module not found" error
```powershell
# Reactivate virtual environment and reinstall
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
Edit `app.py` line and change port from 5000 to 5001 or another available port.

### Issue: Model file not found
The model file `optimized_fetal_health_model.pkl` must exist in the project root. If missing, run the Jupyter notebook first.

---

## 📚 Full Documentation

See `README.md` for detailed documentation, API endpoints, and troubleshooting.

---

**Happy predicting! 🏥**
