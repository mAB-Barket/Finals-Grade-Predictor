# Student Success AI — Savior Mode

This small project predicts final exam scores from student features and includes a "Savior Mode" prescriptive feature: it simulates added study hours to find the minimal increase that flips a predicted Fail → Pass.

Files
- `generate_data.py` — creates `student_data_advanced.csv` (2000 synthetic students).
- `train_model.py` — trains a `RandomForestRegressor` and saves `grade_predictor_model.pkl`.
- `app.py` — Streamlit app with prediction + "Savior Mode" recommendations.
- `requirements.txt` — Python dependencies for the project.

Quick setup (PowerShell)

1. (Optional) Create a venv and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Generate dataset (optional — a CSV is already included if present):

```powershell
python generate_data.py
```

4. Train model (optional — a trained model is already included if present):

```powershell
python train_model.py
```

5. Run the Streamlit app:

```powershell
python -m streamlit run "d:/ Practiceeeee/Programming For Ai Project/app.py"
```

Access the app at: `http://localhost:8501`

Notes
- The app will warn if `grade_predictor_model.pkl` is missing. If you see that, run `train_model.py`.
- "Savior Mode" simulates increasing `Hours_Studied` (up to a limit) and reports the minimal hours needed to reach the passing threshold.

If you want, I can pin versions in `requirements.txt` to match your environment, add a small license header, or create a simple Git commit for these changes.
