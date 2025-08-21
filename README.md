# Heart Disease EDA
Exploratory data analysis of a heart disease dataset using Python, Pandas, Seaborn, Matplotlib, and Plotly.
The project investigates demographics, risk factors, symptoms, and clinical measures to understand their relationship with heart disease.

## Contents
- heart.csv – dataset
- heart_disease.py – analysis & visualizations

## Dataset
Columns include:
- Age, Sex
- Chest pain type
- Resting BP, Cholesterol, Fasting blood sugar
- Resting ECG, Max heart rate, Exercise-induced angina
- ST depression, ST slope
- Number of major vessels, Thallium test
- Condition (0 = No disease, 1 = Disease)

## Analysis Highlights
- Class balance (disease vs no disease)
- Sex distribution and condition breakdown
- Risk factors (blood pressure, cholesterol, max heart rate, fasting blood sugar)
- Symptom features (chest pain, exercise-induced angina)
- Heart function measures (ECG, ST slope, ST depression)
- Advanced plots: pairplot, feature correlations, scatter vs age

## How to Run
- Install dependencies:
pip install pandas numpy matplotlib seaborn plotly
- Place heart.csv in the same folder as heart_disease.py.
- Run:
python heart_disease.py
- Charts will display inline (Jupyter/Colab) or in a window (local).

## License

Academic coursework — for learning and exploration only.
