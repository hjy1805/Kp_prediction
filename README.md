# KAUST Infectious Diseases Epidemiology Lab | Digital Health Tool

## Overview

**KAUST Digital Health | Genomic Risk Prediction** is a research-focused **machine learning platform** developed by the Infectious Diseases Epidemiology Lab at KAUST.  
It predicts **in-hospital mortality** and **ICU admission** from bacterial **genomic biomarkers** using XGBoost models.  

> ⚠️ This platform is intended for **research purposes only** and is **not for clinical decision-making**.

Key features:

- Scan **bacterial genome assemblies (FASTA)** for genomic biomarkers (*unitigs*)  
- Predict **mortality** and **ICU admission risk**  
- Compute **approximate 95% confidence intervals** for predictions  
- Optional **SHAP-based biomarker importance visualization**  
- Parallelized genome scanning for **large datasets**

---

## Project Structure

```plaintext
my-streamlit-app/
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies (pip)
├── Unitigs_predictor_DEATH.csv   # Feature CSV for mortality model
├── Unitigs_predictor_ICU.csv     # Feature CSV for ICU model
├── xgb_fold1_8_Death.joblib      # Pretrained XGBoost mortality model
├── xgb_fold5_8_ICU.joblib        # Pretrained XGBoost ICU model
└── KAUST_Logo.svg                # Logo for app display
```

---

## Installation

### **Using pip**

Ensure Python 3.10+ and pip are installed:

```bash
pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```
### **Docker Usage**

Ensure Docker are installed:

```bash
docker pull hjy1805/ide_app:latest
docker run -p 8501:8501 hjy1805/ide_app:latest
```


## App Workflow
1.	Select clinical outcome:
	•	In-hospital Mortality
	•	ICU Admission
2.	Upload bacterial genome FASTA files (.fa, .fasta, .fna, .fa.gz, .fasta.gz)
3.	Adjust optional parameters:
	•	Decision threshold (probability cutoff)
	•	Confidence interval effective n (controls CI width)
	•	Enable SHAP predictive biomarker analysis (optional)
4.	Run prediction:
	•	Model scans genomes for unitigs
	•	Generates probability prediction & 95% CI
	•	Downloadable CSV with predictions
5.	SHAP Visualisation (optional):
	•	Identify most influential genomic biomarkers
	•	Waterfall or bar plot per sample
	•	Adjustable number of top biomarkers


## Contacts
For inquiries regarding this research, please contact:

Jiayi Huang

Email: jiayi.huang@kaust.edu.sa

PhD student

Infectious Disease Epidemiology Lab

Biological and Environmental Science and Engineering (BESE) Division

King Abdullah University of Science and Technology (KAUST)















