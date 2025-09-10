# 🔒 RWE Privacy & Compliance Playbook

**Automated checks and scorecards to evaluate Real-World Data (RWD/RWE) pipelines against privacy and regulatory requirements (HIPAA, GDPR, DUA).**

---

## 📖 Overview

This playbook extends the [RWE Governance & Analytics Playbook](https://github.com/camontefusco/RWE-governance-and-analytics-playbook-openFDA-clinicaltrials-CDC-OMOP-FHIR-ROI) by adding a **privacy & compliance lens**:

- Detect quasi-identifiers and re-identification risks  
- Evaluate datasets for **HIPAA Safe Harbor** and **GDPR anonymization** requirements  
- Score compliance risks across jurisdictions (US, EU, APAC)  
- Link governance metrics → compliance risk → business impact  

Built with Jupyter notebooks + helper scripts. Beginner-friendly outputs (CSVs, JSONs, PDFs) make it easy for leaders, data stewards, and compliance teams to understand and act.

---

## 🗂️ Repo Structure

```text
RWE-Privacy-and-Compliance-Playbook/
│
├── notebooks/
│   ├── 01_privacy_quasi_identifier_scan.ipynb
│   ├── 02_deidentification_scorecard.ipynb
│   ├── 03_jurisdictional_compliance_matrix.ipynb
│   └── 04_privacy_risk_to_roi.ipynb
│
├── scripts/
│   ├── privacy_checks.py       # identifier scans, k-anonymity, l-diversity
│   ├── compliance_matrix.py    # jurisdiction-specific rules (HIPAA, GDPR, etc.)
│   └── roi_privacy.py          # ROI model for risk vs. safeguard costs
│
├── data/
│   ├── sample_synthetic.csv    # synthetic dataset with identifiers
│   └── compliance_report.json  # generated metrics
│
├── reports/
│   └── privacy_compliance_report.pdf
```
## 🚀 Quickstart

### Clone & setup
```bash
git clone https://github.com/camontefusco/RWE-Privacy-and-Compliance-Playbook.git
cd RWE-Privacy-and-Compliance-Playbook
pip install -r requirements.txt
```
## ▶️ Run notebooks in order

1. `01_privacy_quasi_identifier_scan.ipynb` → detect direct & quasi-identifiers  
2. `02_deidentification_scorecard.ipynb` → compute k-anonymity, l-diversity  
3. `03_jurisdictional_compliance_matrix.ipynb` → score GDPR/HIPAA/DUA compliance  
4. `04_privacy_risk_to_roi.ipynb` → translate compliance gaps into business impact  

---

## 📤 Outputs

- **CSV/JSON** → identifier scans & compliance metrics in `data/`  
- **PDF report** → leadership-facing compliance report in `reports/privacy_compliance_report.pdf`  

---

## 📊 Privacy & Compliance Scorecard

| Metric            | Meaning                                               |
|-------------------|-------------------------------------------------------|
| Identifier risk   | % of columns containing direct or quasi-identifiers   |
| K-anonymity       | Minimum group size ensuring anonymity                 |
| L-diversity       | Diversity of sensitive attributes within groups       |
| HIPAA compliance  | Alignment with HIPAA Safe Harbor identifiers list     |
| GDPR compliance   | Risk-based anonymization readiness                    |

---

## 💰 ROI of Safeguards

Using `roi_privacy.py`, privacy safeguards are translated into business terms:

- **Cost avoided** → regulatory fines, trial delays, reputational loss  
- **Investment required** → de-ID pipelines, privacy-preserving analytics  
- **Net benefit** → readiness score uplift × trial/market value impact  

---

## 📑 Reports & Policies

- **`reports/privacy_compliance_report.pdf`** → beginner-friendly compliance report  
- **Privacy Checklist (md)** → template for data stewards  
- **Compliance Policy (md)** → organizational safeguards & roles  

---

## 🧭 Talking Points for Leadership

- **Proactive compliance** → cheaper than reactive penalties  
- **Interoperability with governance** → complements RWE scorecard from repo #1  
- **Risk-to-value translation** → links privacy safeguards to ROI  
- **Enablement** → empowers clinical, regulatory, HEOR teams to use compliant RWE  

---

## 📫 Contact

**Carlos Victor Montefusco Pereira, PhD**  
- [LinkedIn](https://www.linkedin.com/in/carlos-montefusco-pereira-dr/)  

│
└── README.md
