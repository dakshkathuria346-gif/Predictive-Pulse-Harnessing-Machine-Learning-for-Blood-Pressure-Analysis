# Predictive Pulse: Hypertension Risk Assessment
## Machine Learning Project — Blood Pressure Stage Prediction

---

### Project Structure
```
HYPERTENSION PREDICTION/
├── static/
│   └── style.css              # Application styling
├── templates/
│   └── index.html             # Landing page (Flask Jinja2)
├── app.py                     # Flask application backend
├── logreg_model.pkl           # Trained Logistic Regression model
├── hypertension_model_training.ipynb  # Full ML training notebook
└── requirements.txt           # Python dependencies
```

---

### Dataset
- **Source:** Kaggle.com
- **Link:** https://drive.google.com/file/d/1qYvKqg4w_w4blizSVqmLvwY25m7V7N3_/view?usp=sharing
- **Records:** 1,825 → 1,348 (after removing 477 duplicates)
- **Features:** 13 clinical indicators
- **Target:** Hypertension Stages (Normal, Stage-1, Stage-2, Hypertensive Crisis)

---

### Model Performance
| Algorithm         | Accuracy | Assessment    | Status    |
|-------------------|----------|---------------|-----------|
| Decision Tree     | 100%     | Overfitted    | ❌ Rejected |
| Random Forest     | 100%     | Overfitted    | ❌ Rejected |
| SVM               | 100%     | Overfitted    | ❌ Rejected |
| KNN               | 98.1%    | Good          | ⚠ Considered |
| **Logistic Regression** | **95.2%** | **Excellent** | ✅ **Selected** |
| Ridge Classifier  | 90.0%    | Good          | ⚠ Considered |
| Naive Bayes       | 84.4%    | Good          | ⚠ Considered |

**Selected: Logistic Regression** — Best balance of accuracy and generalization.

---

### Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model** (run the Jupyter notebook first):
   ```bash
   jupyter notebook hypertension_model_training.ipynb
   ```
   This will generate `logreg_model.pkl`.

3. **Run the Flask app:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   ```
   http://localhost:5000
   ```

---

### Features
- 13 clinical input features across 4 sections
- Real-time hypertension stage prediction
- Color-coded risk assessment results
- Detailed clinical recommendations per stage
- Confidence score display
- Responsive medical-grade UI

---

### Hypertension Stages
| Stage | Description | Action |
|-------|-------------|--------|
| Normal | BP within healthy range | Maintain lifestyle |
| Stage-1 | Mild elevation | Lifestyle + consultation |
| Stage-2 | Significant elevation | Urgent medical care |
| Hypertensive Crisis | Dangerous elevation | Emergency care |

---

*For educational and research purposes only. Always consult a qualified healthcare professional.*
