---

```markdown
# 🚀 Voyage Analytics

Voyage Analytics is a Machine Learning & MLOps project designed to predict flight prices, recommend travel options, and classify travel-related data using advanced ML models.  
It integrates **MLflow** for experiment tracking, **Docker** for containerization, and provides both **REST API** (FastAPI/Flask) and **Streamlit** interfaces for user interaction.

---

## 🛠 Tech Stack

- **Programming Language:** Python 3.10
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn
- **Natural Language Processing:** nltk
- **MLOps:** mlflow
- **APIs & Web Frameworks:** fastapi, flask
- **Deployment & Serving:** uvicorn, python-multipart
- **Frontend:** streamlit
- **Containerization:** Docker

---

## 📂 Project Structure

```

Voyage\_Analytics/
│
├── MLOPS Project/Data/            # Datasets
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
│
├── Notebooks/                     # Jupyter Notebooks for EDA & Model Training
│
├── saved\_models/                   # Trained Models
│
├── templates/                      # HTML Templates
│
├── flight\_price\_mlflow\.py          # Flight price prediction with MLflow tracking
├── Travel\_app.py                    # Travel recommendation logic
├── Gen\_app.py                       # Generative AI application logic
├── main.py                          # Entry point
│
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Docker multi-container setup
├── Dockerfile.mlflow                # MLflow container setup
├── README.md

````

---

## ⚙️ Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/shubham01s2/Voyage_Analytics.git
cd Voyage_Analytics
````

2️⃣ Create a virtual environment & activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### **Run Locally**

```bash
python main.py
```

### **Run with Docker**

```bash
docker-compose up --build
```

### **Run MLflow Tracking Server**

```bash
docker run -d --name mlflow-server -p 5000:5000 shubham01s2/mlflow-server:v1
```

Access MLflow UI at: [http://localhost:5000](http://localhost:5000)

---

## 🌟 Features

* ✈ **Flight Price Prediction** using XGBoost
* 🏨 **Hotel Recommendation**
* 🗺 **Travel Recommendation Engine**
* 📊 **MLflow Experiment Tracking**
* 🐳 **Dockerized Deployment**
* 🎨 **Streamlit & HTML Frontend**
* ⚡ **FastAPI & Flask API Endpoints**
* 📂 Organized MLOps-ready project structure

---

## 📌 Example

Flight price prediction model with MLflow logging:

```python
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("flight_price_prediction")
```

---

## 👤 Author

**Shubham Sharma**
[LinkedIn Profile](https://www.linkedin.com/in/shubham-sharma611/)

---

## 📜 License

This project is licensed under the MIT License.

```
