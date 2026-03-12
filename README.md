# 🍳 Recipe Recommender System

A **machine learning based recommendation system** that suggests recipes based on user preferences using both **content-based filtering and collaborative filtering** techniques.

This project combines **TF-IDF, cosine similarity, and LightFM** to provide personalized recipe recommendations from a dataset of 1000+ recipes.

---

## 🚀 Features

* Hybrid recommendation system
* Content-based filtering using **TF-IDF + cosine similarity**
* Collaborative filtering using **LightFM**
* Interactive web interface
* Flask API for serving recommendations
* Personalized recipe suggestions based on user preferences

---

## 🧠 How It Works

1️⃣ Recipe dataset is processed and cleaned

2️⃣ Ingredients and recipe descriptions are converted into vectors using **TF-IDF**

3️⃣ Similar recipes are identified using **cosine similarity**

4️⃣ Collaborative filtering using **LightFM** improves personalization

5️⃣ A **Flask API** serves the recommendations to the web interface

---

## 🛠 Tech Stack

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* LightFM
* TF-IDF Vectorizer
* Cosine Similarity

**Data Processing**

* Pandas
* NumPy

**Backend**

* Flask

**Frontend**

* HTML
* CSS
* JavaScript

---

## 📂 Project Structure
```
recipe-recommender/
│
├── backend/                # Backend logic and ML pipeline
│   │
│   ├── data/               # Dataset and preprocessing files
│   │
│   ├── model/              # Trained models and recommendation logic
│   │
│   ├── utils/              # Helper functions and utilities
│   │
│   ├── testing/            # Testing scripts for the system
│   │
│   └── main.py / app.py    # Backend entry point
│
├── frontend/               # User interface
│
├── .github/                # GitHub workflows / configs
│
└── README.md               # Project documentation
```
---

## 📊 Model Performance

Evaluation metrics for the hybrid recommendation model:

* **HitRate@10:** 0.8
* **Precision@10:** 0.07
* **Recall@10:** 1.0

These metrics indicate strong recommendation accuracy for top results.

---

## ⚙️ Installation

Clone the repository

```
git clone https://github.com/Kritikaa09/recipe-recommender.git
```

Navigate into the project directory

```
cd recipe-recommender
```

Install required dependencies

```
pip install -r requirements.txt
```

Run the application

```
python app.py
```

---

## 📸 Demo

(Add screenshots or demo GIF here)

Example:

![App Screenshot](screenshots/demo.png)

---

## 🔮 Future Improvements

* Add deep learning based recommendation models
* Integrate user login and personalization
* Deploy the application on cloud platforms
* Improve UI with React or Next.js
* Add nutrition-based filtering

---

## 👩‍💻 Author

**Kritika Dadheech**

B.Tech Computer Science (Data Science)
Swami Keshvanand Institute of Technology, Jaipur

GitHub: https://github.com/Kritikaa09
LinkedIn: https://linkedin.com/in/kritika-dadheech9

---

⭐ If you found this project interesting, consider giving it a **star**!
