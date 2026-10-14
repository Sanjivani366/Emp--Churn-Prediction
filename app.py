import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Gaussian Naive Bayes model safely
# Change 'naive.pkl' to match your actual filename if needed
model_path = os.path.join(os.path.dirname(__file__), 'naive.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'naive.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Modern and responsive Dark HTML/CSS Template
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .card {
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            color: #f8fafc;
        }
        .form-control, .form-select {
            background-color: #1f2937;
            border: 1px solid #4b5563;
            color: #f8fafc;
        }
        .form-control:focus, .form-select:focus {
            background-color: #1f2937;
            color: #f8fafc;
            border-color: #6366f1;
            box-shadow: 0 0 0 0.25rem rgba(99, 102, 241, 0.25);
        }
        .form-control::placeholder {
            color: #9ca3af;
        }
        .btn-custom {
            background-color: #6366f1;
            color: white;
            transition: background 0.3s ease;
        }
        .btn-custom:hover {
            background-color: #4f46e5;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="card p-4 p-md-5">
                    <div class="text-center mb-4">
                        <h2 class="fw-bold text-light">Employee Predictor</h2>
                        <p class="text-secondary">Enter employee details to run predictions</p>
                    </div>

                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Education</label>
                            <select class="form-select" name="Education" required>
                                <option value="" disabled selected>Select Education</option>
                                <option value="Bachelors">Bachelors</option>
                                <option value="Masters">Masters</option>
                                <option value="PHD">PHD</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">City</label>
                            <select class="form-select" name="City" required>
                                <option value="" disabled selected>Select City</option>
                                <option value="Bangalore">Bangalore</option>
                                <option value="Pune">Pune</option>
                                <option value="New Delhi">New Delhi</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Joining Year</label>
                            <input type="number" class="form-control" name="JoiningYear" placeholder="e.g., 2018" required>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Gender</label>
                            <select class="form-select" name="Gender" required>
                                <option value="" disabled selected>Select Gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Ever Benched?</label>
                            <select class="form-select" name="EverBenched" required>
                                <option value="" disabled selected>Select Option</option>
                                <option value="No">No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Experience In Current Domain</label>
                            <input type="number" class="form-control" name="ExperienceInCurrentDomain" placeholder="e.g., 3" required>
                        </div>

                        <div class="mb-4">
                            <label class="form-label fw-semibold">Age</label>
                            <input type="number" class="form-control" name="Age" placeholder="e.g., 28" required>
                        </div>

                        <button type="submit" class="btn btn-custom w-100 py-2 fw-bold">Predict</button>
                    </form>

                    {% if prediction is not none %}
                        <div class="alert mt-4 text-center {% if prediction == 1 %}alert-danger bg-danger text-white border-0{% else %}alert-success bg-success text-white border-0{% endif %}" role="alert">
                            <h4 class="alert-heading fw-bold">Prediction Result</h4>
                            <p class="mb-0 fs-5">
                                {% if prediction == 1 %}
                                    The model predicts: <strong>Likely to leave / Positive Class (1)</strong>
                                {% else %}
                                    The model predicts: <strong>Likely to stay / Negative Class (0)</strong>
                                {% endif %}
                            </p>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract form values matching the model features
            education = request.form['Education']
            city = request.form['City']
            joining_year = float(request.form['JoiningYear'])
            gender = request.form['Gender']
            ever_benched = request.form['EverBenched']
            experience = float(request.form['ExperienceInCurrentDomain'])
            age = float(request.form['Age'])
            
            # Map categorical inputs to numeric values
            edu_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
            city_map = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
            gender_map = {'Male': 0, 'Female': 1}
            bench_map = {'No': 0, 'Yes': 1}
            
            features = np.array([[
                edu_map.get(education, 0),
                city_map.get(city, 0),
                joining_year,
                gender_map.get(gender, 0),
                bench_map.get(ever_benched, 0),
                experience,
                age
            ]])
            
            pred = model.predict(features)
            prediction = int(pred[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    return render_template_string(TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
