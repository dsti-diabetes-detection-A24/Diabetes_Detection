import pandas as pd
import numpy as np
import pickle
import gradio as gr
import time

# Load the models
with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("ada_boost_model.pkl", "rb") as file:
    ada_model = pickle.load(file)

with open("log_model.pkl", "rb") as file:
    log_reg_model = pickle.load(file)

with open("knn_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

with open("decision_tree_model.pkl", "rb") as file:
    desc_tree_model = pickle.load(file)

# Loading the scaler for the Logistic Regression model
with open("log_scaler.pkl", "rb") as file:
    log_scaler = pickle.load(file)

# Prediction function with input validation
def predictor(pregnancies, plasmaglucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age, model):
    inputs = [pregnancies, plasmaglucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age]
    
    # Check if any input is empty
    if any(not isinstance(val, (int, float)) for val in inputs):
        return "⚠️ Error: Please fill in all fields Correctly."
    
    time.sleep(2)  # Simulate processing time
    params = np.array([inputs])
    if model == "SVM":
        chosen_model = svm_model

    elif model == "KNN":
        chosen_model = knn_model

    elif model == "Ada Boost":
        chosen_model = ada_model

    elif model == "Decision Tree":
        chosen_model = desc_tree_model

    elif model == "Logistic Regression":
        chosen_model = log_reg_model
        inputs = log_scaler.transform([inputs]) # Scaling the inputs
        params = inputs

    prediction = chosen_model.predict(params)
    
    return "✅ Positive" if prediction[0] > 0.5 else "❌ Negative"

# Clear function
def clear_all():
    return [None] * 9 + [""]

# ------------------------ UI ------------------------

with gr.Blocks(css="""
    body { font-family: 'Arial', sans-serif; transition: background 0.3s ease-in-out, color 0.3s ease-in-out; }
    body.dark-mode { background-color: #121212; color: #fff; }
    body.light-mode { background-color: #ffffff; color: #000; }
    .btn { font-weight: bold; padding: 12px 24px; border-radius: 8px; border: none; cursor: pointer; transition: 0.3s; }
    .btn-predict { background-color: #00c853; color: white; }
    .btn-clear { background-color: #d32f2f; color: white; }
    .btn-darkmode { background-color: #444; color: white; }
    .btn:hover { opacity: 0.8; }
    .container { text-align: center; margin-bottom: 15px; }
    .card { background: rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); }
""") as demo:

    # Inject JS to toggle dark mode
    gr.HTML("""
        <script>
            function toggleDarkMode() {
                const body = document.body;
                if (body.classList.contains("dark-mode")) {
                    body.classList.remove("dark-mode");
                    body.classList.add("light-mode");
                } else {
                    body.classList.remove("light-mode");
                    body.classList.add("dark-mode");
                }
            }
        </script>
    """)

    with gr.Row():
        dark_mode_btn = gr.Button("🌙 Dark Mode", elem_classes="btn-darkmode", elem_id="darkmode-btn", variant="secondary")

        dark_mode_btn.click(fn=lambda: gr.HTML("<script>toggleDarkMode();</script>"), inputs=[], outputs=[])

    gr.HTML("""
        <div class="container">
            <img src="logo.jpg" width="120" onerror="this.style.display='none';">
        </div>

        <div class="container">
            <h1 style="color:#00c853; font-size: 24px;">Women Diabetes Prediction App</h1>
            <p style="font-size: 16px; max-width: 600px; margin: auto;">
                This tool predicts diabetes risk based on key health metrics.
                Fill in the details below and click <b>Predict</b>.
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            with gr.Group(elem_id="card"):
                pregnancies = gr.Number(label="Number of Pregnancies", interactive=True)
                plasmaglucose = gr.Number(label="Plasma Glucose (mg/dL)", interactive=True)
                diastolic_bp = gr.Number(label="Diastolic Blood Pressure (mm Hg)", interactive=True)
                triceps_thickness = gr.Number(label="Triceps Skinfold Thickness (mm)", interactive=True)

        with gr.Column():
            with gr.Group(elem_id="card"):
                serum_insulin = gr.Number(label="Serum Insulin (μU/mL)", interactive=True)
                bmi = gr.Number(label="Body Mass Index (kg/m²)", interactive=True)
                diabetes_pedigree = gr.Number(label="Diabetes Pedigree Function", interactive=True)
                age = gr.Number(label="Age (years)", interactive=True)
    with gr.Row():
        model_type = gr.Dropdown(["SVM", "Logistic Regression", "Decision Tree", "KNN", "Ada Boost"], label="Model Name")

    with gr.Row():
        predict = gr.Button("🔍 Predict", elem_classes="btn btn-predict")
        clear = gr.Button("🧹 Clear", elem_classes="btn btn-clear")

    loading_message = gr.Textbox(value="", label="", interactive=False, show_label=False)
    output = gr.Textbox(label="Prediction Result", lines=2, interactive=False)

    # Button click actions
    def predict_with_loading(*args):
        return "⏳ Analyzing... Please wait"

    predict.click(
        fn=predict_with_loading,
        inputs=[],
        outputs=loading_message
    ).then(
        fn=predictor,
        inputs=[pregnancies, plasmaglucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age, model_type],
        outputs=output
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=loading_message
    )

    clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[pregnancies, plasmaglucose, diastolic_bp, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age, output, loading_message]
    )


demo.launch(share=True)
