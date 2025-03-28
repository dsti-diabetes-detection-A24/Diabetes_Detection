import pandas as pd
import numpy as np
import pickle
import gradio as gr
import time

# Load the models
with open("models/svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

with open("models/ada_boost_model.pkl", "rb") as file:
    ada_model = pickle.load(file)

with open("models/log_model.pkl", "rb") as file:
    log_reg_model = pickle.load(file)

with open("models/knn_model.pkl", "rb") as file:
    knn_model = pickle.load(file)

with open("models/decision_tree_model.pkl", "rb") as file:
    desc_tree_model = pickle.load(file)

with open("models/random_forest.pkl", "rb") as file:
    random_forest = pickle.load(file)


# Loading the scaler for the Logistic Regression model
with open("models/log_scaler.pkl", "rb") as file:
    log_scaler = pickle.load(file)


# Prediction function with input validation
def predictor(
    pregnancies,
    plasmaglucose,
    diastolic_bp,
    triceps_thickness,
    serum_insulin,
    bmi,
    diabetes_pedigree,
    age,
    model,
):
    inputs = [
        pregnancies,
        plasmaglucose,
        diastolic_bp,
        triceps_thickness,
        serum_insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]

    # Check if any input is empty
    if any(not isinstance(val, (int, float)) for val in inputs):
        return "⚠️ Error: Please fill in all fields Correctly."

    if model not in [
        "SVM",
        "KNN",
        "Ada Boost",
        "Decision Tree",
        "Logistic Regression",
        "Random Forest",
    ]:
        return "⚠️ Error: Please choose a valid Model"

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

        # Create the additional derived features to match the 17 expected by the scaler
        bmi_age_ratio = bmi / age if age > 0 else 0
        isi = (
            1.0 / (serum_insulin / plasmaglucose)
            if serum_insulin > 0 and plasmaglucose > 0
            else 0
        )
        bp_age_ratio = diastolic_bp / age if age > 0 else 0
        high_risk_pregnancy = pregnancies * bmi
        glucose_insulin = plasmaglucose * serum_insulin
        pregnancies_age_ratio = pregnancies / age if age > 0 else 0
        metabolic_risk = (plasmaglucose + bmi + serum_insulin) / 3
        pedigree_glucose = diabetes_pedigree * plasmaglucose
        fat_index = bmi * triceps_thickness / 10

        # Create expanded input with all 17 features
        expanded_inputs = [
            pregnancies,
            plasmaglucose,
            diastolic_bp,
            triceps_thickness,
            serum_insulin,
            bmi,
            diabetes_pedigree,
            age,
            bmi_age_ratio,
            isi,
            bp_age_ratio,
            high_risk_pregnancy,
            glucose_insulin,
            pregnancies_age_ratio,
            metabolic_risk,
            pedigree_glucose,
            fat_index,
        ]

        # Scale the expanded inputs
        params = log_scaler.transform([expanded_inputs])

    elif model == "Random Forest":
        chosen_model = random_forest

    prediction = chosen_model.predict(params)

    return "✅ Positive" if prediction[0] > 0.5 else "❌ Negative"


# Clear function
def clear_all():
    return [None] * 9 + [""]


# ------------------------ UI ------------------------

with gr.Blocks(
    css="""
    :root {
        --primary-light: #0077c2;
        --primary-dark: rgb(0, 200, 83);
        --success: #4caf50;
        --warning: #ff9800;
        --danger: #f44336;
        --background-light: #f8fafc;
        --background-dark: #121826;
        --text-light: #334155;
        --text-dark: #e2e8f0;
        --card-radius: 16px;
        --button-radius: 12px;
        --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    body { 
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; 
        transition: background 0.3s ease-in-out, color 0.3s ease-in-out;
        font-size: 16px;
        line-height: 1.5;
        --primary: var(--primary-light); /* Default primary color for light mode */
    }
    
    /* Dark mode base styles */
    body.dark { 
        background-color: var(--background-dark) !important; 
        color: var(--text-dark) !important;
        --primary: var(--primary-dark) !important; /* Green primary color for dark mode */
    }
    
    body.light { 
        background-color: var(--background-light) !important; 
        color: var(--text-light) !important;
        --primary: var(--primary-light); /* Blue primary color for light mode */
    }
    
    /* Medical themed header */
    h1 { 
        color: var(--primary) !important; 
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    /* Accessible text */
    p, label, button {
        font-size: 1.05rem !important;
    }
    
    input, select {
        font-size: 1.05rem !important;
        height: 3rem !important;
    }
    
    /* Enhanced buttons */
    .btn { 
        font-weight: 600;
        padding: 16px 28px; 
        min-height: 72px;
        border-radius: var(--card-radius) !important; 
        border: none; 
        cursor: pointer; 
        transition: all 0.2s ease-out;
        box-shadow: var(--shadow);
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .btn-predict { 
        background-color: var(--primary) !important; 
        color: white; 
    }
    
    .btn-clear { 
        background-color: var(--danger); 
        color: white; 
    }
    
    .btn:hover { 
        transform: scale(1.01);
        filter: brightness(110%);
    }
    
    .btn:active {
        transform: translateY(1px);
    }
    
    /* Medical themed container styles */
    .container { 
        text-align: center; 
        margin-bottom: 24px; 
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Shadcn-style rounded cards */
    .card { 
        background: #ffffff;
        padding: 32px; 
        border-radius: var(--card-radius) !important; 
        box-shadow: var(--shadow);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
    }
    
    /* Fix for dark mode cards */
    body.dark .card,
    body.dark .gr-box,
    body.dark div.gr-form,
    body.dark div.gr-group,
    body.dark .form-group,
    body.dark .gr-input-label,
    body.dark .gr-input {
        background-color: #27272A !important;
        border-color: #3f3f46 !important;
        color: var(--text-dark) !important;
    }
    
    /* Fix for dark mode labels */
    body.dark label,
    body.dark .gr-label {
        color: var(--text-dark) !important;
    }
    
    /* Apply green accent color to headings in dark mode */
    body.dark h3 {
        color: var(--primary-dark) !important;
    }
    
    /* Make model info box use the proper accent color */
    .model-info-box {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: rgba(0, 119, 194, 0.1);
        border-radius: 8px;
    }
    
    body.dark .model-info-box {
        background-color: rgba(0, 200, 83, 0.15) !important;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Medical themed form groups */
    .form-group {
        background-color: #ffffff;
        border-radius: var(--card-radius);
        box-shadow: var(--shadow);
        padding: 18px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
    }
    .form-drop {
        background-color: #ffffff;
        border-radius: var(--card-radius);
        box-shadow: var(--shadow);
        padding-left: 18px;
        padding-right: 18px;
        padding-top: 8px;
        padding-bottom: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Result display */
    .result-box {
        padding: 1rem;
        boreder-radius: var(--card-radius);
        margin-top: 1rem;
    }
    
    /* Medical footer */
    .footer { 
        margin-top: 40px; 
        font-size: 14px; 
        text-align: center;
        padding: 16px;
        border-top: 1px solid #e2e8f0;
    }
    
    body.dark .footer {
        border-color: #3f3f46;
    }
    
    /* Fix Gradio defaults */
    div.gr-block.gr-box {
        border-radius: var(--card-radius) !important;
    }
    
    /* Fix Gradio dropdown and input colors in dark mode */
    body.dark .gr-dropdown button,
    body.dark .gr-dropdown-label,
    body.dark input.gr-text-input {
        background-color: #27272A !important;
        color: var(--text-dark) !important;
        border-color: #3f3f46 !important;
    }
    
    /* Apply green accent color to Gradio elements in dark mode */
    body.dark .gr-button-primary {
        background-color: var(--primary-dark) !important;
    }
"""
) as demo:

    gr.HTML(
        """
        <div class="container">
            <img src="logo.jpg" width="150" onerror="this.style.display='none';">
        </div>

        <div class="container">
            <h1>Women Diabetes Risk Assessment</h1>
            <p style="font-size: 1.2rem; max-width: 700px; margin: 1.5rem auto;">
                This medical tool evaluates diabetes risk based on clinical parameters.
                Complete all fields below, choose model and click <b>Generate Prediction</b> for an accurate prediction. 
            </p>
        </div>
    """
    )

    with gr.Row():
        with gr.Column():
            gr.HTML(
                     "<h3 style='color: var(--primary); font-size: 1.2rem; margin-top: 0;'>Patient Information</h3>"
                 )
            with gr.Group(elem_id="card", elem_classes="form-group"):
                
                age = gr.Number(label="Age (years)", interactive=True)
                pregnancies = gr.Number(label="Number of Pregnancies", interactive=True)
                diabetes_pedigree = gr.Number(
                    label="Diabetes Pedigree Function", interactive=True
                )
                bmi = gr.Number(label="Body Mass Index (kg/m²)", interactive=True)

        with gr.Column():
            gr.HTML(
                    "<h3 style='color: var(--primary); font-size: 1.2rem; margin-top: 0;'>Clinical Measurements</h3>"
                )
            with gr.Group(elem_id="card", elem_classes="form-group"):
                
                plasmaglucose = gr.Number(
                    label="Plasma Glucose (mg/dL)", interactive=True
                )
                serum_insulin = gr.Number(
                    label="Serum Insulin (μU/mL)", interactive=True
                )
                diastolic_bp = gr.Number(
                    label="Diastolic Blood Pressure (mm Hg)", interactive=True
                )
                triceps_thickness = gr.Number(
                    label="Triceps Skinfold Thickness (mm)", interactive=True
                )

    gr.HTML(
        "<h3 style='color: var(--primary); font-size: 1.2rem; margin-top: 0;'>Prediction Settings</h3>"
        )
    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(elem_id="card", elem_classes="form-drop"):
                    model_type = gr.Dropdown(
                        [
                            "SVM",
                            "Logistic Regression",
                            "Decision Tree",
                            "KNN",
                            "Ada Boost",
                            "Random Forest",
                        ],
                        label="Model Name"
                    )
                    
        with gr.Column(scale = 1):
            predict = gr.Button(
                        "📈 Generate Prediction", elem_classes="btn btn-predict"
                    )
            clear = gr.Button("🗑️ Reset Form", elem_classes="btn btn-clear")
    # gr.HTML(
    #         """
    #         <div style="padding: 18px; background-color: rgba(0,119,194,0.1); border-radius: 12px;">
    #             <p style="margin: 0; font-size: 0.9rem !important;">
    #                 <strong>Model Selection Guide:</strong><br>
    #                 • <strong>Random Forest</strong>: Most accurate for complex cases<br>
    #                 • <strong>SVM</strong>: High precision for positive cases<br>
    #                 • <strong>Logistic Regression</strong>: Not the best overall performance<br>
    #             </p>
    #         </div>
    #     """
    #     )
        

    

    

    # gr.HTML(
    #     """
    #     <div class="container">
    #     </div>
    #     """
    # )

    with gr.Group(elem_id="card", elem_classes="form-group"):
        loading_message = gr.Textbox(
            value="",
            label="Progress",
            interactive=False,
            show_label=True,
        )
        output = gr.Textbox(
            label="Assessment Result", interactive=False, show_label=True
        )

    # Button click actions
    predict.click(
        fn=lambda: "⏳ Analyzing clinical parameters... Please wait",
        inputs=[],
        outputs=loading_message,
    ).then(
        fn=predictor,
        inputs=[
            pregnancies,
            plasmaglucose,
            diastolic_bp,
            triceps_thickness,
            serum_insulin,
            bmi,
            diabetes_pedigree,
            age,
            model_type,
        ],
        outputs=output,
    ).then(
        fn=lambda: "", inputs=[], outputs=loading_message
    )

    clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            pregnancies,
            plasmaglucose,
            diastolic_bp,
            triceps_thickness,
            serum_insulin,
            bmi,
            diabetes_pedigree,
            age,
            model_type,
            output,
            loading_message,
        ],
    )

    
    gr.HTML(
        """
        <div class="footer">
            <p>This tool is for educational purposes only and should not replace professional medical advice.</p>
            <p>© 2024/2025 Diabetes Detection Project - DSTI</p>
        </div>
    """
    )

demo.launch(share=True)
