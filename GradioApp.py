import os
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import shap
from reportlab.lib.pagesizes import letter
import re
import requests
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.colors import HexColor

api_Key = "Enter Your Gemini API Key Here"


def create_custom_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor("#1a237e"),
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=16,
        textColor=HexColor("#303f9f"),
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=12,
        spaceBefore=6,
        spaceAfter=6,
        leading=14,
    )

    status_style = ParagraphStyle(
        "StatusStyle",
        parent=styles["Normal"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=12,
        textColor=HexColor("#1b5e20"),
        borderWidth=1,
        borderColor=HexColor("#1b5e20"),
        borderPadding=8,
        borderRadius=8,
    )
    bullet_style = ParagraphStyle(
        "BulletStyle", parent=normal_style, leftIndent=20, firstLineIndent=0
    )

    return {
        "title": title_style,
        "heading": heading_style,
        "normal": normal_style,
        "status": status_style,
        "bullet": bullet_style,
    }


def process_markdown_text(text):
    """
    Process markdown-style formatting in text
    """
    # Convert markdown to ReportLab's paragraph markup

    # Handle bold text with double asterisks
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Handle single asterisk for italic
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)

    # Handle bullet points with asterisk or hyphen
    text = re.sub(r"^\s*[\*\-]\s+", "&bull;&nbsp;", text)

    return text


def generate_pdf_report(
    company,
    year,
    predicted_status,
    shap_summary_df,
    filename="Output/BankruptcyReport.pdf",
    top_n=5,
):
    """
    Generate a professionally formatted PDF report for bankruptcy prediction with proper markdown support.
    """
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="FinGuard",
        author="",  
    )

    styles = create_custom_styles()
    story = []
    story.append(Paragraph("Bankruptcy Prediction Report", styles["title"]))
    story.append(Spacer(1, 20))
    status_text = f"Predicted Status: {predicted_status}"
    story.append(Paragraph(status_text, styles["status"]))
    story.append(Spacer(1, 20))
    top_features = (
        shap_summary_df[["Feature", "SHAP Value"]].head(top_n).values.tolist()
    )
    generated_text = call_gemini_api(company, year, predicted_status, top_features)
    sections = generated_text.splitlines()
    in_bullet_list = False

    for section in sections:
        if not section.strip():
            story.append(Spacer(1, 6))
            continue
        processed_text = process_markdown_text(section.strip())

        if "###" in section or "##" in section:
            # Section heading - remove the ### or ## markers and any extra spaces
            heading_text = re.sub(r"^#{2,3}\s*", "", section).strip()
            story.append(Spacer(1, 12))  # Add space before heading
            story.append(Paragraph(heading_text, styles["heading"]))
            story.append(Spacer(1, 6))  # Add space after heading
            in_bullet_list = False

        elif section.strip().startswith(("*", "-")) or "**X" in section:
            # Bullet points or financial ratios
            if not in_bullet_list:
                story.append(Spacer(1, 6))
                in_bullet_list = True
            story.append(Paragraph(processed_text, styles["bullet"]))

        else:
            # Normal paragraph text
            in_bullet_list = False
            if section.startswith("**") and section.endswith("**"):
                # Metadata lines (company, date) or bold text
                story.append(Paragraph(processed_text, styles["normal"]))
            else:
                story.append(Paragraph(processed_text, styles["normal"]))
            story.append(Spacer(1, 6))

    doc.build(story)

with open("Output/voting_classifier_model.pkl", "rb") as f:
    voting_clf = pickle.load(f)
with open("Output/logistic_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("Output/xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("Output/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
X_train = np.load("Output/X_train_data.npy")


feature_names = [
    "X1: Current Assets / Current Liabilities",
    "X2: (Current Assets - Inventories) / Current Liabilities",
    "X3: Cash and Cash Equivalents / Current Liabilities",
    "X4: Total Liabilities / Total Equity",
    "X5: Current Liabilities / Total Liabilities",
    "X6: Equity Share Capital / Fixed Assets",
    "X7: Net Sales / Average Total Assets",
    "X8: Net Sales / Average Current Assets",
    "X9: Gross Profit / Net Sales",
    "X10: Operating Profit / Net Sales",
    "X11: Net Profit / Net Sales",
    "X12: Net Profit / Total Assets",
    "X13: Total Debt / Total Assets",
    "X14: Working Capital / Total Assets",
    "X15: Sales / Total Assets",
    "X16: (Total Assets - Total Assets Previous Year) / Total Assets Previous Year",
    "X17: Net Profit / Net Sales",
    "X18: Cash & Short Term Investment / Total Assets",
    "X19: Cash & Short Term Investment / (Equity Share Capital + Total Liability)",
    "X20: Cash / Total Assets",
    "X21: Cash / Current Liabilities",
    "X22: (Inventory - Inventory Previous Year) / Inventory Previous Year",
    "X23: Inventory / Sales",
    "X24: (Current Liabilities - Cash) / Total Asset",
    "X25: Current Liabilities / Sales",
    "X26: Total Liabilities / Total Assets",
    "X27: Total Liabilities / (Equity Share Capital + Total Liabilities)",
    "X28: Net Income / (Equity Share Capital + Total Liabilities)",
    "X29: Operating Income / Total Assets",
    "X30: Operating Income / Sales",
    "X31: Quick Assets / Current Liabilities",
    "X32: Dividends / Net Income",
    "X33: EBIT / Overall Capital Employed",
    "X34: Net Cash Flow / Revenue",
    "X35: Cash Flow from Operations / Total Debt",
    "X36: EBT / Current Liabilities",
    "X37: EBT / Total Equity",
    "X38: Equity / Total Liabilities",
    "X39: (Gross Profit + Depreciation) / Sales",
    "X40: Quick Assets / Total Assets",
    "X41: Gross Profit / Total Assets",
    "X42: Operating Expenses / Total Liabilities",
    "X43: (Current Assets - Inventory) / Short term Liabilities",
    "X44: Current Assets / Total Liabilities",
    "X45: Short term Liabilities / Total Assets",
    "X46: (Current Assets - Inventory - Short term Liabilities) / (Sales - Gross Profit - Depreciation)",
    "X47: (Net Profit + Depreciation) / Total Liabilities",
    "X48: Working Capital / Fixed Assets",
    "X49: (Total Liabilities - Cash) / Sales",
    "X50: Long term Liability / Equity Capital",
    "X51: Current Assets / Total Assets",
    "X52: Current Liabilities / Assets",
    "X53: Inventory / Working Capital",
    "X54: Inventory / Current Liability",
    "X55: Current Liabilities / Total Liability",
    "X56: Working Capital / Equity Capital",
    "X57: Current Liabilities / Equity Share Capital",
    "X58: Long term Liability / Current Assets",
    "X59: Total Income / Total Expense",
    "X60: Total Expense / Assets",
    "X61: Net Sales / Quick Assets",
    "X62: Sales / Working Capital",
    "X63: Inflation Rate",
    "X64: Unemployment Rate",
    "X65: Real Interest Rate",
    "X66: GDP",
]


def predict_company_status(company, year, *features):
    user_data = pd.DataFrame([features], columns=feature_names)
    user_input_np = user_data.to_numpy()
    prediction = voting_clf.predict(user_input_np)

    # SHAP explanation (keeping your original code)
    explainer_lr = shap.Explainer(lr_model, X_train)
    explainer_xgb = shap.Explainer(xgb_model, X_train)
    explainer_rf = shap.Explainer(rf_model, X_train)

    shap_values_lr = explainer_lr(user_input_np).values
    shap_values_xgb = explainer_xgb(user_input_np).values
    shap_values_rf = explainer_rf(user_input_np).values
    shap_values_ensemble = (shap_values_lr + shap_values_xgb + shap_values_rf) / 3

    status_map = {
        0: "Bankrupt",
        1: "Financial Distress",
        2: "Healthy",
        3: "Probable Bankrupt",
    }
    predicted_status = status_map[prediction[0]]
    shap_summary_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values_ensemble[0, :, prediction[0]],
        }
    ).sort_values(by="SHAP Value", key=abs, ascending=False)

    generate_pdf_report(company, year, predicted_status, shap_summary_df)

    return predicted_status

def call_gemini_api(company, year, predicted_status, top_features):
    global api_Key
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key="
        + api_Key
    )
    headers = {"Content-Type": "application/json"}

    input_text = f"""
Generate a bankruptcy report in the following structured format:
    
## Bankruptcy Prediction Report
    
**Company:** {company}
**Financial Year:** {year}

### Predicted Status
{predicted_status}

### Analysis
Provide an analysis based on the following top contributing financial ratios and SHAP values. Each ratio should include a brief explanation in full sentences and no more than two sentences per ratio.

Top Contributing Financial Ratios:
{', '.join([f"{feature} ({value:.2f})" for feature, value in top_features])}

### Recommendations
Based on the predicted status, provide detailed recommendations. Recommendations should be presented as clear, actionable steps, each in a separate bullet point.

### Disclaimer
End with a standard disclaimer stating that the report is based on model predictions and should be used as a tool for analysis, not as financial advice.
    """

    data = {"contents": [{"parts": [{"text": input_text}]}]}

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)
        return "Error in response from Gemini API."

    response_json = response.json()

    try:
        generated_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        print("Error: Unexpected response format from Gemini API.")
        generated_text = "The Gemini API failed to generate a response."

    return generated_text


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue", secondary_hue="gray", font=gr.themes.GoogleFont("Inter")
    )
) as app:
    gr.Markdown(
        """
        # Financial Health Prediction System
        
        This advanced system analyzes company financial health using 66 key financial ratios and machine learning models. 
        Enter your financial metrics below to receive a comprehensive analysis with a downloadable report.
        """
    )

    # Add input fields for company name and financial year
    company_input = gr.Textbox(label="Company Name")
    year_input = gr.Textbox(label="Financial Year")

    with gr.Tabs():
        with gr.TabItem("Liquidity & Working Capital (1-13)"):
            inputs_1_13 = [
                gr.Slider(
                    0.0, 10.0, value=0.5, label=feature_names[i], info="Enter value"
                )
                for i in range(13)
            ]

        with gr.TabItem("Profitability & Asset Management (14-26)"):
            inputs_14_26 = [
                gr.Slider(
                    0.0, 10.0, value=0.5, label=feature_names[i], info="Enter value"
                )
                for i in range(13, 26)
            ]

        with gr.TabItem("Leverage & Capital Structure (27-39)"):
            inputs_27_39 = [
                gr.Slider(
                    0.0, 10.0, value=0.5, label=feature_names[i], info="Enter value"
                )
                for i in range(26, 39)
            ]

        with gr.TabItem("Operational Efficiency (40-52)"):
            inputs_40_52 = [
                gr.Slider(
                    0.0, 10.0, value=0.5, label=feature_names[i], info="Enter value"
                )
                for i in range(39, 52)
            ]

        with gr.TabItem("Market & Economic Indicators (53-66)"):
            inputs_53_66 = [
                gr.Slider(
                    0.0, 10.0, value=0.5, label=feature_names[i], info="Enter value"
                )
                for i in range(52, 66)
            ]

    with gr.Row():
        with gr.Column():
            predict_btn = gr.Button("Generate Prediction", variant="primary", scale=2)

    with gr.Row():
        output_status = gr.Textbox(label="Predicted Company Status", interactive=False)
        output_file = gr.File(label="Download Detailed Report", interactive=False)

    # Combine company name, year input, and all feature sliders in the correct order
    all_inputs = (
        [company_input, year_input]
        + inputs_1_13
        + inputs_14_26
        + inputs_27_39
        + inputs_40_52
        + inputs_53_66
    )

    # Keep your original prediction function call
    predict_btn.click(
        fn=lambda company, year, *features: (
            predict_company_status(company, year, *features),
            "Output/BankruptcyReport.pdf",
        ),
        inputs=all_inputs,
        outputs=[output_status, output_file],
    )

    gr.Markdown(
        """
        ### About the System
        
        This prediction system uses an ensemble of machine learning models including:
        - Random Forest
        - XGBoost
        - Logistic Regression
        
        The downloadable report includes:
        - Detailed financial health analysis
        - Key contributing factors
        - Recommendations based on the analysis
        """
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use Render's PORT if available
    app.launch(server_name="0.0.0.0", server_port=port)