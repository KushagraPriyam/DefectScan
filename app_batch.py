import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="DefectScan ",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: black;
    }
    
    /* Card-like containers */
    .st-emotion-cache-183lzff {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        background-color: #0068C9;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0055A4;
    }
    
    /* Expander styling */
    .st-expander {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
    .st-expander header {
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the clean, inference-ready Keras model."""
    model_path = 'metal_defect_model_INFERENCE.h5' # Recommended to use .keras
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

# --- Constants & Defect Info ---
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches']
DEFECT_EMOJIS = {'Crazing': '🕸️', 'Inclusion': '⚫', 'Patches': '🟤', 'Pitted_Surface': '⚫', 'Rolled-in_Scale': '📏', 'Scratches': '⚡'}
IMG_SIZE = (224, 224)

# --- PDF Report Generation (No changes needed here) ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Metal Surface Defect Analysis Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(results_df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Analysis Summary', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Total Images Analyzed: {len(results_df)}", 0, 1)
    defect_counts = results_df['Prediction'].value_counts()
    pdf.cell(0, 10, "Defect Distribution:", 0, 1)
    for defect, count in defect_counts.items():
        pdf.cell(0, 8, f"  - {defect}: {count} ({count/len(results_df):.1%})", 0, 1)
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Detailed Results', 0, 1)
    for index, row in results_df.iterrows():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"File: {row['Filename']}", 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"  Predicted Defect: {row['Prediction']}", 0, 1)
        pdf.cell(0, 8, f"  Confidence: {row['Confidence']:.2f}%", 0, 1)
        try:
            image = Image.open(row['Image_Bytes']).convert("RGB")
            with BytesIO() as temp_buffer:
                image.save(temp_buffer, format="PNG")
                temp_buffer.seek(0)
                page_width = pdf.w - 2 * pdf.l_margin
                pdf.image(temp_buffer, x=pdf.get_x(), w=page_width*0.8)
        except Exception as e:
            pdf.cell(0, 8, f"  (Could not display image: {e})", 0, 1)
        pdf.ln(5)
    return bytes(pdf.output(dest='S'))

# --- Main Application UI ---
# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ Quality Control System")
    st.markdown("---")
    st.header("About")
    st.info("""
    This application uses a deep learning model (fine-tuned MobileNetV2) to perform automated quality control on metal surfaces. 
    It supports batch processing and generates a downloadable summary report.
    """)
    st.header("How to Use")
    st.markdown("""
    1. **Upload Images:** Drag and drop one or more metal surface images.
    2. **Analyze:** Click the 'Analyze Images' button to start the process.
    3. **Review Dashboard:** Check the summary metrics and detailed results.
    4. **Download Report:** Click the download button for a formal PDF summary.
    """)

# --- Main Page ---
#st.title("DefectScan: Metal Surface Quality Control & Reporting System")

# Title using H1 (largest)
st.markdown("<h1 style='text-align:center;'>DefectScan🛡️ </h1>", unsafe_allow_html=True)

# Subtitle using H3 (smaller than H1)
st.markdown("<h3 style='text-align:center;'>A Metal Surface Quality Control & Reporting System</h3>", unsafe_allow_html=True)

#st.markdown("Upload multiple metal surface images to classify defects and generate a summary report.")
st.markdown("<h6 style='text-align:left;'>Upload multiple metal surface images to classify defects and generate a summary report.</h3>", unsafe_allow_html=True)


model = load_model()

if model is not None:
    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Drag and Drop or Click to Upload Images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button(f"🔬 Analyze {len(uploaded_files)} Images", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0, text="Starting analysis...")

                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file).convert("RGB")
                    image_resized = image.resize(IMG_SIZE)
                    img_array = np.array(image_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    prediction = model.predict(processed_img, verbose=0)
                    pred_index = np.argmax(prediction[0])
                    pred_name = CLASS_NAMES[pred_index]
                    confidence = np.max(prediction) * 100
                    
                    with BytesIO() as buffer:
                        image.save(buffer, format="PNG")
                        results.append({
                            "Filename": uploaded_file.name,
                            "Image": image,
                            "Prediction": pred_name,
                            "Confidence": confidence,
                            "Image_Bytes": BytesIO(buffer.getvalue())
                        })

                    progress_text = f"Analyzing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}"
                    progress_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
                
                progress_bar.empty()
                st.success("Batch analysis complete!")
                st.session_state['analysis_results'] = results

    # --- Display Results Dashboard ---
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        results_df = pd.DataFrame(results)

        st.markdown("---")
        st.header("📊 Analysis Dashboard")
        
        # --- Top Level Metrics ---
        defect_counts = results_df['Prediction'].value_counts()
        most_common_defect = defect_counts.idxmax() if not defect_counts.empty else "None"
        
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Images Analyzed", len(results_df))
        metric2.metric("Defects Detected", len(results_df[results_df['Confidence'] > 50]))
        metric3.metric("Most Common Defect", most_common_defect)
        st.markdown("---")

        # --- Main Dashboard Layout ---
        col1, col2 = st.columns([1, 1.5])

        with col1:
            with st.container(border=True):
                st.subheader("Defect Distribution")
                st.bar_chart(defect_counts, color="#0068C9")
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=generate_pdf_report(results_df),
                    file_name="defect_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        with col2:
            with st.container(border=True):
                st.subheader("Individual Image Results")
                for index, row in results_df.iterrows():
                    emoji = DEFECT_EMOJIS.get(row['Prediction'], '❔')
                    with st.expander(f"{emoji} {row['Filename']}  ➔  **{row['Prediction']}** ({row['Confidence']:.1f}%)"):
                        res_col1, res_col2 = st.columns([1, 2])
                        with res_col1:
                            st.image(row['Image'], width=150)
                        with res_col2:
                            st.metric("Confidence", f"{row['Confidence']:.2f}%")
                            st.progress(int(row['Confidence']))

