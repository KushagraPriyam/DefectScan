import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="DefectScan Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS for Beautiful UI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%);
        color: #FAFAFA;
    }
    
    /* Headers with Gradient Text */
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* Sidebar with glassmorphism effect */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.9) 0%, rgba(26, 31, 46, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.9) 0%, rgba(45, 48, 59, 0.9) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 15px;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .stButton > button:hover:before {
        left: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* Enhanced File Uploader */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.5) 0%, rgba(45, 48, 59, 0.5) 100%);
        border: 2px dashed #667eea;
        border-radius: 20px;
        padding: 40px;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.7) 0%, rgba(45, 48, 59, 0.7) 100%);
    }
    
    /* Enhanced Expanders */
    .st-expander {
        background: linear-gradient(135deg, rgba(45, 48, 59, 0.6) 0%, rgba(38, 39, 48, 0.6) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    .st-expander:hover {
        border-color: rgba(102, 126, 234, 0.4);
    }
    .st-expander header {
        color: #FAFAFA;
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% 100%;
        animation: progress-animation 2s linear infinite;
    }
    
    @keyframes progress-animation {
        0% { background-position: 0% 0%; }
        100% { background-position: 200% 0%; }
    }

    /* Enhanced Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(38, 39, 48, 0.7) 0%, rgba(45, 48, 59, 0.7) 100%);
        border-radius: 15px;
        padding: 8px;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #bbb;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Container styling */
    .main-container {
        background: rgba(20, 24, 33, 0.5);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Card hover effects */
    .image-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 15px;
        overflow: hidden;
    }
    .image-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    /* Enhanced notification styles */
    .success-notification {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(129, 199, 132, 0.2) 100%);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Loading animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the clean, inference-ready Keras model."""
    model_path = 'metal_defect_model_inference.keras'
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model from '{model_path}': {e}")
        return None

# --- Constants & Defect Info ---
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches']
DEFECT_EMOJIS = {'Crazing': '🕸️', 'Inclusion': '⚫', 'Patches': '🟤', 'Pitted_Surface': '🕳️', 'Rolled-in_Scale': '📏', 'Scratches': '⚡'}
DEFECT_COLORS = {'Crazing': '#FF6B6B', 'Inclusion': '#4ECDC4', 'Patches': '#45B7D1', 'Pitted_Surface': '#96CEB4', 'Rolled-in_Scale': '#FECA57', 'Scratches': '#DDA0DD'}
DEFECT_DESCRIPTIONS = {'Crazing': 'Network of fine cracks on the surface', 'Inclusion': 'Foreign material embedded in the metal', 'Patches': 'Irregular colored areas on the surface', 'Pitted_Surface': 'Small holes or pits in the surface', 'Rolled-in_Scale': 'Scale pressed into the surface during rolling', 'Scratches': 'Linear marks on the surface'}
IMG_SIZE = (224, 224)

# --- Initialize Session State ---
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'previous_file_count' not in st.session_state:
    st.session_state.previous_file_count = 0

# --- PDF Report Generation ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'DefectScan Pro - Quality Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | DefectScan Pro', 0, 0, 'C')

def generate_pdf_report(results_df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Total Images Analyzed: {len(results_df)}", 0, 1)
    pdf.cell(0, 8, f"Average Confidence: {results_df['Confidence'].mean():.2f}%", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Defect Distribution', 0, 1)
    pdf.set_font('Arial', '', 11)
    defect_counts = results_df['Prediction'].value_counts()
    for defect, count in defect_counts.items():
        percentage = (count/len(results_df))*100
        pdf.cell(0, 8, f"  - {defect}: {count} occurrences ({percentage:.1f}%)", 0, 1)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detailed Analysis Results', 0, 1)
    pdf.ln(5)
    for index, row in results_df.iterrows():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Image: {row['Filename']}", 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 6, f"  Defect Type: {row['Prediction']}", 0, 1)
        pdf.cell(0, 6, f"  Confidence: {row['Confidence']:.2f}%", 0, 1)
        pdf.cell(0, 6, f"  Description: {DEFECT_DESCRIPTIONS.get(row['Prediction'], 'N/A')}", 0, 1)
        pdf.ln(3)
    return bytes(pdf.output(dest='S'))

# --- Enhanced Visualization Functions ---
def create_pie_chart(results_df):
    defect_counts = results_df['Prediction'].value_counts()
    colors = [DEFECT_COLORS.get(defect, '#808080') for defect in defect_counts.index]
    fig = go.Figure(data=[go.Pie(
        labels=defect_counts.index, 
        values=defect_counts.values, 
        hole=0.4, 
        marker=dict(colors=colors, line=dict(color='#0E1117', width=2)),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        pull=[0.1 if i == 0 else 0 for i in range(len(defect_counts))]
    )])
    fig.update_layout(
        title={'text': '🎯 Defect Distribution Analysis', 'x': 0.5, 'font': {'size': 24}},
        showlegend=True, height=450,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="white", size=14),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(t=80, b=40, l=40, r=150)
    )
    return fig

def create_confidence_chart(results_df):
    fig = go.Figure(data=[go.Bar(
        x=results_df['Filename'],
        y=results_df['Confidence'],
        marker=dict(
            color=results_df['Confidence'],
            colorscale='Viridis',
            colorbar=dict(title="Confidence %", thickness=15),
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=results_df['Confidence'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
    )])
    fig.update_layout(
        title={'text': '📊 Detection Confidence by Image', 'x': 0.5, 'font': {'size': 24}},
        xaxis_title="Image File", yaxis_title="Confidence (%)",
        height=450,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="white", size=14),
        xaxis=dict(tickangle=-45, color="white", gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[0, 105], gridcolor='rgba(255,255,255,0.1)', color="white"),
        margin=dict(t=80, b=100)
    )
    return fig

def create_radar_chart(results_df):
    defect_counts = results_df['Prediction'].value_counts()
    all_defects = pd.Series(0, index=CLASS_NAMES)
    all_defects.update(defect_counts)
    
    fig = go.Figure(data=go.Scatterpolar(
        r=all_defects.values,
        theta=all_defects.index,
        fill='toself',
        marker=dict(color='#667eea', size=10),
        line=dict(color='#667eea', width=3),
        hovertemplate='<b>%{theta}</b><br>Count: %{r}<extra></extra>',
        name='Defect Count'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[max(all_defects.values) * 0.5] * len(all_defects),
        theta=all_defects.index,
        fill='toself',
        fillcolor='rgba(118, 75, 162, 0.1)',
        line=dict(color='rgba(118, 75, 162, 0.3)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(all_defects.values) + 1] if max(all_defects.values) > 0 else [0, 1],
                gridcolor='rgba(255,255,255,0.1)',
                gridwidth=1,
                tickfont=dict(color='white', size=10)
            ),
            angularaxis=dict(color="white", gridcolor='rgba(255,255,255,0.1)')
        ),
        showlegend=False,
        title={'text': '🎯 Defect Type Radar Analysis', 'x': 0.5, 'font': {'size': 24}},
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="white", size=14),
        margin=dict(t=80, b=40)
    )
    return fig

def create_time_series_chart():
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    defect_data = [{'Date': date, 'Defect': defect, 'Count': random.randint(0, 10)} for date in dates for defect in CLASS_NAMES]
    df_time = pd.DataFrame(defect_data)
    
    fig = px.line(df_time, x='Date', y='Count', color='Defect',
                  title='📈 Defect Trends Over Time (Simulated)',
                  color_discrete_map=DEFECT_COLORS,
                  line_shape='spline')
    
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        height=450,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="white", size=14),
        title={'x': 0.5, 'font': {'size': 24}},
        hovermode='x unified',
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Defect Count'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Date'),
        legend=dict(title='Defect Type', bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
        margin=dict(t=80)
    )
    return fig

# --- Main Application ---
st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'><span class='gradient-text'>DefectScan Pro</span> 🔍</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 1.3rem; margin-top: 0;'>A Metal Surface Quality Control & Reporting System</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: none; height: 3px; background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent); border-radius: 3px; margin: 2rem 0;'>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
        <h2 style='color: white; margin: 0; font-size: 1.8rem;'>🛡️ DefectScan Pro</h2>
        <p style='color: rgba(255,255,255,0.95); margin: 8px 0 0 0; font-size: 0.95rem;'>Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Analyzed", st.session_state.total_analyzed, 
                  delta=f"+{len(st.session_state.get('analysis_results', []))}" if 'analysis_results' in st.session_state else None)
    with col2:
        st.metric("Session Active", "✅ Live")
    
    st.markdown("---")
    st.markdown("### ✨ Features")
    features = [
        "🤖 AI-Powered Detection",
        "📈 Real-time Analytics",
        "📊 Interactive Visualizations",
        "📄 PDF Report Generation",
        "🎯 6 Defect Types",
        "⚡ Batch Processing",
        "🔄 Auto-clear on New Upload"
    ]
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.markdown("---")
    with st.expander("ℹ️ About DefectScan Pro", expanded=False):
        st.markdown("**DefectScan Pro** uses state-of-the-art deep learning (MobileNetV2) to identify and classify metal surface defects.")
        st.markdown("**Supported Defect Types:**")
        for defect, desc in DEFECT_DESCRIPTIONS.items():
            st.markdown(f"**{DEFECT_EMOJIS.get(defect, '🔍')} {defect}:** {desc}")
    
    with st.expander("📖 Usage Guidelines", expanded=False):
        st.markdown("""
        1. **Upload** one or more images
        2. **Click** 'Analyze Images'
        3. **View** instant analytics
        4. **Download** comprehensive PDF report
        
        *Graphs auto-clear when you remove/change images*
        """)
    
    # Clear data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        if 'analysis_results' in st.session_state:
            del st.session_state['analysis_results']
        st.rerun()

# --- Model and Main Interface ---
model = load_model()
if model is not None:
    with st.container():
        uploaded_files = st.file_uploader(
            "Drag and drop your images here or click to browse",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            help="Support for JPG, JPEG, PNG, and BMP formats • Graphs auto-clear on file change",
            key="file_uploader"
        )
        
        # Clear results if files are removed or changed
        current_file_count = len(uploaded_files) if uploaded_files else 0
        if current_file_count == 0 and st.session_state.previous_file_count > 0:
            # Files were removed
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.session_state.previous_file_count = 0
            st.rerun()  # Force rerun to clear the display immediately
        elif current_file_count != st.session_state.previous_file_count and current_file_count > 0:
            # Different files uploaded
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            st.session_state.previous_file_count = current_file_count
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} image(s) ready for analysis")
            
            analyze_button = st.button(
                f"🚀 Analyze {len(uploaded_files)} Image(s)",
                type="primary",
                use_container_width=True
            )
            
            if analyze_button:
                results = []
                progress_bar = st.progress(0, text="🔄 Initializing analysis...")
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.markdown(f"<p style='color: #667eea; font-weight: 600;'>🔍 Analyzing: {uploaded_file.name}</p>", unsafe_allow_html=True)
                    
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
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Image_Bytes": BytesIO(buffer.getvalue())
                        })
                    
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress, text=f"⚡ Progress: {int(progress * 100)}%")
                
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['analysis_results'] = results
                st.session_state.total_analyzed += len(results)
                st.session_state.analysis_history.extend(results)
                st.session_state.previous_file_count = len(uploaded_files)
                
                st.success(f"✅ Successfully analyzed {len(results)} image(s)!")
                st.balloons()
        else:
            # Show placeholder when no files are uploaded
            st.markdown("""
            <div style='text-align: center; padding: 60px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        border-radius: 20px; border: 2px dashed rgba(102, 126, 234, 0.3); margin: 20px 0;'>
                <h3 style='color: #667eea;'>👆 Upload images to begin analysis</h3>
                <p style='color: #a0aec0;'>Drag and drop or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
    
    # --- Results Display (Only show if analysis_results exist) ---
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        results = st.session_state['analysis_results']
        results_df = pd.DataFrame(results)
        
        st.markdown("<h2 style='text-align: center; color: #FAFAFA; margin: 40px 0; font-size: 2.5rem;'>📊 Analysis Results Dashboard</h2>", unsafe_allow_html=True)
        
        # Metrics with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        avg_confidence = results_df['Confidence'].mean()
        defect_counts = results_df['Prediction'].value_counts()
        most_common = defect_counts.idxmax() if not defect_counts.empty else "None"
        high_confidence = len(results_df[results_df['Confidence'] > 90])
        
        with col1:
            st.metric("📸 Images Analyzed", len(results_df), delta=f"Session: {st.session_state.total_analyzed}")
        with col2:
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%", delta=f"{avg_confidence-50:.1f}% vs baseline")
        with col3:
            st.metric("🔝 Most Common", f"{DEFECT_EMOJIS.get(most_common, '❓')} {most_common}")
        with col4:
            st.metric("✨ High Confidence", high_confidence, delta=f"{(high_confidence/len(results_df)*100):.0f}% of total")
        
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent); margin: 40px 0;'>", unsafe_allow_html=True)
        
        # Enhanced Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Confidence", "🎯 Radar View", "🖼️ Details"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_pie_chart(results_df), use_container_width=True)
            with col2:
                defect_counts = results_df['Prediction'].value_counts()
                fig_bar = go.Figure(data=[go.Bar(
                    x=defect_counts.index,
                    y=defect_counts.values,
                    marker=dict(
                        color=[DEFECT_COLORS.get(defect, '#808080') for defect in defect_counts.index],
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ),
                    text=defect_counts.values,
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )])
                fig_bar.update_layout(
                    title={'text': '📈 Defect Count Distribution', 'x': 0.5, 'font': {'size': 20}},
                    xaxis_title="Defect Type", yaxis_title="Count",
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif", color="white", size=14),
                    xaxis=dict(tickangle=-45, color="white", gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color="white"),
                    margin=dict(t=80, b=100)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_confidence_chart(results_df), use_container_width=True)
            
            # Additional confidence statistics
            st.markdown("### 📊 Confidence Statistics")
            conf_stats = pd.DataFrame({
                'Metric': ['Average', 'Minimum', 'Maximum', 'Standard Deviation'],
                'Value': [
                    f"{results_df['Confidence'].mean():.2f}%",
                    f"{results_df['Confidence'].min():.2f}%",
                    f"{results_df['Confidence'].max():.2f}%",
                    f"{results_df['Confidence'].std():.2f}%"
                ]
            })
            st.dataframe(conf_stats, use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_radar_chart(results_df), use_container_width=True)
            
            # Defect summary table
            st.markdown("### 🎯 Defect Summary")
            summary_df = pd.DataFrame({
                'Defect Type': defect_counts.index,
                'Count': defect_counts.values,
                'Percentage': (defect_counts.values / len(results_df) * 100).round(1),
                'Emoji': [DEFECT_EMOJIS.get(defect, '❓') for defect in defect_counts.index]
            })
            st.dataframe(summary_df, use_container_width=True)
        
        
        
        with tab4:
            st.markdown("### 🖼️ Detailed Results")
            
            # Display images in a grid with results
            cols = st.columns(3)
            for idx, result in enumerate(results):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"**{result['Filename']}**")
                        # Version-safe image display
                        try:
                            st.image(result['Image'], use_column_width=True)
                        except:
                            st.image(result['Image'])
                        
                        # Result card
                        defect = result['Prediction']
                        confidence = result['Confidence']
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(38, 39, 48, 0.8) 0%, rgba(45, 48, 59, 0.8) 100%);
                                    border: 1px solid {DEFECT_COLORS.get(defect, '#808080')};
                                    border-radius: 12px; padding: 15px; margin: 10px 0;'>
                            <h4 style='color: {DEFECT_COLORS.get(defect, '#FAFAFA')}; margin: 0;'>
                                {DEFECT_EMOJIS.get(defect, '❓')} {defect}
                            </h4>
                            <p style='color: #a0aec0; margin: 5px 0;'>Confidence: {confidence:.1f}%</p>
                            <p style='color: #d0d0d0; margin: 5px 0; font-size: 0.9em;'>
                                {DEFECT_DESCRIPTIONS.get(defect, 'N/A')}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # PDF Download Section
        st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent); margin: 40px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📄 Generate Report")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📄 Download PDF Report", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating comprehensive PDF report..."):
                    pdf_bytes = generate_pdf_report(results_df)
                    st.download_button(
                        label="⬇️ Download Report",
                        data=pdf_bytes,
                        file_name=f"DefectScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Report generated successfully!")
else:
    st.error("❌ Model could not be loaded. Please check if 'metal_defect_model_inference.keras' exists in the directory.")
