# DefectScan Pro🔍
A Metal Surface Quality Control & Reporting System


- **DefectScan** is an end-to-end deep learning application engineered to automate and enhance the process of industrial quality control for metal surfaces. This project was developed to address the inherent limitations of traditional manual inspection methods, which are often time-consuming, costly, and susceptible to human error and inconsistency.

- The system leverages a fine-tuned computer vision model, built upon **MobileNetV2 architecture**, to accurately classify a range of metal sheet defects from digital images with high precision. At its core, the model has been trained on a specialized dataset to reliably distinguish between **six distinct categories** of manufacturing flaws.

- Furthermore, the application is equipped with capabilities for batch processing, allowing operators to analyze **multiple images** simultaneously in a workflow that simulates a real-world, high-throughput production environment. 

- The system's output is an interactive analytics dashboard that provides an immediate, **at a glance summary** of the findings, and includes the automated generation of professional summary reports in **PDF format** for documentation and quality assurance records.

---

## 🎥 Live Demonstration  


👉 A deployed instance of the application is available for interactive use at the following address: 
  **https://defectscan.onrender.com**  

> ⚠️ *Note: The app is hosted on a free Render service. Initial load may take 30–60 seconds due to cold start.*  

---

## ✨ Core Features  

- 🎯 **High-Accuracy Deep Learning Model** – Fine-tuned **MobileNetV2** for defect classification (6 defect types).  
- 📂 **Batch Image Processing** – Simultaneous inspection of multiple images for high-throughput workflows.  
- 📊 **Interactive Analytics Dashboard** – Real-time performance metrics & defect distribution.  
- 🔍 **Detailed Results View** – Image-level prediction with model confidence scores.  
- 📑 **Automated PDF Reporting** – Generate professional-grade inspection reports.  
- ☁️ **Containerized Deployment** – Dockerized & deployed on **Render Cloud** for portability and scalability.  

---

## 🛠️ Tech Stack  

| Component             | Technology / Library       |
|-----------------------|----------------------------|
| **Backend & ML**      | Python, TensorFlow, Keras |
| **Web Framework**     | Streamlit                 |
| **Data Handling**     | Pandas, NumPy             |
| **Image Processing**  | OpenCV, Pillow            |
| **PDF Generation**    | FPDF2                     |
| **Containerization**  | Docker                    |
| **Deployment**        | Render                    |

---

## 📂 Project Structure  

```bash
defect-deployment-app/
│── Dockerfile                      # Defines Docker environment
│── app.py                          # Main Streamlit app
│── metal_defect_model_inference.keras # Trained deep learning model
│── requirements.txt                # Dependencies
```

---

## ⚙️ Setup & Local Execution

- 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/defectscan.git
cd defectscan
```

- 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

- 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

- 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

- Expand defect categories beyond 6 classes

- Add REST API for external system integration

- Support for real-time video stream inspection

- Multi-language support for reports

---

## 👨‍💻 Developed with ❤️ by Kushagra Priyam
