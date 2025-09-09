# Step 1: Start with a specific, stable version of Python.
# This solves the Python version mismatch error we saw on Streamlit Cloud.
FROM python:3.10-slim

# Step 2: Set the working directory inside the container.
# This is like creating a project folder on the cloud server.
WORKDIR /app

# Step 3: Install system libraries needed by Python packages.
# This command installs the library that OpenCV needs to run.
RUN apt-get update && apt-get install -y libgl1

# Step 4: Copy the requirements file into the container.
COPY requirements.txt ./

# Step 5: Install all the Python libraries listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy all the other project files (app_batch.py, model file) into the container.
COPY . .

# Step 7: Tell the cloud platform that the application will be accessible on port 8501.
EXPOSE 8501

# Step 8: Define the command that will start your Streamlit app.
CMD ["streamlit", "run", "app_batch.py", "--server.port=8501", "--server.address=0.0.0.0"]

