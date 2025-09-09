# Use the EXACT Python version from your training environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install the system library needed by OpenCV
RUN apt-get update && apt-get install -y libgl1

# Copy the requirements file first
COPY requirements.txt ./

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the other project files
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

