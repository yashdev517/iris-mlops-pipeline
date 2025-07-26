# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only requirements first (this rarely changes)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Now copy rest of the project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run API
CMD ["python", "src/predict_api.py"]