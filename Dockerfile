# Use official Python image
FROM python:3.11.5

# Set working directory inside the container
WORKDIR /app

# Copy all files from project to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit’s default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "model.py", "--server.port=8501", "--server.address=0.0.0.0"]