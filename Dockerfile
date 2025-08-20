# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the static data (FAISS index and URLs file)
COPY faiss_index/ /app/faiss_index/
COPY urls.txt /app/urls.txt

# Copy the application code
COPY . /app

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application with Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]