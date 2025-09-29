# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file (if exists) or create one
COPY requirements.txt* .

# If requirements.txt doesn't exist, create a default one
RUN if [ ! -f requirements.txt ]; then \
    echo "fastapi==0.95.0" > requirements.txt && \
    echo "uvicorn==0.21.1" >> requirements.txt && \
    echo "joblib==1.2.0" >> requirements.txt && \
    echo "scikit-learn==1.2.2" >> requirements.txt && \
    echo "pydantic==1.10.7" >> requirements.txt; \
    fi

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY mini_project.py .
COPY regression.joblib .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "mini_project:app", "--host", "0.0.0.0", "--port", "8000"]