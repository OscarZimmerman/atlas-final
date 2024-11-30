# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy analysis code
COPY analysis/ ./analysis/
COPY run.sh .



# Create output directory
RUN mkdir -p /app/output

# Run the analysis
CMD ["bash", "run.sh"]
