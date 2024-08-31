FROM python:3.11-slim

# Set the working directory
WORKDIR /code/app

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

# Copy your application code
COPY ./app /code/app

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
