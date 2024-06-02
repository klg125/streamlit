FROM --platform=linux/x86-64 python:3.10-slim

WORKDIR /

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 80
EXPOSE 80

# Command to run the Streamlit app on port 80
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType=none", "--server.port=80"]
