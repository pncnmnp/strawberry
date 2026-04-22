FROM python:3.12-slim
RUN pip install --no-cache-dir numpy pandas matplotlib requests
WORKDIR /sandbox
COPY server.py .
CMD ["python", "server.py"]
