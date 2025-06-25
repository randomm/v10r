FROM python:3.11-slim

WORKDIR /app

# Copy the mock server script
COPY mock_server.py /app/

# Expose port 80
EXPOSE 80

# Run the server
CMD ["python", "-u", "mock_server.py"] 