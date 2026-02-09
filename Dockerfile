# 1. Start with a lightweight Python "Base"
FROM python:3.9-slim

# 2. Set the "Home" directory inside the container
WORKDIR /app

# 3. Install dependencies first (better for caching!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your project folders into the container
COPY src/ src/
COPY model/ model/

# 5. The command that starts your API when the container boots
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]