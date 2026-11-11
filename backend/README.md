# NEST Agricultural Prediction System - Backend

FastAPI backend service powering the **NEST Agricultural Prediction System** (IoT-Based Agricultural Intelligence).

---

## Directory Structure

```text
backend/
├── app/
│   ├── main.py        # FastAPI initialization, CORS, and root endpoint
│   ├── routes/        # API endpoints (e.g., /health)
│   ├── services/      # Business logic services
│   ├── ml/            # Machine learning models & inference (future phase)
│   └── utils/         # Configuration and helper utilities
├── requirements.txt   # Python package dependencies
└── README.md          # Setup and run guide
```

---

## Prerequisites

- **Python**: version 3.10 or higher (recommended: Python 3.11+)
- **pip** and `venv`

---

## Setup Instructions

### 1. Create a Python Virtual Environment

Navigate to the `backend/` directory:

```bash
cd backend
```

Create a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv

# macOS / Linux
python3 -m venv venv
```

### 2. Activate the Virtual Environment

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Backend

Start the development server with Uvicorn:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, access:
- **Root Endpoint**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Documentation (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## CORS Configuration

By default, the backend allows origins:
- `http://localhost:5173`
- `http://127.0.0.1:5173`
- `http://localhost:3000`

Configured in `app/utils/config.py`.
