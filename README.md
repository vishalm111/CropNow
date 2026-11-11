# NEST Agricultural Prediction System

**IoT-Based Agricultural Intelligence**

A modular platform combining IoT sensor telemetry, data-driven agricultural intelligence, and predictive insights.

---

## Project Structure

```text
.
├── backend/                    # Python FastAPI application
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint, CORS & root endpoint
│   │   ├── routes/             # API routes (e.g. GET /health)
│   │   ├── services/           # Business logic layer
│   │   ├── ml/                 # Machine learning models (future phase)
│   │   └── utils/              # Configuration & helpers
│   ├── requirements.txt        # Python backend dependencies
│   └── README.md               # Backend-specific instructions
│
├── frontend/                   # React + Vite application
│   ├── src/
│   │   ├── services/api.js     # Health check & backend communication
│   │   ├── App.jsx             # Landing page with live connection status
│   │   ├── App.css             # Responsive styling
│   │   ├── index.css           # Base styles
│   │   └── main.jsx            # React root mount
│   ├── package.json            # Frontend dependencies & scripts
│   ├── vite.config.js          # Vite config (configured for port 5173)
│   └── README.md               # Frontend-specific instructions
│
├── .gitignore                  # Git exclusions for Python and Node
└── README.md                   # This project guide
```

---

## Quick Start Guide

You will need **two terminal windows** to run the backend and frontend simultaneously.

### 1. Start the FastAPI Backend

In your first terminal:

```bash
# Navigate to backend
cd backend

# Create virtual environment (first time only)
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server on port 8000
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify backend in your browser:
- Root Overview: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Start the React Frontend

In your second terminal:

```bash
# Navigate to frontend
cd frontend

# Install dependencies (first time only)
npm install

# Start Vite dev server on port 5173
npm run dev
```

Open your browser at:
- **Frontend Dashboard**: [http://localhost:5173/](http://localhost:5173/)

---

## Verification & Features

- **Port Configuration**:
  - React + Vite runs strictly on port `5173`.
  - FastAPI runs on port `8000`.
- **Backend Health Check**:
  - React automatically connects to `GET http://127.0.0.1:8000/health`.
  - When the backend is running, the status badge displays **Connected** (green) along with latency and backend service metadata.
  - When the backend is offline, the status badge displays **Disconnected** (red) with startup hints and a "Check Connection Now" retry button.
  - Heartbeat automatically refreshes every 10 seconds.
- **CORS Support**:
  - FastAPI is pre-configured to accept requests from `http://localhost:5173` and `http://127.0.0.1:5173`.
- **Modular & Clean**:
  - Backend follows a separation of concerns (`routes`, `services`, `ml`, `utils`).
  - No ML models or mock data are implemented at this foundational phase.
