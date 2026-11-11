# NEST Agricultural Prediction System - Frontend

React + Vite frontend dashboard for the **NEST Agricultural Prediction System** (IoT-Based Agricultural Intelligence).

---

## Features

- **Framework**: React 19 + Vite 6
- **Server Port**: Configured strictly to run on port `5173`
- **Backend Health Integration**: Automatically monitors and verifies connection to the FastAPI backend at `/health`
- **Live Status Display**: Displays visual connection badge (`Connected` / `Disconnected`), latency, and backend details
- **Auto-polling**: Checks health status every 10 seconds with instant manual refresh option

---

## Getting Started

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment (Optional)

By default, the frontend connects to `http://127.0.0.1:8000`. You can configure a custom backend URL by creating a `.env` file:

```bash
cp .env.example .env
```

`.env` content:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Run Development Server

```bash
npm run dev
```

The frontend will be available at [http://localhost:5173/](http://localhost:5173/).

### 4. Build for Production

```bash
npm run build
```
