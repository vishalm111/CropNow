/**
 * API service for communicating with the FastAPI backend.
 */

let activeBaseUrl =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Perform a fetch with automatic fallback between 127.0.0.1 and localhost
 * to handle IPv4/IPv6 loopback differences in Windows environments.
 */
async function safeFetch(endpoint, options = {}) {
  try {
    const res = await fetch(`${activeBaseUrl}${endpoint}`, options);
    return res;
  } catch (err) {
    if (activeBaseUrl.includes('127.0.0.1:8000')) {
      const fallbackUrl = activeBaseUrl.replace('127.0.0.1:8000', 'localhost:8000');
      try {
        const fallbackRes = await fetch(`${fallbackUrl}${endpoint}`, options);
        activeBaseUrl = fallbackUrl;
        return fallbackRes;
      } catch {
        throw err;
      }
    } else if (activeBaseUrl.includes('localhost:8000')) {
      const fallbackUrl = activeBaseUrl.replace('localhost:8000', '127.0.0.1:8000');
      try {
        const fallbackRes = await fetch(`${fallbackUrl}${endpoint}`, options);
        activeBaseUrl = fallbackUrl;
        return fallbackRes;
      } catch {
        throw err;
      }
    }
    throw err;
  }
}

/**
 * Checks backend health by querying the /health endpoint.
 *
 * @param {number} timeoutMs - Request timeout in milliseconds (default 5000ms)
 * @returns {Promise<{ isConnected: boolean, data: object | null, error: string | null, latencyMs: number }>}
 */
export async function checkBackendHealth(timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const startTime = performance.now();

  try {
    const response = await safeFetch('/health', {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal: controller.signal,
    });

    const latencyMs = Math.round(performance.now() - startTime);
    clearTimeout(timeoutId);

    if (!response.ok) {
      return {
        isConnected: false,
        data: null,
        error: `HTTP Error ${response.status}: ${response.statusText}`,
        latencyMs,
      };
    }

    const data = await response.json();
    return {
      isConnected: true,
      data,
      error: null,
      latencyMs,
    };
  } catch (err) {
    clearTimeout(timeoutId);
    const latencyMs = Math.round(performance.now() - startTime);

    let errorMessage = 'Unable to reach backend service';
    if (err.name === 'AbortError') {
      errorMessage = 'Connection timed out';
    } else if (err.message) {
      errorMessage = err.message;
    }

    return {
      isConnected: false,
      data: null,
      error: errorMessage,
      latencyMs,
    };
  }
}

/**
 * Fetch available synthetic scenarios from GET /stability/scenarios.
 *
 * @returns {Promise<{ success: boolean, scenarios: Array<object>, note: string, error: string | null }>}
 */
export async function fetchStabilityScenarios() {
  try {
    const response = await safeFetch('/stability/scenarios', {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      return {
        success: false,
        scenarios: [],
        note: '',
        error: `HTTP ${response.status}: Failed to fetch scenarios`,
      };
    }

    const data = await response.json();
    return {
      success: true,
      scenarios: data.scenarios || [],
      note: data.note || 'SYNTHETIC',
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      scenarios: [],
      note: '',
      error: err.message || 'Network error fetching stability scenarios',
    };
  }
}

/**
 * Fetch 48 packets for a specific predefined scenario from GET /stability/scenarios/:scenarioId.
 *
 * @param {string} scenarioId - Predefined scenario identifier (e.g. scenario_01_normal)
 * @returns {Promise<{ success: boolean, data: object | null, error: string | null }>}
 */
export async function fetchScenarioPackets(scenarioId) {
  try {
    const response = await safeFetch(`/stability/scenarios/${encodeURIComponent(scenarioId)}`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    if (!response.ok) {
      return {
        success: false,
        data: null,
        error: `HTTP ${response.status}: Failed to fetch scenario packets`,
      };
    }

    const data = await response.json();
    return {
      success: true,
      data,
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      data: null,
      error: err.message || 'Network error fetching scenario packets',
    };
  }
}

/**
 * Analyze 48 packets through the dual-probe signal conditioning pipeline via POST /stability/analyze.
 *
 * @param {Array<object>} packets - Exactly 48 consecutive 30-min telemetry packets
 * @returns {Promise<{ success: boolean, data: object | null, error: string | null }>}
 */
export async function analyzeStabilityPackets(packets) {
  try {
    const response = await safeFetch('/stability/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ packets }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => null);
      const detail = errJson?.detail || errJson?.error || `HTTP ${response.status}`;
      return {
        success: false,
        data: null,
        error: detail,
      };
    }

    const data = await response.json();
    return {
      success: true,
      data,
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      data: null,
      error: err.message || 'Network error analyzing stability',
    };
  }
}

/**
 * Run unified agricultural predictions (irrigation/stability, fungal risk, pest risk)
 * via POST /predict/agricultural.
 *
 * @param {Array<object>} packets - Chronologically ordered 30-min NEST telemetry packets (1 to 2000)
 * @param {string} irrigationModel - Soil moisture model variant ("xgb_residual" | "xgb" | "gru_residual" | "gru")
 * @returns {Promise<{ success: boolean, data: object | null, error: string | null }>}
 */
export async function predictAgricultural(packets, irrigationModel = 'xgb_residual') {
  try {
    const response = await safeFetch('/predict/agricultural', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        packets,
        irrigation_model: irrigationModel,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => null);
      const detail = errJson?.detail || errJson?.error || `HTTP ${response.status}`;
      return {
        success: false,
        data: null,
        error: typeof detail === 'string' ? detail : JSON.stringify(detail),
      };
    }

    const data = await response.json();
    return {
      success: true,
      data,
      error: null,
    };
  } catch (err) {
    return {
      success: false,
      data: null,
      error: err.message || 'Network error running agricultural prediction',
    };
  }
}
