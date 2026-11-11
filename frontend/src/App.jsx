import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import {
  checkBackendHealth,
  fetchStabilityScenarios,
  fetchScenarioPackets,
  analyzeStabilityPackets,
  predictAgricultural,
} from './services/api';
import SensorCards from './components/SensorCards';
import AgriculturalPredictions from './components/AgriculturalPredictions';
import TechnicalDetails from './components/TechnicalDetails';
import InfoTip from './components/InfoTip';
import './App.css';

const CURRENT_YEAR = new Date().getFullYear();

export default function App() {
  const [status, setStatus] = useState('checking'); // 'connected' | 'disconnected' | 'checking'
  const [latency, setLatency] = useState(null);

  // Stability & Scenario API State
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState('');
  const [isLoadingScenarios, setIsLoadingScenarios] = useState(false);
  const [scenariosError, setScenariosError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [stabilityResult, setStabilityResult] = useState(null);
  const [stabilityError, setStabilityError] = useState(null);

  // Unified Agricultural Inference State (POST /predict/agricultural)
  const [irrigationModel, setIrrigationModel] = useState('xgb_residual');
  const [isPredictingAgri, setIsPredictingAgri] = useState(false);
  const [agriculturalResult, setAgriculturalResult] = useState(null);
  const [agriculturalError, setAgriculturalError] = useState(null);
  const [dataSourceLabel, setDataSourceLabel] = useState('SYNTHETIC');
  const [sourceDescription, setSourceDescription] = useState('');
  const [uploadError, setUploadError] = useState(null);
  const [activePackets, setActivePackets] = useState([]);
  const [lastDataUpdate, setLastDataUpdate] = useState(null);

  const performHealthCheck = useCallback(async () => {
    const result = await checkBackendHealth();
    setStatus(result.isConnected ? 'connected' : 'disconnected');
    setLatency(result.latencyMs);
  }, []);

  const loadScenarios = useCallback(async () => {
    setIsLoadingScenarios(true);
    setScenariosError(null);
    try {
      const res = await fetchStabilityScenarios();
      if (res.success && Array.isArray(res.scenarios) && res.scenarios.length > 0) {
        setScenarios(res.scenarios);
        setSelectedScenarioId((prev) => {
          const match = res.scenarios.find((s) => s.id === prev);
          return match ? match.id : res.scenarios[0].id;
        });
        setScenariosError(null);
      } else {
        setScenariosError(res.error || 'Failed to load scenarios from backend');
      }
    } catch (err) {
      setScenariosError(err.message || 'Error connecting to stability scenarios API');
    } finally {
      setIsLoadingScenarios(false);
    }
  }, []);

  useEffect(() => {
    let isSubscribed = true;
    const runInitialChecks = async () => {
      await performHealthCheck();
      if (isSubscribed) {
        await loadScenarios();
      }
    };
    runInitialChecks();

    const intervalId = setInterval(performHealthCheck, 10000);
    return () => {
      isSubscribed = false;
      clearInterval(intervalId);
    };
  }, [performHealthCheck, loadScenarios]);

  useEffect(() => {
    if (status === 'connected' && scenarios.length === 0 && !isLoadingScenarios) {
      const timer = setTimeout(() => {
        loadScenarios();
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [status, scenarios.length, isLoadingScenarios, loadScenarios]);

  // Run selected synthetic 48-packet scenario through stability + agricultural pipelines
  const handleAnalyzeScenario = async () => {
    if (!selectedScenarioId) return;
    setIsAnalyzing(true);
    setIsPredictingAgri(true);
    setStabilityError(null);
    setStabilityResult(null);
    setAgriculturalError(null);
    setUploadError(null);

    const packetRes = await fetchScenarioPackets(selectedScenarioId);
    if (!packetRes.success || !packetRes.data || !packetRes.data.packets) {
      const msg = packetRes.error || 'Failed to fetch 48 packets for scenario';
      setStabilityError(msg);
      setAgriculturalError(msg);
      setIsAnalyzing(false);
      setIsPredictingAgri(false);
      return;
    }

    const packets = packetRes.data.packets;
    setActivePackets(packets);
    setDataSourceLabel('SYNTHETIC');
    setSourceDescription(selectedScenarioMeta?.name || selectedScenarioId);
    setLastDataUpdate(new Date().toLocaleTimeString());

    const [analyzeRes, agriRes] = await Promise.all([
      analyzeStabilityPackets(packets),
      predictAgricultural(packets, irrigationModel),
    ]);

    if (!analyzeRes.success || !analyzeRes.data) {
      setStabilityError(analyzeRes.error || 'Failed to analyze soil moisture graphs');
    } else {
      setStabilityResult(analyzeRes.data);
    }

    if (!agriRes.success || !agriRes.data) {
      setAgriculturalError(agriRes.error || 'Failed to evaluate farm predictions');
    } else {
      setAgriculturalResult(agriRes.data);
    }

    setIsAnalyzing(false);
    setIsPredictingAgri(false);
  };

  // Optional custom JSON upload (1 to 2,000 genuine packets, e.g. 48, 336, 1,344)
  const handleTelemetryUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;

    setUploadError(null);
    setAgriculturalError(null);

    let parsed;
    try {
      const text = await file.text();
      parsed = JSON.parse(text);
    } catch {
      setUploadError('Invalid JSON file. Expected an array of NEST packets or {"packets": [...]}.');
      return;
    }

    const packets = Array.isArray(parsed) ? parsed : parsed?.packets;
    if (!Array.isArray(packets) || packets.length === 0) {
      setUploadError('Uploaded JSON must contain a non-empty array of NEST telemetry packets.');
      return;
    }
    if (packets.length > 2000) {
      setUploadError(`Uploaded ${packets.length} packets exceeds maximum limit of 2,000.`);
      return;
    }

    setIsPredictingAgri(true);
    setActivePackets(packets);
    setDataSourceLabel('UPLOADED JSON');
    setSourceDescription(`${file.name} (${packets.length.toLocaleString()} packets)`);
    setLastDataUpdate(new Date().toLocaleTimeString());

    const agriPromise = predictAgricultural(packets, irrigationModel);

    if (packets.length >= 48) {
      setIsAnalyzing(true);
      setStabilityError(null);
      const [agriRes, analyzeRes] = await Promise.all([
        agriPromise,
        analyzeStabilityPackets(packets.slice(-48)),
      ]);

      if (!agriRes.success || !agriRes.data) {
        setAgriculturalError(agriRes.error || 'Failed to evaluate uploaded packets');
      } else {
        setAgriculturalResult(agriRes.data);
      }

      if (!analyzeRes.success || !analyzeRes.data) {
        setStabilityError(analyzeRes.error || 'Sensor graphs failed on latest 48 uploaded packets');
        setStabilityResult(null);
      } else {
        setStabilityResult(analyzeRes.data);
      }
      setIsAnalyzing(false);
    } else {
      setStabilityResult(null);
      setStabilityError(
        `Uploaded ${packets.length} packets: Sensor graphs require at least 48 consecutive 30-min packets. No history invented.`
      );
      const agriRes = await agriPromise;
      if (!agriRes.success || !agriRes.data) {
        setAgriculturalError(agriRes.error || 'Failed to evaluate uploaded packets');
      } else {
        setAgriculturalResult(agriRes.data);
      }
    }

    setIsPredictingAgri(false);
  };

  const selectedScenarioMeta = scenarios.find((s) => s.id === selectedScenarioId);
  const latestActivePacket = activePackets.length > 0 ? activePackets[activePackets.length - 1] : null;
  const headerDeviceId = agriculturalResult?.device_id || latestActivePacket?.device_id || null;
  const headerFarmId = agriculturalResult?.farm_id || latestActivePacket?.farm_id || null;
  const headerCropId =
    agriculturalResult?.crop_id || stabilityResult?.crop_id || latestActivePacket?.crop_id || null;
  const headerPacketTs = latestActivePacket?.timestamp
    ? String(latestActivePacket.timestamp).replace('T', ' ').slice(0, 16)
    : null;

  // Format 48-packet timeline data for Recharts
  const chartData = useMemo(() => {
    if (!stabilityResult || !stabilityResult.timeline) return [];
    return stabilityResult.timeline.map((row) => {
      let timeLabel = `#${row.packet_index}`;
      if (row.timestamp) {
        const parts = row.timestamp.split('T');
        if (parts[1]) {
          timeLabel = parts[1].substring(0, 5);
        }
      }
      return {
        step: row.packet_index,
        time: timeLabel,
        fullTime: row.timestamp || `Step ${row.packet_index}`,
        raw1: row.raw_soil_moisture_1,
        filt1: row.filtered_soil_moisture_1,
        raw2: row.raw_soil_moisture_2,
        filt2: row.filtered_soil_moisture_2,
        rawAvg: row.raw_average,
        stabilizedAvg: row.stabilized_average,
        irrigation: row.irrigation_decision ? 1 : 0,
        sudden: row.is_sudden_change_probe_1 || row.is_sudden_change_probe_2,
        madp: stabilityResult.madp_threshold,
      };
    });
  }, [stabilityResult]);

  return (
    <div className="dashboard-page">
      {/* 1. Simple CropDesk Header & Device Status */}
      <header className="cropdesk-header-card">
        <div className="topbar-primary-row">
          <div className="topbar-titles">
            <div className="topbar-badge-row">
              <span className="badge-pill">CropDesk • CropNow</span>
              <span className={dataSourceLabel === 'UPLOADED JSON' ? 'uploaded-badge' : 'synthetic-badge'}>
                {dataSourceLabel === 'UPLOADED JSON'
                  ? 'UPLOADED FILE TELEMETRY'
                  : 'SYNTHETIC DEMO (NOT LIVE SENSOR)'}
              </span>
            </div>
            <h1 className="main-title">NEST Agricultural Intelligence</h1>
            <p className="subtitle">Plain-language irrigation, fungal, and pest guidance from farm sensors</p>
          </div>

          <div className="topbar-status-strip">
            <div className={`status-badge ${status}`} aria-live="polite">
              <span className="indicator-dot" />
              <span className="status-text">
                {status === 'connected' && 'Connected'}
                {status === 'disconnected' && 'Disconnected'}
                {status === 'checking' && 'Checking...'}
              </span>
              {latency !== null && <span className="status-latency">• {latency} ms</span>}
            </div>
          </div>
        </div>

        <div className="topbar-telemetry-strip" aria-label="Farm and Device Identification">
          <div className="header-meta-pill">
            <span className="header-meta-label">Farm / Crop</span>
            <strong className="header-meta-value">
              {headerFarmId || '— (Not loaded)'}
              {headerCropId ? ` • ${headerCropId.toUpperCase()}` : ''}
            </strong>
          </div>
          <div className="header-meta-pill">
            <span className="header-meta-label">Device ID</span>
            <strong className="header-meta-value">
              {headerDeviceId || '— (Awaiting telemetry)'}
            </strong>
          </div>
          <div className="header-meta-pill">
            <span className="header-meta-label">Last Data Update</span>
            <strong className="header-meta-value">
              {lastDataUpdate
                ? `${lastDataUpdate}${headerPacketTs ? ` (Pkt: ${headerPacketTs})` : ''}`
                : 'Awaiting scenario or JSON upload'}
            </strong>
          </div>
        </div>
      </header>

      <main className="dashboard-main">
        {/* 2. Demo Scenario Selector & Packet Upload Bar */}
        <section className="demo-controls-card" aria-label="Scenario Selection">
          <div className="demo-controls-top">
            <div className="demo-select-wrap">
              <label htmlFor="scenario-select" className="control-label">
                Demo Scenario (Synthetic 24h NEST Samples)
              </label>
              <div className="control-row">
                <select
                  id="scenario-select"
                  className="scenario-dropdown"
                  value={selectedScenarioId}
                  onChange={(e) => setSelectedScenarioId(e.target.value)}
                  disabled={isLoadingScenarios || isAnalyzing || (scenarios.length === 0 && !isLoadingScenarios)}
                >
                  {isLoadingScenarios && <option value="">Loading scenarios...</option>}
                  {!isLoadingScenarios && scenariosError && <option value="">Failed to load scenarios</option>}
                  {!isLoadingScenarios && !scenariosError && scenarios.length === 0 && (
                    <option value="">No scenarios available</option>
                  )}
                  {scenarios.map((sc) => (
                    <option key={sc.id} value={sc.id}>
                      {sc.name}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="analyze-btn"
                  onClick={handleAnalyzeScenario}
                  disabled={isAnalyzing || !selectedScenarioId || isLoadingScenarios || scenarios.length === 0}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="spinner-mini" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    'Run Demo Analysis'
                  )}
                </button>
                <div className="upload-control">
                  <label htmlFor="telemetry-json-upload" className="upload-btn-label">
                    Upload Packet JSON (48 / 336 / 1,344)
                  </label>
                  <input
                    id="telemetry-json-upload"
                    type="file"
                    accept=".json,application/json"
                    className="visually-hidden"
                    onChange={handleTelemetryUpload}
                    disabled={isAnalyzing || isPredictingAgri}
                  />
                </div>
              </div>
            </div>
          </div>

          {selectedScenarioMeta && (
            <p className="scenario-description">
              <strong>Selected Scenario:</strong> {selectedScenarioMeta.description}
            </p>
          )}

          {scenariosError && (
            <div className="scenario-error-banner" role="alert">
              <span>{scenariosError}</span>
              <button
                type="button"
                className="retry-btn"
                onClick={loadScenarios}
                disabled={isLoadingScenarios}
              >
                {isLoadingScenarios ? 'Retrying...' : 'Retry'}
              </button>
            </div>
          )}

          {uploadError && (
            <div className="scenario-error-banner" role="alert">
              <span>{uploadError}</span>
            </div>
          )}
        </section>

        {/* 3. Four Easy-to-Read Sensor Cards */}
        <SensorCards
          activePackets={activePackets}
          stabilityResult={stabilityResult}
          agriculturalResult={agriculturalResult}
        />

        {/* 4. Three Prominent Agricultural Prediction Cards */}
        <div className="predictions-card-shell">
          <AgriculturalPredictions
            agriculturalResult={agriculturalResult}
            agriculturalError={agriculturalError}
            isPredicting={isPredictingAgri}
            dataSourceLabel={dataSourceLabel}
            sourceDescription={sourceDescription}
          />
        </div>

        {/* 5. Raw vs Filtered Sensor Graphs */}
        <section className="graphs-card-shell" aria-labelledby="graphs-section-heading">
          <div className="section-heading-row">
            <div>
              <h2 id="graphs-section-heading" className="section-title">
                Soil Moisture &amp; Watering Valve Graphs
              </h2>
              <p className="section-helper">
                Compares raw sensor readings (dashed) against cleaned trends (solid) over the latest 24 hours.
              </p>
            </div>
            <span className={dataSourceLabel === 'UPLOADED JSON' ? 'uploaded-badge' : 'synthetic-badge'}>
              {dataSourceLabel === 'UPLOADED JSON' ? 'UPLOADED FILE DATA' : 'SYNTHETIC RESULTS'}
            </span>
          </div>

          {!isAnalyzing && !stabilityError && !stabilityResult && (
            <div className="empty-graphs-box">
              Click <strong>Run Demo Analysis</strong> above (or upload a packet JSON file with at least 48 readings) to view the raw vs. cleaned sensor graphs.
            </div>
          )}

          {isAnalyzing && (
            <div className="analyzing-banner" aria-live="polite">
              <div className="spinner-medium" />
              <div className="analyzing-text">
                <strong>Updating Sensor Graphs...</strong>
                <span>Processing 48 consecutive sensor packets through spike removal and smoothing.</span>
              </div>
            </div>
          )}

          {stabilityError && (
            <div className="health-details error" role="alert">
              <div className="details-header">Sensor Graph Error</div>
              <p className="error-message">{stabilityError}</p>
              <button
                type="button"
                className="retry-btn"
                onClick={handleAnalyzeScenario}
                disabled={isAnalyzing}
              >
                Retry Analysis
              </button>
            </div>
          )}

          {stabilityResult && (
            <div className="charts-section">
              {/* Graph 1: Sensor 1 Raw vs Filtered */}
              <div className="chart-card">
                <div className="chart-header">
                  <div className="chart-title-group">
                    <span className="chart-title-row">
                      <span className="chart-title">Soil Moisture Sensor 1: Raw vs Filtered</span>
                      <InfoTip
                        label="Sensor 1 chart"
                        text="The dashed red line shows raw sensor readings (including sudden electrical glitches). The solid blue line shows the cleaned soil trend used for crop advice."
                      />
                    </span>
                    <span className="chart-subtitle">
                      Smooths sudden hardware spikes while tracking real watering changes
                    </span>
                  </div>
                  <span className="sensor-tag sensor-1">Sensor 1 (Shallow)</span>
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        interval={3}
                        label={{ value: 'Time (HH:MM / 30m steps)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#52705d' }}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        unit="%"
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 12 }}
                        formatter={(val, name) => [`${Number(val).toFixed(2)}%`, name]}
                        labelFormatter={(lbl, p) => `Step: ${lbl} (${p[0]?.payload?.fullTime || ''}) [${dataSourceLabel}]`}
                      />
                      <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 12 }} />
                      <ReferenceLine y={stabilityResult.madp_threshold} stroke="#155e34" strokeDasharray="4 4" label={{ value: `Target (${stabilityResult.madp_threshold}%)`, fill: '#155e34', fontSize: 10, position: 'right' }} />
                      <Line
                        type="monotone"
                        dataKey="raw1"
                        name="Raw Sensor 1"
                        stroke="#dc2626"
                        strokeDasharray="3 3"
                        dot={false}
                        strokeWidth={1.5}
                      />
                      <Line
                        type="monotone"
                        dataKey="filt1"
                        name="Cleaned Sensor 1"
                        stroke="#2563eb"
                        dot={false}
                        strokeWidth={2.5}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Graph 2: Sensor 2 Raw vs Filtered */}
              <div className="chart-card">
                <div className="chart-header">
                  <div className="chart-title-group">
                    <span className="chart-title-row">
                      <span className="chart-title">Soil Moisture Sensor 2: Raw vs Filtered</span>
                      <InfoTip
                        label="Sensor 2 chart"
                        text="Second probe measures root-zone moisture independently as a backup check so a single faulty probe does not trigger false irrigation."
                      />
                    </span>
                    <span className="chart-subtitle">
                      Backup probe channel smoothed independently for sensor reliability
                    </span>
                  </div>
                  <span className="sensor-tag sensor-2">Sensor 2 (Root Zone)</span>
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        interval={3}
                        label={{ value: 'Time (HH:MM / 30m steps)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#52705d' }}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        unit="%"
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 12 }}
                        formatter={(val, name) => [`${Number(val).toFixed(2)}%`, name]}
                        labelFormatter={(lbl, p) => `Step: ${lbl} (${p[0]?.payload?.fullTime || ''}) [${dataSourceLabel}]`}
                      />
                      <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 12 }} />
                      <ReferenceLine y={stabilityResult.madp_threshold} stroke="#155e34" strokeDasharray="4 4" label={{ value: `Target (${stabilityResult.madp_threshold}%)`, fill: '#155e34', fontSize: 10, position: 'right' }} />
                      <Line
                        type="monotone"
                        dataKey="raw2"
                        name="Raw Sensor 2"
                        stroke="#d97706"
                        strokeDasharray="3 3"
                        dot={false}
                        strokeWidth={1.5}
                      />
                      <Line
                        type="monotone"
                        dataKey="filt2"
                        name="Cleaned Sensor 2"
                        stroke="#15803d"
                        dot={false}
                        strokeWidth={2.5}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Graph 3: Irrigation Actuator ON/OFF Timeline */}
              <div className="chart-card">
                <div className="chart-header">
                  <div className="chart-title-group">
                    <span className="chart-title-row">
                      <span className="chart-title">Irrigation Valve Activity</span>
                      <InfoTip
                        label="Irrigation Valve Activity chart"
                        text="Shows when automated watering turns ON (1) or OFF (0) using a small safety buffer around the target level so pumps never flicker repeatedly."
                      />
                    </span>
                    <span className="chart-subtitle">
                      1 = Valve ON (Watering) • 0 = Valve OFF (Standby) — prevents rapid valve chatter
                    </span>
                  </div>
                  <span className="sensor-tag sensor-actuator">Valve Control</span>
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height={160}>
                    <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis
                        dataKey="time"
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        interval={3}
                        label={{ value: 'Time (HH:MM / 30m steps)', position: 'insideBottom', offset: -10, fontSize: 11, fill: '#52705d' }}
                      />
                      <YAxis
                        ticks={[0, 1]}
                        domain={[0, 1.2]}
                        tick={{ fontSize: 11, fill: '#52705d' }}
                        tickFormatter={(v) => (v === 1 ? 'ON' : 'OFF')}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#ffffff', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 12 }}
                        formatter={(val) => [val === 1 ? 'WATERING (Valve OPEN)' : 'STANDBY (Valve CLOSED)', 'Valve State']}
                        labelFormatter={(lbl, p) => `Step: ${lbl} (${p[0]?.payload?.fullTime || ''}) [${dataSourceLabel}]`}
                      />
                      <Area
                        type="stepAfter"
                        dataKey="irrigation"
                        name="Irrigation Valve State"
                        stroke="#15803d"
                        fill="#dcfce7"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* 6. Collapsible Technical Details (Advanced ML Metrics & Telemetry Table) */}
        <TechnicalDetails
          irrigationModel={irrigationModel}
          onChangeIrrigationModel={setIrrigationModel}
          isBusy={isAnalyzing || isPredictingAgri}
          agriculturalResult={agriculturalResult}
          stabilityResult={stabilityResult}
          dataSourceLabel={dataSourceLabel}
        />
      </main>

      <footer className="app-footer">
        <p>CropDesk by CropNow &copy; {CURRENT_YEAR} • NEST Agricultural Prediction System • Data marked SYNTHETIC is simulated demo telemetry.</p>
      </footer>
    </div>
  );
}
