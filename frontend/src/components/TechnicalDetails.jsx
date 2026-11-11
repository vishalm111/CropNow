import React, { useState } from 'react';

function ModelProbabilityBreakdown({ title, riskData }) {
  if (!riskData || !riskData.ready || !riskData.probabilities) {
    return (
      <div className="tech-subcard">
        <div className="tech-subcard-head">
          <strong>{title}</strong>
          <span className="tech-code-tag">{riskData?.status || 'insufficient_history'}</span>
        </div>
        <p className="tech-empty-note">
          Requires {riskData?.packets_required?.toLocaleString() || '—'} packets; received{' '}
          {riskData?.packets_provided?.toLocaleString() || 0}.
        </p>
      </div>
    );
  }

  return (
    <div className="tech-subcard">
      <div className="tech-subcard-head">
        <strong>{title}</strong>
        <code className="tech-code-tag">{riskData.model_used || 'N/A'}</code>
      </div>
      <div className="agri-prob-bars">
        {['LOW', 'MEDIUM', 'HIGH'].map((cls) => {
          const prob = riskData.probabilities?.[cls] ?? 0;
          const pct = Math.round(prob * 1000) / 10;
          return (
            <div key={cls} className="agri-prob-row">
              <span className="agri-prob-label">{cls}</span>
              <div className="agri-prob-track" aria-hidden="true">
                <div
                  className={`agri-prob-fill risk-fill-${cls.toLowerCase()}`}
                  style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                />
              </div>
              <span className="agri-prob-pct">{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
      {riskData.window_start_timestamp && riskData.window_end_timestamp && (
        <div className="tech-window-meta">
          Window: <code>{riskData.window_start_timestamp}</code> &rarr;{' '}
          <code>{riskData.window_end_timestamp}</code>
        </div>
      )}
    </div>
  );
}

export default function TechnicalDetails({
  irrigationModel,
  onChangeIrrigationModel,
  isBusy,
  agriculturalResult,
  stabilityResult,
  dataSourceLabel,
}) {
  const [isOpen, setIsOpen] = useState(false);

  const irr = agriculturalResult?.irrigation;
  const irrLatestTimeline = stabilityResult?.timeline?.slice(-1)[0] || null;
  const isStressed = Boolean(
    irrLatestTimeline?.irrigation_advisory ??
      (irrLatestTimeline && irrLatestTimeline.stabilized_average < stabilityResult?.madp_threshold)
  );

  return (
    <section id="technical-details" className="tech-details-card" aria-labelledby="tech-details-heading">
      <div className="tech-header-bar">
        <div className="tech-title-block">
          <h2 id="tech-details-heading" className="tech-heading">
            Technical Details
          </h2>
          <span className="tech-subtitle">
            ML model settings, class probabilities, signal-noise metrics, and raw packet logs
          </span>
        </div>
        <button
          type="button"
          className="tech-toggle-btn"
          aria-expanded={isOpen}
          aria-controls="technical-details-panel"
          onClick={() => setIsOpen((prev) => !prev)}
        >
          {isOpen ? 'Hide Technical Details' : 'Show Technical Details'}
        </button>
      </div>

      <div id="technical-details-panel" className="tech-panel-body" hidden={!isOpen}>
        {/* 1. Model Variant Selection */}
        <div className="tech-block">
          <h3 className="tech-block-title">Forecasting Model Selection</h3>
          <div className="inline-control-group">
            <label htmlFor="irrigation-model-select" className="inline-control-label">
              Irrigation Model Variant:
            </label>
            <select
              id="irrigation-model-select"
              className="compact-select"
              value={irrigationModel}
              onChange={(e) => onChangeIrrigationModel(e.target.value)}
              disabled={isBusy}
            >
              <option value="xgb_residual">xgb_residual (Recommended Default)</option>
              <option value="xgb">xgb (Absolute Baseline)</option>
              <option value="gru_residual">gru_residual (Neural Residual)</option>
              <option value="gru">gru (Neural Absolute)</option>
            </select>
          </div>
          {irr?.ready && (
            <div className="tech-kv-row">
              <span>Active Model: <code>{irr.model_used}</code></span>
              <span>Raw Soil Moisture: <strong>{irr.raw_soil_moisture?.toFixed(2) ?? '—'}%</strong></span>
              <span>24h Predicted Change (Δy): <strong>{irr.change_predicted?.toFixed(3) ?? '—'}%</strong></span>
              {irr.unavailable_features?.length > 0 && (
                <span>
                  Imputed (z=0.0): <code>{irr.unavailable_features.join(', ')}</code>
                </span>
              )}
            </div>
          )}
        </div>

        {/* 2. Classifier Probability Breakdowns */}
        <div className="tech-block">
          <h3 className="tech-block-title">Biological Classifier Outputs &amp; Windows</h3>
          <div className="tech-prob-grid">
            <ModelProbabilityBreakdown
              title="Fungal Risk (XGBoost • 38 Features • 336 Pkts)"
              riskData={agriculturalResult?.fungal_risk}
            />
            <ModelProbabilityBreakdown
              title="Pest Risk (LogisticRegression • 26 Features • 1,344 Pkts)"
              riskData={agriculturalResult?.pest_risk}
            />
          </div>
        </div>

        {/* 3. Quantitative Signal Stability Metrics */}
        {stabilityResult ? (
          <>
            <div className="tech-block">
              <h3 className="tech-block-title">
                Dual-Probe Signal Conditioning Metrics ({stabilityResult.packets_analyzed} Packets • Crop:{' '}
                {stabilityResult.crop_id} • MADP: {stabilityResult.madp_threshold}%)
              </h3>
              <div className="metric-grid stability-metrics-grid">
                <div className="metric-item highlight-metric">
                  <span className="metric-title">High-Freq Jitter Reduction</span>
                  <span className="metric-value metric-accent">
                    {stabilityResult.metrics.high_frequency_jitter_reduction_pct.toFixed(1)}%
                  </span>
                  <span className="metric-subtext">Second-order variance suppression</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Total Variation Reduction</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.total_variation_reduction_pct.toFixed(1)}%
                  </span>
                  <span className="metric-subtext">Signal roughness smoothing</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Genuine Range Retention</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.dynamic_range_retention_pct.toFixed(1)}%
                  </span>
                  <span className="metric-subtext">Clean signal fidelity (excl. faults)</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Outlier Suppression</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.outliers_detected_count > 0
                      ? `${stabilityResult.metrics.outlier_suppression_pct.toFixed(1)}%`
                      : 'N/A'}
                  </span>
                  <span className="metric-subtext">
                    {stabilityResult.metrics.outliers_detected_count > 0
                      ? `${stabilityResult.metrics.outliers_detected_count} fault spikes suppressed`
                      : '0 isolated spikes detected'}
                  </span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Sudden Events Detected</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.sudden_events_detected_count}
                  </span>
                  <span className="metric-subtext">Wetting fronts fast-tracked</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Probe 1 Noise Reduction</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.probe1_noise_reduction_pct.toFixed(1)}%
                  </span>
                  <span className="metric-subtext">Independent channel</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Probe 2 Noise Reduction</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.probe2_noise_reduction_pct.toFixed(1)}%
                  </span>
                  <span className="metric-subtext">Independent channel</span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Actuator Switches (Raw vs Filt)</span>
                  <span className="metric-value">
                    {stabilityResult.metrics.raw_switch_count} &rarr;{' '}
                    {stabilityResult.metrics.filtered_switch_count}
                  </span>
                  <span className="metric-subtext">
                    Chatter reduction: {stabilityResult.metrics.chatter_reduction_pct.toFixed(0)}%
                  </span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Irrigation Advisory</span>
                  <span className={`metric-value ${isStressed ? 'tag-stress' : 'tag-standby'}`}>
                    {isStressed ? 'IRRIGATION STRESS' : 'NO STRESS'}
                  </span>
                  <span className="metric-subtext">
                    Stabilized: {irrLatestTimeline ? irrLatestTimeline.stabilized_average.toFixed(1) : '—'}% vs MADP{' '}
                    {stabilityResult.madp_threshold}%
                  </span>
                </div>

                <div className="metric-item">
                  <span className="metric-title">Actuator State</span>
                  <span
                    className={`metric-value ${irrLatestTimeline?.irrigation_decision ? 'tag-irrigate' : 'tag-standby'}`}
                  >
                    {irrLatestTimeline?.irrigation_decision ? 'VALVE OPEN (ON)' : 'VALVE CLOSED (OFF)'}
                  </span>
                  <span className="metric-subtext">
                    Hysteresis ({stabilityResult.madp_threshold - 1}% ON /{' '}
                    {stabilityResult.madp_threshold + 1}% OFF)
                  </span>
                </div>
              </div>
            </div>

            {/* 4. Recent 8 Telemetry Steps Table */}
            <div className="tech-block">
              <div className="timeline-table-wrapper">
                <div className="table-caption">
                  <span>
                    Recent 8 Telemetry Steps (Raw vs Stabilized Readings —{' '}
                    {dataSourceLabel === 'UPLOADED JSON' ? 'UPLOADED JSON' : 'SYNTHETIC'})
                  </span>
                  <span
                    className={
                      dataSourceLabel === 'UPLOADED JSON'
                        ? 'uploaded-badge-subtle'
                        : 'synthetic-badge-subtle'
                    }
                  >
                    {dataSourceLabel === 'UPLOADED JSON' ? 'UPLOADED' : 'SYNTHETIC'}
                  </span>
                </div>
                <table className="timeline-table">
                  <thead>
                    <tr>
                      <th scope="col">Step</th>
                      <th scope="col">Timestamp</th>
                      <th scope="col">Raw SM1</th>
                      <th scope="col">Raw SM2</th>
                      <th scope="col">Raw Avg</th>
                      <th scope="col">Filtered SM1</th>
                      <th scope="col">Filtered SM2</th>
                      <th scope="col">Stabilized Avg</th>
                      <th scope="col">Event</th>
                      <th scope="col">Advisory</th>
                      <th scope="col">Actuator State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stabilityResult.timeline.slice(-8).map((row) => {
                      const rowStressed = Boolean(
                        row.irrigation_advisory ??
                          row.stabilized_average < stabilityResult.madp_threshold
                      );
                      return (
                        <tr key={row.packet_index}>
                          <td>#{row.packet_index}</td>
                          <td>
                            <code>
                              {row.timestamp ? row.timestamp.split('T')[1] : `+${row.packet_index * 30}m`}
                            </code>
                          </td>
                          <td>{row.raw_soil_moisture_1.toFixed(1)}%</td>
                          <td>{row.raw_soil_moisture_2.toFixed(1)}%</td>
                          <td>
                            <strong>{row.raw_average.toFixed(1)}%</strong>
                          </td>
                          <td>{row.filtered_soil_moisture_1.toFixed(1)}%</td>
                          <td>{row.filtered_soil_moisture_2.toFixed(1)}%</td>
                          <td className="stabilized-cell">{row.stabilized_average.toFixed(1)}%</td>
                          <td>
                            {row.is_sudden_change_probe_1 || row.is_sudden_change_probe_2 ? (
                              <span className="sudden-tag">FAST-TRACK</span>
                            ) : (
                              <span className="normal-tag">SMOOTH</span>
                            )}
                          </td>
                          <td>
                            <span className={rowStressed ? 'advisory-stress' : 'advisory-ok'}>
                              {rowStressed ? 'STRESS' : 'NO STRESS'}
                            </span>
                          </td>
                          <td>
                            <span className={row.irrigation_decision ? 'actuator-on' : 'actuator-off'}>
                              {row.irrigation_decision ? 'ON' : 'OFF'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          <div className="tech-block">
            <p className="tech-empty-note">
              Run a demo scenario or upload packet JSON to populate signal conditioning metrics and packet step tables.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
