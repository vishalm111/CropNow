import React from 'react';

const STATUS_LABELS = {
  ready: 'READY',
  insufficient_history: 'INSUFFICIENT HISTORY',
  blocked: 'BLOCKED',
};

function formatConfidence(confidence, confidenceLevel) {
  if (typeof confidence === 'number' && Number.isFinite(confidence)) {
    const pct = `${(confidence * 100).toFixed(0)}%`;
    return confidenceLevel && confidenceLevel !== 'unavailable'
      ? `${pct} (${confidenceLevel.toUpperCase()})`
      : pct;
  }
  return 'Unavailable (no prediction)';
}

function getRiskOutcomeClass(status, riskLevel) {
  if (status === 'insufficient_history') return 'outcome-insufficient';
  if (status === 'blocked') return 'outcome-blocked';
  if (riskLevel === 'HIGH') return 'outcome-high';
  if (riskLevel === 'MEDIUM') return 'outcome-medium';
  if (riskLevel === 'LOW') return 'outcome-low';
  return 'outcome-insufficient';
}

function getIrrigationOutcomeClass(status, isRequired) {
  if (status === 'insufficient_history') return 'outcome-insufficient';
  if (status === 'blocked') return 'outcome-blocked';
  return isRequired ? 'outcome-required' : 'outcome-not-required';
}

function PacketHistoryMeter({ provided, required, horizonLabel }) {
  const safeProvided = Math.max(0, Number(provided) || 0);
  const safeRequired = Math.max(1, Number(required) || 1);
  const clampedValue = Math.min(safeProvided, safeRequired);
  const missing = Math.max(0, safeRequired - safeProvided);
  const pct = Math.min(100, Math.round((safeProvided / safeRequired) * 100));

  return (
    <div className="agri-history-block">
      <div className="agri-history-row">
        <span className="agri-field-label">Required History ({horizonLabel})</span>
        <span className="agri-history-ratio">
          {safeProvided.toLocaleString()} / {safeRequired.toLocaleString()} pkts ({pct}%)
        </span>
      </div>
      <progress
        className="agri-progress"
        value={clampedValue}
        max={safeRequired}
        aria-label={`Required packet history (${horizonLabel}): ${safeProvided} of ${safeRequired} packets`}
      />
      {missing > 0 && (
        <span className="agri-history-missing">
          Needs {missing.toLocaleString()} more packets — missing history is never invented.
        </span>
      )}
    </div>
  );
}

function PlainWarningsList({ warnings = [], domainLimitations = [] }) {
  const seen = new Set();
  const items = [];

  for (const raw of [...domainLimitations, ...warnings]) {
    const text = String(raw || '').trim();
    if (text && !seen.has(text)) {
      seen.add(text);
      items.push({
        text,
        isDomain:
          domainLimitations.includes(raw) ||
          text.startsWith('DOMAIN_LIMITATION') ||
          text.startsWith('OUT_OF_DISTRIBUTION'),
      });
    }
  }

  if (items.length === 0) return null;

  return (
    <ul className="agri-warnings-list" aria-label="Warnings and domain notices">
      {items.map((item, idx) => (
        <li
          key={`${idx}-${item.text.slice(0, 20)}`}
          className={item.isDomain ? 'agri-warning-item domain' : 'agri-warning-item'}
        >
          {item.text}
        </li>
      ))}
    </ul>
  );
}

function RiskCard({
  title,
  horizonLabel,
  requiredPackets,
  data,
  totalPacketsReceived,
  conceptDescription,
  plainSummaryReady,
  nextActionByState,
  domainPlainNote,
}) {
  const status = data?.status || 'insufficient_history';
  const ready = Boolean(data?.ready);
  const provided = data?.packets_provided ?? totalPacketsReceived ?? 0;
  const required = data?.packets_required ?? requiredPackets;
  const riskLevel = ready ? data?.risk_level : null;

  let headlineText = 'INSUFFICIENT HISTORY';
  let statusMeaning = `Only ${provided.toLocaleString()} of ${required.toLocaleString()} packets (${horizonLabel}) are available, so no risk prediction is shown.`;
  let nextAction = nextActionByState.insufficient_history;

  if (status === 'blocked') {
    headlineText = 'BLOCKED';
    statusMeaning = 'Sensor gaps or faulty values appeared inside this time window, so the result was paused safely.';
    nextAction = nextActionByState.blocked;
  } else if (ready && riskLevel) {
    headlineText = riskLevel;
    statusMeaning =
      plainSummaryReady[riskLevel] || `Assessed as ${riskLevel} risk from recent weather patterns.`;
    nextAction = nextActionByState[riskLevel] || nextActionByState.MEDIUM;
  }

  const outcomeClass = getRiskOutcomeClass(status, riskLevel);

  return (
    <article className={`agri-card status-${status}`}>
      <header className="agri-card-header">
        <div className="agri-card-title-wrap">
          <h3 className="agri-card-title">{title}</h3>
          <span className="agri-card-sub">{horizonLabel} outlook</span>
        </div>
        <span className={`agri-readiness-pill readiness-${status}`}>
          {STATUS_LABELS[status] || status.toUpperCase()}
        </span>
      </header>

      <div className="agri-card-content">
        <div className={`agri-outcome-box ${outcomeClass}`}>
          <span className="agri-outcome-kicker">Risk Level</span>
          <span className="agri-outcome-value">{headlineText}</span>
        </div>

        {/* Farmer-Friendly Meaning & Next Action Box */}
        <div className="agri-meaning-box">
          <div className="agri-meaning-title">What does this mean?</div>
          <p className="agri-meaning-text">
            {conceptDescription} {statusMeaning}
          </p>
          <div className="agri-next-action">
            <strong>Next Action:</strong> {nextAction}
          </div>
        </div>

        <dl className="agri-meta-list">
          <div className="agri-meta-row">
            <dt>Readiness</dt>
            <dd>{ready ? 'Ready' : status === 'blocked' ? 'Blocked' : 'Insufficient History'}</dd>
          </div>
          <div className="agri-meta-row">
            <dt>Confidence</dt>
            <dd>{formatConfidence(data?.confidence, data?.confidence_level)}</dd>
          </div>
        </dl>

        <PacketHistoryMeter
          provided={provided}
          required={required}
          horizonLabel={horizonLabel}
        />

        <div className="agri-alerts-box">
          <p className="agri-domain-caption">{domainPlainNote}</p>
          <PlainWarningsList
            domainLimitations={data?.domain_limitations || []}
            warnings={
              data?.warnings && data.warnings.length > 0
                ? data.warnings
                : !ready
                  ? [`${provided.toLocaleString()} of ${required.toLocaleString()} packets provided.`]
                  : []
            }
          />
        </div>
      </div>
    </article>
  );
}

function IrrigationCard({ data, totalPacketsReceived }) {
  const status = data?.status || 'insufficient_history';
  const ready = Boolean(data?.ready);
  const provided = data?.packets_provided ?? totalPacketsReceived ?? 0;
  const required = data?.packets_required ?? 48;
  const isRequired = ready
    ? Boolean(data?.irrigation_advisory || data?.irrigation_advice?.irrigate)
    : false;
  const madp = data?.irrigation_advice?.madp_threshold ?? 20;

  let headlineText = 'INSUFFICIENT HISTORY';
  let statusMeaning = `Needs 24 hours (${required} half-hour readings) of continuous data before advising on irrigation.`;
  let nextAction =
    'Select a 48-packet scenario or let your field station log 24 hours of readings before irrigating automatically.';

  if (status === 'blocked') {
    headlineText = 'BLOCKED';
    statusMeaning =
      'Watering advice paused because the sensor log includes invalid readings (such as negative humidity or extreme temperature spikes).';
    nextAction =
      'Inspect soil/weather sensors for loose wiring and inspect the cleaned probe graph below before irrigating.';
  } else if (ready) {
    headlineText = isRequired ? 'REQUIRED' : 'NOT REQUIRED';
    if (isRequired) {
      statusMeaning = `Root-zone moisture (${data.stabilized_soil_moisture?.toFixed(1) ?? '—'}%) or the 24-hour outlook (${data.predicted_soil_moisture_24h?.toFixed(1) ?? '—'}%) is dipping below your ${madp}% crop comfort floor.`;
      nextAction = `Schedule or turn on watering soon so root-zone moisture recovers safely above ${madp}%.`;
    } else {
      statusMeaning = `Current moisture (${data.stabilized_soil_moisture?.toFixed(1) ?? '—'}%) and the 24-hour outlook (${data.predicted_soil_moisture_24h?.toFixed(1) ?? '—'}%) stay comfortably above your ${madp}% threshold.`;
      nextAction = 'No irrigation needed right now. Check back after the next sensor update.';
    }
  }

  const outcomeClass = getIrrigationOutcomeClass(status, isRequired);

  return (
    <article className={`agri-card status-${status}`}>
      <header className="agri-card-header">
        <div className="agri-card-title-wrap">
          <h3 className="agri-card-title">Irrigation</h3>
          <span className="agri-card-sub">Next 24h watering guidance</span>
        </div>
        <span className={`agri-readiness-pill readiness-${status}`}>
          {STATUS_LABELS[status] || status.toUpperCase()}
        </span>
      </header>

      <div className="agri-card-content">
        <div className={`agri-outcome-box ${outcomeClass}`}>
          <span className="agri-outcome-kicker">Watering Advice</span>
          <span className="agri-outcome-value">{headlineText}</span>
        </div>

        {/* Farmer-Friendly Meaning & Next Action Box */}
        <div className="agri-meaning-box">
          <div className="agri-meaning-title">What does this mean?</div>
          <p className="agri-meaning-text">
            Checks how fast your soil is drying over the last 24 hours and forecasts tomorrow&apos;s moisture. {statusMeaning}
          </p>
          <div className="agri-next-action">
            <strong>Next Action:</strong> {nextAction}
          </div>
        </div>

        <dl className="agri-meta-list">
          <div className="agri-meta-row">
            <dt>Readiness</dt>
            <dd>{ready ? 'Ready' : status === 'blocked' ? 'Blocked' : 'Insufficient History'}</dd>
          </div>
          <div className="agri-meta-row">
            <dt>Confidence</dt>
            <dd>{formatConfidence(data?.confidence, data?.confidence_level)}</dd>
          </div>
          {ready && (
            <div className="agri-meta-row">
              <dt>Valve State</dt>
              <dd>{data?.actuator_decision ? 'Open (Watering)' : 'Closed (Standby)'}</dd>
            </div>
          )}
        </dl>

        <PacketHistoryMeter
          provided={provided}
          required={required}
          horizonLabel="24 hours"
        />

        <div className="agri-alerts-box">
          <p className="agri-domain-caption">
            {data?.unavailable_features?.length
              ? 'Unavailable field channels (soil EC, pH, deep probe temp, past flow volume) are filled at the training baseline (z=0.0), which can reduce accuracy.'
              : 'Uses dual soil probe smoothing and 24-hour moisture forecasting.'}
          </p>
          <PlainWarningsList
            warnings={
              data?.warnings && data.warnings.length > 0
                ? data.warnings
                : !ready
                  ? [`${provided.toLocaleString()} of ${required.toLocaleString()} packets provided.`]
                  : []
            }
          />
        </div>
      </div>
    </article>
  );
}

export default function AgriculturalPredictions({
  agriculturalResult,
  agriculturalError,
  isPredicting,
  dataSourceLabel,
  sourceDescription,
}) {
  const totalReceived = agriculturalResult?.total_packets_received ?? 0;
  const cropId = agriculturalResult?.crop_id || 'chilli';
  const isSyntheticSource = !dataSourceLabel || dataSourceLabel === 'SYNTHETIC';

  return (
    <section
      className="agri-section"
      aria-labelledby="agri-predictions-heading"
      aria-live="polite"
    >
      <div className="section-heading-row">
        <div>
          <h2 id="agri-predictions-heading" className="section-title">
            Farm Predictions &amp; Recommended Actions
          </h2>
          <p className="section-helper">
            Clear guidance for watering, fungal disease risk, and insect pests — never guessed when history is short.
          </p>
        </div>
        <div className="agri-header-badges">
          <span className={isSyntheticSource ? 'synthetic-badge' : 'uploaded-badge'}>
            {isSyntheticSource ? 'SYNTHETIC DEMO RESULTS' : 'UPLOADED TELEMETRY RESULTS'}
          </span>
          {agriculturalResult && (
            <span className="agri-packet-pill">
              {totalReceived.toLocaleString()} Packets Checked • {cropId}
            </span>
          )}
        </div>
      </div>

      {/* Explicit Synthetic vs Real Data & Crop-Domain Notice */}
      <div className="agri-limitation-banner" role="note" aria-label="Synthetic data and crop-domain limitations">
        <div className="agri-limitation-title">
          {isSyntheticSource
            ? 'Demo Warning: Viewing Synthetic Simulated Results (Not Real Farm Device Readings)'
            : `Uploaded File Telemetry (${sourceDescription || `${totalReceived} packets`}) — Cross-Crop Caution`}
        </div>
        <p className="agri-limitation-summary">
          {isSyntheticSource
            ? 'Built-in scenarios come from 24-hour synthetic simulations rather than a live NEST sensor station. Because Fungal (7 days) and Pest (28 days) models need longer continuous logs, they honestly display INSUFFICIENT HISTORY instead of inventing numbers. '
            : 'These uploaded multi-day packets are not verified live NEST chilli station logs unless recorded from your hardware. '}
          Fungal risk was trained on wild flax rust (10.2–18.9°C) and Pest risk on Rice Black Bug — verify locally before field decisions on <strong>{cropId}</strong>.
        </p>
      </div>

      {agriculturalError && (
        <div className="health-details error" role="alert">
          <div className="details-header">Could Not Run Predictions</div>
          <p className="error-message">{agriculturalError}</p>
        </div>
      )}

      {isPredicting && (
        <div className="analyzing-banner">
          <div className="spinner-medium" />
          <div className="analyzing-text">
            <strong>Checking Farm Conditions...</strong>
            <span>Running Irrigation, Fungal, and Pest checks independently.</span>
          </div>
        </div>
      )}

      <div className="agri-cards-grid">
        <RiskCard
          title="Pest Risk"
          horizonLabel="28 days"
          requiredPackets={1344}
          data={agriculturalResult?.pest_risk}
          totalPacketsReceived={totalReceived}
          conceptDescription="Looks at 4 full weeks of heat, humidity, rain, wind, and soil wetness to spot conditions when crop-eating insects tend to multiply."
          plainSummaryReady={{
            LOW: 'Weather over the last 4 weeks suggests low insect pest pressure.',
            MEDIUM: 'Recent 4-week weather favors moderate insect activity.',
            HIGH: '4-week warmth and moisture match conditions linked with high insect pest outbreaks.',
          }}
          nextActionByState={{
            LOW: 'Continue routine weekly field walks; no emergency spraying is suggested.',
            MEDIUM: 'Check leaf undersides and sticky traps twice this week so early pests do not spread.',
            HIGH: 'Inspect crops across all rows today, verify insect counts on leaves, and consult your agronomist on targeted control.',
            insufficient_history:
              'Keep sensors recording until 28 continuous days (1,344 packets) are collected, or upload a 1,344-packet JSON file to test.',
            blocked:
              'Fix missing or invalid weather readings in the older 4-week log, then re-run.',
          }}
          domainPlainNote="Trained on Rice Black Bug in wetland rice (not field-calibrated for chilli pests)."
        />

        <IrrigationCard
          data={agriculturalResult?.irrigation}
          totalPacketsReceived={totalReceived}
        />

        <RiskCard
          title="Fungal Risk"
          horizonLabel="7 days"
          requiredPackets={336}
          data={agriculturalResult?.fungal_risk}
          totalPacketsReceived={totalReceived}
          conceptDescription="Tracks the past 7 days of humidity, wet leaves, rain, and temperature swings that allow fungal spores to sprout on foliage."
          plainSummaryReady={{
            LOW: 'Past 7-day leaf wetness and humidity do not strongly favor fungal spread.',
            MEDIUM: 'Moisture and humidity over the last 7 days show moderate fungal risk.',
            HIGH: 'Prolonged dampness and leaf wetness over the last 7 days match high fungal pressure.',
          }}
          nextActionByState={{
            LOW: 'Keep normal field checks and avoid watering foliage late in the evening.',
            MEDIUM: 'Check shaded lower leaves for early spots or mildew and improve airflow or drainage if crowded.',
            HIGH: 'Inspect leaves carefully for spots, stem rot, or mold today and consult your agronomist about preventive treatment before more damp weather arrives.',
            insufficient_history:
              'Keep logging continuously for 7 full days (336 packets), or upload a 336-packet JSON file to test.',
            blocked:
              'Check humidity, temperature, and leaf-wetness probes for errors in the 7-day log.',
          }}
          domainPlainNote="Trained on cool subalpine wild flax rust (10.2–18.9°C), not chilli fungal diseases."
        />
      </div>
    </section>
  );
}
