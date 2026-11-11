import React from 'react';

const STATUS_LABELS = {
  ready: 'Ready',
  insufficient_history: 'Not Enough Data Yet',
  blocked: 'Paused (Sensor Issue)',
};

const RISK_DISPLAY_LABELS = {
  LOW: 'Low Risk',
  MEDIUM: 'Medium Risk',
  HIGH: 'High Risk',
};

function formatConfidence(confidence, confidenceLevel) {
  if (typeof confidence === 'number' && Number.isFinite(confidence)) {
    const pct = `${(confidence * 100).toFixed(0)}%`;
    if (confidenceLevel && confidenceLevel !== 'unavailable') {
      const level =
        confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1).toLowerCase();
      return `${pct} (${level})`;
    }
    return pct;
  }
  return 'Not available yet';
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

function formatWarningText(rawText) {
  if (rawText.startsWith('OUT_OF_DISTRIBUTION_CLIMATE:')) {
    return rawText.replace('OUT_OF_DISTRIBUTION_CLIMATE:', 'Weather Difference Notice:');
  }
  if (rawText.startsWith('DOMAIN_LIMITATION')) {
    return rawText.replace(/^DOMAIN_LIMITATION\s*\(([^)]+)\):\s*/, 'Crop Difference ($1): ');
  }
  return rawText;
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
        <span className="agri-field-label">Data Collected ({horizonLabel})</span>
        <span className="agri-history-ratio">
          {safeProvided.toLocaleString()} / {safeRequired.toLocaleString()} readings ({pct}%)
        </span>
      </div>
      <progress
        className="agri-progress"
        value={clampedValue}
        max={safeRequired}
        aria-label={`Sensor readings collected (${horizonLabel}): ${safeProvided} of ${safeRequired} readings`}
      />
      {missing > 0 && (
        <span className="agri-history-missing">
          Needs {missing.toLocaleString()} more readings — missing days are never guessed.
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
        text: formatWarningText(text),
        isDomain:
          domainLimitations.includes(raw) ||
          text.startsWith('DOMAIN_LIMITATION') ||
          text.startsWith('OUT_OF_DISTRIBUTION'),
      });
    }
  }

  if (items.length === 0) return null;

  return (
    <ul className="agri-warnings-list" aria-label="Warnings and notices">
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
  plainSummaryReady,
  nextActionByState,
}) {
  const status = data?.status || 'insufficient_history';
  const ready = Boolean(data?.ready);
  const provided = data?.packets_provided ?? totalPacketsReceived ?? 0;
  const required = data?.packets_required ?? requiredPackets;
  const riskLevel = ready ? data?.risk_level : null;

  let headlineText = 'Not Enough Data Yet';
  let outcomeSub = `${provided.toLocaleString()} of ${required.toLocaleString()} readings (${horizonLabel} needed)`;
  let statusMeaning = `Needs ${horizonLabel} of steady sensor readings before checking this risk.`;
  let nextAction = nextActionByState.insufficient_history;

  if (status === 'blocked') {
    headlineText = 'Paused for Safety';
    outcomeSub = 'Missing or faulty sensor values found';
    statusMeaning = 'Some sensor readings in this window had gaps or errors, so the check was paused safely.';
    nextAction = nextActionByState.blocked;
  } else if (ready && riskLevel) {
    headlineText = RISK_DISPLAY_LABELS[riskLevel] || riskLevel;
    outcomeSub = `Based on the past ${horizonLabel}`;
    statusMeaning =
      plainSummaryReady[riskLevel] || `Recent field weather shows ${riskLevel.toLowerCase()} risk.`;
    nextAction = nextActionByState[riskLevel] || nextActionByState.MEDIUM;
  }

  const outcomeClass = getRiskOutcomeClass(status, riskLevel);
  const hasNotices =
    (data?.domain_limitations && data.domain_limitations.length > 0) ||
    (data?.warnings && data.warnings.length > 0);

  return (
    <article className={`agri-card status-${status}`}>
      <header className="agri-card-header">
        <div className="agri-card-title-wrap">
          <h3 className="agri-card-title">{title}</h3>
          <span className="agri-card-sub">Past {horizonLabel} check</span>
        </div>
        <span className={`agri-readiness-pill readiness-${status}`}>
          {STATUS_LABELS[status] || status}
        </span>
      </header>

      <div className="agri-card-content">
        <div className={`agri-outcome-box ${outcomeClass}`}>
          <span className="agri-outcome-kicker">Current Result</span>
          <span className="agri-outcome-value">{headlineText}</span>
          <span className="agri-outcome-note">{outcomeSub}</span>
        </div>

        <div className="agri-meaning-box">
          <div className="agri-meaning-title">What this means</div>
          <p className="agri-meaning-text">{statusMeaning}</p>
          <div className="agri-next-action">
            <strong>What to do next:</strong> {nextAction}
          </div>
        </div>

        <dl className="agri-meta-list">
          <div className="agri-meta-row">
            <dt>How Sure Is This?</dt>
            <dd>{formatConfidence(data?.confidence, data?.confidence_level)}</dd>
          </div>
        </dl>

        <PacketHistoryMeter
          provided={provided}
          required={required}
          horizonLabel={horizonLabel}
        />

        {hasNotices && (
          <div className="agri-alerts-box">
            <PlainWarningsList
              domainLimitations={data?.domain_limitations || []}
              warnings={data?.warnings || []}
            />
          </div>
        )}
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

  const currentSm = data?.stabilized_soil_moisture?.toFixed(1) ?? '—';
  const forecastSm = data?.predicted_soil_moisture_24h?.toFixed(1) ?? '—';

  let headlineText = 'Not Enough Data Yet';
  let outcomeSub = `${provided.toLocaleString()} of ${required.toLocaleString()} readings (24 hours needed)`;
  let statusMeaning = 'Needs 1 full day (48 half-hour readings) of soil readings before giving watering advice.';
  let nextAction = 'Pick a sample day above and click Show Demo Results, or let your field sensor record for 24 hours.';

  if (status === 'blocked') {
    headlineText = 'Paused for Safety';
    outcomeSub = 'Faulty sensor values found';
    statusMeaning = 'Watering advice stopped safely because the readings include impossible values (like negative air moisture or extreme heat spikes).';
    nextAction = 'Check the soil and weather sensors for loose wires before watering.';
  } else if (ready) {
    headlineText = isRequired ? 'Water Crop Soon' : 'No Water Needed Today';
    outcomeSub = `Now: ${currentSm}% • Tomorrow: ${forecastSm}% (Goal: ${madp}%)`;
    if (isRequired) {
      statusMeaning = `Soil water (${currentSm}%) or tomorrow's level (${forecastSm}%) drops below your ${madp}% crop comfort goal.`;
      nextAction = `Turn on watering or plan irrigation soon so root moisture rises safely above ${madp}%.`;
    } else {
      statusMeaning = `Soil water now (${currentSm}%) and tomorrow's outlook (${forecastSm}%) stay safely above your ${madp}% goal.`;
      nextAction = 'Keep the water off for now and check again after new readings arrive.';
    }
  }

  const outcomeClass = getIrrigationOutcomeClass(status, isRequired);
  const hasWarnings = data?.warnings && data.warnings.length > 0;

  return (
    <article className={`agri-card status-${status}`}>
      <header className="agri-card-header">
        <div className="agri-card-title-wrap">
          <h3 className="agri-card-title">Watering Advice</h3>
          <span className="agri-card-sub">Next 24 hours check</span>
        </div>
        <span className={`agri-readiness-pill readiness-${status}`}>
          {STATUS_LABELS[status] || status}
        </span>
      </header>

      <div className="agri-card-content">
        <div className={`agri-outcome-box ${outcomeClass}`}>
          <span className="agri-outcome-kicker">Watering Guidance</span>
          <span className="agri-outcome-value">{headlineText}</span>
          <span className="agri-outcome-note">{outcomeSub}</span>
        </div>

        <div className="agri-meaning-box">
          <div className="agri-meaning-title">What this means</div>
          <p className="agri-meaning-text">{statusMeaning}</p>
          <div className="agri-next-action">
            <strong>What to do next:</strong> {nextAction}
          </div>
        </div>

        <dl className="agri-meta-list">
          <div className="agri-meta-row">
            <dt>How Sure Is This?</dt>
            <dd>{formatConfidence(data?.confidence, data?.confidence_level)}</dd>
          </div>
          {ready && (
            <div className="agri-meta-row">
              <dt>Watering Switch</dt>
              <dd>{data?.actuator_decision ? 'On (Watering)' : 'Off (Closed)'}</dd>
            </div>
          )}
        </dl>

        <PacketHistoryMeter
          provided={provided}
          required={required}
          horizonLabel="24 hours"
        />

        {hasWarnings && (
          <div className="agri-alerts-box">
            <PlainWarningsList warnings={data.warnings} />
          </div>
        )}
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
    >
      <div className="section-heading-row">
        <div>
          <h2 id="agri-predictions-heading" className="section-title">
            Farm Advice &amp; Risk Checks
          </h2>
          <p className="section-helper">
            Simple next steps for watering, crop pests, and crop disease. Missing data is never guessed.
          </p>
        </div>
        <div className="agri-header-badges">
          <span className={isSyntheticSource ? 'synthetic-badge' : 'uploaded-badge'}>
            {isSyntheticSource ? 'SIMULATED DEMO — NOT LIVE FARM DATA' : 'UPLOADED FILE READINGS'}
          </span>
          {agriculturalResult && (
            <span className="agri-packet-pill">
              {totalReceived.toLocaleString()} Readings Checked • {cropId.toUpperCase()}
            </span>
          )}
        </div>
      </div>

      {/* Explicit Simulated Demo & Crop Limitation Notice */}
      <div className="agri-limitation-banner" role="note" aria-label="Simulated demo notice and limits">
        <div className="agri-limitation-title">
          {isSyntheticSource
            ? 'Simulated Demo Notice: Example Readings Only (Not Live Field Data)'
            : `Uploaded File Readings (${sourceDescription || `${totalReceived} readings`}) — Check Locally`}
        </div>
        <p className="agri-limitation-summary">
          {isSyntheticSource
            ? 'These 24-hour examples are computer-made for testing, not live readings from a farm sensor. Because Disease Risk (7 days) and Pest Risk (28 days) need longer logs, they show Not Enough Data Yet instead of guessing. '
            : 'Uploaded readings are checked over 1 day, 7 days, and 28 days without inventing missing hours. '}
          Also note: disease and pest checks were trained on other plants (wild flax and rice), so check your <strong>{cropId}</strong> plants directly before taking action.
        </p>
      </div>

      {agriculturalError && (
        <div className="health-details error" role="alert">
          <div className="details-header">Could Not Run Farm Checks</div>
          <p className="error-message">{agriculturalError}</p>
        </div>
      )}

      {isPredicting && (
        <div className="analyzing-banner">
          <div className="spinner-medium" />
          <div className="analyzing-text">
            <strong>Checking Field Conditions...</strong>
            <span>Checking watering (1 day), crop disease (7 days), and crop pests (28 days).</span>
          </div>
        </div>
      )}

      <div className="agri-cards-grid">
        <RiskCard
          title="Risk of Crop Pests"
          horizonLabel="28 days"
          requiredPackets={1344}
          data={agriculturalResult?.pest_risk}
          totalPacketsReceived={totalReceived}
          plainSummaryReady={{
            LOW: 'Weather over the past 4 weeks looks unfavorable for heavy insect pests.',
            MEDIUM: 'Warmth and moisture over the past 4 weeks allow moderate insect activity.',
            HIGH: 'Past 4 weeks of weather match conditions tied to high pest outbreaks.',
          }}
          nextActionByState={{
            LOW: 'Keep doing your normal weekly walk-through. No emergency spray needed.',
            MEDIUM: 'Check the undersides of leaves and traps twice this week for early bugs.',
            HIGH: 'Walk your rows today to check bug numbers, and ask an advisor before spraying.',
            insufficient_history:
              'Keep recording until 28 straight days (1,344 readings) are saved, or upload a 28-day file.',
            blocked:
              'Fix missing or faulty weather readings in the older 4-week log, then try again.',
          }}
        />

        <IrrigationCard
          data={agriculturalResult?.irrigation}
          totalPacketsReceived={totalReceived}
        />

        <RiskCard
          title="Risk of Crop Disease"
          horizonLabel="7 days"
          requiredPackets={336}
          data={agriculturalResult?.fungal_risk}
          totalPacketsReceived={totalReceived}
          plainSummaryReady={{
            LOW: 'Leaf wetness and humidity over the last 7 days do not favor leaf fungus.',
            MEDIUM: 'Damp air and wet leaves over the last 7 days show moderate disease risk.',
            HIGH: 'Long wet-leaf and humid hours over the last 7 days raise fungal disease risk.',
          }}
          nextActionByState={{
            LOW: 'Keep your normal checks and avoid wetting leaves late in the evening.',
            MEDIUM: 'Look at shaded lower leaves for spots or mold and keep rows well ventilated.',
            HIGH: 'Inspect leaves closely for spots or rot today and talk to an advisor before more damp weather arrives.',
            insufficient_history:
              'Keep recording for 7 straight days (336 readings), or upload a 7-day file to check disease risk.',
            blocked:
              'Check your humidity, temperature, and leaf-wetness sensors for errors in the 7-day log.',
          }}
        />
      </div>
    </section>
  );
}
