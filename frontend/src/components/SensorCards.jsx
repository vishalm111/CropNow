import React from 'react';
import InfoTip from './InfoTip';

function fmt(value, decimals = 1, unit = '') {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return `${value.toFixed(decimals)}${unit}`;
  }
  return '—';
}

export default function SensorCards({
  activePackets,
  stabilityResult,
  agriculturalResult,
}) {
  const count = Array.isArray(activePackets) ? activePackets.length : 0;
  const hasReadings = count > 0;
  const latestPkt = hasReadings ? activePackets[count - 1] : null;
  const latestTimeline = stabilityResult?.timeline?.slice(-1)[0] || null;

  const cleanedSoilWater =
    agriculturalResult?.irrigation?.stabilized_soil_moisture ??
    latestTimeline?.stabilized_average ??
    null;
  const rawSm1 = latestPkt?.soil_moisture_1;
  const rawSm2 = latestPkt?.soil_moisture_2;
  const displaySm =
    cleanedSoilWater !== null
      ? cleanedSoilWater
      : typeof rawSm1 === 'number' && typeof rawSm2 === 'number'
        ? (rawSm1 + rawSm2) / 2
        : null;

  const targetLevel =
    stabilityResult?.madp_threshold ??
    agriculturalResult?.irrigation?.irrigation_advice?.madp_threshold ??
    null;

  let soilPlainHint = 'Pick a demo day or upload a file above to see soil water levels.';
  let soilTone = 'neutral';
  if (hasReadings && displaySm !== null) {
    if (targetLevel !== null && displaySm < targetLevel) {
      soilPlainHint = `Getting dry (below ${targetLevel}% goal). Two soil sensors read ${fmt(rawSm1, 1, '%')} and ${fmt(rawSm2, 1, '%')}.`;
      soilTone = 'warn';
    } else {
      soilPlainHint = `Soil has enough water. Two soil sensors read ${fmt(rawSm1, 1, '%')} and ${fmt(rawSm2, 1, '%')}.`;
      soilTone = 'good';
    }
  }

  let tempPlainHint = 'Waiting for air and soil temperature readings.';
  if (typeof latestPkt?.temperature === 'number') {
    if (latestPkt.temperature >= 32) {
      tempPlainHint = `Hot day — soil dries out faster. Soil warmth: ${fmt(latestPkt?.soil_temperature, 1, ' degrees')}.`;
    } else if (latestPkt.temperature <= 18) {
      tempPlainHint = `Cool weather — soil dries slowly. Soil warmth: ${fmt(latestPkt?.soil_temperature, 1, ' degrees')}.`;
    } else {
      tempPlainHint = `Mild weather for crops. Soil warmth: ${fmt(latestPkt?.soil_temperature, 1, ' degrees')}.`;
    }
  }

  let humPlainHint = 'Waiting for air moisture and wet-leaf readings.';
  if (typeof latestPkt?.humidity === 'number') {
    const lwText = fmt(latestPkt?.leaf_wetness, 0, ' out of 100');
    humPlainHint =
      latestPkt.humidity >= 80
        ? `Damp air — check lower leaves often. Leaf wetness: ${lwText}.`
        : `Normal air moisture. Leaf wetness: ${lwText}.`;
  }

  let rainPlainHint = 'Waiting for rain, sunlight, and wind readings.';
  if (hasReadings) {
    const solar = fmt(latestPkt?.solar_radiation, 0);
    const wind = fmt(latestPkt?.wind_speed, 1);
    rainPlainHint =
      typeof latestPkt?.rainfall === 'number' && latestPkt.rainfall > 0
        ? `Rain recorded recently. Wind: ${wind} meters per second • Sunlight: ${solar}.`
        : `No rain recently. Wind: ${wind} meters per second • Sunlight: ${solar}.`;
  }

  const cards = [
    {
      label: 'Water in the Soil',
      tip: 'How wet the soil is around plant roots (%). When it drops below your crop goal, plants can soon run short of water.',
      value: fmt(displaySm, 1, '%'),
      sub: soilPlainHint,
      badge: cleanedSoilWater !== null ? 'Cleaned' : hasReadings ? 'Raw Average' : 'Waiting',
      tone: soilTone,
    },
    {
      label: 'Air Warmth',
      tip: 'Current air temperature in the field (degrees Celsius). Hot weather dries soil faster; cool damp weather can help plant disease grow.',
      value: fmt(latestPkt?.temperature, 1, ' °C'),
      sub: tempPlainHint,
      badge: hasReadings ? 'Latest' : 'Waiting',
      tone: 'neutral',
    },
    {
      label: 'Air Moisture & Wet Leaves',
      tip: 'How damp the air is (%) and how wet plant leaves feel (0 to 100). Leaves that stay wet for many hours raise the risk of crop disease.',
      value: fmt(latestPkt?.humidity, 0, '%'),
      sub: humPlainHint,
      badge:
        typeof latestPkt?.humidity === 'number' && latestPkt.humidity >= 80 ? 'Very Damp' : hasReadings ? 'Normal' : 'Waiting',
      tone: typeof latestPkt?.humidity === 'number' && latestPkt.humidity >= 80 ? 'warn' : 'neutral',
    },
    {
      label: 'Rain & Sunlight',
      tip: 'Rain measured in the last 30 minutes (millimeters), plus sunlight strength and wind speed.',
      value: fmt(latestPkt?.rainfall, 2, ' mm'),
      sub: rainPlainHint,
      badge:
        typeof latestPkt?.rainfall === 'number' && latestPkt.rainfall > 0 ? 'Rain' : hasReadings ? 'Dry' : 'Waiting',
      tone: 'neutral',
    },
  ];

  return (
    <section aria-labelledby="sensor-cards-heading" className="sensor-cards-section">
      <div className="section-heading-row">
        <h2 id="sensor-cards-heading" className="section-title">
          Latest Field Conditions
        </h2>
        <span className="section-helper">
          {hasReadings
            ? `Showing the newest reading (#${count.toLocaleString()} in this set)`
            : 'Choose a simulated demo or upload a sensor file above to view readings'}
        </span>
      </div>

      <div className="simple-sensor-grid">
        {cards.map((card) => (
          <article key={card.label} className={`simple-sensor-card tone-${card.tone}`}>
            <div className="simple-sensor-top">
              <span className="simple-sensor-label-group">
                <span className="simple-sensor-label">{card.label}</span>
                <InfoTip label={card.label} text={card.tip} />
              </span>
              <span className={`simple-sensor-badge badge-${card.tone}`}>{card.badge}</span>
            </div>
            <div className="simple-sensor-value">{card.value}</div>
            <p className="simple-sensor-desc">{card.sub}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
