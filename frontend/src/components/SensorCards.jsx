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
  const hasTelemetry = count > 0;
  const latestPkt = hasTelemetry ? activePackets[count - 1] : null;
  const latestTimeline = stabilityResult?.timeline?.slice(-1)[0] || null;

  const stabilizedSm =
    agriculturalResult?.irrigation?.stabilized_soil_moisture ??
    latestTimeline?.stabilized_average ??
    null;
  const rawSm1 = latestPkt?.soil_moisture_1;
  const rawSm2 = latestPkt?.soil_moisture_2;
  const displaySm =
    stabilizedSm !== null
      ? stabilizedSm
      : typeof rawSm1 === 'number' && typeof rawSm2 === 'number'
        ? (rawSm1 + rawSm2) / 2
        : null;

  const madp =
    stabilityResult?.madp_threshold ??
    agriculturalResult?.irrigation?.irrigation_advice?.madp_threshold ??
    null;

  let soilPlainHint = 'Run a demo scenario or upload JSON to view readings.';
  let soilTone = 'neutral';
  if (hasTelemetry && displaySm !== null) {
    if (madp !== null && displaySm < madp) {
      soilPlainHint = `Dry soil (below ${madp}% target). Probes: ${fmt(rawSm1, 1, '%')} / ${fmt(rawSm2, 1, '%')}.`;
      soilTone = 'warn';
    } else {
      soilPlainHint = `Healthy moisture level. Probes: ${fmt(rawSm1, 1, '%')} / ${fmt(rawSm2, 1, '%')}.`;
      soilTone = 'good';
    }
  }

  let tempPlainHint = 'Awaiting sensor data.';
  if (typeof latestPkt?.temperature === 'number') {
    if (latestPkt.temperature >= 32) {
      tempPlainHint = `Warm air conditions. Soil temp: ${fmt(latestPkt?.soil_temperature, 1, ' °C')}.`;
    } else if (latestPkt.temperature <= 18) {
      tempPlainHint = `Cool air conditions. Soil temp: ${fmt(latestPkt?.soil_temperature, 1, ' °C')}.`;
    } else {
      tempPlainHint = `Mild field temperature. Soil temp: ${fmt(latestPkt?.soil_temperature, 1, ' °C')}.`;
    }
  }

  let humPlainHint = 'Awaiting humidity & canopy wetness data.';
  if (typeof latestPkt?.humidity === 'number') {
    const lwText = fmt(latestPkt?.leaf_wetness, 0, '/100');
    humPlainHint =
      latestPkt.humidity >= 80
        ? `Damp air — monitor leaves closely. Leaf wetness: ${lwText}.`
        : `Moderate air moisture. Leaf wetness: ${lwText}.`;
  }

  let rainPlainHint = 'Awaiting rain & sunlight sensors.';
  if (hasTelemetry) {
    const solar = fmt(latestPkt?.solar_radiation, 0, ' W/m²');
    const wind = fmt(latestPkt?.wind_speed, 1, ' m/s');
    rainPlainHint =
      typeof latestPkt?.rainfall === 'number' && latestPkt.rainfall > 0
        ? `Recent rainfall recorded. Light: ${solar} • Wind: ${wind}.`
        : `Dry window (no rain). Light: ${solar} • Wind: ${wind}.`;
  }

  const cards = [
    {
      label: 'Soil Moisture',
      tip: 'How wet the soil is around the roots (%). Readings below the crop target mean plants may struggle to take up water.',
      value: fmt(displaySm, 1, '%'),
      sub: soilPlainHint,
      badge: stabilizedSm !== null ? 'Stabilized' : hasTelemetry ? 'Raw Avg' : 'No Data',
      tone: soilTone,
    },
    {
      label: 'Air Temperature',
      tip: 'Current air warmth in the field (°C). Hot spells dry soil faster; cool humid periods can favor fungal growth.',
      value: fmt(latestPkt?.temperature, 1, ' °C'),
      sub: tempPlainHint,
      badge: hasTelemetry ? 'Live Snapshot' : 'No Data',
      tone: 'neutral',
    },
    {
      label: 'Humidity & Leaf Wetness',
      tip: 'Moisture in the air (%) and surface water film on leaves (0–100). Wet leaves over many hours raise fungal disease risk.',
      value: fmt(latestPkt?.humidity, 0, '%'),
      sub: humPlainHint,
      badge:
        typeof latestPkt?.humidity === 'number' && latestPkt.humidity >= 80 ? 'High RH' : hasTelemetry ? 'Normal' : 'No Data',
      tone: typeof latestPkt?.humidity === 'number' && latestPkt.humidity >= 80 ? 'warn' : 'neutral',
    },
    {
      label: 'Rainfall & Sunlight',
      tip: 'Rain measured during the last 30-minute step alongside sunlight intensity (W/m²) and wind speed.',
      value: fmt(latestPkt?.rainfall, 2, ' mm'),
      sub: rainPlainHint,
      badge:
        typeof latestPkt?.rainfall === 'number' && latestPkt.rainfall > 0 ? 'Rain' : hasTelemetry ? 'Clear' : 'No Data',
      tone: 'neutral',
    },
  ];

  return (
    <section aria-labelledby="sensor-cards-heading" className="sensor-cards-section">
      <div className="section-heading-row">
        <h2 id="sensor-cards-heading" className="section-title">
          Current Field Conditions
        </h2>
        <span className="section-helper">
          {hasTelemetry
            ? `Latest reading from packet #${count}`
            : 'Shown after you run a demo scenario or upload packet JSON'}
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
