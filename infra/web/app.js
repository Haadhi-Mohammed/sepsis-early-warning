// Sepsis early warning dashboard: a static page served from S3 through
// CloudFront. Requests go to /api/*, which CloudFront forwards to the Lambda
// Function URL in front of the SageMaker Serverless endpoint.

// "?api=https://..." overrides the API base (handy for local testing)
const API = new URLSearchParams(location.search).get('api') || '/api';
const HOURS = 6;
const REQUEST_TIMEOUT_MS = 75000;   // CloudFront waits at most 60 s for the API

const ROWS = [
  { section: 'Vital signs' },
  { key: 'HR',    label: 'Heart rate',       unit: 'bpm',    normal: [60, 100],   step: 1 },
  { key: 'O2Sat', label: 'O2 saturation',    unit: '%',      normal: [95, 100],   step: 1 },
  { key: 'Temp',  label: 'Temperature',      unit: '°C',     normal: [36.1, 37.8], step: 0.1 },
  { key: 'SBP',   label: 'Systolic BP',      unit: 'mmHg',   normal: [100, 140],  step: 1 },
  { key: 'MAP',   label: 'Mean arterial BP', unit: 'mmHg',   normal: [65, 105],   step: 1 },
  { key: 'DBP',   label: 'Diastolic BP',     unit: 'mmHg',   normal: [60, 90],    step: 1 },
  { key: 'Resp',  label: 'Respiratory rate', unit: '/min',   normal: [12, 20],    step: 1 },
  { section: 'Labs (optional)' },
  { key: 'Lactate',    label: 'Lactate',    unit: 'mmol/L', normal: [0.5, 2],   step: 0.1 },
  { key: 'WBC',        label: 'WBC',        unit: '10³/µL', normal: [4, 11],    step: 0.1 },
  { key: 'Creatinine', label: 'Creatinine', unit: 'mg/dL',  normal: [0.6, 1.2], step: 0.1 },
  { key: 'Glucose',    label: 'Glucose',    unit: 'mg/dL',  normal: [70, 140],  step: 1 },
  { key: 'pH',         label: 'pH',         unit: '',       normal: [7.35, 7.45], step: 0.01 },
  { key: 'Hgb',        label: 'Haemoglobin', unit: 'g/dL',  normal: [12, 17],   step: 0.1 },
];
const FIELDS = ROWS.filter(r => r.key);
const TRENDS = ['HR', 'Resp', 'Temp', 'SBP', 'MAP', 'O2Sat'];

const series = (from, to, digits = 0) =>
  Array.from({ length: HOURS }, (_, i) => +(from + (to - from) * i / (HOURS - 1)).toFixed(digits));
const PRESETS = {
  stable: {
    HR: series(80, 78), O2Sat: series(98, 98), Temp: series(36.9, 36.8, 1),
    SBP: series(124, 122), MAP: series(86, 85), DBP: series(70, 69), Resp: series(16, 15),
    Glucose: [null, null, 110, null, null, null],
  },
  deteriorating: {
    HR: series(99, 119), O2Sat: series(95, 92), Temp: series(38.1, 39.6, 1),
    SBP: series(105, 80), MAP: series(74, 56), DBP: series(53, 40), Resp: series(22, 32),
    Lactate: [null, null, null, null, 2.4, null], WBC: [null, null, null, null, 13.5, null],
  },
  septic: {
    HR: series(112, 128), O2Sat: series(92, 89), Temp: series(38.9, 39.4, 1),
    SBP: series(95, 78), MAP: series(65, 55), DBP: series(52, 44), Resp: series(26, 32),
    Lactate: [null, 3.5, null, null, 4.8, null], WBC: [null, 16.2, null, null, null, null],
    Creatinine: [null, 1.6, null, null, 2.1, null],
  },
};

const $ = id => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

// ── Input table ──────────────────────────────────────────────────────────
function buildTable() {
  const head = el('tr', {}, el('th', { scope: 'col', textContent: '' }));
  for (let h = 0; h < HOURS; h++) {
    head.append(el('th', { scope: 'col', textContent: h === HOURS - 1 ? 'Latest' : `-${HOURS - 1 - h} h` }));
  }
  $('readingsTable').tHead.append(head);
  const body = $('readingsTable').tBodies[0];
  for (const row of ROWS) {
    if (row.section) {
      body.append(el('tr', { className: 'section' }, el('th', { colSpan: HOURS + 1, textContent: row.section })));
      continue;
    }
    const tr = el('tr', {}, el('th', { scope: 'row' }, row.label, el('small', { textContent: row.unit })));
    for (let h = 0; h < HOURS; h++) {
      tr.append(el('td', {}, el('input', {
        type: 'number', step: row.step, inputMode: 'decimal', id: `${row.key}-${h}`,
        ariaLabel: `${row.label}, ${h === HOURS - 1 ? 'latest hour' : `${HOURS - 1 - h} hours earlier`}`,
      })));
    }
    body.append(tr);
  }
}

function applyPreset(name) {
  const preset = PRESETS[name];
  for (const { key } of FIELDS) {
    for (let h = 0; h < HOURS; h++) {
      const v = preset[key]?.[h];
      $(`${key}-${h}`).value = v ?? '';
    }
  }
  $('patientId').value = { stable: 'P-STABLE', deteriorating: 'P-DETER', septic: 'P-SHOCK' }[name];
}

function readForm() {
  const lastHour = Math.max(HOURS, parseInt($('lastHour').value, 10) || HOURS);
  const num = v => (v === '' || v == null || isNaN(+v) ? null : +v);
  const readings = [];
  for (let h = 0; h < HOURS; h++) {
    const r = {
      Age: num($('age').value), Gender: num($('gender').value),
      HospAdmTime: num($('hospAdm').value) == null ? null : -num($('hospAdm').value),
      ICULOS: lastHour - (HOURS - 1 - h),
    };
    for (const { key } of FIELDS) r[key] = num($(`${key}-${h}`).value);
    readings.push(r);
  }
  return { patient_id: $('patientId').value.trim() || 'P001', readings };
}

// ── API calls ────────────────────────────────────────────────────────────
async function callApi(path, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(API + path, { ...options, signal: ctrl.signal });
    const body = await res.json().catch(() => ({}));
    return { ok: res.ok, status: res.status, body };
  } catch (err) {
    return { ok: false, status: 0, body: { detail: err.name === 'AbortError' ? 'timed out' : String(err) } };
  } finally {
    clearTimeout(timer);
  }
}

const postPredict = payload => callApi('/predict', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
});
// 502/503/504 or no response usually means the endpoint was still starting
const retryable = r => r.status === 0 || r.status >= 502;

function setStatus(state, text) {
  $('modelStatus').className = `status ${state}`;
  $('modelStatus').querySelector('.label').textContent = text;
}

// Wake the serverless endpoint as soon as the page opens, so it is usually
// ready by the time the form has been filled in.
async function warmUp() {
  const health = await callApi('/health');
  if (!health.ok) return setStatus('down', 'API unreachable');
  if (health.body.served_model) $('servedModel').textContent = `Serving registry model ${health.body.served_model}.`;
  setStatus('waking', 'Waking model…');
  const ping = { patient_id: 'warm-up', readings: [{ HR: 80, ICULOS: 1 }] };
  for (let attempt = 0; attempt < 2; attempt++) {
    const r = await postPredict(ping);
    if (r.ok) return setStatus('ready', 'Model ready');
    if (!retryable(r)) break;
  }
  setStatus('down', 'Model not responding');
}

// ── Results ──────────────────────────────────────────────────────────────
function renderFactors(listId, factors) {
  const list = $(listId);
  list.replaceChildren();
  if (!factors.length) return list.append(el('li', { className: 'none', textContent: 'None' }));
  const max = Math.max(...factors.map(f => Math.abs(f.shap_value)), 1e-9);
  for (const f of factors) {
    list.append(el('li', {},
      el('div', { className: 'row' }, el('span', { textContent: f.display_name }),
        el('span', { className: 'val', textContent: f.contribution })),
      el('div', { className: 'track' }, el('div', {
        className: 'fill', style: `width:${Math.max(4, 100 * Math.abs(f.shap_value) / max)}%` }))));
  }
}

function sparkline(values, normal) {
  const known = values.filter(v => v != null);
  const lo = Math.min(...known, normal[0]), hi = Math.max(...known, normal[1]);
  const pad = (hi - lo) * 0.1 || 1, min = lo - pad, max = hi + pad;
  const x = i => (i / (HOURS - 1)) * 100;
  const y = v => 40 - ((v - min) / (max - min)) * 36;
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 100 42');
  svg.setAttribute('preserveAspectRatio', 'none');
  const band = document.createElementNS(ns, 'rect');
  Object.entries({ class: 'band', x: 0, width: 100, y: y(normal[1]), height: y(normal[0]) - y(normal[1]) })
    .forEach(([k, v]) => band.setAttribute(k, v));
  const line = document.createElementNS(ns, 'polyline');
  line.setAttribute('class', 'line');
  line.setAttribute('points', values.map((v, i) => (v == null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(' '));
  svg.append(band, line);
  return svg;
}

function renderTrends(readings) {
  const box = $('trends');
  box.replaceChildren();
  for (const key of TRENDS) {
    const row = FIELDS.find(f => f.key === key);
    const values = readings.map(r => r[key]);
    const last = [...values].reverse().find(v => v != null);
    if (last == null) continue;
    const out = last < row.normal[0] || last > row.normal[1];
    box.append(el('div', { className: `trend${out ? ' out' : ''}` },
      el('div', { className: 't-head' }, el('span', { className: 't-name', textContent: row.label }),
        el('span', { className: 't-last', textContent: `${last} ${row.unit}` })),
      sparkline(values, row.normal)));
  }
}

function renderNotes(readings) {
  const latest = key => [...readings].reverse().map(r => r[key]).find(v => v != null);
  const v = Object.fromEntries(FIELDS.map(f => [f.key, latest(f.key)]));
  const notes = [];
  if (v.HR > 100) notes.push(`Tachycardia: heart rate ${v.HR} bpm`);
  if (v.Temp > 38.3) notes.push(`Fever: ${v.Temp} °C`);
  if (v.Temp != null && v.Temp < 36) notes.push(`Hypothermia: ${v.Temp} °C`);
  if (v.Resp >= 22) notes.push(`Respiratory rate ${v.Resp}/min (qSOFA criterion: ≥ 22)`);
  if (v.SBP <= 100) notes.push(`Systolic BP ${v.SBP} mmHg (qSOFA criterion: ≤ 100)`);
  if (v.MAP != null && v.MAP < 65) notes.push(`Mean arterial pressure ${v.MAP} mmHg is below 65`);
  if (v.O2Sat != null && v.O2Sat < 92) notes.push(`Low oxygen saturation: ${v.O2Sat}%`);
  if (v.Lactate > 2) notes.push(`Raised lactate: ${v.Lactate} mmol/L`);
  const list = $('notes');
  list.replaceChildren(...(notes.length
    ? notes.map(n => el('li', { textContent: n }))
    : [el('li', { className: 'ok', textContent: 'No abnormal values in the latest readings.' })]));
}

function renderResult(result, payload, seconds) {
  $('resultEmpty').hidden = true;
  $('resultBody').hidden = false;
  $('alertBanner').className = `banner ${result.alert_level}`;
  $('alertLevel').textContent = result.alert_level;
  $('alertMessage').textContent = result.alert_message;
  $('alertMeta').textContent =
    `Patient ${result.patient_id} · ${result.hours_of_data} hours · answered in ${seconds.toFixed(1)} s`;
  $('riskValue').textContent = `${(result.risk_score * 100).toFixed(1)}%`;
  renderFactors('riskUp', result.top_risk_factors || []);
  renderFactors('riskDown', result.protective_factors || []);
  renderTrends(payload.readings);
  renderNotes(payload.readings);
}

function renderError(message) {
  $('resultEmpty').hidden = false;
  $('resultBody').hidden = true;
  $('resultEmpty').replaceChildren(el('p', { className: 'error', textContent: message }));
}

async function onSubmit(event) {
  event.preventDefault();
  const payload = readForm();
  if (!payload.readings.some(r => FIELDS.some(f => r[f.key] != null))) {
    return renderError('Enter at least one reading, or choose an example patient.');
  }
  const button = $('submitBtn');
  button.disabled = true;
  button.textContent = 'Assessing…';
  const started = performance.now();
  let r = await postPredict(payload);
  if (retryable(r)) {
    // A cold start can outlast CloudFront's 60 s wait; by now the model is
    // usually up, so one retry succeeds quickly.
    button.textContent = 'Model is starting, retrying…';
    setStatus('waking', 'Waking model…');
    r = await postPredict(payload);
  }
  button.disabled = false;
  button.textContent = 'Assess sepsis risk';
  if (r.ok) {
    setStatus('ready', 'Model ready');
    renderResult(r.body, payload, (performance.now() - started) / 1000);
  } else {
    renderError(`Could not get a prediction (${r.status || 'no response'}): ${r.body.detail || 'please try again'}`);
  }
}

buildTable();
applyPreset('deteriorating');
document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => applyPreset(b.dataset.preset)));
$('patientForm').addEventListener('submit', onSubmit);
warmUp();
