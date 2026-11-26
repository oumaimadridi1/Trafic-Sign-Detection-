(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PAL = ['#2F6BFF','#00A6C8','#6C5CE7','#0F9D6E','#E8A31A','#E5484D','#0E7490','#C2410C','#4F46E5','#0891B2'];
const hue = id => PAL[((id % PAL.length) + PAL.length) % PAL.length];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const pct = x => (x * 100).toFixed(0) + '%';
const S = { info: null, src: 'webcam', running: false, stream: null, lastDets: [], stable: null, recent: [], espN: -1, espTimer: null, vidJob: null, shown: [] };

async function J(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch {} throw new Error(t || r.status); }
  return r.json();
}
let toastT;
function toast(msg, bad) {
  const t = $('#toast'); t.textContent = msg; t.className = 'toast' + (bad ? ' bad' : ''); t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => t.hidden = true, 4000);
}

/* ============================== tabs ============================== */
$('#tabs').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  $$('#tabs button').forEach(x => x.classList.toggle('on', x === b));
  $$('.tab').forEach(t => t.hidden = t.id !== 'tab-' + b.dataset.tab);
  if (b.dataset.tab === 'history') loadHistory(true);
  if (b.dataset.tab === 'model') loadModel();
});

/* ============================== canvas ============================== */
const view = $('#view'), vctx = view.getContext('2d');
function setSize(w, h) { if (view.width !== w || view.height !== h) { view.width = w; view.height = h; } }
function drawBoxes(dets) {
  const W = view.width, H = view.height, fs = Math.max(13, Math.round(W / 55));
  vctx.lineWidth = Math.max(2, W / 320); vctx.font = `600 ${fs}px system-ui`;
  for (const d of dets) {
    const x = d.x1 * W, y = d.y1 * H, w = (d.x2 - d.x1) * W, h = (d.y2 - d.y1) * H, col = hue(d.cls_id);
    vctx.strokeStyle = col; vctx.strokeRect(x, y, w, h);
    const t = `${d.label} ${pct(d.conf)}`, tw = vctx.measureText(t).width + 10, th = fs + 8, ty = y - th < 0 ? y : y - th, tx = Math.max(0, Math.min(x, W - tw));
    vctx.fillStyle = col; vctx.fillRect(tx, ty, tw, th);
    vctx.fillStyle = '#fff'; vctx.fillText(t, tx + 5, ty + th - 7);
  }
}
function showBitmap(bmp, dets) {
  setSize(bmp.width, bmp.height); vctx.drawImage(bmp, 0, 0); drawBoxes(dets || []); $('#stageEmpty').hidden = true;
}
function clearStage(msg) {
  vctx.clearRect(0, 0, view.width, view.height);
  const e = $('#stageEmpty'); e.textContent = msg || ''; e.hidden = !msg;
}
async function showSaved(id) {           // show a saved, already annotated frame on the stage
  const img = new Image(); img.src = `/api/frame/${id}/annotated?w=1280`;
  await img.decode(); showBitmap(img, []);
}

/* ============================== speed card / alerts ============================== */
function renderSpeed(label) {
  const m = label && label.match(/\d+/);
  $('#speedBox').innerHTML = label
    ? `<div class="sign">${m ? m[0] : '?'}</div><div class="txt"><div class="big">${esc(label)}</div><div class="sub">confirmed over several frames</div></div>`
    : `<div class="sign none">–</div><div class="txt"><div class="big">No speed limit yet</div><div class="sub">Shown once a sign is seen in 3 of 5 frames</div></div>`;
}
let actx;
function alertSpeed(label) {
  if (!$('#audio').checked) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    const o = actx.createOscillator(), g = actx.createGain(); o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g); g.connect(actx.destination); o.start(); o.stop(actx.currentTime + 0.18);
    const m = label.match(/\d+/);
    if ('speechSynthesis' in window && m) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance('Speed limit ' + m[0])); }
  } catch {}
}
function handleResult(res, source) {
  if (res.stable_speed !== S.stable) { S.stable = res.stable_speed; renderSpeed(S.stable); if (S.stable) alertSpeed(S.stable); }
  $('#seen').textContent = res.detections.length ? res.detections.map(d => d.label).join(', ') : 'nothing';
  if (res.saved_frame_id && res.detections.length) {
    const d = res.detections[0];
    S.recent.unshift({ id: res.saved_frame_id, t: new Date().toLocaleTimeString(), label: d.label, conf: d.conf, source });
    S.recent = S.recent.slice(0, 8); renderRecent();
  }
}
function renderRecent() {
  $('#recent').innerHTML = S.recent.length ? S.recent.map(r =>
    `<div class="it" data-id="${r.id}"><span>${esc(r.label)} ${pct(r.conf)}</span><span class="muted">${esc(r.source)}, ${r.t}</span></div>`).join('')
    : '<span class="muted small">Saved frames will appear here.</span>';
}
$('#recent').addEventListener('click', e => { const it = e.target.closest('.it'); if (it) openFrame(+it.dataset.id); });

/* ============================== live pill ============================== */
function setLive(on, txt) { $('#stage').classList.toggle('live', on); $('#liveTxt').textContent = txt || (on ? 'Live' : 'Standby'); }

/* ============================== stats ============================== */
const lat = [];
function drawSpark() {
  const c = $('#spark'), g = c.getContext('2d'), W = c.width, H = c.height; g.clearRect(0, 0, W, H);
  if (lat.length < 2) return;
  const max = Math.max(...lat, 1) * 1.15, step = W / (lat.length - 1);
  const grad = g.createLinearGradient(0, 0, 0, H); grad.addColorStop(0, 'rgba(47,107,255,.32)'); grad.addColorStop(1, 'rgba(47,107,255,0)');
  g.beginPath(); lat.forEach((v, i) => { const x = i * step, y = H - 4 - (v / max) * (H - 8); i ? g.lineTo(x, y) : g.moveTo(x, y); });
  g.strokeStyle = '#2f6bff'; g.lineWidth = 2.4; g.lineJoin = 'round'; g.stroke();
  g.lineTo(W, H); g.lineTo(0, H); g.closePath(); g.fillStyle = grad; g.fill();
}
function renderStats(s) {
  const x = s.session;
  const tiles = [['Frames processed', x.frames_total], ['With a sign', x.frames_with_signs], ['Ignored (empty)', x.frames_ignored],
                 ['Saved', x.frames_saved], ['Frames per second', x.fps], ['Average latency', x.latency_avg_ms + ' ms']];
  $('#hudFps').textContent = x.fps; $('#hudMs').textContent = x.latency_avg_ms;
  if (x.frames_total) { lat.push(x.latency_avg_ms); if (lat.length > 40) lat.shift(); drawSpark(); }
  $('#tiles').innerHTML = tiles.map(([k, v]) => `<div class="tile"><b>${v}</b><span>${k}</span></div>`).join('');
  const max = Math.max(1, ...x.classes.map(c => c.n));
  const idOf = l => (S.info?.classes || []).indexOf(l);
  $('#classBars').innerHTML = x.classes.length ? x.classes.slice(0, 10).map(c =>
    `<div class="cbar" style="--c:${hue(Math.max(0, idOf(c.label)))}"><span>${esc(c.label)}</span><span class="muted">${c.n} found, avg ${pct(c.avg_conf)}</span><div class="track"><i style="width:${c.n / max * 100}%"></i></div></div>`).join('')
    : '<span class="muted small">Nothing detected yet.</span>';
  const fm = $('#fModel');
  if (fm && s.db.models.length !== fm.options.length - 1) {
    const cur = fm.value; fm.innerHTML = '<option value="">All</option>' + s.db.models.map(m => `<option>${esc(m)}</option>`).join(''); fm.value = cur;
  }
}
async function refreshStats() { try { renderStats(await J('/api/stats')); } catch {} }
setInterval(refreshStats, 1500);
$('#resetBtn').onclick = async () => { await J('/api/stats/reset', { method: 'POST' }); S.stable = null; renderSpeed(null); refreshStats(); };

/* ============================== settings ============================== */
let setT;
function pushSettings() {
  $('#confVal').textContent = (+$('#conf').value).toFixed(2); $('#cdVal').textContent = (+$('#cd').value).toFixed(1) + ' s';
  clearTimeout(setT);
  setT = setTimeout(() => J('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conf: +$('#conf').value, cooldown: +$('#cd').value, save: $('#saveOn').checked }) }).catch(() => {}), 200);
}
['conf', 'cd', 'saveOn'].forEach(id => $('#' + id).addEventListener('input', pushSettings));

/* ============================== source switching ============================== */
$('#srcSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  setSource(b.dataset.src);
});
function setSource(src) {
  stopCam(); stopEsp(); setLive(false); S.src = src;
  $$('#srcSeg button').forEach(x => x.classList.toggle('on', x.dataset.src === src));
  ['webcam', 'upload', 'esp32'].forEach(s => $('#panel-' + s).hidden = s !== src);
  clearStage({ webcam: 'Press “Start camera”.', upload: 'Choose or drop an image or a video.', esp32: 'Waiting for the ESP32-CAM…' }[src]);
  if (src === 'esp32') startEsp();
}

/* ============================== webcam ============================== */
const video = document.createElement('video'); video.muted = true; video.playsInline = true;
async function listCams() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const cur = $('#camSel').value;
    $('#camSel').innerHTML = '<option value="">Default camera</option>' + devs.map((d, i) => `<option value="${d.deviceId}">${esc(d.label || 'Camera ' + (i + 1))}</option>`).join('');
    $('#camSel').value = cur;
  } catch {}
}
async function startCam() {
  try {
    const id = $('#camSel').value;
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: id ? { exact: id } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) { toast('Camera error: ' + e.message + ' (allow camera access in the browser)', true); return; }
  video.srcObject = S.stream; await video.play(); listCams();
  S.running = true; setLive(true, 'Live'); $('#camBtn').textContent = 'Stop camera'; $('#stageEmpty').hidden = true;
  requestAnimationFrame(renderLoop); sendLoop();
}
function stopCam() {
  S.running = false; S.lastDets = []; setLive(false);
  if (S.stream) { S.stream.getTracks().forEach(t => t.stop()); S.stream = null; }
  $('#camBtn').textContent = 'Start camera';
}
$('#camBtn').onclick = () => S.running ? (stopCam(), clearStage('Camera stopped.')) : startCam();
function renderLoop() {
  if (!S.running) return;
  if (video.videoWidth) { setSize(video.videoWidth, video.videoHeight); vctx.drawImage(video, 0, 0); drawBoxes(S.lastDets); }
  requestAnimationFrame(renderLoop);
}
async function sendLoop() {
  const oc = document.createElement('canvas'), octx = oc.getContext('2d');
  while (S.running) {
    const t0 = performance.now();
    try {
      if (video.videoWidth) {
        const w = Math.min(960, video.videoWidth); oc.width = w; oc.height = Math.round(video.videoHeight * w / video.videoWidth);
        octx.drawImage(video, 0, 0, oc.width, oc.height);
        const blob = await new Promise(r => oc.toBlob(r, 'image/jpeg', 0.8));
        const res = await J('/api/frame?source=webcam&mode=auto', { method: 'POST', body: blob });
        S.lastDets = res.detections; handleResult(res, 'webcam');
      }
    } catch (e) { toast('Server error: ' + e.message, true); await sleep(1500); }
    await sleep(Math.max(30, 1000 / +$('#camFps').value - (performance.now() - t0)));
  }
}

/* ============================== upload (image / video) ============================== */
$('#fileIn').addEventListener('change', e => handleFiles([...e.target.files]));
const stage = $('#stage');
['dragenter', 'dragover'].forEach(ev => stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.add('drag'); }));
['dragleave', 'drop'].forEach(ev => stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.remove('drag'); }));
stage.addEventListener('drop', e => { if (S.src !== 'upload') setSource('upload'); handleFiles([...e.dataTransfer.files]); });

async function handleFiles(files) {
  for (const f of files) {
    if (f.type.startsWith('image/')) await doImage(f);
    else if (f.type.startsWith('video/')) await doVideo(f);
  }
  $('#fileIn').value = '';
}
async function doImage(f) {
  try {
    const bmp = await createImageBitmap(f);
    const res = await J('/api/frame?source=upload&mode=dets', { method: 'POST', body: f });
    showBitmap(bmp, res.detections);
    S.shown.unshift({ name: f.name, bmp, dets: res.detections, id: res.saved_frame_id }); S.shown = S.shown.slice(0, 20);
    renderUploadList(); refreshStats();
  } catch (e) { toast(f.name + ': ' + e.message, true); }
}
function renderUploadList() {
  $('#uploadList').innerHTML = S.shown.map((s, i) =>
    `<div class="it" data-i="${i}"><span>${esc(s.name)}</span><span class="muted">${s.dets.length ? esc(s.dets.map(d => d.label + ' ' + pct(d.conf)).join(', ')) : 'no sign found'}${s.id ? ', saved' : ''}</span></div>`).join('');
}
$('#uploadList').addEventListener('click', e => {
  const it = e.target.closest('.it'); if (!it) return; const s = S.shown[+it.dataset.i]; showBitmap(s.bmp, s.dets);
});
async function doVideo(f) {
  const fd = new FormData(); fd.append('file', f); fd.append('sample_fps', $('#vidFps').value);
  $('#vidBox').hidden = false; $('#vidMeta').textContent = 'Uploading ' + f.name + '…'; $('#vidBar').style.width = '0';
  $('#vidEvents').innerHTML = '';
  let job; try { job = (await J('/api/video', { method: 'POST', body: fd })).job; } catch (e) { toast(e.message, true); return; }
  S.vidJob = job; let lastShown = null;
  while (S.vidJob === job) {
    let st; try { st = await J('/api/video/' + job); } catch { break; }
    $('#vidBar').style.width = (st.progress * 100) + '%';
    setLive(st.status === 'running', 'Analysing video');
    $('#vidMeta').textContent = `${st.filename}: ${st.status}, ${(st.progress * 100).toFixed(0)}% done. ${st.processed} frames analysed, ${st.with_signs} with signs, ${st.events.length} saved.`;
    $('#vidEvents').innerHTML = st.events.map(ev => `<div class="it" data-id="${ev.frame_id}"><span>At ${ev.t} s: ${esc(ev.label)} ${pct(ev.conf)}</span><span class="muted">frame ${ev.frame_id}</span></div>`).join('');
    if (st.last_frame_id && st.last_frame_id !== lastShown && st.status === 'running') { lastShown = st.last_frame_id; showSaved(st.last_frame_id).catch(() => {}); }
    if (['done', 'error', 'cancelled'].includes(st.status)) { if (st.status === 'error') toast(st.error, true); break; }
    await sleep(600);
  }
  refreshStats();
}
$('#vidCancel').onclick = async () => { if (S.vidJob) { await J(`/api/video/${S.vidJob}/cancel`, { method: 'POST' }); S.vidJob = null; } };
$('#vidEvents').addEventListener('click', e => { const it = e.target.closest('.it'); if (it) showSaved(+it.dataset.id); });

/* ============================== ESP32 ============================== */
function startEsp() {
  const ip = (S.info?.ips || ['<PC-IP>'])[0];
  $('#espUrl').textContent = `http://${ip}:8000/ingest`;
  S.espN = -1; S.espBusy = false;
  S.espTimer = setInterval(async () => {
    if (S.espBusy) return; S.espBusy = true;
    try {
      const j = await J('/api/esp32/latest.json');
      if (j.n && j.n !== S.espN) {
        S.espN = j.n;
        const img = new Image(); img.src = '/api/esp32/latest.jpg?n=' + j.n; await img.decode();
        showBitmap(img, j.result.detections); setLive(true, 'Receiving'); handleResult(j.result, 'esp32');
      }
      $('#espStatus').textContent = j.n ? `${j.n} frames received, the last one ${j.age_s} s ago` : 'Waiting for the first frame…';
    } catch {} finally { S.espBusy = false; }
  }, 600);
}
function stopEsp() { clearInterval(S.espTimer); S.espTimer = null; setLive(false); }

/* ============================== modal ============================== */
const modal = $('#modal');
function openModal(html) { $('#modalCard').innerHTML = html; modal.hidden = false; }
function closeModal() { modal.hidden = true; $('#modalCard').innerHTML = ''; }
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

async function openFrame(id) {
  const fr = await J('/api/frame/' + id).catch(e => { toast(e.message, true); return null; }); if (!fr) return;
  const v = Date.now();
  openModal(`<div class="m-grid">
    <img src="/api/frame/${id}/annotated?w=1280&v=${v}">
    <div>
      <h3>Frame #${id} <button class="ghost tiny" data-act="close">Close</button></h3>
      <div class="muted small">${esc(fr.ts_iso)}, from ${esc(fr.source)}, model ${esc(fr.model || 'unknown')}, ${fr.width}×${fr.height}${fr.video_t != null ? ', video time ' + fr.video_t.toFixed(1) + ' s' : ''}</div>
      <div class="dets">${fr.detections.length ? fr.detections.map(d => `<div class="det"><span><span class="chip" style="--c:${hue(d.cls_id)}">${esc(d.label)}</span> ${pct(d.conf)}</span>
        <button data-act="wrong" data-id="${d.id}" data-w="${d.wrong ? 0 : 1}" class="${d.wrong ? 'danger' : ''}">${d.wrong ? 'Marked wrong ✓ (undo)' : 'Wrong detection'}</button></div>`).join('') : '<span class="muted">No detections in this frame.</span>'}</div>
      <div class="btns">
        <button data-act="flag" data-f="${fr.flag === 'missed' ? '' : 'missed'}" class="${fr.flag === 'missed' ? 'danger' : ''}">${fr.flag === 'missed' ? 'Missed sign ✓ (undo)' : 'A sign was missed'}</button>
        <button data-act="flag" data-f="${fr.flag === 'ok' ? '' : 'ok'}">${fr.flag === 'ok' ? 'Marked OK ✓ (undo)' : 'All correct'}</button>
      </div>
      <div class="btns"><button data-act="cmp" data-s="a">Use as A in Compare</button><button data-act="cmp" data-s="b">Use as B</button></div>
      <div class="btns"><a class="btn" href="/api/frame/${id}/image" download="frame_${id}.jpg">Download original</a><button class="danger" data-act="del">Delete</button></div>
      <p class="muted small">“Wrong” and “missed” marks are used when you export the dataset for Roboflow, so you can fix these cases and retrain.</p>
    </div></div>`);
  $('#modalCard').onclick = async e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const a = b.dataset.act;
    try {
      if (a === 'close') closeModal();
      else if (a === 'wrong') { await J(`/api/detection/${b.dataset.id}/wrong`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ wrong: b.dataset.w === '1' }) }); openFrame(id); refreshHistoryCard(id); }
      else if (a === 'flag') { await J(`/api/frame/${id}/flag`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ flag: b.dataset.f || null }) }); openFrame(id); refreshHistoryCard(id); }
      else if (a === 'del') { if (confirm('Delete this frame?')) { await J('/api/frame/' + id, { method: 'DELETE' }); closeModal(); loadHistory(true); } }
      else if (a === 'cmp') { closeModal(); setSlot(b.dataset.s, id); $('#tabs [data-tab=compare]').click(); }
    } catch (err) { toast(err.message, true); }
  };
}
function openPicker(cb) {
  openModal('<h3>Pick a frame <button class="ghost tiny" data-x>Close</button></h3><div class="picker" id="pickGrid">Loading…</div>');
  $('#modalCard [data-x]').onclick = closeModal;
  J('/api/history?limit=80').then(h => {
    $('#pickGrid').innerHTML = h.items.length ? h.items.map(f => `<img loading="lazy" data-id="${f.id}" title="#${f.id} ${esc(f.ts_iso)}" src="/api/frame/${f.id}/annotated?w=320">`).join('') : '<span class="muted">History is empty.</span>';
    $('#pickGrid').onclick = e => { const im = e.target.closest('img'); if (im) { closeModal(); cb(+im.dataset.id); } };
  });
}

/* ============================== history ============================== */
const H = { offset: 0, total: 0 };
function histParams() {
  const p = new URLSearchParams();
  [['label', 'fLabel'], ['source', 'fSource'], ['model', 'fModel'], ['flag', 'fFlag'], ['date_from', 'fFrom'], ['date_to', 'fTo']].forEach(([k, id]) => { if ($('#' + id).value) p.set(k, $('#' + id).value); });
  if (+$('#fConf').value > 0) p.set('min_conf', $('#fConf').value);
  return p;
}
function card(f) {
  const chips = f.detections.slice(0, 3).map(d => `<span class="chip${d.wrong ? ' bad' : ''}" style="--c:${hue(d.cls_id)}">${esc(d.label)} ${pct(d.conf)}</span>`).join('') + (f.detections.length > 3 ? `<span class="muted">+${f.detections.length - 3}</span>` : '');
  const wrong = f.detections.some(d => d.wrong);
  const badges = (f.flag === 'missed' ? '<span class="badge missed">missed</span> ' : '') + (f.flag === 'ok' ? '<span class="badge ok">ok</span> ' : '') + (wrong ? '<span class="badge wrong">wrong</span>' : '');
  return `<article class="card" data-id="${f.id}"><img loading="lazy" src="/api/frame/${f.id}/annotated?w=400&v=${f.flag || ''}${wrong ? 'w' : ''}">
    <div class="meta"><div>${chips || '<span class="muted">no detections</span>'}</div><div style="display:flex;justify-content:space-between"><span>${esc(f.ts_iso)}</span><span>${esc(f.source)} ${badges}</span></div></div></article>`;
}
async function loadHistory(reset) {
  if (reset) { H.offset = 0; $('#histGrid').innerHTML = ''; }
  const p = histParams(); p.set('limit', 40); p.set('offset', H.offset);
  try {
    const h = await J('/api/history?' + p); H.total = h.total;
    $('#histGrid').insertAdjacentHTML('beforeend', h.items.map(card).join(''));
    H.offset += h.items.length;
    $('#histInfo').textContent = `${h.total} saved frame${h.total === 1 ? '' : 's'} match your filters`;
    $('#moreBtn').hidden = H.offset >= h.total;
    const q = histParams().toString();
    $('#expCsv').href = '/api/history.csv?' + q; $('#expZip').href = '/api/export/dataset.zip?' + q;
  } catch (e) { toast(e.message, true); }
}
async function refreshHistoryCard(id) {
  const el = $(`.card[data-id="${id}"]`); if (!el) return;
  const f = await J('/api/frame/' + id); const t = document.createElement('div'); t.innerHTML = card(f); el.replaceWith(t.firstElementChild);
}
$('#fApply').onclick = () => loadHistory(true);
$('#moreBtn').onclick = () => loadHistory(false);
$('#histGrid').addEventListener('click', e => { const c = e.target.closest('.card'); if (c) openFrame(+c.dataset.id); });

/* ============================== compare ============================== */
const C = { a: null, b: null, mode: 'side' };
function setSlot(which, id) {
  C[which] = id;
  $('#slot' + which.toUpperCase() + 'img').innerHTML = `<img src="/api/frame/${id}/annotated?w=640">`;
  if (C.a && C.b) runCompare();
}
$$('.slot').forEach(sl => {
  const which = sl.id === 'slotA' ? 'a' : 'b';
  $('.pick', sl).onclick = () => openPicker(id => setSlot(which, id));
  $('.up', sl).addEventListener('change', async e => {
    const f = e.target.files[0]; if (!f) return;
    try { const r = await J('/api/frame?source=compare&mode=always', { method: 'POST', body: f }); setSlot(which, r.saved_frame_id); }
    catch (err) { toast(err.message, true); }
    e.target.value = '';
  });
});
$('#cmpMode').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return; C.mode = b.dataset.m;
  $$('#cmpMode button').forEach(x => x.classList.toggle('on', x === b)); renderCmpView();
});
function renderCmpView() {
  if (!C.a || !C.b) return;
  const A = `/api/frame/${C.a}/annotated?w=960`, B = `/api/frame/${C.b}/annotated?w=960`;
  if (C.mode === 'side') $('#cmpView').innerHTML = `<div class="side"><div><div class="muted small">Frame A (#${C.a})</div><img src="${A}"></div><div><div class="muted small">Frame B (#${C.b})</div><img src="${B}"></div></div>`;
  else {
    $('#cmpView').innerHTML = `<div class="cmp" id="cmpBox"><img src="${B}"><img id="cmpTop" src="${A}" style="clip-path:inset(0 50% 0 0)"><div class="handle" id="cmpHandle" style="left:50%"></div></div>
      <input class="cmp-range" id="cmpRange" type="range" min="0" max="100" value="50" aria-label="Slide between A (left) and B (right)">
      <div class="muted small">Frame A on the left, frame B on the right</div>`;
    $('#cmpRange').oninput = e => { const v = e.target.value; $('#cmpTop').style.clipPath = `inset(0 ${100 - v}% 0 0)`; $('#cmpHandle').style.left = v + '%'; };
  }
}
async function runCompare() {
  try {
    const r = await J(`/api/compare?a=${C.a}&b=${C.b}`);
    $('#cmpResult').hidden = false; $('#cmpHint').hidden = true; renderCmpView();
    const sum = f => `#${f.id} from ${esc(f.ts_iso)}, ${esc(f.source)}, model ${esc(f.model || 'unknown')}, ${f.n_det} sign(s), best confidence ${pct(f.max_conf)}`;
    const st = { both: 'in both', only_a: 'only in A', only_b: 'only in B' };
    $('#cmpTable').innerHTML = `<p class="muted small">A: ${sum(r.a)}<br>B: ${sum(r.b)}</p>` + (r.rows.length ? `<table><thead><tr><th>Class</th><th>Frame A (count, best confidence)</th><th>Frame B (count, best confidence)</th><th>Δ confidence</th><th></th></tr></thead><tbody>` +
      r.rows.map(x => `<tr><td>${esc(x.label)}</td><td>${x.a_n ? x.a_n + ' found, ' + pct(x.a_conf) : '—'}</td><td>${x.b_n ? x.b_n + ' found, ' + pct(x.b_conf) : '—'}</td>
        <td class="${x.delta > 0 ? 'pos' : x.delta < 0 ? 'neg' : ''}">${x.delta == null ? '—' : (x.delta > 0 ? '+' : '') + (x.delta * 100).toFixed(1) + ' pts'}</td><td>${st[x.status]}</td></tr>`).join('') + '</tbody></table>' : '<p class="muted">No detections in either frame.</p>');
  } catch (e) { toast(e.message, true); }
}

/* ============================== model quality ============================== */
async function loadModel() {
  const body = $('#modelBody'); body.innerHTML = '<p class="muted">Loading…</p>';
  const r = await J('/api/model-metrics').catch(e => { body.innerHTML = esc(e.message); return null; }); if (!r) return;
  const m = r.metrics;
  if (!m) { body.innerHTML = `<div class="note">No metrics file found for <b>${esc(r.model)}</b>. Run <code>python evaluate_model.py models/${esc(r.model)}</code> to create <code>models/${esc(r.model.replace(/\.pt$/, ''))}.metrics.json</code>.</div>`; return; }
  const o = m.overall;
  const tile = (k, v, d) => `<div class="mtile"><b>${v}</b><span>${k}${d ? ' · ' + d : ''}</span></div>`;
  const evald = new Set(m.classes.map(c => c.name));
  const missing = (m.class_names || []).filter(n => !evald.has(n));
  const cls = [...m.classes].sort((a, b) => b.map50 - a.map50);
  const total = m.classes.reduce((s, c) => s + c.instances, 0);
  const cell = v => `<div class="mbar"><i class="${v < .3 ? 'bad' : v < .6 ? 'low' : ''}" style="width:${v * 100}%"></i></div>`;
  body.innerHTML = `
    ${r.model && S.info?.demo ? '<div class="note">Demo mode is on, so these are the saved numbers of your first trained model (best_v1), not of the fake demo detector.</div>' : ''}
    <div class="note">${esc(m.note || '')} <b>${total}</b> labelled signs were evaluated in total. A score based on 1 to 3 signs can swing wildly, so treat it as a rough indication.</div>
    <div class="mtiles">
      ${tile('Precision', o.precision.toFixed(2), 'of the boxes drawn, how many were right')}
      ${tile('Recall', o.recall.toFixed(2), 'of the real signs, how many were found')}
      ${tile('mAP50', o.map50.toFixed(2), 'main accuracy score')}
      ${tile('mAP50-95', o.map50_95.toFixed(2), 'strict box accuracy')}
    </div>
    <div class="panel"><h3>Per class (${esc(m.split || 'test')} set)</h3>
      <table><thead><tr><th>Class</th><th>Signs in set</th><th>Precision</th><th>Recall</th><th>mAP50</th><th></th></tr></thead><tbody>
      ${cls.map(c => `<tr><td>${esc(c.name)}</td><td>${c.instances}${c.instances < 5 ? ' <span class="badge missed">too few</span>' : ''}</td><td>${c.precision.toFixed(2)}</td><td>${c.recall.toFixed(2)}</td><td>${c.map50.toFixed(2)}</td><td>${cell(c.map50)}</td></tr>`).join('')}
      </tbody></table>
      ${missing.length ? `<p class="muted small"><b>Not evaluated</b> (no example in this set): ${missing.map(esc).join(', ')}</p>` : ''}
    </div>
    <div class="panel"><h3>How to read this</h3><p class="muted">Precision and recall depend on the confidence threshold; mAP summarises both. Live video has no “correct answers”, so these numbers come from your labelled test images, not from the camera. For live use, look at the confidence of each detection and mark mistakes in History with “Wrong detection” / “A sign was missed”.</p></div>
    ${Object.keys(r.assets).length ? `<div class="assets">${Object.values(r.assets).map(u => `<img src="${u}" alt="">`).join('')}</div>` : '<p class="muted small">Tip: copy <code>confusion_matrix.png</code> and <code>results.png</code> from your runs folder into <code>models/assets/</code> to show them here.</p>'}`;
}

/* ============================== header / model switch ============================== */
async function loadInfo() {
  const i = S.info = await J('/api/info');
  $('#device').textContent = i.device; $('#device').title = i.device;
  $('#modelSel').innerHTML = i.models.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join('') + '<option value="__demo__">Demo (fake results)</option>';
  $('#modelSel').value = i.demo ? '__demo__' : i.model;
  const d = $('#demo'); d.hidden = !i.demo;
  if (i.demo) d.textContent = 'DEMO MODE — detections are FAKE. ' + (i.load_error ? i.load_error + '. ' : '') + 'Put your best.pt into the models/ folder, restart, and pick it above.';
  $('#fLabel').innerHTML = '<option value="">All</option>' + i.classes.map(c => `<option>${esc(c)}</option>`).join('');
  $('#conf').value = i.settings.conf; $('#cd').value = i.settings.cooldown; $('#saveOn').checked = i.settings.save;
  $('#confVal').textContent = (+i.settings.conf).toFixed(2); $('#cdVal').textContent = (+i.settings.cooldown).toFixed(1) + ' s';
}
$('#modelSel').onchange = async e => {
  try { const r = await J('/api/models/select', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: e.target.value }) }); toast('Model loaded: ' + r.model); S.stable = null; renderSpeed(null); await loadInfo(); }
  catch (err) { toast(err.message, true); await loadInfo(); }
};

/* ============================== init ============================== */
renderSpeed(null); renderRecent(); clearStage('Press “Start camera”.');
loadInfo().then(refreshStats);
})();
