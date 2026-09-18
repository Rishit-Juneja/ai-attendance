(() => {
  const CAP_W = 640, CAP_H = 480;
  // keep in sync with the tokens in static/css/app.css
  const COLOR_MATCH = '#22c55e', COLOR_UNKNOWN = '#f59e0b', COLOR_SPOOF = '#ef4444';

  const webcam = document.getElementById('webcam');
  const overlay = document.getElementById('overlay');
  const stageOff = document.getElementById('stage-off');
  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const statusPill = document.getElementById('live-status-pill');
  const attendeeList = document.getElementById('attendee-list');
  const alertList = document.getElementById('alert-list');
  const statFps = document.getElementById('stat-fps');
  const statInf = document.getElementById('stat-inf');
  const statPresent = document.getElementById('stat-present');
  const statAlerts = document.getElementById('stat-alerts');

  const captureCanvas = document.createElement('canvas');
  let stream = null;
  let loopTimer = null;
  let busy = false;
  let frameCount = 0;
  let fpsTimer = performance.now();

  function captureFrame() {
    const vw = webcam.videoWidth, vh = webcam.videoHeight;
    if (!vw || !vh) return null;
    const targetRatio = CAP_W / CAP_H, srcRatio = vw / vh;
    let sx, sy, sw, sh;
    if (srcRatio > targetRatio) { sh = vh; sw = vh * targetRatio; sx = (vw - sw) / 2; sy = 0; }
    else { sw = vw; sh = vw / targetRatio; sx = 0; sy = (vh - sh) / 2; }
    captureCanvas.width = CAP_W; captureCanvas.height = CAP_H;
    captureCanvas.getContext('2d').drawImage(webcam, sx, sy, sw, sh, 0, 0, CAP_W, CAP_H);
    return captureCanvas.toDataURL('image/jpeg', 0.7);
  }

  function drawOverlay(detections) {
    overlay.width = CAP_W; overlay.height = CAP_H;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, CAP_W, CAP_H);
    ctx.lineWidth = 2;
    ctx.font = '13px -apple-system, sans-serif';
    ctx.textBaseline = 'bottom';
    for (const d of detections) {
      const [x1, y1, x2, y2] = d.bbox;
      const color = d.is_spoof ? COLOR_SPOOF : (d.name !== 'Unknown' ? COLOR_MATCH : COLOR_UNKNOWN);
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = d.name !== 'Unknown' ? `${d.name} ${d.score.toFixed(2)}` : 'Unknown';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 18), tw + 8, 18);
      ctx.fillStyle = '#0a0a0b';
      ctx.fillText(label, x1 + 4, Math.max(18, y1));
    }
  }

  function renderAttendance(summary) {
    statPresent.textContent = summary.present_now || 0;
    statAlerts.textContent = summary.alerts || 0;
    const persons = (summary.persons || []).slice().sort((a, b) => {
      if (a.present !== b.present) return a.present ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    attendeeList.innerHTML = persons.length ? persons.map(p => `
      <div class="person-row">
        <div class="person-info">
          <div class="n">${escapeHtml(p.name)}</div>
          <div class="r">${escapeHtml(p.roll)} · in ${escapeHtml(p.entry || '--')}${p.exit ? ' → ' + escapeHtml(p.exit) : ''}</div>
        </div>
        <span class="badge ${p.present ? 'badge-green' : 'badge-neutral'}">${p.present ? 'Present' : 'Left'}</span>
      </div>`).join('') : '<div class="empty-state"><span>No one detected yet.</span></div>';

    const alerts = (summary.alerts_log || []).slice().reverse().slice(0, 15);
    alertList.innerHTML = alerts.length ? alerts.map(a => `
      <div class="alert-row">
        <div class="a-time">${escapeHtml(a.time)}</div>
        <div class="a-type ${a.type}">${a.type.replace(/_/g, ' ')}</div>
        <div class="a-detail">${escapeHtml(a.details)}</div>
      </div>`).join('') : '<div class="empty-state"><span>No alerts.</span></div>';
  }

  async function tick() {
    if (busy) return;
    const dataUrl = captureFrame();
    if (!dataUrl) return;
    busy = true;
    const t0 = performance.now();
    try {
      const res = await fetch('/api/live/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl }),
      });
      const data = await res.json();
      if (data.ok) {
        drawOverlay(data.detections);
        renderAttendance(data.summary);
        statInf.textContent = data.inference_ms.toFixed(0);
        frameCount++;
        const elapsed = (performance.now() - fpsTimer) / 1000;
        if (elapsed >= 1) { statFps.textContent = (frameCount / elapsed).toFixed(1); frameCount = 0; fpsTimer = performance.now(); }
      }
    } catch (err) {
      console.error('live frame failed', err);
    } finally {
      busy = false;
    }
  }

  startBtn.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 960 } });
    } catch (err) {
      // No /dev/video device is the normal case on a desktop with no webcam —
      // point at the network camera box instead of a dead-end alert.
      const hint = document.getElementById('cam-status');
      if (hint) hint.textContent =
        'No webcam on this machine (' + err.name + '). Use the Network camera box below — '
        + 'paste your phone or CCTV URL and press Connect.';
      const url = document.getElementById('cam-url');
      if (url) { url.focus(); url.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
      return;
    }
    webcam.srcObject = stream;
    stageOff.style.display = 'none';
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusPill.classList.add('on');
    statusPill.innerHTML = '<span class="dot"></span> Live';

    await fetch('/api/live/start', { method: 'POST' });
    loopTimer = setInterval(tick, 400); // ~2.5 fps, within the 2-5fps target
  });

  stopBtn.addEventListener('click', async () => {
    clearInterval(loopTimer);
    if (stream) stream.getTracks().forEach(t => t.stop());
    webcam.srcObject = null;
    stageOff.style.display = 'flex';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusPill.classList.remove('on');
    statusPill.innerHTML = '<span class="dot"></span> Camera off';

    const res = await fetch('/api/live/stop', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      statusPill.innerHTML = `<span class="dot"></span> Saved (${data.session})`;
    }
  });

  // ---------------- enrollment ----------------
  const dzEnroll = document.getElementById('dz-enroll');
  const enrollInput = document.getElementById('enroll-input');
  const enrollName = document.getElementById('enroll-name');
  const enrollRoll = document.getElementById('enroll-roll');
  const enrollBtn = document.getElementById('enroll-btn');
  const enrollStatus = document.getElementById('enroll-status');
  const galleryList = document.getElementById('gallery-list');
  const galleryCount = document.getElementById('gallery-count');
  let enrollFiles = [];

  function updateEnrollState() {
    enrollBtn.disabled = !(enrollFiles.length && enrollName.value.trim() && enrollRoll.value.trim());
  }
  enrollName.addEventListener('input', updateEnrollState);
  enrollRoll.addEventListener('input', updateEnrollState);

  dzEnroll.addEventListener('dragover', (e) => { e.preventDefault(); dzEnroll.classList.add('dragover'); });
  dzEnroll.addEventListener('dragleave', () => dzEnroll.classList.remove('dragover'));
  dzEnroll.addEventListener('drop', (e) => {
    e.preventDefault();
    dzEnroll.classList.remove('dragover');
    if (e.dataTransfer.files.length) { enrollInput.files = e.dataTransfer.files; handleEnrollFiles([...e.dataTransfer.files]); }
  });
  enrollInput.addEventListener('change', () => { if (enrollInput.files.length) handleEnrollFiles([...enrollInput.files]); });

  function handleEnrollFiles(files) {
    // Accumulate so photos can be added one at a time; they get averaged server
    // side, which is what lets a distant CCTV face clear the match threshold.
    const seen = new Set(enrollFiles.map(f => f.name + f.size));
    enrollFiles = enrollFiles.concat(files.filter(f => !seen.has(f.name + f.size)));
    dzEnroll.querySelectorAll('img.preview, .preview-tag').forEach(el => el.remove());
    const img = document.createElement('img');
    img.className = 'preview';
    img.src = URL.createObjectURL(enrollFiles[0]);
    dzEnroll.prepend(img);
    const tag = document.createElement('div');
    tag.className = 'preview-tag';
    tag.textContent = enrollFiles.length === 1 ? enrollFiles[0].name : `${enrollFiles.length} photos — averaged`;
    dzEnroll.appendChild(tag);
    updateEnrollState();
  }

  enrollBtn.addEventListener('click', async () => {
    enrollBtn.disabled = true;
    enrollBtn.innerHTML = '<span class="spinner"></span> Adding…';
    enrollStatus.textContent = '';
    const fd = new FormData();
    fd.append('name', enrollName.value.trim());
    fd.append('roll', enrollRoll.value.trim());
    enrollFiles.forEach(f => fd.append('photo', f));
    try {
      const res = await fetch('/api/gallery', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        enrollStatus.textContent = data.error || 'Could not enroll that photo.';
      } else {
        enrollName.value = ''; enrollRoll.value = ''; enrollFiles = [];
        enrollInput.value = '';
        dzEnroll.querySelectorAll('img.preview, .preview-tag').forEach(el => el.remove());
        await refreshGallery();
        enrollStatus.textContent = `Enrolled from ${data.photos} photo(s).`;
      }
    } catch (err) {
      enrollStatus.textContent = 'Request failed: ' + err.message;
    } finally {
      enrollBtn.innerHTML = 'Add to database';
      updateEnrollState();
    }
  });

  async function refreshGallery() {
    const res = await fetch('/api/gallery');
    const data = await res.json();
    galleryCount.textContent = data.gallery.length;
    galleryList.innerHTML = data.gallery.length ? data.gallery.map(p => `
      <div class="person-row" data-roll="${escapeAttr(p.roll)}">
        <img class="person-thumb" src="${p.photo_url}" alt="${escapeAttr(p.name)}">
        <div class="person-info">
          <div class="n">${escapeHtml(p.name)}</div>
          <div class="r">${escapeHtml(p.roll)}</div>
        </div>
        <button class="btn btn-danger-ghost btn-sm remove-person" data-roll="${escapeAttr(p.roll)}" aria-label="Remove ${escapeAttr(p.name)}">
          <svg class="icon" style="width:15px;height:15px" viewBox="0 0 24 24"><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-8 0 1 13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1l1-13"/></svg>
        </button>
      </div>`).join('') : '<div class="empty-state"><span>No one enrolled yet.</span></div>';
  }

  galleryList.addEventListener('click', async (e) => {
    const btn = e.target.closest('.remove-person');
    if (!btn) return;
    const roll = btn.dataset.roll;
    if (!confirm(`Remove this person (${roll}) from the database?`)) return;
    await fetch(`/api/gallery/${encodeURIComponent(roll)}`, { method: 'DELETE' });
    await refreshGallery();
  });

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML;
  }
  function escapeAttr(s) { return escapeHtml(s).replace(/"/g, '&quot;'); }

  // ---------------- network camera (phone MJPEG / CCTV RTSP) ----------------
  // Captured and processed server-side, so it needs no /dev/video device and no
  // browser permission — the same path a real CCTV camera uses.
  const camUrl = document.getElementById('cam-url');
  const camBtn = document.getElementById('cam-btn');
  const camStatus = document.getElementById('cam-status');
  const camStream = document.getElementById('cam-stream');
  let camOn = false;
  let camPoll = null;
  let feedTimer = null;

  async function camStart() {
    const url = camUrl.value.trim();
    if (!url) return;
    camBtn.disabled = true;
    camStatus.textContent = 'Connecting…';
    try {
      const res = await fetch('/api/live/stream_start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) { camStatus.textContent = data.error || 'Could not connect.'; return; }
      camOn = true;
      camBtn.textContent = 'Disconnect';
      camStatus.textContent = 'Connected to ' + data.url;
      if (stageOff) stageOff.style.display = 'none';
      camStream.style.display = 'block';
      startFeed();
      camPoll = setInterval(camTick, 1500);
    } catch (err) {
      camStatus.textContent = 'Request failed: ' + err.message;
    } finally {
      camBtn.disabled = false;
    }
  }

  // Poll single JPEGs rather than holding one long multipart response open.
  // Chain each request off the previous load so a slow frame can't pile up.
  function startFeed() {
    if (feedTimer) return;
    const tick = () => {
      if (!camOn) return;
      const img = new Image();
      img.onload = () => { camStream.src = img.src; feedTimer = setTimeout(tick, 120); };
      img.onerror = () => { feedTimer = setTimeout(tick, 500); };
      img.src = '/api/live/snapshot?t=' + Date.now();
    };
    tick();
  }

  function stopFeed() {
    clearTimeout(feedTimer); feedTimer = null;
  }

  async function camTick() {
    try {
      const d = await (await fetch('/api/live/stream_status')).json();
      if (d.error) { camStatus.textContent = d.error; await camStop(); return; }
      if (d.summary) {
        const present = document.getElementById('stat-present');
        const alerts = document.getElementById('stat-alerts');
        if (present) present.textContent = d.summary.present_now ?? 0;
        if (alerts) alerts.textContent = d.summary.alerts ?? 0;
        const names = [...new Set((d.summary.persons || []).map(p => p.name).filter(n => n && n !== 'Unknown'))];
        if (names.length) camStatus.textContent = 'Present: ' + names.join(', ');
      }
      if (!d.running) { camStatus.textContent = 'Stream ended.'; await camStop(); return; }
    } catch (err) { /* transient poll failure is not fatal */ }
  }

  async function camStop() {
    clearInterval(camPoll); camPoll = null;
    camOn = false;
    stopFeed();
    camStream.src = ''; camStream.style.display = 'none';
    if (stageOff) stageOff.style.display = '';
    camBtn.textContent = 'Connect';
    try {
      const d = await (await fetch('/api/live/stream_stop', { method: 'POST' })).json();
      camStatus.textContent = d.session ? `Stopped. Log saved as ${d.session}.` : 'Stopped.';
    } catch (err) { camStatus.textContent = 'Stopped.'; }
  }

  camBtn.addEventListener('click', () => (camOn ? camStop() : camStart()));

  // Tell the user which control to use before they click the one that can't work.
  (async () => {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      if (!devs.some(d => d.kind === 'videoinput')) {
        startBtn.title = 'No webcam detected on this machine — use the Network camera box below';
        startBtn.classList.remove('btn-primary');
        startBtn.classList.add('btn-outline');
        camBtn.classList.add('btn-primary');
        camBtn.classList.remove('btn-outline');
      }
    } catch (err) { /* enumerateDevices unavailable; leave buttons as-is */ }
  })();

  // ---------------- zone editor ----------------
  // Zones are stored normalized 0..1, so they survive the camera being
  // reconfigured to a different resolution.

  const stage = document.getElementById('video-stage');
  const zoneName = document.getElementById('zone-name');
  const zoneDraw = document.getElementById('zone-draw');
  const zoneUndo = document.getElementById('zone-undo');
  const zoneSave = document.getElementById('zone-save');
  const zoneClear = document.getElementById('zone-clear');
  const zoneStatus = document.getElementById('zone-status');

  let drawing = false, pts = [], savedZones = [];

  // The feed is object-fit:contain, so the image is letterboxed inside the
  // element. Mapping a click against the element's own rect would skew every
  // point by the size of the bars.
  function imageBox() {
    const showingStream = camStream && camStream.style.display !== 'none';
    const el = showingStream ? camStream : webcam;
    const r = el.getBoundingClientRect();
    const natW = el.naturalWidth || el.videoWidth || CAP_W;
    const natH = el.naturalHeight || el.videoHeight || CAP_H;
    if (!natW || !natH) return { left: r.left, top: r.top, w: r.width, h: r.height };
    const scale = Math.min(r.width / natW, r.height / natH);
    const w = natW * scale, h = natH * scale;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, w, h };
  }

  function drawPolygon() {
    const sr = stage.getBoundingClientRect();
    overlay.width = sr.width;
    overlay.height = sr.height;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!pts.length) return;
    const b = imageBox();
    const sx = x => b.left - sr.left + x * b.w;
    const sy = y => b.top - sr.top + y * b.h;

    ctx.lineWidth = 2;
    ctx.strokeStyle = COLOR_UNKNOWN;
    ctx.fillStyle = 'rgba(245,158,11,.15)';
    ctx.beginPath();
    ctx.moveTo(sx(pts[0][0]), sy(pts[0][1]));
    for (const p of pts.slice(1)) ctx.lineTo(sx(p[0]), sy(p[1]));
    if (pts.length > 2) { ctx.closePath(); ctx.fill(); }
    ctx.stroke();
    ctx.fillStyle = COLOR_UNKNOWN;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  async function refreshZones() {
    try {
      const d = await (await fetch('/api/zones')).json();
      savedZones = d.zones || [];
      zoneStatus.textContent = savedZones.length
        ? 'Active: ' + savedZones.map(z => z.name).join(', ')
        : 'No zones — the whole frame counts.';
    } catch (err) { /* leave the hint as-is */ }
  }

  function setDrawing(on) {
    drawing = on;
    zoneDraw.textContent = on ? 'Stop drawing' : 'Draw';
    stage.classList.toggle('drawing', on);
    if (on) zoneStatus.textContent = 'Click the feed to place corners.';
  }

  stage.addEventListener('click', ev => {
    if (!drawing) return;
    const b = imageBox();
    const x = (ev.clientX - b.left) / b.w;
    const y = (ev.clientY - b.top) / b.h;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;   // clicked the letterbox bar
    pts.push([+x.toFixed(4), +y.toFixed(4)]);
    zoneUndo.disabled = false;
    zoneSave.disabled = pts.length < 3;
    zoneStatus.textContent = pts.length + (pts.length === 1 ? ' point' : ' points')
      + (pts.length < 3 ? ' — need at least 3' : '');
    drawPolygon();
  });

  zoneDraw.addEventListener('click', () => setDrawing(!drawing));

  zoneUndo.addEventListener('click', () => {
    pts.pop();
    zoneUndo.disabled = !pts.length;
    zoneSave.disabled = pts.length < 3;
    drawPolygon();
  });

  zoneSave.addEventListener('click', async () => {
    const name = (zoneName.value || '').trim();
    if (!name) { zoneStatus.textContent = 'Give the zone a name first.'; return; }
    const body = { zones: savedZones.concat([{ name, points: pts }]) };
    try {
      const r = await (await fetch('/api/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })).json();
      if (!r.ok) { zoneStatus.textContent = r.error || 'Could not save.'; return; }
      pts = [];
      zoneName.value = '';
      zoneSave.disabled = true;
      zoneUndo.disabled = true;
      setDrawing(false);
      drawPolygon();
      await refreshZones();
    } catch (err) { zoneStatus.textContent = 'Could not reach the server.'; }
  });

  zoneClear.addEventListener('click', async () => {
    pts = [];
    zoneSave.disabled = true;
    zoneUndo.disabled = true;
    setDrawing(false);
    drawPolygon();
    try {
      await fetch('/api/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ zones: [] }),
      });
      await refreshZones();
    } catch (err) { zoneStatus.textContent = 'Could not reach the server.'; }
  });

  window.addEventListener('resize', drawPolygon);
  refreshZones();

  // The stream lives on the server, so a page refresh leaves it running with the
  // client unaware — reattach instead of reporting "already running".
  (async () => {
    try {
      const d = await (await fetch('/api/live/stream_status')).json();
      if (!d.running) return;
      camOn = true;
      camUrl.value = d.url;
      camBtn.textContent = 'Disconnect';
      camStatus.textContent = 'Reattached to ' + d.url;
      if (stageOff) stageOff.style.display = 'none';
      camStream.style.display = 'block';
      startFeed();
      camPoll = setInterval(camTick, 1500);
    } catch (err) { /* nothing running */ }
  })();

})();
