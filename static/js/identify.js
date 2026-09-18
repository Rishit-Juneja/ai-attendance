(() => {
  const refInput = document.getElementById('ref-input');
  const dzRef = document.getElementById('dz-ref');
  const targetInput = document.getElementById('target-input');
  const dzTarget = document.getElementById('dz-target');
  const dzTargetLabel = document.getElementById('dz-target-label');
  const dzTargetHint = document.getElementById('dz-target-hint');
  const tabs = document.querySelectorAll('#target-tabs .tab');
  const runBtn = document.getElementById('run-btn');
  const resultsEl = document.getElementById('results');

  let refFiles = [];
  let targetFile = null;
  let mode = 'image';

  function setupDropzone(dz, input, onFiles) {
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
    dz.addEventListener('drop', (e) => {
      e.preventDefault();
      dz.classList.remove('dragover');
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; onFiles([...e.dataTransfer.files]); }
    });
    input.addEventListener('change', () => { if (input.files.length) onFiles([...input.files]); });
  }

  function showPreview(dz, file, tagText) {
    dz.querySelectorAll('img.preview, .preview-tag').forEach(el => el.remove());
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.className = 'preview';
      img.src = url;
      dz.prepend(img);
    }
    const tag = document.createElement('div');
    tag.className = 'preview-tag';
    tag.textContent = tagText;
    dz.appendChild(tag);
  }

  setupDropzone(dzRef, refInput, (files) => {
    // Accumulate across picks so the user can add photos one at a time.
    const seen = new Set(refFiles.map(f => f.name + f.size));
    refFiles = refFiles.concat(files.filter(f => !seen.has(f.name + f.size)));
    showPreview(dzRef, refFiles[0],
      refFiles.length === 1 ? refFiles[0].name : `${refFiles.length} photos — averaged`);
    updateRunState();
  });

  setupDropzone(dzTarget, targetInput, ([f]) => {
    targetFile = f;
    if (mode === 'image') showPreview(dzTarget, f, f.name);
    else {
      dzTarget.querySelectorAll('img.preview, .preview-tag').forEach(el => el.remove());
      const tag = document.createElement('div');
      tag.className = 'preview-tag';
      tag.textContent = f.name;
      dzTarget.appendChild(tag);
    }
    updateRunState();
  });

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      mode = tab.dataset.mode;
      targetFile = null;
      targetInput.value = '';
      dzTarget.querySelectorAll('img.preview, .preview-tag').forEach(el => el.remove());
      if (mode === 'image') {
        targetInput.accept = 'image/*';
        dzTargetLabel.textContent = 'Photo to search';
        dzTargetHint.textContent = 'A crowd photo, CCTV still, etc.';
      } else {
        targetInput.accept = 'video/*';
        dzTargetLabel.textContent = 'Video to search';
        dzTargetHint.textContent = 'Sampled at ~3 fps, first 30s analyzed';
      }
      updateRunState();
    });
  });

  function updateRunState() {
    runBtn.disabled = !(refFiles.length && targetFile);
  }

  function setLoading(loading) {
    runBtn.disabled = loading;
    runBtn.innerHTML = loading
      ? '<span class="spinner"></span> Analyzing…'
      : '<svg class="icon" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.6-4.6"/></svg> Run identification';
  }

  runBtn.addEventListener('click', async () => {
    if (!refFiles.length || !targetFile) return;
    setLoading(true);
    resultsEl.style.display = 'none';

    const fd = new FormData();
    refFiles.forEach(f => fd.append('reference', f));
    fd.append('target', targetFile);

    try {
      const url = mode === 'image' ? '/api/identify/image' : '/api/identify/video';
      const res = await fetch(url, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        renderError(data.error || 'Something went wrong processing that file.');
      } else if (mode === 'image') {
        renderImageResult(data);
      } else {
        renderVideoResult(data);
      }
    } catch (err) {
      renderError('Request failed: ' + err.message);
    } finally {
      setLoading(false);
      updateRunState();
    }
  });

  // Advisory, never a filter. A single reference photo is the biggest limiter on
  // a distant match, so name that before the user concludes the person is absent.
  function weakRefBanner(data) {
    // Mixed people is a correctness problem, not a quality one — lead with it.
    if (data.ref_odd && data.ref_odd.length) {
      // All flagged means they simply disagree with each other and we can't tell
      // which is the intruder; a subset means those are the odd ones out.
      const detail = data.ref_odd.length === data.ref_count
        ? `Your reference photos don't match each other.`
        : `Reference photo ${data.ref_odd.map(i => `#${i + 1}`).join(', ')} doesn't match the others.`;
      return `<div class="result-banner warn">
          <svg class="icon" viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17h.01"/></svg>
          <div><div class="rb-title">Your reference photos look like different people</div>
          <div class="rb-sub">${detail} Each reference contributes its <em>largest</em> face — so a group photo contributes whoever stands closest to the camera, not your subject. Use photos where the person you want is the biggest face.</div></div>
        </div>`;
    }
    const reasons = [];
    if (data.ref_count < 2) reasons.push(`only <strong>1 reference photo</strong> — adding 2–3 more of the same person typically lifts a distant face by 0.05–0.07`);
    if (data.ref_width < data.weak_ref_px) reasons.push(`the reference face is only <strong>${data.ref_width}px</strong> wide (ideally ≥${data.weak_ref_px}px)`);
    return `<div class="result-banner warn">
        <svg class="icon" viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17h.01"/></svg>
        <div><div class="rb-title">Scores here are limited by the reference, not the target</div>
        <div class="rb-sub">You gave ${reasons.join(', and ')}. Every score below is computed against that embedding, so the right person can land under the threshold.</div></div>
      </div>`;
  }

  function renderError(msg) {
    resultsEl.style.display = 'block';
    resultsEl.innerHTML = `
      <div class="result-banner nomatch">
        <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>
        <div><div class="rb-title">Couldn't complete identification</div><div class="rb-sub">${escapeHtml(msg)}</div></div>
      </div>`;
  }

  function renderImageResult(data) {
    resultsEl.style.display = 'block';
    const matched = data.matches.filter(m => m.matched).length;
    const bestIdx = data.matches.reduce((best, m, i, arr) => m.score > arr[best].score ? i : best, 0);

    // A weak reference makes every score below it read low — say so before the
    // user concludes the person isn't there. Advisory: nothing is filtered out.
    const refWarning = data.ref_weak ? weakRefBanner(data) : '';

    // Who the uploaded reference is, according to the enrolled database.
    const who = data.ref_identity;
    const refIdentity = who
      ? `<div class="result-banner match">
           <svg class="icon" viewBox="0 0 24 24"><path d="M12 3a4 4 0 1 1 0 8 4 4 0 0 1 0-8M4 21a8 8 0 0 1 16 0"/></svg>
           <div><div class="rb-title">Reference is enrolled: ${escapeHtml(who.name)}</div>
           <div class="rb-sub">Matched the database at ${who.score.toFixed(3)}${who.roll ? ` · roll ${escapeHtml(who.roll)}` : ''}</div></div>
         </div>`
      : `<div class="result-banner nomatch">
           <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="8" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/></svg>
           <div><div class="rb-title">Reference is not in the database</div>
           <div class="rb-sub">Enroll this person on the Live &amp; Database page to have them recognised automatically.</div></div>
         </div>`;

    const banner = matched > 0
      ? `<div class="result-banner match">
           <svg class="icon" viewBox="0 0 24 24"><path d="M5 13l4 4 10-10"/></svg>
           <div><div class="rb-title">Match found${who ? `: ${escapeHtml(who.name)}` : ''}</div><div class="rb-sub">${matched} of ${data.matches.length} detected face(s) matched the reference photo (best score ${data.best_score.toFixed(3)}, threshold ${data.threshold})${data.known_count ? ` · ${data.known_count} face(s) recognised from the database` : ''}</div></div>
         </div>`
      : `<div class="result-banner nomatch">
           <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/></svg>
           <div><div class="rb-title">No match found</div><div class="rb-sub">${data.matches.length} face(s) detected, none above threshold ${data.threshold}. Best score: ${data.best_score.toFixed(3)}${data.known_count ? ` · ${data.known_count} face(s) recognised from the database` : ''}</div></div>
         </div>`;

    const sorted = [...data.matches].sort((a, b) => b.score - a.score);
    const scoreRows = sorted.map((m, i) => {
      const isBest = m === data.matches[bestIdx];
      const cls = m.matched ? 'match' : (isBest ? 'best' : '');
      const verdict = m.matched ? 'YES' : (isBest ? 'best' : 'no');
      const id = m.identity
        ? `<strong>${escapeHtml(m.identity.name)}</strong> <span class="hint">${m.identity.score.toFixed(3)}</span>`
        : '<span class="hint">—</span>';
      return `<tr class="${cls}"><td>${i + 1}</td><td>${m.score.toFixed(4)}</td><td>${m.width}px</td><td>${verdict}</td><td>${id}</td></tr>`;
    }).join('');

    resultsEl.innerHTML = `
      ${refWarning}
      ${refIdentity}
      ${banner}
      <div class="result-image-wrap">
        <img src="data:image/jpeg;base64,${data.annotated_image}" alt="Annotated result">
      </div>
      <div class="card" style="margin-top:1rem;">
        <h2>All detected faces (${data.matches.length})</h2>
        <table style="width:100%;border-collapse:collapse;font-size:.85rem;">
          <thead><tr><th>#</th><th>vs reference</th><th>Face width</th><th>Verdict</th><th>In database</th></tr></thead>
          <tbody>${scoreRows}</tbody>
        </table>
        <p class="hint" style="margin-top:.5rem;">Green = matched your reference. "In database" is a separate lookup against everyone enrolled — a face can be named there even if it isn't the person you searched for. Single-image search scores every face regardless of size; on the Live page, small faces get averaged across a ByteTrack track, which scores them far higher.</p>
      </div>
    `;
  }

  function renderVideoResult(data) {
    resultsEl.style.display = 'block';
    const matchedFrames = data.frames.filter(f => f.matched);
    const refWarning = data.ref_weak ? weakRefBanner(data) : '';
    const banner = matchedFrames.length > 0
      ? `<div class="result-banner match">
           <svg class="icon" viewBox="0 0 24 24"><path d="M5 13l4 4 10-10"/></svg>
           <div><div class="rb-title">Person found in ${matchedFrames.length} of ${data.frames.length} sampled frame(s)</div><div class="rb-sub">First seen at ${matchedFrames[0].t.toFixed(1)}s · best score ${data.best_score.toFixed(3)}</div></div>
         </div>`
      : `<div class="result-banner nomatch">
           <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m9 9 6 6m0-6-6 6"/></svg>
           <div><div class="rb-title">Not found in this video</div><div class="rb-sub">${data.frames.length} frame(s) sampled, best score ${data.best_score.toFixed(3)}</div></div>
         </div>`;

    const strip = data.frames.map(f => `
      <div class="filmstrip-item ${f.matched ? 'matched' : ''}">
        <img src="data:image/jpeg;base64,${f.thumb}" alt="frame at ${f.t.toFixed(1)}s">
        <div class="t">${f.t.toFixed(1)}s${f.matched ? ' · ' + f.score.toFixed(2) : ''}</div>
      </div>`).join('');

    resultsEl.innerHTML = `
      ${refWarning}
      ${banner}
      <div class="card">
        <h2>Sampled frames (${data.frames.length})</h2>
        <div class="filmstrip">${strip}</div>
      </div>
    `;
  }

  function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }
})();
