/* ── Scroll reveal ─────────────────────────────────────────────────── */
(function () {
  const obs = new IntersectionObserver(
    entries => entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); } }),
    { threshold: 0.12 }
  );
  document.querySelectorAll('.scroll-reveal').forEach(el => obs.observe(el));
})();

/* ── Predict page ─────────────────────────────────────────────────── */
(function () {
  if (!document.getElementById('predict-btn')) return;  // not on predict page

  // ── Tab switching ────────────────────────────────────────────────────
  const tabBtns   = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');
  const predictBtn = document.getElementById('predict-btn');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });

  // ── Upload tab ───────────────────────────────────────────────────────
  const uploadZone    = document.getElementById('upload-zone');
  const fileInput     = document.getElementById('file-input');
  const uploadPreview = document.getElementById('upload-preview');
  let   selectedBlob  = null;

  uploadZone.addEventListener('click',     () => fileInput.click());
  uploadZone.addEventListener('dragover',  e  => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) setUploadFile(f);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) setUploadFile(fileInput.files[0]); });

  function setUploadFile(file) {
    selectedBlob = file;
    const reader = new FileReader();
    reader.onload = ev => { uploadPreview.src = ev.target.result; uploadPreview.classList.remove('hidden'); };
    reader.readAsDataURL(file);
    predictBtn.disabled = false;
  }

  // ── Upload → POST /api/predict ───────────────────────────────────────
  const resultsEl   = document.getElementById('results');
  const emptyEl     = document.getElementById('results-empty');
  const predictLabel = document.getElementById('predict-label');

  predictBtn.addEventListener('click', async () => {
    if (!selectedBlob) return;
    predictBtn.disabled = true;
    predictBtn.classList.add('loading');
    predictLabel.textContent = 'Classifying…';

    const form = new FormData();
    form.append('file', selectedBlob, 'sign.jpg');
    try {
      const res  = await fetch('/api/predict', { method: 'POST', body: form });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || res.statusText); }
      showUploadResult(await res.json());
    } catch (err) {
      alert('Prediction failed: ' + err.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.classList.remove('loading');
      predictLabel.textContent = 'Classify Sign';
    }
  });

  function showUploadResult(data) {
    document.getElementById('result-letter').textContent = data.letter;
    document.getElementById('result-conf').textContent   = data.confidence.toFixed(1) + '%';

    const bar = document.getElementById('conf-bar');
    bar.style.width = '0%';
    requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = data.confidence + '%'; }));

    const top5El = document.getElementById('top5');
    top5El.innerHTML = '';
    data.top5.forEach(item => {
      const row = document.createElement('div');
      row.className = 'top5-row';
      row.innerHTML = `
        <span class="top5-letter">${item.letter}</span>
        <div class="top5-bar-wrap"><div class="top5-bar" style="width:0%"></div></div>
        <span class="top5-conf">${item.confidence.toFixed(1)}%</span>`;
      top5El.appendChild(row);
    });
    requestAnimationFrame(() => requestAnimationFrame(() => {
      top5El.querySelectorAll('.top5-bar').forEach((b, i) => { b.style.width = data.top5[i].confidence + '%'; });
    }));

    emptyEl.classList.add('hidden');
    resultsEl.classList.remove('hidden');
  }

  // ── Live webcam (WebSocket → server → MediaPipe + CNN) ───────────────
  const liveVideo    = document.getElementById('live-video');
  const liveOverlay  = document.getElementById('live-overlay');
  const liveBadge    = document.getElementById('live-badge');
  const liveHint     = document.getElementById('live-hint');
  const liveLetterEl = document.getElementById('live-letter');
  const liveConfEl   = document.getElementById('live-conf');
  const liveTopRest  = document.getElementById('live-top-rest');
  const liveStatusTxt = document.getElementById('live-status-text');
  const liveDot      = document.getElementById('live-dot');
  const startBtn     = document.getElementById('live-start-btn');
  const stopBtn      = document.getElementById('live-stop-btn');

  let ws         = null;
  let camStream  = null;
  let liveActive = false;
  const capCanvas = document.createElement('canvas');

  function setStatus(text, active) {
    liveStatusTxt.textContent = text;
    liveDot.classList.toggle('active', active);
  }

  async function startLive() {
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      liveVideo.srcObject = camStream;
      await liveVideo.play();
    } catch (err) {
      alert('Camera error: ' + err.message);
      return;
    }

    setStatus('Connecting…', false);
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/live`);

    ws.onopen = () => {
      liveActive = true;
      setStatus('Live', true);
      startBtn.classList.add('hidden');
      stopBtn.classList.remove('hidden');
      liveHint.classList.remove('hidden');
      sendFrame();
    };

    ws.onmessage = ev => {
      const data = JSON.parse(ev.data);
      if (data.error) { console.error('[WS]', data.error); }
      drawFrame(data);
      if (liveActive) sendFrame();  // chain: send next only after result received
    };

    ws.onerror = () => setStatus('Error', false);
    ws.onclose = () => { if (liveActive) stopLive(); };
  }

  function stopLive() {
    liveActive = false;
    if (ws)        { ws.close(); ws = null; }
    if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
    liveVideo.srcObject = null;
    clearOverlay();
    liveBadge.classList.add('hidden');
    liveHint.classList.add('hidden');
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    setStatus('Camera off', false);
    // Restore empty state on results side
    resultsEl.classList.add('hidden');
    emptyEl.classList.remove('hidden');
  }

  function sendFrame() {
    if (!liveActive || !liveVideo.videoWidth) return;
    capCanvas.width  = liveVideo.videoWidth;
    capCanvas.height = liveVideo.videoHeight;
    capCanvas.getContext('2d').drawImage(liveVideo, 0, 0);
    capCanvas.toBlob(blob => {
      if (ws && ws.readyState === WebSocket.OPEN)
        blob.arrayBuffer().then(buf => ws.send(buf));
    }, 'image/jpeg', 0.72);
  }

  function clearOverlay() {
    const ctx = liveOverlay.getContext('2d');
    liveOverlay.width  = liveOverlay.offsetWidth  || 1;
    liveOverlay.height = liveOverlay.offsetHeight || 1;
    ctx.clearRect(0, 0, liveOverlay.width, liveOverlay.height);
  }

  // MediaPipe hand skeleton connections (same as Livestream.py HAND_CONNECTIONS)
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],          // thumb
    [0,5],[5,6],[6,7],[7,8],          // index
    [0,9],[9,10],[10,11],[11,12],     // middle
    [0,13],[13,14],[14,15],[15,16],   // ring
    [0,17],[17,18],[18,19],[19,20],   // pinky
    [5,9],[9,13],[13,17],             // palm
  ];

  function drawFrame(data) {
    liveOverlay.width  = liveVideo.offsetWidth  || liveVideo.videoWidth  || 1;
    liveOverlay.height = liveVideo.offsetHeight || liveVideo.videoHeight || 1;
    const ctx = liveOverlay.getContext('2d');
    const W = liveOverlay.width, H = liveOverlay.height;
    ctx.clearRect(0, 0, W, H);

    if (!data.hand_detected) {
      liveBadge.classList.add('hidden');
      liveHint.classList.remove('hidden');
      return;
    }

    liveHint.classList.add('hidden');

    // ── Landmark skeleton ─────────────────────────────────────────────
    if (data.landmarks && data.landmarks.length === 21) {
      const lms = data.landmarks;

      // Connections
      ctx.strokeStyle = 'rgba(0, 229, 176, 0.75)';
      ctx.lineWidth   = 1.8;
      ctx.lineJoin    = 'round';
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo(lms[a][0] * W, lms[a][1] * H);
        ctx.lineTo(lms[b][0] * W, lms[b][1] * H);
        ctx.stroke();
      });

      // Joint dots
      lms.forEach(([x, y], i) => {
        ctx.beginPath();
        ctx.arc(x * W, y * H, i === 0 ? 5.5 : 3.5, 0, Math.PI * 2);
        // Fingertips (4,8,12,16,20) → accent colour; wrist → purple; rest → teal
        ctx.fillStyle = i === 0 ? '#7c6aff'
                      : [4,8,12,16,20].includes(i) ? '#ffffff'
                      : '#00e5b0';
        ctx.shadowBlur  = i === 0 ? 10 : 6;
        ctx.shadowColor = i === 0 ? '#7c6aff' : '#00e5b0';
        ctx.fill();
        ctx.shadowBlur = 0;
      });
    }

    // ── Corner bracket bbox ───────────────────────────────────────────
    if (data.bbox) {
      const { x1, y1, x2, y2 } = data.bbox;
      const bx = x1 * W, by = y1 * H;
      const bw = (x2 - x1) * W, bh = (y2 - y1) * H;
      const cs = Math.min(bw, bh) * 0.16;

      const grd = ctx.createLinearGradient(bx, by, bx + bw, by + bh);
      grd.addColorStop(0, '#7c6aff');
      grd.addColorStop(1, '#00e5b0');
      ctx.strokeStyle = grd;
      ctx.lineWidth   = 2.5;
      ctx.shadowBlur  = 10;
      ctx.shadowColor = '#7c6aff88';
      ctx.beginPath();
      [[bx,by,1,1],[bx+bw,by,-1,1],[bx,by+bh,1,-1],[bx+bw,by+bh,-1,-1]].forEach(([cx,cy,dx,dy]) => {
        ctx.moveTo(cx + dx * cs, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy * cs);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // ── Floating badge ────────────────────────────────────────────────
    liveLetterEl.textContent = data.letter;
    liveConfEl.textContent   = data.confidence.toFixed(0) + '%';
    if (data.top5 && data.top5.length > 1) {
      liveTopRest.textContent = data.top5.slice(1, 3)
        .map(x => `${x.letter} ${x.confidence.toFixed(0)}%`)
        .join('  ·  ');
    }
    liveBadge.classList.remove('hidden');
  }

  startBtn.addEventListener('click', startLive);
  stopBtn.addEventListener('click', stopLive);

  // Stop stream if user navigates away
  window.addEventListener('beforeunload', () => { if (liveActive) stopLive(); });
})();
