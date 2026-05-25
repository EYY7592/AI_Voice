/* ── 智慧語音詐騙檢測系統 — Frontend JavaScript v3.1 ── */

/** @type {File|null} */
let selectedFile = null;
const STEPS = [
  { id: 'preprocess', icon: '🔧', title: '音頻預處理', desc: '載入 → 降噪 → 特徵萃取' },
  { id: 'transcribe', icon: '📝', title: '語音轉錄', desc: 'Whisper + 繁簡轉換' },
  { id: 'agent', icon: '🤖', title: '三 Agent 分析', desc: '聲紋 · 語義 · 記憶' },
  { id: 'fusion', icon: '⚡', title: 'SE-Attention 融合', desc: '動態權重判決' },
  { id: 'report', icon: '📊', title: '報告生成', desc: 'Markdown 報告 + 歷史記錄' },
];

/* ── Init ─────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  renderSteps();
  loadHistory();
});

/* ── Page Navigation ─────────────────────────────── */
function switchPage(pageId, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById('page-' + pageId);
  if (page) page.classList.add('active');
  if (el) el.classList.add('active');
}

/* ── File Handling ────────────────────────────────── */
function onDragOver(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.add('drag-over');
}
function onDragLeave(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
}
function onFileSelected(e) {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
}
function handleFile(file) {
  const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a', 'audio/flac', 'audio/ogg', 'audio/wave', 'audio/x-wav'];
  const ext = file.name.split('.').pop().toLowerCase();
  const validExts = ['wav', 'mp3', 'm4a', 'flac', 'ogg'];
  if (!validExts.includes(ext) && !validTypes.includes(file.type)) {
    setStatus('不支援的檔案格式', 'badge-red');
    return;
  }
  selectedFile = file;
  document.getElementById('fileInfo').style.display = 'flex';
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileMeta').textContent =
    `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || ext}`;
  document.getElementById('analyzeBtn').disabled = false;
  setStatus('檔案已就緒', 'badge-green');
}
function clearFile() {
  selectedFile = null;
  document.getElementById('fileInfo').style.display = 'none';
  document.getElementById('fileInput').value = '';
  document.getElementById('analyzeBtn').disabled = true;
  setStatus('', '');
}
function setStatus(text, cls) {
  const el = document.getElementById('uploadStatus');
  el.textContent = text;
  el.className = 'badge ' + cls;
  el.style.fontSize = '13px';
}

/* ── Steps ────────────────────────────────────────── */
function renderSteps() {
  const list = document.getElementById('stepList');
  list.innerHTML = STEPS.map(s => `
    <div class="step-card" id="step-${s.id}">
      <div class="step-icon">${s.icon}</div>
      <div>
        <div class="step-title">${s.title}</div>
        <div class="step-desc">${s.desc}</div>
      </div>
      <div class="step-status">
        <span class="badge" id="step-status-${s.id}" style="font-size:12px;">等待中</span>
      </div>
    </div>
  `).join('');
}

function updateStep(idx, state, text) {
  const s = STEPS[idx];
  const card = document.getElementById('step-' + s.id);
  const badge = document.getElementById('step-status-' + s.id);
  card.className = 'step-card ' + state;
  if (state === 'running') {
    badge.innerHTML = '<span class="spinner"></span>';
    badge.className = 'badge badge-accent';
  } else if (state === 'done') {
    badge.textContent = text || '完成';
    badge.className = 'badge badge-green';
  } else if (state === 'error') {
    badge.textContent = text || '失敗';
    badge.className = 'badge badge-red';
  }
}

function updateProgress(pct, label) {
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent = Math.round(pct) + '%';
  if (label) document.getElementById('progressLabel').textContent = label;
}

/* ── Console ─────────────────────────────────────── */
function logConsole(msg, type) {
  const c = document.getElementById('console');
  const time = new Date().toLocaleTimeString('zh-TW', { hour12: false });
  const cls = type === 'err' ? 'err' : type === 'warn' ? 'warn' : type === 'info' ? 'info' : '';
  c.innerHTML += `<span class="${cls}">[${time}] ${msg}</span>\n`;
  c.scrollTop = c.scrollHeight;
}
function clearConsole() {
  document.getElementById('console').innerHTML = '═══ 智慧語音詐騙檢測系統 v3.1 ═══\n';
}

/* ── Gauge ────────────────────────────────────────── */
function setGauge(pct) {
  const arc = document.getElementById('gaugeArc');
  const txt = document.getElementById('gaugePct');
  const fill = 251 * (pct / 100);
  arc.setAttribute('stroke-dasharray', `${fill} ${251 - fill}`);
  txt.textContent = Math.round(pct) + '%';
}

/* ── Analysis ────────────────────────────────────── */
async function startAnalysis() {
  if (!selectedFile) return;

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;

  switchPage('progress', document.querySelector('[data-page="progress"]'));
  renderSteps();
  clearConsole();
  logConsole('開始分析: ' + selectedFile.name, 'info');

  try {
    const formData = new FormData();
    formData.append('audio', selectedFile);
    formData.append('language', document.getElementById('langSelect').value);

    updateStep(0, 'running');
    updateProgress(10, '音頻預處理中...');
    logConsole('Step 1: 音頻預處理 — 載入、降噪、特徵萃取', 'info');

    const resp = await fetch('/api/analyze', {
      method: 'POST',
      body: formData
    });

    if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `伺服器錯誤 (${resp.status})`);
    }

    const data = await resp.json();

    updateStep(0, 'done', `${data.duration?.toFixed(1)}s`);
    updateProgress(20, '音頻預處理完成');
    logConsole(`  音頻: ${data.duration?.toFixed(2)}s, SNR: ${data.snr?.toFixed(1)}dB`);

    await sleep(200);
    updateStep(1, 'running');
    updateProgress(40, '語音轉錄中...');
    logConsole('Step 2: Whisper 語音轉錄', 'info');
    await sleep(300);
    updateStep(1, 'done');
    logConsole(`  轉錄結果: ${(data.transcript || '').substring(0, 50)}...`);

    await sleep(200);
    updateStep(2, 'running');
    updateProgress(60, '三 Agent 分析中...');
    logConsole('Step 3: 三 Agent 並行分析', 'info');
    await sleep(400);
    updateStep(2, 'done');
    if (data.agents) {
      data.agents.forEach(a => {
        logConsole(`  ${a.name}: P=${(a.fraud_probability * 100).toFixed(1)}%, C=${(a.confidence * 100).toFixed(0)}%`);
      });
    }

    await sleep(200);
    updateStep(3, 'running');
    updateProgress(80, 'SE-Attention 融合中...');
    logConsole('Step 4: SE-Attention 動態融合', 'info');
    await sleep(300);
    updateStep(3, 'done');
    logConsole(`  融合結果: ${data.risk_level}, P=${(data.fraud_probability * 100).toFixed(1)}%`);

    await sleep(200);
    updateStep(4, 'running');
    updateProgress(95, '生成報告...');
    logConsole('Step 5: 報告生成', 'info');
    await sleep(200);
    updateStep(4, 'done');
    updateProgress(100, '分析完成 ✓');
    logConsole(`分析完成! 風險等級: ${data.risk_level}`, data.risk_level === '高風險' ? 'err' : 'info');

    const fill = document.getElementById('progressFill');
    fill.classList.add('success');

    renderResults(data);
    addToHistory(data);

    await sleep(1000);
    switchPage('result', document.querySelector('[data-page="result"]'));

  } catch (err) {
    logConsole('分析失敗: ' + err.message, 'err');
    updateProgress(0, '分析失敗');
    STEPS.forEach((s, i) => {
      const card = document.getElementById('step-' + s.id);
      if (card && card.classList.contains('running')) updateStep(i, 'error', err.message);
    });
  } finally {
    btn.disabled = false;
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ── Render Results ──────────────────────────────── */
function renderResults(data) {
  document.getElementById('resultEmpty').style.display = 'none';
  document.getElementById('resultContent').style.display = 'block';

  document.getElementById('resultTimestamp').textContent =
    `分析時間: ${new Date().toLocaleString('zh-TW')} · 耗時 ${data.elapsed?.toFixed(2)}s`;

  const pct = (data.fraud_probability || 0) * 100;
  setGauge(pct);
  const badge = document.getElementById('riskBadge');
  badge.textContent = data.risk_level || '未知';
  badge.className = 'badge ' + riskBadgeClass(data.risk_level);

  document.getElementById('fraudProb').textContent = `${pct.toFixed(1)}%`;
  document.getElementById('snrValue').textContent = `${data.snr?.toFixed(1) || '—'}dB`;
  document.getElementById('audioDuration').textContent = `${data.duration?.toFixed(1) || '—'}s`;
  document.getElementById('elapsedTime').textContent = `${data.elapsed?.toFixed(2) || '—'}s`;

  renderAgents(data.agents || []);
  document.getElementById('transcript').textContent = data.transcript || '（無轉錄結果）';
  renderWeights(data.weights || {});
  renderProsody(data.prosody || {});
}

function riskBadgeClass(level) {
  switch (level) {
    case '高風險': return 'badge-red';
    case '中風險': return 'badge-amber';
    case '低風險': return 'badge-green';
    default: return '';
  }
}

function renderAgents(agents) {
  const agentStyles = {
    voiceprint: { icon: '🔊', name: '聲紋分析', method: '韻律 + 深偽偵測', cls: 'agent-voiceprint' },
    semantic: { icon: '📝', name: '語義分析', method: 'BERT 分類', cls: 'agent-semantic' },
    memory: { icon: '🧠', name: '記憶比對', method: 'FAISS 檢索', cls: 'agent-memory' },
  };

  const container = document.getElementById('agentResults');
  container.innerHTML = agents.map(a => {
    const style = agentStyles[a.name] || { icon: '❓', name: a.name, method: '—', cls: '' };
    const probColor = a.fraud_probability > 0.6 ? 'var(--coral)' :
      a.fraud_probability > 0.3 ? 'var(--amber)' : 'var(--accent2)';
    const confPct = (a.confidence * 100).toFixed(0);
    const sqPct = (a.signal_quality * 100).toFixed(0);
    const confColor = a.confidence > 0.5 ? 'var(--accent)' : 'var(--text3)';
    const sqColor = a.signal_quality > 0.5 ? 'var(--blue)' : 'var(--text3)';

    return `
    <div class="agent-card ${style.cls}">
      <div class="agent-header">
        <div class="agent-icon">${style.icon}</div>
        <div>
          <div class="agent-name">${style.name}</div>
          <div class="agent-method">${style.method}</div>
        </div>
      </div>
      <div class="agent-metrics">
        <div class="metric-item">
          <div class="metric-label">詐騙機率</div>
          <div class="metric-value" style="color:${probColor}">${(a.fraud_probability * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">信心度</div>
          <div class="metric-value">${confPct}%</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${confPct}%;background:${confColor}"></div></div>
        </div>
        <div class="metric-item">
          <div class="metric-label">信號品質</div>
          <div class="metric-value">${sqPct}%</div>
          <div class="bar-wrap"><div class="bar-fill" style="width:${sqPct}%;background:${sqColor}"></div></div>
        </div>
      </div>
      <div class="agent-explain">${a.explanation || '—'}</div>
    </div>`;
  }).join('');
}

function renderWeights(weights) {
  const container = document.getElementById('weightDisplay');
  const items = Object.entries(weights);
  if (items.length === 0) {
    container.textContent = '—';
    return;
  }
  const nameMap = { voiceprint: '聲紋', semantic: '語義', memory: '記憶' };
  const colorMap = { voiceprint: 'var(--blue)', semantic: 'var(--accent)', memory: 'var(--accent2)' };

  container.innerHTML = items.map(([k, v]) => {
    const pct = (v * 100).toFixed(1);
    const name = nameMap[k] || k;
    const color = colorMap[k] || 'var(--text2)';
    return `
    <div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <span style="font-size:14px;font-weight:600;">${name}</span>
        <span style="font-size:14px;font-weight:700;color:${color}">${pct}%</span>
      </div>
      <div class="bar-wrap" style="height:8px;">
        <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
    </div>`;
  }).join('');
}

function renderProsody(prosody) {
  const container = document.getElementById('prosodyDisplay');
  const items = Object.entries(prosody);
  if (items.length === 0) {
    container.textContent = '—';
    return;
  }
  container.innerHTML = items.map(([k, v]) => `
    <div class="stat-box">
      <div class="stat-label">${k}</div>
      <div class="stat-value" style="font-size:16px;">${typeof v === 'number' ? v.toFixed(3) : v}</div>
    </div>
  `).join('');
}

/* ── History & Multi-Tier Memory ──────────────────────── */

let sessionHistory = [];
let permanentHistory = [];
let currentTab = 'session'; // 'session' or 'permanent'

// 重新整理守衛
window.addEventListener('beforeunload', (e) => {
  if (sessionHistory.length > 0) {
    e.preventDefault();
    e.returnValue = '短期會話中尚有未存檔的分析記錄，重新整理將會遺失。確定要離開嗎？';
    return e.returnValue;
  }
});

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    if (res.ok) {
      const data = await res.json();
      permanentHistory = data.history || [];
      if (currentTab === 'permanent') renderHistory();
    }
  } catch (err) {
    console.error("載入紀錄失敗", err);
  }
}

function switchMemoryTab(tab) {
  currentTab = tab;
  const sBtn = document.getElementById('tab-session');
  const pBtn = document.getElementById('tab-permanent');
  
  if (tab === 'session') {
    sBtn?.classList.remove('btn-outline');
    sBtn?.classList.add('btn-primary');
    pBtn?.classList.remove('btn-primary');
    pBtn?.classList.add('btn-outline');
  } else {
    sBtn?.classList.remove('btn-primary');
    sBtn?.classList.add('btn-outline');
    pBtn?.classList.remove('btn-outline');
    pBtn?.classList.add('btn-primary');
    loadHistory(); 
  }
  renderHistory();
}

function renderHistory() {
  const tbody = document.getElementById('historyBody');
  const data = currentTab === 'session' ? sessionHistory : permanentHistory;
  
  if (!data || data.length === 0) {
    const msg = currentTab === 'session' ? '本次會話尚無分析記錄' : '尚無長期防詐記錄';
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--text2);padding:40px;">${msg}</td></tr>`;
    return;
  }
  
  tbody.innerHTML = data.map((h, idx) => {
    const rType = h.fraud_type || h.risk || '—';
    let cls = '';
    if (rType.includes('高風險')) cls = 'badge-red';
    else if (rType.includes('中風險')) cls = 'badge-amber';
    else if (rType.includes('低風險')) cls = 'badge-green';
    
    const preview = h.text ? (h.text.length > 40 ? h.text.substring(0, 40) + '...' : h.text) : '無文字';
    const timeStr = h.timestamp || h.time || '—';
    const id = h.id || `idx-${idx}`;
    
    return `<tr>
      <td style="font-size:13px;color:var(--text2)">${timeStr}</td>
      <td><span class="badge ${cls}">${rType}</span></td>
      <td style="max-width:300px;text-overflow:ellipsis;overflow:hidden;white-space:nowrap;font-size:14px;" title="${(h.text || '').replace(/"/g, '&quot;')}">${preview}</td>
      <td>
        <button class="btn btn-outline btn-sm" style="color:var(--red);border-color:var(--red);padding:4px 12px;" onclick="deleteHistory('${id}', '${currentTab}')">
          🗑️ 刪除
        </button>
      </td>
    </tr>`;
  }).join('');
}

function addToHistory(data) {
  sessionHistory.unshift({
    id: `session-${Date.now()}`,
    time: new Date().toLocaleString('zh-TW'),
    risk: data.risk_level,
    text: data.transcript,
  });
  if (currentTab === 'session') renderHistory();
  setTimeout(loadHistory, 2000); 
}

async function deleteHistory(id, target) {
  const confirmMsg = target === 'session' 
    ? '確定要從本次會話中移除這筆記錄嗎？' 
    : '確定要從 FAISS 長期記憶庫中永久刪除這筆案例嗎？\n(刪除後 AI 將不再學習此通話特徵)';
    
  if (!confirm(confirmMsg)) return;
  
  if (target === 'session') {
    sessionHistory = sessionHistory.filter(h => h.id !== id);
    renderHistory();
  } else {
    try {
      const res = await fetch('/api/history/' + id, { method: 'DELETE' });
      if (res.ok) {
        await loadHistory();
      } else {
        const err = await res.json().catch(() => ({}));
        alert('刪除失敗: ' + (err.detail || '未知錯誤'));
      }
    } catch (err) {
      console.error('刪除錯誤', err);
      alert('網路錯誤');
    }
  }
}
