const state = { type: "text", proposal: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  })[char]);
}

function setStatus(message, isError = false) {
  const node = $("#statusMessage");
  node.textContent = message;
  node.classList.toggle("error", isError);
}

$$('.tab').forEach(button => button.addEventListener('click', () => {
  state.type = button.dataset.type;
  $$('.tab').forEach(item => {
    const selected = item === button;
    item.classList.toggle('active', selected);
    item.setAttribute('aria-selected', String(selected));
  });
  $$('.input-pane').forEach(pane => pane.classList.toggle('active', pane.dataset.pane === state.type));
  setStatus('');
}));

$('#textInput').addEventListener('input', event => {
  $('#textCount').textContent = event.target.value.length.toLocaleString('zh-TW');
});

$('#analyzeButton').addEventListener('click', async () => {
  const form = new FormData();
  if (state.type === 'text') {
    const text = $('#textInput').value.trim();
    if (!text) return setStatus('請先貼上文字。', true);
    form.append('text', text);
  } else {
    const input = state.type === 'image' ? $('#imageInput') : $('#audioInput');
    if (!input.files[0]) return setStatus('請先選擇檔案。', true);
    form.append('upload', input.files[0]);
  }
  await submit(form);
});

async function submit(form) {
  $('#analyzeButton').disabled = true;
  setStatus('本機分析中……');
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '分析失敗');
    if (data.status === 'needs_confirmation') return showCorrection(data);
    showResult(data);
    setStatus('分析完成。');
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $('#analyzeButton').disabled = false;
  }
}

function showCorrection(data) {
  state.proposal = data;
  $('#originalCorrection').textContent = data.original_text;
  $('#suggestedCorrection').textContent = data.suggested_text;
  $('#correctionChanges').innerHTML = (data.corrections || []).map(change =>
    `<span><del>${escapeHtml(change.before || '∅')}</del> → <ins>${escapeHtml(change.after || '∅')}</ins></span>`
  ).join('');
  $('#correctionPanel').classList.remove('hidden');
  $('#resultPanel').classList.add('hidden');
  $('#correctionPanel').scrollIntoView({ behavior: 'smooth' });
  setStatus('請確認模型建議的文字修正。');
}

async function analyzeAccepted(text) {
  const form = new FormData();
  form.append('text', text);
  form.append('source_type', state.proposal.input_type);
  form.append('correction_confirmed', 'true');
  $('#correctionPanel').classList.add('hidden');
  await submit(form);
}

$('#useOriginal').addEventListener('click', () => analyzeAccepted(state.proposal.original_text));
$('#useSuggested').addEventListener('click', () => analyzeAccepted(state.proposal.suggested_text));

function showResult(data) {
  $('#correctionPanel').classList.add('hidden');
  $('#resultPanel').classList.remove('hidden');
  $('#riskScore').textContent = data.risk_score ?? '—';
  $('#riskLevel').textContent = data.risk_level;
  $('#riskLevel').className = `risk-level ${data.risk_level === '高風險' ? 'high' : data.risk_level === '中風險' ? 'medium' : 'low'}`;
  $('#categories').innerHTML = (data.categories || []).map(item => `<span>${escapeHtml(item)}</span>`).join('');
  $('#evidence').innerHTML = (data.evidence || []).map(item => `<li><mark>${escapeHtml(item.text)}</mark> <small>${escapeHtml(item.kind)}</small></li>`).join('') || '<li>未命中明確話術</li>';
  $('#safetyActions').innerHTML = (data.safety_actions || []).map(item => `<li>${escapeHtml(item)}</li>`).join('');
  $('#analysisText').textContent = data.analysis_text || '無有效文字';
  $('#disclaimer').textContent = data.disclaimer;
  $('#resultPanel').scrollIntoView({ behavior: 'smooth' });
}
