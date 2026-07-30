/* Dashboard JavaScript */

const API_BASE = window.location.origin;

let priceChart = null;

async function fetchJSON(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function formatCurrency(value) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    if (isNaN(num)) return '--';
    const sign = num >= 0 ? '+' : '';
    return `${sign}$${num.toFixed(2)}`;
}

function formatPips(value) {
    if (value === null || value === undefined) return '--';
    const num = parseFloat(value);
    if (isNaN(num)) return '--';
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(1)} pips`;
}

function colorizeLogLine(line) {
    let html = line
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    const levelMatch = html.match(/\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b/);
    if (levelMatch) {
        const level = levelMatch[1];
        const cls = `log-level-${level.toLowerCase()}`;
        html = html.replace(levelMatch[0], `<span class="${cls}">${level}</span>`);
    }

    const tsMatch = html.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/);
    if (tsMatch) {
        html = html.replace(tsMatch[0], `<span class="log-time">${tsMatch[0]}</span>`);
    }

    return html;
}

async function loadStats() {
    try {
        const data = await fetchJSON('/api/stats');
        document.getElementById('stat-total').textContent = data.total_trades ?? 0;
        document.getElementById('stat-winrate').textContent = (data.win_rate ?? 0) + '%';
        const netEl = document.getElementById('stat-net-pnl');
        netEl.textContent = formatCurrency(data.net_pnl);
        netEl.className = `text-2xl font-bold ${(data.net_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`;
        const avgEl = document.getElementById('stat-avg-pnl');
        avgEl.textContent = formatCurrency(data.avg_pnl);
        avgEl.className = `text-2xl font-bold ${(data.avg_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`;

        if (data.last_trade) {
            const lt = data.last_trade;
            const pnlColor = (lt.gross_profit ?? 0) >= 0 ? 'text-green-400' : 'text-red-400';
            document.getElementById('last-trade').innerHTML = `
                <div class="grid grid-cols-2 gap-2">
                    <div><span class="text-gray-500">Type:</span> <span class="text-white">${lt.type}</span></div>
                    <div><span class="text-gray-500">P/L:</span> <span class="${pnlColor}">${formatCurrency(lt.gross_profit)}</span></div>
                    <div><span class="text-gray-500">Entry:</span> <span class="text-white">${lt.entry_price}</span></div>
                    <div><span class="text-gray-500">Exit:</span> <span class="text-white">${lt.close_price}</span></div>
                    <div class="col-span-2"><span class="text-gray-500">Time:</span> <span class="text-white">${lt.timestamp}</span></div>
                </div>
            `;
        }
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

async function loadChart() {
    try {
        const data = await fetchJSON('/api/chart');
        const ctx = document.getElementById('priceChart').getContext('2d');
        const emptyEl = document.getElementById('chart-empty');

        if (!data.points || data.points.length === 0) {
            emptyEl.classList.remove('hidden');
            return;
        }
        emptyEl.classList.add('hidden');

        const pricePoints = data.points.map(p => ({ x: new Date(p.x).getTime(), y: p.y }));
        const entryMarkers = data.markers.filter(m => m.type === 'entry').map(m => ({ x: new Date(m.x).getTime(), y: m.y }));
        const exitMarkers = data.markers.filter(m => m.type === 'exit').map(m => ({ x: new Date(m.x).getTime(), y: m.y }));

        if (priceChart) {
            priceChart.destroy();
        }

        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Price',
                        data: pricePoints,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    },
                    {
                        label: 'Entry',
                        data: entryMarkers,
                        borderColor: '#22c55e',
                        backgroundColor: '#22c55e',
                        pointStyle: 'triangle',
                        pointRadius: 8,
                        showLine: false,
                    },
                    {
                        label: 'Exit',
                        data: exitMarkers,
                        borderColor: '#ef4444',
                        backgroundColor: '#ef4444',
                        pointStyle: 'rectRot',
                        pointRadius: 8,
                        showLine: false,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#9ca3af', boxWidth: 12 }
                    },
                    tooltip: {
                        backgroundColor: '#1f2937',
                        titleColor: '#f3f4f6',
                        bodyColor: '#d1d5db',
                        borderColor: '#374151',
                        borderWidth: 1,
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true,
                            },
                            mode: 'x',
                        },
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM d HH:mm',
                                hour: 'MMM d HH:mm',
                            },
                        },
                        ticks: { color: '#6b7280', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
                        grid: { color: '#1f2937' },
                        title: {
                            display: true,
                            text: 'Time',
                            color: '#9ca3af',
                        }
                    },
                    y: {
                        ticks: { color: '#6b7280' },
                        grid: { color: '#1f2937' },
                        title: {
                            display: true,
                            text: 'Price',
                            color: '#9ca3af',
                        }
                    }
                }
            }
        });
    } catch (err) {
        console.error('Failed to load chart:', err);
    }
}

async function loadModelInfo() {
    try {
        const data = await fetchJSON('/api/model-info');
        const container = document.getElementById('model-info');
        if (!data.model_exists) {
            container.innerHTML = '<div class="text-gray-500">No trained model found.</div>';
            return;
        }
        container.innerHTML = `
            <div class="flex justify-between"><span class="text-gray-400">Type</span><span class="text-white">${data.model_type || '--'}</span></div>
            <div class="flex justify-between"><span class="text-gray-400">Features</span><span class="text-white">${data.n_features || '--'}</span></div>
            <div class="flex justify-between"><span class="text-gray-400">Classes</span><span class="text-white">${(data.classes || []).join(', ') || '--'}</span></div>
            <div class="flex justify-between"><span class="text-gray-400">Last Modified</span><span class="text-white">${data.last_modified || '--'}</span></div>
            ${data.feature_names?.length ? `<div class="mt-2"><span class="text-gray-400">Feature names:</span><div class="text-gray-300 mt-1">${data.feature_names.join(', ')}</div></div>` : ''}
        `;
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

async function loadLogs() {
    try {
        const data = await fetchJSON('/api/logs');
        const container = document.getElementById('log-container');
        if (!data.lines || data.lines.length === 0) {
            container.innerHTML = '<div class="text-gray-500">No logs available.</div>';
            return;
        }
        container.innerHTML = data.lines.map(line => `<div class="log-line">${colorizeLogLine(line)}</div>`).join('');
        container.scrollTop = container.scrollHeight;
    } catch (err) {
        console.error('Failed to load logs:', err);
    }
}

document.getElementById('refresh-logs').addEventListener('click', loadLogs);

async function copyLogs() {
    const container = document.getElementById('log-container');
    const text = container.innerText || container.textContent;
    try {
        await navigator.clipboard.writeText(text);
        const btn = document.getElementById('copy-logs');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = originalText; }, 1500);
    } catch (err) {
        console.error('Failed to copy logs:', err);
        alert('Failed to copy logs to clipboard');
    }
}

document.getElementById('copy-logs').addEventListener('click', copyLogs);

async function loadTraderStatus() {
    try {
        const data = await fetchJSON('/api/trader/status');
        const statusEl = document.getElementById('trader-status');
        const startBtn = document.getElementById('btn-start-trader');
        const stopBtn = document.getElementById('btn-stop-trader');

        const status = (data.status || 'unknown').toLowerCase();
        statusEl.textContent = data.status || 'Unknown';

        statusEl.className = 'px-2 py-1 rounded-full text-xs border';
        if (status === 'running') {
            statusEl.classList.add('bg-green-900/30', 'text-green-400', 'border-green-800');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else if (status === 'exited' || status === 'dead' || status === 'not_found') {
            statusEl.classList.add('bg-red-900/30', 'text-red-400', 'border-red-800');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        } else {
            statusEl.classList.add('bg-yellow-900/30', 'text-yellow-400', 'border-yellow-800');
            startBtn.disabled = true;
            stopBtn.disabled = true;
        }
    } catch (err) {
        console.error('Failed to load trader status:', err);
    }
}

async function controlTrader(action) {
    const startBtn = document.getElementById('btn-start-trader');
    const stopBtn = document.getElementById('btn-stop-trader');
    startBtn.disabled = true;
    stopBtn.disabled = true;

    try {
        const res = await fetch(`/api/trader/${action}`, { method: 'POST' });
        const data = await res.json();
        if (!data.ok) {
            alert(`Failed: ${data.detail || 'Unknown error'}`);
        }
        await loadTraderStatus();
    } catch (err) {
        alert(`Error: ${err.message}`);
        loadTraderStatus();
    }
}

document.getElementById('btn-start-trader').addEventListener('click', () => controlTrader('start'));
document.getElementById('btn-stop-trader').addEventListener('click', () => controlTrader('stop'));

document.getElementById('reset-chart-zoom').addEventListener('click', () => {
    if (priceChart) {
        priceChart.resetZoom();
    }
});

async function init() {
    await loadStats();
    await loadChart();
    await loadModelInfo();
    await loadLogs();
    await loadTraderStatus();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
