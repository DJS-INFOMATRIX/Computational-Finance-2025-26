const API_URL = "http://127.0.0.1:8000/api";

// Initial Setup
window.onload = () => {
    console.log("App Loaded");
    addLeg(); // Add one default leg
    // analyzeStrategy(false); // Initial analysis (optional, let's wait for user input)
};

// UI Functions
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    // Show selected
    const tab = document.getElementById(`tab-${tabId}`);
    if (tab) tab.classList.add('active');

    // Update button state
    // We strictly match the onclick attribute in HTML to find the button
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => {
        if (btn.getAttribute('onclick').includes(tabId)) {
            btn.classList.add('active');
        }
    });

    // If Explain tab, maybe refresh?
    if (tabId === 'explain') {
        const hasData = document.getElementById('val-price').innerText !== '-';
        if (!hasData) analyzeStrategy(false);
    }
}

function updateVolInput(val) {
    document.getElementById('volatility').value = val;
}
function updateVolSlider(val) {
    document.getElementById('vol-slider').value = val;
}

// --- Leg Management ---
function addLeg() {
    const container = document.getElementById('legs-container');
    const id = Date.now() + Math.random().toString(16).slice(2);

    const div = document.createElement('div');
    div.className = 'leg-card glass-panel';
    div.id = `leg-${id}`;

    div.innerHTML = `
        <button class="remove-leg" onclick="removeLeg('${id}')">&times;</button>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;">
            <select class="leg-side">
                <option value="long">Buy</option>
                <option value="short">Sell</option>
            </select>
            <select class="leg-type">
                <option value="call">Call</option>
                <option value="put">Put</option>
            </select>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
            <div>
                <label>Strike</label>
                <input type="number" class="leg-strike" value="450">
            </div>
            <div>
                <label>Exp (Yrs)</label>
                <input type="number" class="leg-t" value="0.08" step="0.01">
            </div>
            <div>
                <label>Qty</label>
                <input type="number" class="leg-qty" value="1">
            </div>
        </div>
    `;

    container.appendChild(div);
}

function removeLeg(id) {
    const leg = document.getElementById(`leg-${id}`);
    if (leg) leg.remove();
}

function getStrategyData() {
    const ticker = document.getElementById('ticker').value;
    const S = parseFloat(document.getElementById('underlying_price').value);
    const sigma = parseFloat(document.getElementById('volatility').value);
    const r = parseFloat(document.getElementById('rate').value) || 0.045;

    const legs = [];
    document.querySelectorAll('.leg-card').forEach(card => {
        legs.push({
            side: card.querySelector('.leg-side').value, // 'long' or 'short'
            type: card.querySelector('.leg-type').value, // 'call' or 'put'
            K: parseFloat(card.querySelector('.leg-strike').value),
            T: parseFloat(card.querySelector('.leg-t').value),
            quantity: parseInt(card.querySelector('.leg-qty').value)
        });
    });

    return {
        ticker: ticker,
        underlying_price: S,
        volatility: sigma,
        risk_free_rate: r,
        legs: legs
    };
}

// --- API Calls ---

async function analyzeStrategy(autoSwitch = true) {
    const data = getStrategyData();
    console.log("Analyzing:", data);

    try {
        const response = await fetch(`${API_URL}/analyze-strategy`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        if (result.status === 'success') {
            updateDashboard(result.greeks, result.payoff_diagram, result.risk_analysis);
            if (autoSwitch) switchTab('explain');
        } else {
            console.error("API Error:", result);
            alert("Analysis failed. Check console.");
        }
    } catch (e) {
        console.error("Analysis failed", e);
        alert("Server connection failed. Is the backend running?");
    }
}

async function runSimulation() {
    const data = getStrategyData();
    const chartDiv = document.getElementById('sim-chart');
    chartDiv.innerHTML = '<p style="text-align:center; padding-top:20px;">Running Simulation...</p>';

    try {
        const response = await fetch(`${API_URL}/run-simulation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: data.ticker,
                legs: data.legs,
                days: 30,
                volatility: data.volatility
            })
        });

        const result = await response.json();
        if (result.status === 'success') {
            const res = result.results;
            renderSimulationChart(res.history);

            // Add stats overlay
            const statsDiv = document.createElement('div');
            statsDiv.style.textAlign = 'center';
            statsDiv.style.marginTop = '10px';
            statsDiv.innerHTML = `
                <span style="color: #38bdf8; margin-right: 15px;">Total PnL: $${res.total_pnl.toFixed(2)}</span>
                <span style="color: #22c55e; margin-right: 15px;">ROI: ${res.final_roi_pct.toFixed(1)}%</span>
                <span style="color: #ef4444;">Max DD: ${res.max_drawdown_pct.toFixed(1)}%</span>
            `;
            chartDiv.appendChild(statsDiv);
        }
    } catch (e) {
        console.error("Simulation failed", e);
        chartDiv.innerHTML = '<p>Simulation Failed (Check Console)</p>';
    }
}

async function runStressTest() {
    const data = getStrategyData();
    document.getElementById('stress-results').innerHTML = 'Running Scenarios...';

    try {
        const response = await fetch(`${API_URL}/stress-test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        if (result.status === 'success') {
            renderStressResults(result.report);
        }
    } catch (e) {
        console.error("Stress Test failed", e);
        document.getElementById('stress-results').innerText = "Error running stress test.";
    }
}

async function generateReport() {
    const data = getStrategyData();
    try {
        const response = await fetch(`${API_URL}/generate-report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "strategy_report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
        } else {
            alert("Report generation failed");
        }
    } catch (e) {
        console.error("Report gen failed", e);
    }
}

// --- Rendering ---

function updateDashboard(greeks, payoffData, riskAnalysis) {
    // Update Stats
    document.getElementById('val-price').innerText = `$${greeks.price.toFixed(2)}`;
    document.getElementById('val-delta').innerText = greeks.delta.toFixed(2);
    document.getElementById('val-theta').innerText = greeks.theta.toFixed(2);
    document.getElementById('val-vega').innerText = greeks.vega.toFixed(2);

    // Update Risk Analysis Box
    const riskBox = document.getElementById('risk-summary');
    if (riskBox) {
        riskBox.innerHTML = `
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-bottom: 12px;">
                <div><small style="color:#94a3b8">Max Profit</small><br>${riskAnalysis.max_profit}</div>
                <div><small style="color:#94a3b8">Max Loss</small><br><span style="color: #ef4444">${riskAnalysis.max_loss}</span></div>
                <div><small style="color:#94a3b8">POP</small><br>${(riskAnalysis.pop * 100).toFixed(1)}%</div>
                <div><small style="color:#94a3b8">Breakevens</small><br>${riskAnalysis.breakevens.join(', ')}</div>
            </div>
            <div style="font-size: 0.95em; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px; line-height: 1.5;">
                <i class="fa-solid fa-wand-magic-sparkles"></i> ${riskAnalysis.risk_summary}
            </div>
         `;
    }

    // Render Payoff Chart
    const x = payoffData.map(p => p.underlying_price);
    const y = payoffData.map(p => p.pnl);

    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#38bdf8', width: 3 }
    };

    const layout = {
        title: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        margin: { t: 20, l: 40, r: 20, b: 40 },
        xaxis: { title: 'Price', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'PnL', gridcolor: 'rgba(255,255,255,0.1)' }
    };

    Plotly.newPlot('payoff-chart', [trace], layout, { responsive: true, displayModeBar: false });
}

function renderStressResults(report) {
    const container = document.getElementById('stress-results');

    let html = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 15px; align-items:center;">
            <div style="text-align:center;">
                <span style="display:block; font-size:0.8rem; color:#94a3b8;">Risk Score</span>
                <span style="font-size:1.8rem; font-weight:bold; color: ${report.risk_score > 7 ? '#ef4444' : '#22c55e'}">${report.risk_score}<span style="font-size:1rem;color:#64748b">/10</span></span>
            </div>
            <div>
                 <span style="display:block; font-size:0.8rem; color:#94a3b8;">Worst Case</span>
                 <span style="font-size:1.1rem; color:#ef4444;">$${report.worst_case.toFixed(0)}</span>
            </div>
            <div>
                 <span style="display:block; font-size:0.8rem; color:#94a3b8;">Best Case</span>
                 <span style="font-size:1.1rem; color:#22c55e;">$${report.best_case.toFixed(0)}</span>
            </div>
        </div>
        <table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
            <tbody>
    `;

    report.scenarios.forEach(sc => {
        const color = sc.pnl >= 0 ? '#22c55e' : '#ef4444';
        html += `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                <td style="padding: 6px 0;">${sc.scenario}</td>
                <td style="padding: 6px 0; text-align: right; color: ${color};">$${sc.pnl.toFixed(0)}</td>
            </tr>
        `;
    });

    html += `</tbody></table>`;
    container.innerHTML = html;
}

function renderSimulationChart(history) {
    const x = history.map(r => r.date);
    const y = history.map(r => r.pnl);

    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#c084fc', width: 2 },
    };

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#94a3b8' },
        margin: { t: 20, l: 40, r: 20, b: 20 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'PnL ($)', gridcolor: 'rgba(255,255,255,0.1)' }
    };

    Plotly.newPlot('sim-chart', [trace], layout, { responsive: true, displayModeBar: false });
}
