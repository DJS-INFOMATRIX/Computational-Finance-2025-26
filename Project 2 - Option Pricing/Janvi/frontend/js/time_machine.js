// Time Machine JavaScript

let strategyLegs = [];

/**
 * Check available date range for a ticker
 */
async function checkDateRange() {
    try {
        const ticker = validateRequired(document.getElementById('ticker').value, 'Stock ticker').toUpperCase();
        
        const result = await apiRequest(`/data/date-range/${ticker}`, 'GET');
        
        if (result.available) {
            const infoHTML = `
                <div class="alert alert-success">
                    ✓ Data available for ${ticker}<br>
                    <strong>Date Range:</strong> ${result.earliest_date} to ${result.latest_date}<br>
                    <strong>Total Days:</strong> ${result.total_days.toLocaleString()}
                </div>
            `;
            document.getElementById('dateRangeInfo').innerHTML = infoHTML;
            
            // Set max date on date picker
            document.getElementById('startDate').max = result.latest_date;
            document.getElementById('startDate').min = result.earliest_date;
        } else {
            const errorHTML = `
                <div class="alert alert-danger">
                    ✗ No data available for ${ticker}<br>
                    ${result.error}
                </div>
            `;
            document.getElementById('dateRangeInfo').innerHTML = errorHTML;
        }
        
    } catch (error) {
        showError(error.message);
    }
}

/**
 * Add option leg to strategy
 */
function addLeg() {
    try {
        const leg = {
            option_type: document.getElementById('optionType').value,
            position: document.getElementById('position').value,
            strike: validateNumber(document.getElementById('strike').value, 0.01, Infinity, 'Strike price'),
            quantity: validateNumber(document.getElementById('quantity').value, 1, 1000, 'Quantity'),
            expiration_days: validateNumber(document.getElementById('expirationDays').value, 1, 3650, 'Days to expiration')
        };

        strategyLegs.push(leg);
        updateLegsDisplay();
        showSuccess(`Added ${leg.position} ${leg.option_type} leg!`);
        
    } catch (error) {
        showError(error.message);
    }
}

/**
 * Remove leg from strategy
 */
function removeLeg(index) {
    strategyLegs.splice(index, 1);
    updateLegsDisplay();
}

/**
 * Update legs display
 */
function updateLegsDisplay() {
    const container = document.getElementById('legsContainer');
    
    if (strategyLegs.length === 0) {
        container.innerHTML = '<p class="text-center" style="color: var(--text-secondary);">No legs added yet.</p>';
        return;
    }

    container.innerHTML = strategyLegs.map((leg, index) => {
        const position = leg.position.charAt(0).toUpperCase() + leg.position.slice(1);
        const optionType = leg.option_type.charAt(0).toUpperCase() + leg.option_type.slice(1);
        
        return `
            <div class="option-leg ${leg.position}" data-index="${index}">
                <div class="leg-header">
                    <span class="leg-type">${position} ${optionType}</span>
                    <button class="leg-remove" onclick="removeLeg(${index})">Remove</button>
                </div>
                <div class="grid grid-3">
                    <div><strong>Strike:</strong> ${formatCurrency(leg.strike)}</div>
                    <div><strong>Quantity:</strong> ${leg.quantity}</div>
                    <div><strong>Expiration:</strong> ${leg.expiration_days} days</div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Run historical simulation
 */
async function runSimulation() {
    try {
        // Validate inputs
        if (strategyLegs.length === 0) {
            showError('Please add at least one option leg');
            return;
        }

        const ticker = validateRequired(document.getElementById('ticker').value, 'Stock ticker').toUpperCase();
        const startDate = validateRequired(document.getElementById('startDate').value, 'Start date');
        const volatility = document.getElementById('volatility').value ? 
            validateNumber(document.getElementById('volatility').value, 0.01, 5, 'Volatility') : null;
        const riskFreeRate = validateNumber(document.getElementById('riskFreeRate').value, 0, 1, 'Risk-free rate');

        // Build request
        const requestData = {
            ticker: ticker,
            start_date: startDate,
            strategy: {
                stock_price: 0,  // Will be set by backend from historical data
                risk_free_rate: riskFreeRate,
                volatility: volatility,
                legs: strategyLegs
            }
        };

        // Show loading
        setButtonLoading('simulateBtn', true, '🚀 Run Historical Simulation');

        // Call API
        const result = await apiRequest('/simulate/historical', 'POST', requestData);

        // Display results
        displaySimulationResults(result);
        
        showSuccess('Simulation completed successfully!');

    } catch (error) {
        showError(error.message);
    } finally {
        setButtonLoading('simulateBtn', false, '🚀 Run Historical Simulation');
    }
}

/**
 * Display simulation results
 */
function displaySimulationResults(result) {
    // Show result cards
    document.getElementById('resultsCard').style.display = 'block';
    document.getElementById('chartCard').style.display = 'block';
    document.getElementById('stockChartCard').style.display = 'block';

    // Performance metrics
    const isProfitable = result.final_pnl > 0;
    const returnPct = result.return_pct;
    
    const performanceHTML = `
        ${createMetricRow('Initial Stock Price', formatCurrency(result.initial_stock_price))}
        ${createMetricRow('Final Stock Price', formatCurrency(result.final_stock_price))}
        ${createMetricRow('Price Change', formatPercent((result.final_stock_price - result.initial_stock_price) / result.initial_stock_price))}
        <div style="margin: 20px 0; border-top: 2px solid var(--border-color); padding-top: 20px;"></div>
        ${createMetricRow('Initial Cost', formatCurrency(result.initial_cost), result.initial_cost > 0)}
        ${createMetricRow('Final P&L', formatCurrency(result.final_pnl), isProfitable)}
        ${createMetricRow('Return %', returnPct.toFixed(2) + '%', isProfitable)}
        ${createMetricRow('Max Profit', formatCurrency(result.max_profit), true)}
        ${createMetricRow('Max Drawdown', formatCurrency(result.max_drawdown), false)}
    `;
    
    document.getElementById('performanceResults').innerHTML = performanceHTML;

    // Explanation
    const explanationHTML = `
        <h4>📖 Simulation Analysis</h4>
        <p>${result.explanation}</p>
        <p style="margin-top: 10px;">
            <strong>Volatility Used:</strong> ${(result.volatility_used * 100).toFixed(2)}%
            ${result.volatility_used ? '(calculated from historical data)' : ''}
        </p>
    `;
    document.getElementById('resultExplanation').innerHTML = explanationHTML;

    // Draw charts
    drawPerformanceChart(result.time_series);
    drawStockChart(result.time_series);

    // Scroll to results
    document.getElementById('resultsCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Draw performance chart
 */
function drawPerformanceChart(timeSeries) {
    const datasets = [
        {
            label: 'Strategy P&L',
            data: timeSeries.pnl,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.1
        },
        {
            label: 'Zero Line',
            data: new Array(timeSeries.dates.length).fill(0),
            borderColor: '#6b7280',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        }
    ];

    createLineChart('performanceChart', timeSeries.dates, datasets, {
        xLabel: 'Date',
        yLabel: 'Profit / Loss',
        scales: {
            x: {
                ticks: {
                    maxTicksLimit: 10
                }
            }
        }
    });
}

/**
 * Draw stock price chart
 */
function drawStockChart(timeSeries) {
    const datasets = [
        {
            label: 'Stock Price',
            data: timeSeries.stock_prices,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
        }
    ];

    createLineChart('stockChart', timeSeries.dates, datasets, {
        xLabel: 'Date',
        yLabel: 'Stock Price',
        scales: {
            x: {
                ticks: {
                    maxTicksLimit: 10
                }
            },
            y: {
                ticks: {
                    callback: function(value) {
                        return formatCurrency(value);
                    }
                }
            }
        }
    });
}

// Set default date to 1 year ago
document.addEventListener('DOMContentLoaded', () => {
    const oneYearAgo = new Date();
    oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    document.getElementById('startDate').value = formatDate(oneYearAgo);
    
    console.log('Time Machine loaded');
});
