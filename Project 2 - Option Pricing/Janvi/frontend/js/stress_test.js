// Stress Test JavaScript

let strategyLegs = [];
let baseStrategyConfig = null;
let stressResults = null;

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
        
        // Hide results
        hideResults();
        
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
    hideResults();
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
 * Hide all result cards
 */
function hideResults() {
    document.getElementById('summaryCard').style.display = 'none';
    document.getElementById('priceStressCard').style.display = 'none';
    document.getElementById('volStressCard').style.display = 'none';
    document.getElementById('timeDecayCard').style.display = 'none';
    document.getElementById('interactiveCard').style.display = 'none';
}

/**
 * Run all stress tests
 */
async function runStressTests() {
    try {
        // Validate
        if (strategyLegs.length === 0) {
            showError('Please add at least one option leg');
            return;
        }

        const stockPrice = validateNumber(document.getElementById('stockPrice').value, 0.01, Infinity, 'Stock price');
        const volatility = validateNumber(document.getElementById('volatility').value, 0.01, 5, 'Volatility');
        const riskFreeRate = validateNumber(document.getElementById('riskFreeRate').value, 0, 1, 'Risk-free rate');
        const ticker = document.getElementById('ticker').value.trim();

        // Build strategy config
        baseStrategyConfig = {
            stock_price: stockPrice,
            ticker: ticker,
            risk_free_rate: riskFreeRate,
            volatility: volatility,
            legs: strategyLegs
        };

        // Show loading
        setButtonLoading('stressBtn', true, '⚡ Run Stress Tests');

        // Call comprehensive stress test API
        const result = await apiRequest('/stress/comprehensive', 'POST', {
            strategy: baseStrategyConfig
        });

        stressResults = result;

        // Display results
        displayStressResults(result);
        
        showSuccess('Stress tests completed!');

    } catch (error) {
        showError(error.message);
    } finally {
        setButtonLoading('stressBtn', false, '⚡ Run Stress Tests');
    }
}

/**
 * Display stress test results
 */
function displayStressResults(results) {
    // Show all cards
    document.getElementById('summaryCard').style.display = 'block';
    document.getElementById('priceStressCard').style.display = 'block';
    document.getElementById('volStressCard').style.display = 'block';
    document.getElementById('timeDecayCard').style.display = 'block';
    document.getElementById('interactiveCard').style.display = 'block';

    // Risk summary
    displayRiskSummary(results.risk_summary);

    // Price stress test
    displayPriceStress(results.price_stress);

    // Volatility stress test
    displayVolStress(results.volatility_stress);

    // Time decay
    displayTimeDecay(results.time_decay);

    // Initialize interactive sliders
    resetSliders();
    updateInteractive();
}

/**
 * Display risk summary
 */
function displayRiskSummary(summary) {
    const summaryHTML = `
        ${createMetricRow('Risk Level', getRiskBadgeHTML(summary.risk_level))}
        ${createMetricRow('Worst Case (Price)', formatCurrency(summary.worst_case_price_shock), false)}
        ${createMetricRow('Best Case (Price)', formatCurrency(summary.best_case_price_shock), true)}
        ${createMetricRow('Worst Case (Vol)', formatCurrency(summary.worst_case_vol_shock), false)}
        ${createMetricRow('Best Case (Vol)', formatCurrency(summary.best_case_vol_shock), true)}
        ${createMetricRow('Time Decay Impact', formatCurrency(summary.total_time_decay))}
    `;
    
    document.getElementById('riskSummary').innerHTML = summaryHTML;
    
    document.getElementById('overallExplanation').innerHTML = `
        <h4>📖 Overall Risk Assessment</h4>
        <p>${summary.risk_level === 'EXTREME' || summary.risk_level === 'HIGH' ? '⚠️ ' : ''}${results.overall_explanation || 'See individual stress tests below for detailed analysis.'}</p>
    `;
}

/**
 * Display price stress test
 */
function displayPriceStress(priceStress) {
    // Draw chart
    const datasets = [
        {
            label: 'P&L vs Price Change',
            data: priceStress.pnl,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.1
        },
        {
            label: 'Zero Line',
            data: new Array(priceStress.price_changes.length).fill(0),
            borderColor: '#6b7280',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        }
    ];

    const labels = priceStress.price_changes.map(pc => (pc * 100).toFixed(0) + '%');

    createLineChart('priceStressChart', labels, datasets, {
        xLabel: 'Stock Price Change',
        yLabel: 'Profit / Loss'
    });

    // Explanation
    document.getElementById('priceExplanation').innerHTML = `
        <h4>📖 Price Sensitivity Analysis</h4>
        <p>${priceStress.explanation}</p>
    `;
}

/**
 * Display volatility stress test
 */
function displayVolStress(volStress) {
    // Draw chart
    const datasets = [
        {
            label: 'P&L vs Volatility',
            data: volStress.pnl,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.1
        },
        {
            label: 'Zero Line',
            data: new Array(volStress.vol_changes.length).fill(0),
            borderColor: '#6b7280',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        }
    ];

    const labels = volStress.volatilities.map(v => v.toFixed(1) + '%');

    createLineChart('volStressChart', labels, datasets, {
        xLabel: 'Volatility Level',
        yLabel: 'Profit / Loss'
    });

    // Explanation
    document.getElementById('volExplanation').innerHTML = `
        <h4>📖 Volatility Sensitivity Analysis</h4>
        <p>${volStress.explanation}</p>
        <p><strong>Base Volatility:</strong> ${volStress.base_volatility}%</p>
    `;
}

/**
 * Display time decay analysis
 */
function displayTimeDecay(timeDecay) {
    // Draw chart
    const datasets = [
        {
            label: 'Strategy Value',
            data: timeDecay.strategy_values,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.2)',
            borderWidth: 3,
            fill: true,
            tension: 0.1
        },
        {
            label: 'P&L',
            data: timeDecay.pnl,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.1
        }
    ];

    createLineChart('timeDecayChart', timeDecay.days, datasets, {
        xLabel: 'Days Forward',
        yLabel: 'Value'
    });

    // Explanation
    document.getElementById('timeExplanation').innerHTML = `
        <h4>📖 Time Decay Analysis</h4>
        <p>${timeDecay.explanation}</p>
        <p><strong>Total Decay:</strong> ${formatCurrency(timeDecay.total_decay)}</p>
    `;
}

/**
 * Reset interactive sliders
 */
function resetSliders() {
    document.getElementById('priceSlider').value = 0;
    document.getElementById('volSlider').value = 0;
    document.getElementById('timeSlider').value = 0;
    
    document.getElementById('priceValue').textContent = '0%';
    document.getElementById('volValue').textContent = '0%';
    document.getElementById('timeValue').textContent = '0 days';
}

/**
 * Update interactive scenario (debounced)
 */
const updateInteractive = debounce(async function() {
    if (!baseStrategyConfig) return;

    try {
        // Get slider values
        const priceChange = parseInt(document.getElementById('priceSlider').value) / 100;
        const volChange = parseInt(document.getElementById('volSlider').value) / 100;
        const daysForward = parseInt(document.getElementById('timeSlider').value);

        // Update displayed values
        document.getElementById('priceValue').textContent = (priceChange * 100).toFixed(0) + '%';
        document.getElementById('volValue').textContent = (volChange * 100).toFixed(0) + '%';
        document.getElementById('timeValue').textContent = daysForward + ' days';

        // Build modified strategy config
        const modifiedConfig = {
            stock_price: baseStrategyConfig.stock_price * (1 + priceChange),
            risk_free_rate: baseStrategyConfig.risk_free_rate,
            volatility: baseStrategyConfig.volatility * (1 + volChange),
            legs: baseStrategyConfig.legs.map(leg => ({
                ...leg,
                expiration_days: Math.max(1, leg.expiration_days - daysForward)
            }))
        };

        // Calculate strategy value
        const result = await apiRequest('/strategy/build', 'POST', modifiedConfig);

        // Display scenario metrics
        const isProfitable = result.total_cost > 0;
        
        const metricsHTML = `
            ${createMetricRow('Modified Stock Price', formatCurrency(modifiedConfig.stock_price))}
            ${createMetricRow('Modified Volatility', (modifiedConfig.volatility * 100).toFixed(2) + '%')}
            ${createMetricRow('Days Remaining', modifiedConfig.legs[0].expiration_days)}
            <div style="margin: 20px 0; border-top: 2px solid var(--border-color); padding-top: 20px;"></div>
            ${createMetricRow('Strategy Value', formatCurrency(-result.total_cost))}
            ${createMetricRow('Max Profit', formatCurrency(result.risk_analysis.max_profit), true)}
            ${createMetricRow('Max Loss', formatCurrency(Math.abs(result.risk_analysis.max_loss)), false)}
            ${createMetricRow('Risk Level', getRiskBadgeHTML(result.risk_analysis.risk_level))}
        `;
        
        document.getElementById('scenarioMetrics').innerHTML = metricsHTML;

    } catch (error) {
        console.error('Interactive update error:', error);
    }
}, 300);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Stress Test loaded');
});
