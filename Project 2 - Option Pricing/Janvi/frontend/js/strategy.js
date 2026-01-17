// Strategy Builder JavaScript

// Store strategy legs
let strategyLegs = [];

/**
 * Add a new option leg to the strategy
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
        showSuccess(`Added ${leg.position} ${leg.option_type} leg successfully!`);
        
        // Clear results when adding new leg
        document.getElementById('resultsCard').style.display = 'none';
        document.getElementById('chartCard').style.display = 'none';
        
    } catch (error) {
        showError(error.message);
    }
}

/**
 * Remove a leg from the strategy
 */
function removeLeg(index) {
    strategyLegs.splice(index, 1);
    updateLegsDisplay();
    
    // Clear results when removing leg
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('chartCard').style.display = 'none';
}

/**
 * Update the display of current strategy legs
 */
function updateLegsDisplay() {
    const container = document.getElementById('legsContainer');
    
    if (strategyLegs.length === 0) {
        container.innerHTML = '<p class="text-center" style="color: var(--text-secondary);">No legs added yet. Add your first option above.</p>';
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
 * Calculate the complete strategy
 */
async function calculateStrategy() {
    try {
        // Validate we have legs
        if (strategyLegs.length === 0) {
            showError('Please add at least one option leg to your strategy');
            return;
        }

        // Get form data
        const stockPrice = validateNumber(document.getElementById('stockPrice').value, 0.01, Infinity, 'Stock price');
        const volatility = validateNumber(document.getElementById('volatility').value, 0.01, 5, 'Volatility');
        const riskFreeRate = validateNumber(document.getElementById('riskFreeRate').value, 0, 1, 'Risk-free rate');
        const ticker = document.getElementById('ticker').value.trim();

        // Build strategy config
        const strategyConfig = {
            stock_price: stockPrice,
            ticker: ticker,
            risk_free_rate: riskFreeRate,
            volatility: volatility,
            legs: strategyLegs
        };

        // Show loading state
        setButtonLoading('calculateBtn', true, 'Calculate Strategy');

        // Call API
        const result = await apiRequest('/strategy/build', 'POST', strategyConfig);

        // Display results
        displayResults(result);

        // Show success
        showSuccess('Strategy calculated successfully!');

    } catch (error) {
        showError(error.message);
    } finally {
        setButtonLoading('calculateBtn', false, 'Calculate Strategy');
    }
}

/**
 * Display strategy results
 */
function displayResults(result) {
    // Show results card
    document.getElementById('resultsCard').style.display = 'block';
    document.getElementById('chartCard').style.display = 'block';

    // Display cost analysis
    const costHTML = `
        ${createMetricRow('Total Strategy Cost', formatCurrency(result.total_cost), result.total_cost > 0)}
        ${createMetricRow('Cost Type', result.risk_analysis.cost_type.toUpperCase())}
    `;
    document.getElementById('costResults').innerHTML = costHTML;

    // Display risk metrics
    const riskHTML = `
        ${createMetricRow('Maximum Profit', formatCurrency(result.risk_analysis.max_profit), true)}
        ${createMetricRow('Maximum Loss', formatCurrency(Math.abs(result.risk_analysis.max_loss)), false)}
        ${createMetricRow('Risk Level', getRiskBadgeHTML(result.risk_analysis.risk_level))}
        ${createMetricRow('Breakeven Points', result.risk_analysis.breakeven_points.map(p => formatCurrency(p)).join(', ') || 'None')}
    `;
    document.getElementById('riskResults').innerHTML = riskHTML;

    // Display explanation
    document.getElementById('explanationText').innerHTML = `
        ${result.risk_analysis.explanation}<br><br>
        <strong>Recommendation:</strong> ${result.risk_analysis.recommendation}
    `;

    // Display individual leg details
    displayLegDetails(result.legs);

    // Draw payoff chart
    drawPayoffChart(result.payoff_diagram, result.stock_price);
}

/**
 * Display individual leg pricing details
 */
function displayLegDetails(legs) {
    const container = document.getElementById('legDetails');
    
    const html = legs.map((leg, index) => {
        const position = leg.position.charAt(0).toUpperCase() + leg.position.slice(1);
        const optionType = leg.option_type.charAt(0).toUpperCase() + leg.option_type.slice(1);
        
        return `
            <div class="option-leg ${leg.position}">
                <div class="leg-header">
                    <span class="leg-type">${position} ${optionType} @ ${formatCurrency(leg.strike)}</span>
                </div>
                <div class="grid grid-2">
                    <div>
                        <strong>Option Price:</strong> ${formatCurrency(leg.price)}<br>
                        <strong>Total Cost:</strong> ${formatCurrency(leg.cost)}<br>
                        <strong>Intrinsic Value:</strong> ${formatCurrency(leg.details.intrinsic_value)}<br>
                        <strong>Time Value:</strong> ${formatCurrency(leg.details.time_value)}
                    </div>
                    <div>
                        <strong>Delta:</strong> ${leg.greeks.delta}<br>
                        <strong>Gamma:</strong> ${leg.greeks.gamma}<br>
                        <strong>Theta:</strong> ${leg.greeks.theta}/day<br>
                        <strong>Vega:</strong> ${leg.greeks.vega}
                    </div>
                </div>
                <div class="explanation" style="margin-top: 10px;">
                    <small>${leg.details.explanation}</small>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

/**
 * Draw payoff diagram
 */
function drawPayoffChart(payoffData, currentPrice) {
    const datasets = [
        {
            label: 'Payoff at Expiration',
            data: payoffData.payoffs,
            borderColor: '#2563eb',
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.1
        },
        {
            label: 'Zero Line',
            data: new Array(payoffData.stock_prices.length).fill(0),
            borderColor: '#6b7280',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
        }
    ];

    // Add vertical line at current price
    const currentPriceIndex = payoffData.stock_prices.findIndex(p => p >= currentPrice);
    
    createLineChart('payoffChart', payoffData.stock_prices, datasets, {
        xLabel: 'Stock Price at Expiration',
        yLabel: 'Profit / Loss',
        plugins: {
            annotation: {
                annotations: {
                    currentPrice: {
                        type: 'line',
                        xMin: currentPriceIndex,
                        xMax: currentPriceIndex,
                        borderColor: '#10b981',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        label: {
                            enabled: true,
                            content: 'Current Price',
                            position: 'top'
                        }
                    }
                }
            }
        }
    });

    // Add chart explanation
    const chartExplanation = `
        <h4>Understanding the Payoff Diagram</h4>
        <p>
            This chart shows your profit or loss at expiration for different stock prices.
            The <strong style="color: #2563eb;">blue line</strong> represents your total payoff.
            Points above zero indicate profit, below zero indicate loss.
            ${payoffData.breakeven_points.length > 0 ? 
                `Your breakeven point(s) are at: ${payoffData.breakeven_points.map(p => formatCurrency(p)).join(', ')}.` : 
                'This strategy has no breakeven point within the displayed range.'}
        </p>
    `;
    document.getElementById('chartExplanation').innerHTML = chartExplanation;
}

/**
 * Quick Strategy Templates
 */
function loadTemplate(templateName) {
    strategyLegs = [];
    const stockPrice = parseFloat(document.getElementById('stockPrice').value) || 100;
    
    switch(templateName) {
        case 'long-call':
            strategyLegs.push({
                option_type: 'call',
                position: 'buy',
                strike: stockPrice * 1.05,
                quantity: 1,
                expiration_days: 30
            });
            break;
            
        case 'bull-call-spread':
            strategyLegs.push({
                option_type: 'call',
                position: 'buy',
                strike: stockPrice * 1.00,
                quantity: 1,
                expiration_days: 30
            });
            strategyLegs.push({
                option_type: 'call',
                position: 'sell',
                strike: stockPrice * 1.10,
                quantity: 1,
                expiration_days: 30
            });
            break;
            
        case 'iron-condor':
            strategyLegs.push({
                option_type: 'put',
                position: 'buy',
                strike: stockPrice * 0.90,
                quantity: 1,
                expiration_days: 30
            });
            strategyLegs.push({
                option_type: 'put',
                position: 'sell',
                strike: stockPrice * 0.95,
                quantity: 1,
                expiration_days: 30
            });
            strategyLegs.push({
                option_type: 'call',
                position: 'sell',
                strike: stockPrice * 1.05,
                quantity: 1,
                expiration_days: 30
            });
            strategyLegs.push({
                option_type: 'call',
                position: 'buy',
                strike: stockPrice * 1.10,
                quantity: 1,
                expiration_days: 30
            });
            break;
    }
    
    updateLegsDisplay();
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Strategy Builder loaded');
});
