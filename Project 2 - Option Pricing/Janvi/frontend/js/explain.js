// Explain Page JavaScript

/**
 * Calculate example option pricing with full transparency
 */
async function calculateExample() {
    try {
        // Get inputs
        const S = validateNumber(document.getElementById('exampleS').value, 0.01, Infinity, 'Stock price');
        const K = validateNumber(document.getElementById('exampleK').value, 0.01, Infinity, 'Strike price');
        const days = validateNumber(document.getElementById('exampleDays').value, 1, 3650, 'Days to expiration');
        const sigma = validateNumber(document.getElementById('exampleSigma').value, 0.01, 5, 'Volatility');
        const r = validateNumber(document.getElementById('exampleR').value, 0, 1, 'Risk-free rate');
        const optionType = document.getElementById('exampleType').value;

        // Show loading
        setButtonLoading('calculateExampleBtn', true, 'Calculate Option Price');

        // Prepare request
        const data = {
            stock_price: S,
            strike: K,
            expiration_days: days,
            volatility: sigma,
            risk_free_rate: r
        };

        // Call appropriate API endpoint
        const endpoint = optionType === 'call' ? '/price/call' : '/price/put';
        const result = await apiRequest(endpoint, 'POST', data);

        // Display results
        displayExampleResults(result, optionType, S, K);

    } catch (error) {
        showError(error.message);
    } finally {
        setButtonLoading('calculateExampleBtn', false, 'Calculate Option Price');
    }
}

/**
 * Display example calculation results
 */
function displayExampleResults(result, optionType, S, K) {
    // Show results section
    document.getElementById('exampleResults').style.display = 'block';

    // Format option type
    const typeDisplay = optionType.charAt(0).toUpperCase() + optionType.slice(1);

    // Build metrics HTML
    const metricsHTML = `
        ${createMetricRow('Option Price', formatCurrency(result.price))}
        ${createMetricRow('Intrinsic Value', formatCurrency(result.intrinsic_value))}
        ${createMetricRow('Time Value', formatCurrency(result.time_value))}
        <div style="margin: 20px 0; border-top: 2px solid var(--border-color); padding-top: 20px;">
            <h4 style="color: var(--primary-color); margin-bottom: 15px;">Intermediate Calculations</h4>
            ${createMetricRow('d₁', result.d1 !== null ? result.d1 : 'N/A')}
            ${createMetricRow('d₂', result.d2 !== null ? result.d2 : 'N/A')}
            ${createMetricRow('N(d₁)', result.N_d1 !== null ? result.N_d1 : 'N/A')}
            ${createMetricRow('N(d₂)', result.N_d2 !== null ? (result.N_d2 * 100).toFixed(2) + '% probability' : 'N/A')}
        </div>
        ${result.greeks ? `
        <div style="margin: 20px 0; border-top: 2px solid var(--border-color); padding-top: 20px;">
            <h4 style="color: var(--primary-color); margin-bottom: 15px;">Greeks (Risk Metrics)</h4>
            ${createMetricRow('Delta (Δ)', result.greeks.delta)}
            ${createMetricRow('Gamma (Γ)', result.greeks.gamma)}
            ${createMetricRow('Theta (Θ)', formatCurrency(result.greeks.theta) + '/day')}
            ${createMetricRow('Vega (ν)', result.greeks.vega)}
            ${createMetricRow('Rho (ρ)', result.greeks.rho)}
        </div>
        ` : ''}
    `;

    document.getElementById('exampleMetrics').innerHTML = metricsHTML;

    // Build explanation
    const moneyness = S > K ? 'in-the-money' : (S === K ? 'at-the-money' : 'out-of-the-money');
    
    let explanationHTML = `
        <h4>📖 Breaking Down the Calculation</h4>
        <p>
            This <strong>${typeDisplay}</strong> option is currently <strong>${moneyness}</strong> 
            with the stock at ${formatCurrency(S)} and strike at ${formatCurrency(K)}.
        </p>
        <p>${result.explanation}</p>
    `;

    if (result.d1 !== null && result.d2 !== null) {
        explanationHTML += `
            <p style="margin-top: 15px;">
                <strong>Understanding the Math:</strong><br>
                • d₁ = ${result.d1} (standardized distance to strike)<br>
                • d₂ = ${result.d2} (adjusted for volatility)<br>
                • N(d₂) = ${(result.N_d2 * 100).toFixed(1)}% - This is the estimated probability that this 
                option will expire in-the-money (be worth exercising).<br>
                • The ${formatCurrency(result.time_value)} time value reflects the market's expectation 
                of potential future gains before expiration.
            </p>
        `;
    }

    if (result.greeks) {
        explanationHTML += `
            <p style="margin-top: 15px;">
                <strong>What the Greeks Mean:</strong><br>
                • <strong>Delta (${result.greeks.delta}):</strong> If the stock moves $1, this option's price 
                will change by approximately ${formatCurrency(result.greeks.delta)}.<br>
                • <strong>Theta (${formatCurrency(result.greeks.theta)}/day):</strong> This option loses 
                approximately ${formatCurrency(Math.abs(result.greeks.theta))} in value per day due to time decay.<br>
                • <strong>Vega (${result.greeks.vega}):</strong> If volatility increases by 1%, this option's 
                price will increase by approximately ${formatCurrency(result.greeks.vega)}.
            </p>
        `;
    }

    document.getElementById('exampleExplanation').innerHTML = explanationHTML;

    // Scroll to results
    document.getElementById('exampleResults').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Explain page loaded');
});
