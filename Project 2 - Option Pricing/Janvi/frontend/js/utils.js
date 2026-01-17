// Shared Utility Functions for Glass Box Option Strategist

const API_BASE_URL = 'http://localhost:5001/api';

/**
 * Make an API request to the backend
 */
async function apiRequest(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Format currency values
 */
function formatCurrency(value) {
    const absValue = Math.abs(value);
    const formatted = absValue.toLocaleString('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    return value < 0 ? `-${formatted}` : formatted;
}

/**
 * Format percentage values
 */
function formatPercent(value) {
    return `${(value * 100).toFixed(2)}%`;
}

/**
 * Display error message
 */
function showError(message, containerId = 'error-container') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-danger">
            <strong>Error:</strong> ${message}
        </div>
    `;
    container.classList.remove('hidden');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        container.classList.add('hidden');
    }, 5000);
}

/**
 * Display success message
 */
function showSuccess(message, containerId = 'success-container') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-success">
            ${message}
        </div>
    `;
    container.classList.remove('hidden');
    
    setTimeout(() => {
        container.classList.add('hidden');
    }, 3000);
}

/**
 * Display info message
 */
function showInfo(message, containerId = 'info-container') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="alert alert-info">
            ${message}
        </div>
    `;
    container.classList.remove('hidden');
}

/**
 * Show loading state on button
 */
function setButtonLoading(buttonId, isLoading, originalText = 'Calculate') {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = `<span class="loading"></span> Processing...`;
    } else {
        button.disabled = false;
        button.innerHTML = originalText;
    }
}

/**
 * Get risk badge HTML
 */
function getRiskBadgeHTML(riskLevel) {
    const levelLower = riskLevel.toLowerCase();
    return `<span class="risk-badge risk-${levelLower}">${riskLevel}</span>`;
}

/**
 * Create a metric row for display
 */
function createMetricRow(label, value, isPositive = null) {
    let valueClass = '';
    if (isPositive === true) valueClass = 'positive';
    if (isPositive === false) valueClass = 'negative';
    
    return `
        <div class="metric">
            <span class="metric-label">${label}</span>
            <span class="metric-value ${valueClass}">${value}</span>
        </div>
    `;
}

/**
 * Validate numeric input
 */
function validateNumber(value, min = -Infinity, max = Infinity, fieldName = 'Field') {
    const num = parseFloat(value);
    
    if (isNaN(num)) {
        throw new Error(`${fieldName} must be a valid number`);
    }
    
    if (num < min) {
        throw new Error(`${fieldName} must be at least ${min}`);
    }
    
    if (num > max) {
        throw new Error(`${fieldName} must be at most ${max}`);
    }
    
    return num;
}

/**
 * Validate required field
 */
function validateRequired(value, fieldName = 'Field') {
    if (!value || value.toString().trim() === '') {
        throw new Error(`${fieldName} is required`);
    }
    return value;
}

/**
 * Build strategy configuration object
 */
function buildStrategyConfig(formData, legs) {
    return {
        stock_price: validateNumber(formData.stockPrice, 0.01, Infinity, 'Stock price'),
        ticker: formData.ticker || '',
        risk_free_rate: validateNumber(formData.riskFreeRate || 0.05, 0, 1, 'Risk-free rate'),
        volatility: validateNumber(formData.volatility || 0.20, 0.01, 5, 'Volatility'),
        legs: legs
    };
}

/**
 * Format option leg for display
 */
function formatOptionLeg(leg, index) {
    const position = leg.position.charAt(0).toUpperCase() + leg.position.slice(1);
    const optionType = leg.option_type.charAt(0).toUpperCase() + leg.option_type.slice(1);
    
    return `
        <div class="option-leg ${leg.position}" data-index="${index}">
            <div class="leg-header">
                <span class="leg-type">${position} ${optionType}</span>
                <button class="leg-remove" onclick="removeLeg(${index})">Remove</button>
            </div>
            <div class="grid grid-2">
                <div>
                    <strong>Strike:</strong> ${formatCurrency(leg.strike)}<br>
                    <strong>Quantity:</strong> ${leg.quantity} contract(s)<br>
                    <strong>Expiration:</strong> ${leg.expiration_days} days
                </div>
                <div>
                    <strong>Price:</strong> ${formatCurrency(leg.price || 0)}<br>
                    <strong>Cost:</strong> ${formatCurrency(leg.cost || 0)}<br>
                    ${leg.details ? `<strong>Intrinsic:</strong> ${formatCurrency(leg.details.intrinsic_value)}` : ''}
                </div>
            </div>
        </div>
    `;
}

/**
 * Create a line chart using Chart.js
 */
function createLineChart(canvasId, labels, datasets, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart if it exists
    if (window.charts && window.charts[canvasId]) {
        window.charts[canvasId].destroy();
    }
    
    // Initialize charts object if it doesn't exist
    if (!window.charts) {
        window.charts = {};
    }
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            },
            tooltip: {
                mode: 'index',
                intersect: false
            }
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: options.xLabel || 'X Axis'
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: options.yLabel || 'Y Axis'
                },
                ticks: {
                    callback: function(value) {
                        return formatCurrency(value);
                    }
                }
            }
        }
    };
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: { ...defaultOptions, ...options }
    });
    
    window.charts[canvasId] = chart;
    return chart;
}

/**
 * Export data to CSV
 */
function exportToCSV(data, filename) {
    const csv = convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

/**
 * Convert array of objects to CSV
 */
function convertToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvRows = [];
    
    // Add header row
    csvRows.push(headers.join(','));
    
    // Add data rows
    for (const row of data) {
        const values = headers.map(header => {
            const value = row[header];
            return typeof value === 'string' ? `"${value}"` : value;
        });
        csvRows.push(values.join(','));
    }
    
    return csvRows.join('\n');
}

/**
 * Debounce function to limit rate of function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Format date to YYYY-MM-DD
 */
function formatDate(date) {
    if (typeof date === 'string') {
        return date;
    }
    
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    
    return `${year}-${month}-${day}`;
}

/**
 * Calculate days between two dates
 */
function daysBetween(date1, date2) {
    const d1 = new Date(date1);
    const d2 = new Date(date2);
    const diffTime = Math.abs(d2 - d1);
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
}
