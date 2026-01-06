// Global state
let uploadedData = null;
let manualInputData = [];
let predictionResults = null;

// Tab functionality
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');

        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabId}-tab`).classList.add('active');
    });
});

// File upload functionality
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const clearFileBtn = document.getElementById('clear-file');
const csvPreview = document.getElementById('csv-preview');

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle file
function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    // Show file info
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = 'flex';

    // Read and parse CSV
    const reader = new FileReader();
    reader.onload = (e) => {
        const csvContent = e.target.result;
        parseCSV(csvContent);
    };
    reader.readAsText(file);
}

// Clear file
clearFileBtn.addEventListener('click', () => {
    fileInput.value = '';
    fileInfo.style.display = 'none';
    csvPreview.style.display = 'none';
    uploadedData = null;
    updatePredictButton();
});

// Parse CSV
function parseCSV(csvContent) {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index];
        });
        data.push(row);
    }

    uploadedData = { headers, data };
    displayCSVPreview(headers, data);
    updatePredictButton();
}

// Display CSV preview
function displayCSVPreview(headers, data) {
    const previewTable = document.getElementById('preview-table');
    const previewHeader = document.getElementById('preview-header');
    const previewBody = document.getElementById('preview-body');
    const rowCount = document.getElementById('row-count');

    // Clear previous content
    previewHeader.innerHTML = '';
    previewBody.innerHTML = '';

    // Add headers
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    previewHeader.appendChild(headerRow);

    // Add data rows (show first 10)
    const previewCount = Math.min(data.length, 10);
    for (let i = 0; i < previewCount; i++) {
        const row = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = data[i][header] || '';
            row.appendChild(td);
        });
        previewBody.appendChild(row);
    }

    rowCount.textContent = `Showing ${previewCount} of ${data.length} rows`;
    csvPreview.style.display = 'block';
}

// Sample files
document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const sampleFile = btn.getAttribute('data-file');
        loadSampleFile(sampleFile);
    });
});

// Load sample file
async function loadSampleFile(filename) {
    try {
        // Try to fetch from backend API endpoint first
        const apiUrl = `http://localhost:8000/sample/${filename}`;
        let response = await fetch(apiUrl);

        if (!response.ok) {
            throw new Error('Sample file endpoint not available');
        }

        const csvContent = await response.text();

        fileName.textContent = filename;
        fileSize.textContent = 'Sample file';
        fileInfo.style.display = 'flex';

        parseCSV(csvContent);
    } catch (error) {
        showNotification('Sample files not available. Please upload your own CSV file.', 'error');
        console.error('Error loading sample:', error);
    }
}

// Manual input form
const manualForm = document.getElementById('manual-form');
const inputDate = document.getElementById('input-date');
const inputTime = document.getElementById('input-time');
const nsmInput = document.getElementById('nsm');

// Auto-calculate NSM from time
inputTime.addEventListener('change', () => {
    if (inputTime.value) {
        const [hours, minutes] = inputTime.value.split(':').map(Number);
        const nsm = hours * 3600 + minutes * 60;
        nsmInput.value = nsm;
    }
});

// Auto-detect day of week and week status
inputDate.addEventListener('change', () => {
    if (inputDate.value) {
        const date = new Date(inputDate.value);
        const dayOfWeek = date.toLocaleDateString('en-US', { weekday: 'long' });
        const dayNum = date.getDay();

        document.getElementById('day-of-week').value = dayOfWeek;
        document.getElementById('week-status').value = (dayNum === 0 || dayNum === 6) ? 'Weekend' : 'Weekday';
    }
});

// Fill sample data
document.getElementById('fill-sample').addEventListener('click', () => {
    // Set current date and time
    const now = new Date();
    inputDate.value = now.toISOString().split('T')[0];
    inputTime.value = '09:30';

    // Fill with sample values
    document.getElementById('lagging-power').value = '5.5';
    document.getElementById('leading-power').value = '1.2';
    document.getElementById('lagging-pf').value = '63.5';
    document.getElementById('leading-pf').value = '98.5';
    document.getElementById('co2').value = '0';
    document.getElementById('load-type').value = 'Light_Load';

    // Trigger events
    inputTime.dispatchEvent(new Event('change'));
    inputDate.dispatchEvent(new Event('change'));
});

// Reset form
document.getElementById('reset-form').addEventListener('click', () => {
    manualForm.reset();
    nsmInput.value = '';
});

// Submit manual form
manualForm.addEventListener('submit', (e) => {
    e.preventDefault();

    // Format date for CSV
    const dateValue = inputDate.value;
    const timeValue = inputTime.value;
    const [year, month, day] = dateValue.split('-');
    const formattedDate = `${day}/${month}/${year} ${timeValue}`;

    const row = {
        'date': formattedDate,
        'Lagging_Current_Reactive.Power_kVarh': document.getElementById('lagging-power').value,
        'Leading_Current_Reactive_Power_kVarh': document.getElementById('leading-power').value,
        'CO2(tCO2)': document.getElementById('co2').value,
        'Lagging_Current_Power_Factor': document.getElementById('lagging-pf').value,
        'Leading_Current_Power_Factor': document.getElementById('leading-pf').value,
        'NSM': nsmInput.value,
        'WeekStatus': document.getElementById('week-status').value,
        'Day_of_week': document.getElementById('day-of-week').value,
        'Load_Type': document.getElementById('load-type').value
    };

    manualInputData.push(row);
    displayManualPreview();
    manualForm.reset();
    nsmInput.value = '';
    updatePredictButton();

    showNotification('Input added successfully!', 'success');
});

// Display manual input preview
function displayManualPreview() {
    if (manualInputData.length === 0) {
        document.getElementById('manual-preview').style.display = 'none';
        return;
    }

    const manualPreview = document.getElementById('manual-preview');
    const manualHeader = document.getElementById('manual-header');
    const manualBody = document.getElementById('manual-body');

    manualHeader.innerHTML = '';
    manualBody.innerHTML = '';

    const headers = Object.keys(manualInputData[0]);

    // Add headers
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    manualHeader.appendChild(headerRow);

    // Add data
    manualInputData.forEach((row, index) => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header];
            tr.appendChild(td);
        });
        manualBody.appendChild(tr);
    });

    manualPreview.style.display = 'block';
}

// Clear manual inputs
document.getElementById('clear-manual').addEventListener('click', () => {
    manualInputData = [];
    displayManualPreview();
    updatePredictButton();
});

// Update predict button state
function updatePredictButton() {
    const predictBtn = document.getElementById('predict-btn');
    const hasData = uploadedData !== null || manualInputData.length > 0;

    predictBtn.disabled = !hasData;

    if (hasData) {
        const count = uploadedData ? uploadedData.data.length : manualInputData.length;
        document.querySelector('.predict-info').textContent =
            `Ready to predict ${count} record${count !== 1 ? 's' : ''}`;
    } else {
        document.querySelector('.predict-info').textContent =
            'Upload CSV or add manual inputs to enable prediction';
    }
}

// Predict button
document.getElementById('predict-btn').addEventListener('click', async () => {
    const dataToPredict = uploadedData ? uploadedData.data : manualInputData;

    // Show loading
    showLoading(true);

    try {
        // Call actual backend API
        await callPredictionAPI(dataToPredict);
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Prediction failed. Using mock data instead.', 'error');
        // Fallback to mock prediction if API fails
        await simulatePrediction(dataToPredict);
    }

    showLoading(false);
});

// Call actual FastAPI backend
async function callPredictionAPI(data) {
    const API_URL = 'http://localhost:8000/predict';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: data })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        predictionResults = result.predictions;
        displayResults(result.predictions);
        showNotification(`Predictions completed! Avg: ${result.statistics.average_usage} kWh`, 'success');

    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Fallback: Simulate prediction (if backend is not running)
async function simulatePrediction(data) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Generate mock predictions
    const results = data.map(row => {
        // Mock prediction based on load type
        let basePrediction = 3.5;
        if (row.Load_Type === 'Medium_Load') basePrediction = 15.0;
        if (row.Load_Type === 'Maximum_Load') basePrediction = 45.0;

        // Add some randomness
        const prediction = basePrediction + (Math.random() - 0.5) * 2;

        return {
            ...row,
            'Predicted_Usage_kWh': prediction.toFixed(2),
            'Confidence': (85 + Math.random() * 10).toFixed(1) + '%'
        };
    });

    predictionResults = results;
    displayResults(results);
    showNotification('Predictions completed (mock data)!', 'success');
}

// Display results
function displayResults(results) {
    const resultsSection = document.getElementById('results-section');
    const resultsHeader = document.getElementById('results-header');
    const resultsBody = document.getElementById('results-body');

    // Hide preview tables when showing predictions
    const csvPreview = document.getElementById('csv-preview');
    const manualPreview = document.getElementById('manual-preview');
    if (csvPreview) csvPreview.style.display = 'none';
    if (manualPreview) manualPreview.style.display = 'none';

    // Calculate statistics
    const predictions = results.map(r => parseFloat(r.Predicted_Usage_kWh));
    const avgUsage = (predictions.reduce((a, b) => a + b, 0) / predictions.length).toFixed(2);
    const minUsage = Math.min(...predictions).toFixed(2);
    const maxUsage = Math.max(...predictions).toFixed(2);

    document.getElementById('total-predictions').textContent = results.length;
    document.getElementById('avg-usage').textContent = avgUsage + ' kWh';
    document.getElementById('min-usage').textContent = minUsage + ' kWh';
    document.getElementById('max-usage').textContent = maxUsage + ' kWh';

    // Build table
    resultsHeader.innerHTML = '';
    resultsBody.innerHTML = '';

    const headers = Object.keys(results[0]);

    // Add headers with highlight for prediction
    const headerRow = document.createElement('tr');
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        if (header === 'Predicted_Usage_kWh' || header === 'Confidence') {
            th.style.color = 'var(--success-color)';
            th.style.fontWeight = '700';
        }
        headerRow.appendChild(th);
    });
    resultsHeader.appendChild(headerRow);

    // Add data
    results.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header];
            if (header === 'Predicted_Usage_kWh') {
                td.style.color = 'var(--success-color)';
                td.style.fontWeight = '600';
            }
            if (header === 'Confidence') {
                td.style.color = 'var(--primary-color)';
            }
            tr.appendChild(td);
        });
        resultsBody.appendChild(tr);
    });

    resultsSection.style.display = 'block';

    // Render charts with the results data
    renderCharts(results);

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Download CSV
document.getElementById('download-csv').addEventListener('click', () => {
    if (!predictionResults) return;

    const headers = Object.keys(predictionResults[0]);
    const csvContent = [
        headers.join(','),
        ...predictionResults.map(row =>
            headers.map(h => row[h]).join(',')
        )
    ].join('\n');

    downloadFile(csvContent, 'predictions.csv', 'text/csv');
    showNotification('CSV downloaded successfully!', 'success');
});

// Download JSON
document.getElementById('download-json').addEventListener('click', () => {
    if (!predictionResults) return;

    const jsonContent = JSON.stringify(predictionResults, null, 2);
    downloadFile(jsonContent, 'predictions.json', 'application/json');
    showNotification('JSON downloaded successfully!', 'success');
});

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function downloadFile(content, filename, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: ${type === 'success' ? 'var(--success-color)' : 'var(--primary-color)'};
        color: white;
        border-radius: 8px;
        box-shadow: var(--shadow-lg);
        z-index: 2000;
        animation: slideIn 0.3s ease-out;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => document.body.removeChild(notification), 300);
    }, 3000);
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize
updatePredictButton();

// Chart instances
let charts = {
    usage: null,
    loadType: null,
    powerFactor: null,
    time: null
};

// Render charts with prediction data
function renderCharts(results) {
    const chartsSection = document.getElementById('charts-section');
    chartsSection.style.display = 'block';

    // Destroy existing charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });

    // Extract data
    const predictions = results.map(r => parseFloat(r.Predicted_Usage_kWh));
    const loadTypes = results.map(r => r.Load_Type);
    const laggingPF = results.map(r => parseFloat(r.Lagging_Current_Power_Factor));
    const leadingPF = results.map(r => parseFloat(r.Leading_Current_Power_Factor));

    // Extract hour from date if available
    const hours = results.map(r => {
        if (r.date) {
            const dateStr = r.date;
            const match = dateStr.match(/\d{1,2}:\d{2}/);
            if (match) {
                return parseInt(match[0].split(':')[0]);
            }
        }
        return null;
    }).filter(h => h !== null);

    // Chart 1: Predicted Usage Line Chart
    const usageCtx = document.getElementById('usage-chart').getContext('2d');
    charts.usage = new Chart(usageCtx, {
        type: 'line',
        data: {
            labels: predictions.map((_, i) => `#${i + 1}`),
            datasets: [{
                label: 'Predicted Usage (kWh)',
                data: predictions,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Usage (kWh)' }
                }
            }
        }
    });

    // Chart 2: Load Type Distribution (Pie)
    const loadTypeCounts = {};
    loadTypes.forEach(type => {
        loadTypeCounts[type] = (loadTypeCounts[type] || 0) + 1;
    });

    const loadTypeCtx = document.getElementById('load-type-chart').getContext('2d');
    charts.loadType = new Chart(loadTypeCtx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(loadTypeCounts),
            datasets: [{
                data: Object.values(loadTypeCounts),
                backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });

    // Chart 3: Power Factor Comparison (Bar)
    const sampleSize = Math.min(10, laggingPF.length);
    const pfCtx = document.getElementById('power-factor-chart').getContext('2d');
    charts.powerFactor = new Chart(pfCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: sampleSize}, (_, i) => `#${i + 1}`),
            datasets: [
                {
                    label: 'Lagging PF',
                    data: laggingPF.slice(0, sampleSize),
                    backgroundColor: '#3b82f6'
                },
                {
                    label: 'Leading PF',
                    data: leadingPF.slice(0, sampleSize),
                    backgroundColor: '#8b5cf6'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Power Factor' }
                }
            }
        }
    });

    // Chart 4: Time of Day Distribution (if hours available)
    if (hours.length > 0) {
        const hourCounts = {};
        hours.forEach(hour => {
            hourCounts[hour] = (hourCounts[hour] || 0) + 1;
        });

        const timeCtx = document.getElementById('time-chart').getContext('2d');
        charts.time = new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(hourCounts).map(h => `${h}:00`),
                datasets: [{
                    label: 'Records',
                    data: Object.values(hourCounts),
                    backgroundColor: '#06b6d4'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Count' },
                        ticks: { stepSize: 1 }
                    },
                    x: {
                        title: { display: true, text: 'Hour of Day' }
                    }
                }
            }
        });
    } else {
        // Alternative: Show average usage by sample
        const timeCtx = document.getElementById('time-chart').getContext('2d');
        const avgByGroup = [];
        const groupSize = Math.ceil(predictions.length / 10);
        for (let i = 0; i < predictions.length; i += groupSize) {
            const group = predictions.slice(i, i + groupSize);
            avgByGroup.push(group.reduce((a, b) => a + b, 0) / group.length);
        }

        charts.time = new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: avgByGroup.map((_, i) => `Group ${i + 1}`),
                datasets: [{
                    label: 'Avg Usage (kWh)',
                    data: avgByGroup,
                    backgroundColor: '#06b6d4'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Avg Usage (kWh)' }
                    }
                }
            }
        });
    }

    // Scroll to charts
    setTimeout(() => {
        chartsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);
}
