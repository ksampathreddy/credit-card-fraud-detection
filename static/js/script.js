document.addEventListener('DOMContentLoaded', function() {
    // Hide loading spinner initially
    document.getElementById('loadingSpinner').style.display = 'none';
    
    // Demo data button
    document.getElementById('demoBtn').addEventListener('click', function(e) {
        e.preventDefault();
        fillDemoData();
    });
    
    // Form submission
    document.getElementById('transactionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        checkFraud();
    });
});

function fillDemoData() {
    const now = new Date();
    const transDate = new Date(now.getTime() - Math.random() * 7 * 24 * 60 * 60 * 1000);
    
    // Format date for datetime-local input
    const formattedDate = transDate.toISOString().slice(0, 16);
    
    // Format date of birth (YYYY-MM-DD)
    const dobDate = new Date(now.getFullYear() - (25 + Math.floor(Math.random() * 30)), 
                           Math.floor(Math.random() * 12), 
                           Math.floor(Math.random() * 28 + 1));
    const formattedDob = dobDate.toISOString().split('T')[0];
    
    const demoData = {
        cc_num: Math.floor(1000000000000000 + Math.random() * 9000000000000000).toString(),
        merchant: ['Walmart', 'Amazon', 'Target', 'Best Buy', 'Starbucks'][Math.floor(Math.random() * 5)],
        amt: (Math.random() * 500 + 10).toFixed(2),
        category: ['grocery_pos', 'shopping_net', 'entertainment', 'food_dining', 'gas_transport'][Math.floor(Math.random() * 5)],
        trans_date_trans_time: formattedDate,
        first: ['John', 'Jane', 'Robert', 'Emily', 'Michael'][Math.floor(Math.random() * 5)],
        last: ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'][Math.floor(Math.random() * 5)],
        gender: ['M', 'F'][Math.floor(Math.random() * 2)],
        dob: formattedDob,
        job: ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager'][Math.floor(Math.random() * 5)],
        street: `${Math.floor(Math.random() * 1000) + 100} Main St`,
        city: ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'][Math.floor(Math.random() * 5)],
        state: ['NY', 'CA', 'IL', 'TX', 'AZ'][Math.floor(Math.random() * 5)],
        zip: Math.floor(10000 + Math.random() * 90000).toString(),
        lat: (39.5 + Math.random() * 10).toFixed(6),
        long: (-100 + Math.random() * 40).toFixed(6),
        merch_lat: (39.5 + Math.random() * 10).toFixed(6),
        merch_long: (-100 + Math.random() * 40).toFixed(6)
    };
    
    // Fill the form
    for (const [key, value] of Object.entries(demoData)) {
        const element = document.getElementById(key);
        if (element) {
            element.value = value;
        }
    }
}

function checkFraud() {
    const form = document.getElementById('transactionForm');
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    data.amt = parseFloat(data.amt);
    data.lat = parseFloat(data.lat);
    data.long = parseFloat(data.long);
    data.merch_lat = parseFloat(data.merch_lat);
    data.merch_long = parseFloat(data.merch_long);
    
    // Show loading spinner
    const spinner = document.getElementById('loadingSpinner');
    spinner.style.display = 'block';
    
    // Disable submit button
    document.getElementById('submitBtn').disabled = true;
    
    // Send data to server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw new Error(err.error || 'Network response was not ok') });
        }
        return response.json();
    })
    .then(results => {
        console.log('Results:', results);
        displayResults(results);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
    })
    .finally(() => {
        spinner.style.display = 'none';
        document.getElementById('submitBtn').disabled = false;
    });
}

function displayResults(results) {
    // Update each model's results
    const models = {
        'knn': 'KNN',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'CNN': 'CNN'
    };
    
    let fraudCount = 0;
    let totalModels = 0;
    
    for (const [modelKey, modelName] of Object.entries(models)) {
        const result = results[modelKey];
        if (!result || result.error) {
            console.error(`Error for ${modelKey}:`, result?.error || 'No result');
            continue;
        }
        
        totalModels++;
        
        const predictionElement = document.getElementById(`${modelKey.toLowerCase().replace(' ', '_')}-prediction`);
        const confidenceElement = document.getElementById(`${modelKey.toLowerCase().replace(' ', '_')}-confidence`);
        const progressElement = document.getElementById(`${modelKey.toLowerCase().replace(' ', '_')}-progress`);
        
        if (predictionElement && confidenceElement && progressElement) {
            predictionElement.textContent = result.prediction;
            confidenceElement.textContent = `${result.probability.toFixed(2)}%`;
            progressElement.style.width = `${result.probability}%`;
            
            // Set color based on prediction and confidence
            if (result.prediction === 'Fraud') {
                predictionElement.className = 'prediction text-danger';
                progressElement.className = result.confidence === 'high' ? 
                    'progress-bar bg-danger' : 'progress-bar bg-warning';
                fraudCount++;
            } else {
                predictionElement.className = 'prediction text-success';
                progressElement.className = result.confidence === 'high' ? 
                    'progress-bar bg-success' : 'progress-bar bg-info';
            }
        }
    }
    
    // Update final decision
    const finalDecisionElement = document.getElementById('finalDecision');
    if (finalDecisionElement) {
        if (totalModels === 0) {
            finalDecisionElement.innerHTML = '<p class="mb-0">Error: No models returned results</p>';
            finalDecisionElement.className = 'alert final-decision alert-danger';
        } else {
            const fraudPercentage = (fraudCount / totalModels) * 100;
            if (fraudPercentage >= 50) {
                finalDecisionElement.innerHTML = `
                    <h4 class="alert-heading"><i class="bi bi-exclamation-triangle"></i> Fraud Detected!</h4>
                    <p>${fraudCount} out of ${totalModels} models flagged this transaction as fraudulent (${fraudPercentage.toFixed(0)}%).</p>
                    <hr>
                    <p class="mb-0">This transaction requires manual review.</p>
                `;
                finalDecisionElement.className = 'alert final-decision alert-danger';
            } else {
                finalDecisionElement.innerHTML = `
                    <h4 class="alert-heading"><i class="bi bi-check-circle"></i> Transaction Approved</h4>
                    <p>Only ${fraudCount} out of ${totalModels} models flagged this transaction (${fraudPercentage.toFixed(0)}%).</p>
                    <hr>
                    <p class="mb-0">This transaction appears to be legitimate.</p>
                `;
                finalDecisionElement.className = 'alert final-decision alert-success';
            }
        }
    }
}