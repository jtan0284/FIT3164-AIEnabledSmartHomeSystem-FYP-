// Set up the temperature chart
var temperatureCtx = document.getElementById('temperatureChart').getContext('2d');
var temperatureData = {
    labels: [],  // Timestamps for x-axis
    datasets: [{
        label: 'Temperature (°C)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        data: []  // Temperature data for y-axis
    }]
};

var temperatureChart = new Chart(temperatureCtx, {
    type: 'line',
    data: temperatureData,
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute'
                }
            },
            y: {
                beginAtZero: false
            }
        }
    }
});

// Set up the humidity chart
var humidityCtx = document.getElementById('humidityChart').getContext('2d');
var humidityData = {
    labels: [],  // Timestamps for x-axis
    datasets: [{
        label: 'Humidity (%)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        data: []  // Humidity data for y-axis
    }]
};

var humidityChart = new Chart(humidityCtx, {
    type: 'line',
    data: humidityData,
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute'
                }
            },
            y: {
                beginAtZero: false
            }
        }
    }
});

var preferredTemperature = null; // Global variable to store the preferred temperature

document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from refreshing the page
    preferredTemperature = parseFloat(document.getElementById('temperature').value);  // Store the preferred temperature
});

function fetchLiveData() {
    fetch('http://127.0.0.1:5000/live_data')
    .then(response => response.json())
    .then(data => {
        var timestamp = new Date();  // Get the current timestamp

        // Update the temperature chart
        temperatureChart.data.labels.push(timestamp);
        temperatureChart.data.datasets[0].data.push(data.temperature);

        // Update the humidity chart
        humidityChart.data.labels.push(timestamp);
        humidityChart.data.datasets[0].data.push(data.humidity);

        // Limit the chart data to the latest 100 data points
        if (temperatureChart.data.labels.length > 100) {
            temperatureChart.data.labels.shift();
            temperatureChart.data.datasets[0].data.shift();
        }
        if (humidityChart.data.labels.length > 100) {
            humidityChart.data.labels.shift();
            humidityChart.data.datasets[0].data.shift();
        }

        // Update the charts
        temperatureChart.update();
        humidityChart.update();

        // Compare live temperature with preferred temperature and update action
        if (preferredTemperature !== null) {
            var actionResult = document.getElementById('action-result');
            if (data.temperature > preferredTemperature) {
                actionResult.innerHTML = "Action: decrease";
                actionResult.style.color = "red";
            } else if (data.temperature < preferredTemperature) {
                actionResult.innerHTML = "Action: increase";
                actionResult.style.color = "green";
            } else {
                actionResult.innerHTML = "Action: do nothing";
                actionResult.style.color = "gray";
            }
        }
    })
    .catch(error => console.log("Error fetching live data:", error));
}

// Fetch subscription status from Flask and update the subscription status element
function fetchSubscriptionStatus() {
    fetch('http://127.0.0.1:5000/subscription_status')
    .then(response => response.json())
    .then(data => {
        var statusElement = document.getElementById("subscription-status");
        if (data.subscribed) {
            statusElement.innerHTML = "Subscription Status: Subscribed";
            statusElement.style.color = "green";
        } else {
            statusElement.innerHTML = "Subscription Status: Not Subscribed";
            statusElement.style.color = "red";
        }
    })
    .catch(error => console.log("Error fetching subscription status:", error));
}

document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from refreshing the page

    // Get the temperature value from the input field
    var temperature = document.getElementById('temperature').value;

    // Prepare form data to send
    var formData = new FormData();
    formData.append('temperature', temperature);

    // Send the POST request to Flask to set the temperature
    fetch('http://127.0.0.1:5000/set_temperature', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Update the action result in the action container
        var actionResult = document.getElementById('action-result');
        actionResult.innerHTML = "Action: " + data.action;

        // Optionally, change the style based on the action
        if (data.action === "increase") {
            actionResult.style.color = "green";
        } else if (data.action === "decrease") {
            actionResult.style.color = "red";
        } else {
            actionResult.style.color = "gray";
        }
    })
    .catch(error => console.log("Error submitting form:", error));
});


// Set intervals to regularly fetch live data and subscription status
setInterval(fetchLiveData, 1000);  // Fetch live data every second
setInterval(fetchSubscriptionStatus, 5000);  // Fetch subscription status every 5 seconds
