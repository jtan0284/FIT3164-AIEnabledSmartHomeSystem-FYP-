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
var preferredHumidity = null; // Global variable to store the preferred humidity

// Form submission to capture both preferred temperature and humidity
document.getElementById('preferences-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from refreshing the page

    // Store the preferred temperature and humidity from the form input fields
    preferredTemperature = parseFloat(document.getElementById('temperature').value);
    preferredHumidity = parseFloat(document.getElementById('humidity').value);

    // Prepare form data to send
    var formData = new FormData();
    formData.append('temperature', preferredTemperature);
    formData.append('humidity', preferredHumidity);

    // Send the POST request to Flask to set the temperature and humidity
    fetch('http://127.0.0.1:5000/set_preferences', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        var actionResult = document.getElementById('action-result');
        actionResult.innerHTML = 
            "Action: Temperature: " + data.temperature_action + ", Humidity: " + data.humidity_action;

        // Update the predicted temperature and humidity
        document.getElementById('predicted-temperature').innerHTML = 
            "Predicted Temperature: " + (data.predicted_temperature !== null ? data.predicted_temperature + "°C" : "N/A");
        document.getElementById('predicted-humidity').innerHTML = 
            "Predicted Humidity: " + (data.predicted_humidity !== null ? data.predicted_humidity + "%" : "N/A");

        // Change the style based on the temperature action
        if (data.temperature_action === "increase temperature") {
            actionResult.style.color = "green";
        } else if (data.temperature_action === "decrease temperature") {
            actionResult.style.color = "red";
        } else {
            actionResult.style.color = "gray";
        }

        // Change the border color based on the humidity action
        if (data.humidity_action === "increase humidity") {
            actionResult.style.border = "2px solid green";
        } else if (data.humidity_action === "decrease humidity") {
            actionResult.style.border = "2px solid red";
        } else {
            actionResult.style.border = "2px solid gray";
        }

    })
    .catch(error => console.log("Error submitting form:", error));
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

        // Compare live temperature and humidity with preferred values and update action
        if (preferredTemperature !== null && preferredHumidity !== null) {
            var actionResult = document.getElementById('action-result');
            
            // Compare temperature
            var tempAction = '';
            if (data.temperature > preferredTemperature) {
                tempAction = "Temperature: decrease";
            } else if (data.temperature < preferredTemperature) {
                tempAction = "Temperature: increase";
            } else {
                tempAction = "Temperature: do nothing";
            }

            // Compare humidity
            var humidityAction = '';
            if (data.humidity > preferredHumidity) {
                humidityAction = "Humidity: decrease";
            } else if (data.humidity < preferredHumidity) {
                humidityAction = "Humidity: increase";
            } else {
                humidityAction = "Humidity: do nothing";
            }

            // Update the action result for both temperature and humidity
            actionResult.innerHTML = `Action: ${tempAction}, ${humidityAction}`;
            
            // Optionally, change the style based on temperature and humidity actions
            if (tempAction.includes("increase")) {
                actionResult.style.color = "green";
            } else if (tempAction.includes("decrease")) {
                actionResult.style.color = "red";
            } else {
                actionResult.style.color = "gray";
            }

            if (humidityAction.includes("increase")) {
                actionResult.style.border = "2px solid green";
            } else if (humidityAction.includes("decrease")) {
                actionResult.style.border = "2px solid red";
            } else {
                actionResult.style.border = "2px solid gray";
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

// Set intervals to regularly fetch live data and subscription status
setInterval(fetchLiveData, 1000);  // Fetch live data every second
setInterval(fetchSubscriptionStatus, 5000);  // Fetch subscription status every 5 seconds
