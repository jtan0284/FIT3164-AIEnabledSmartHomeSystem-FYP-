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

// Fetch temperature data from the Flask API and update the temperature chart
function fetchTemperatureData() {
    fetch('http://127.0.0.1:5000/temperature')
    .then(response => response.json())
    .then(data => {
        var timestamp = new Date();  // Current timestamp for x-axis
        data.forEach(newTemp => {
            // Update chart data
            temperatureChart.data.labels.push(timestamp);
            temperatureChart.data.datasets[0].data.push(newTemp);
            // Limit the chart to the latest 200 data points
            if (temperatureChart.data.labels.length > 200) {
                temperatureChart.data.labels.shift();
                temperatureChart.data.datasets[0].data.shift();
            }
        });
        temperatureChart.update();  // Re-draw the chart
    })
    .catch(error => console.log("Error fetching temperature data:", error));
}

// Fetch humidity data from the Flask API and update the humidity chart
function fetchHumidityData() {
    fetch('http://127.0.0.1:5000/humidity')
    .then(response => response.json())
    .then(data => {
        var timestamp = new Date();  // Current timestamp for x-axis
        data.forEach(newHum => {
            // Update chart dataa
            humidityChart.data.labels.push(timestamp);
            humidityChart.data.datasets[0].data.push(newHum);
            // Limit the chart to the latest 200 data points
            if (humidityChart.data.labels.length > 200) {
                humidityChart.data.labels.shift();
                humidityChart.data.datasets[0].data.shift();
            }
        });
        humidityChart.update();  // Re-draw the chart
    })
    .catch(error => console.log("Error fetching humidity data:", error));
}

// Function to fetch subscription status
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

// Fetch subscription status every 3 seconds
setInterval(fetchSubscriptionStatus, 3000);


// Fetch temperature and humidity data every second
setInterval(fetchTemperatureData, 1000);
setInterval(fetchHumidityData, 1000);
