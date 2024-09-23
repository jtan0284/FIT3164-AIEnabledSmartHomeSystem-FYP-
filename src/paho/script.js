// Initialize the Chart.js context for the canvas element
var ctx = document.getElementById('temperatureChart').getContext('2d');

// Data for the chart (initially empty)
var temperatureData = {
    labels: [],  // Timestamps or data points for x-axis
    datasets: [{
        label: 'Temperature (°C)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
        data: []  // Temperature values for y-axis
    }]
};

// Chart configuration
var config = {
    type: 'line',
    data: temperatureData,
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute'  // You can customize the time unit here
                }
            },
            y: {
                beginAtZero: false
            }
        }
    }
};

// Create the chart using the configuration
var temperatureChart = new Chart(ctx, config);

// Function to update the chart with new temperature data
function updateChart(newTemp, timestamp) {
    // Push new data into the dataset
    temperatureChart.data.labels.push(timestamp);
    temperatureChart.data.datasets[0].data.push(newTemp);

    // Remove old data if more than 50 data points
    if (temperatureChart.data.labels.length > 50) {
        temperatureChart.data.labels.shift();  // Remove the oldest label
        temperatureChart.data.datasets[0].data.shift();  // Remove the oldest data point
    }

    // Update the chart to reflect changes
    temperatureChart.update();
}

// Function to fetch temperature data from the Flask API
function fetchTemperatureData() {
    // You need to specify the full Flask API endpoint here
    fetch('http://127.0.0.1:5000/temperature')
    .then(response => response.json())
    .then(data => {
        var timestamp = new Date();  // Use current time as the timestamp
        data.forEach(newTemp => {
            updateChart(newTemp, timestamp);  // Update the chart with new data
        });
    })
    .catch(error => console.log("Error fetching temperature data:", error));
}

// Fetch temperature data every 5 seconds
setInterval(fetchTemperatureData, 5000);
