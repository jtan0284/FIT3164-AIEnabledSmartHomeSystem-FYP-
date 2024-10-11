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
    var temperatureInput = document.getElementById('temperature').value.trim();  // Trim leading/trailing spaces
    var humidityInput = document.getElementById('humidity').value.trim();  // Trim leading/trailing spaces

    // Parse input values as floats for validation
    preferredTemperature = parseFloat(temperatureInput);
    preferredHumidity = parseFloat(humidityInput);

    // Reference to the flash message element
    var flashMessage = document.getElementById('flash-message');

    // Validate the input values to ensure they are numbers
    if (!isNaN(preferredTemperature) && !isNaN(preferredHumidity) && preferredTemperature >= 0 && preferredTemperature <= 50 && preferredHumidity >= 0 && preferredHumidity <= 100) {
        // Prepare form data to send if inputs are valid
        var formData = new FormData();
        formData.append('temperature', preferredTemperature);
        formData.append('humidity', preferredHumidity);

        // Send the POST request to Flask to set the temperature and humidity
        fetch('http://127.0.0.1:5000/set_preferences', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.message === 'Preferences updated successfully') {
                flashMessage.innerHTML = "Preferences encoded!";
                flashMessage.style.color = "green";
                flashMessage.style.display = "block";

                // Flash the message briefly (e.g., for 3 seconds)
                flashMessage.classList.add('flash');
                setTimeout(() => {
                    flashMessage.classList.remove('flash');
                    flashMessage.style.display = "none";
                }, 3000);
            } else {
                throw new Error(data.message || 'Unexpected error');
            }
        })
        .catch(error => {
            console.log("Error submitting form:", error);
            flashMessage.innerHTML = "Error processing your request. Please try again.";
            flashMessage.style.color = "red";
            flashMessage.style.display = "block";
            flashMessage.classList.add('flash');
            setTimeout(() => {
                flashMessage.classList.remove('flash');
                flashMessage.style.display = "none";
            }, 3000);
        });
    } else {
        // If the input is invalid, display an error message
        flashMessage.innerHTML = "Invalid input! Please enter valid numbers for temperature (0-50°C) and humidity (0-100%).";
        flashMessage.style.color = "red";
        flashMessage.style.display = "block";
        flashMessage.classList.add('flash');

        // Flash the error message briefly (e.g., for 3 seconds)
        setTimeout(() => {
            flashMessage.classList.remove('flash');
            flashMessage.style.display = "none";
        }, 3000);
    }
});

function fetchLiveData() {
    fetch('http://127.0.0.1:5000/live_data')
    .then(response => response.json())
    .then(data => {
        var timestamp = new Date();  // Get the current timestamp

        document.getElementById('live-temperature').innerHTML = 'Temperature: ' + (data.temperature ? data.temperature + '°C' : 'N/A');
        document.getElementById('live-humidity').innerHTML = 'Humidity: ' + (data.humidity ? data.humidity + '%' : 'N/A');

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

// Function to fetch the minute-based preferences from Flask
function fetchMinutePreferences() {
    fetch('http://127.0.0.1:5000/get_minute_preferences')
    .then(response => response.json())
    .then(data => {
        var temperaturePerMinute = data.filtered_temperature;
        var humidityPerMinute = data.filtered_humidity;

        // Get the HTML elements where you want to display the data
        var tempDiv = document.getElementById('predicted-temperature');
        var humidDiv = document.getElementById('predicted-humidity');

        // Clear old content before adding new data
        tempDiv.innerHTML = ''; 
        humidDiv.innerHTML = '';

         // Directly insert the temperature and humidity data into the divs
        tempDiv.innerHTML = `Predicted Temperature: ${temperaturePerMinute}°C`;
        humidDiv.innerHTML = `Predicted Humidity: ${humidityPerMinute}%`;
    })
    .catch(error => console.log("Error fetching minute preferences:", error));
}

// Call the function every 60 seconds to update the data
setInterval(fetchMinutePreferences, 60000);  // 60000 ms = 60 seconds

// Optionally, call the function immediately on page load as well
fetchMinutePreferences();



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

// Function to fetch live temperature and humidity actions
function fetchLiveActions() {
    fetch('http://127.0.0.1:5000/live_action')
    .then(response => response.json())
    .then(data => {
        // Get the live action divs from the HTML
        var liveTempActionDiv = document.getElementById('live-temperature-action');
        var liveHumidActionDiv = document.getElementById('live-humidity-action');

        // Update the content with litemperaturechartve temperature and humidity actions
        liveTempActionDiv.innerHTML = "Temperature Action: " + data.temperature_action;
        liveHumidActionDiv.innerHTML = "Humidity Action: " + data.humidity_action;
    })
    .catch(error => console.log("Error fetching live actions:", error));
}

// Call the fetchLiveActions function every 5 seconds to update live actions
setInterval(fetchLiveActions, 5000);


// Set intervals to regularly fetch live data and subscription status
setInterval(fetchLiveData, 1000);  // Fetch live data every second
setInterval(fetchSubscriptionStatus, 5000);  // Fetch subscription status every 5 seconds
