// Fetch the minute-based preferences from Flask and display them
function fetchMinutePreferences() {
    fetch('http://127.0.0.1:5000/get_minute_preferences')
    .then(response => response.json())
    .then(data => {
        var minutePreferencesDiv = document.getElementById('minute-preferences');
        minutePreferencesDiv.innerHTML = "";  // Clear any previous content

        var tempArray = data.temperature_per_minute;
        var humidArray = data.humidity_per_minute;

        // Loop through the 60 minutes and display the data
        for (var i = 0; i < 60; i++) {
            var minuteDiv = document.createElement('div');
            minuteDiv.classList.add('minute-preference');
            minuteDiv.innerHTML = `Minute ${i}: Temperature = ${tempArray[i] ? tempArray[i] + '°C' : 'N/A'}, Humidity = ${humidArray[i] ? humidArray[i] + '%' : 'N/A'}`;
            minutePreferencesDiv.appendChild(minuteDiv);
        }
    })
    .catch(error => console.log("Error fetching minute preferences:", error));
}

// Call this function to fetch and display minute preferences when the page loads
window.onload = fetchMinutePreferences;
