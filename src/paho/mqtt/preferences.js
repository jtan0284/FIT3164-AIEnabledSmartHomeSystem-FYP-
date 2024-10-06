function fetchMinutePreferences() {
    fetch('http://127.0.0.1:5000/get_minute_preferences')
    .then(response => {
        console.log("Fetching minute preferences response:", response);
        return response.json();
    })
    .then(data => {
        console.log("Minute preferences data:", data);  // Check if the data is correct
        var minutePreferencesDiv = document.getElementById('minute-preferences');
        minutePreferencesDiv.innerHTML = "";  // Clear any previous content

        var tempArray = data.temperature_per_minute;  // Get the full array for temperature
        var humidArray = data.humidity_per_minute;    // Get the full array for humidity

        // Loop through the array and display the data for each minute
        for (var i = 0; i < tempArray.length; i++) {
            var minuteDiv = document.createElement('div');
            minuteDiv.classList.add('minute-preference');

            // Display temperature and humidity per minute (if present, otherwise 'N/A')
            var temp = tempArray[i] !== null ? tempArray[i] + '°C' : 'N/A';
            var humid = humidArray[i] !== null ? humidArray[i] + '%' : 'N/A';
            minuteDiv.innerHTML = `Minute ${i}: Temperature = ${temp}, Humidity = ${humid}`;

            minutePreferencesDiv.appendChild(minuteDiv);
        }
    })
    .catch(error => console.log("Error fetching minute preferences:", error));
}

// Call this function to fetch and display minute preferences every 10 seconds
setInterval(fetchMinutePreferences, 10000);
