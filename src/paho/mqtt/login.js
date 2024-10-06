// Form submission for login
document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from refreshing the page

    // Get the username and password from the form input fields
    var username = document.getElementById('username').value;
    var password = document.getElementById('password').value;

    // Prepare form data to send
    var formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    // Send the POST request to Flask to log in the user
    fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())  // Parse the JSON response from Flask
    .then(data => {
        var actionResult = document.getElementById('flash-message');

        // If login is successful
        if (data.success) {
            actionResult.innerHTML = "Login successful! Redirecting...";
            actionResult.style.color = "green";

            // Redirect to the dashboard (or any other page)
            setTimeout(() => {
                window.location.href = "http://127.0.0.1:5500/mqtt/website.html";
            }, 1000);  // Redirect after 1 second
        } else {
            // If login fails, display the error message
            actionResult.innerHTML = "Login failed: " + data.message;
            actionResult.style.color = "red";
        }
    })
    .catch(error => {
        console.log("Error submitting form:", error);
        var actionResult = document.getElementById('flash-message');
        actionResult.innerHTML = "Error logging in. Please try again.";
        actionResult.style.color = "red";
    });
});
