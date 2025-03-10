async function fetchWeather() {
    let searchInput = document.getElementById("search").value;
    const weatherDataSection = document.getElementById("weather-data");
    weatherDataSection.style.display = "block";
    const apiKey = "redacted"; 

    if (searchInput === "") {
        weatherDataSection.innerHTML = `
        <div>
          <h2>Empty Input!</h2>
          <p>Please try again with a valid <u>city name</u>.</p>
        </div>
        `;
        return;
    }

    async function getLonAndLat() {
        const countryCode = "US"; // Use proper country code, or remove for global search
        const geocodeURL = `https://api.openweathermap.org/geo/1.0/direct?q=${encodeURIComponent(searchInput)},${countryCode}&limit=1&appid=${apiKey}`;
        
        try {
            const response = await fetch(geocodeURL);
            if (!response.ok) {
                console.log("Bad response! ", response.status);
                return null;
            }

            const data = await response.json();
            if (data.length === 0) {
                console.log("Something went wrong here.");
                weatherDataSection.innerHTML = `
                <div>
                  <h2>Invalid Input: "${searchInput}"</h2>
                  <p>Please try again with a valid <u>city name</u>.</p>
                </div>
                `;
                return null;
            } else {
                return data[0];
            }
        } catch (error) {
            console.error("Error fetching geolocation:", error);
            return null;
        }
    }
  
    async function getWeatherData(lon, lat) {
        const weatherURL = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${apiKey}&units=metric`;

        try {
            const response = await fetch(weatherURL);
            if (!response.ok) {
                console.log("Bad response! ", response.status);
                return;
            }

            const data = await response.json();

            weatherDataSection.style.display = "flex";
            weatherDataSection.innerHTML = `
            <img src="https://openweathermap.org/img/wn/${data.weather[0].icon}.png" alt="${data.weather[0].description}" width="100" />
            <div>
            <h2>${data.name}</h2>
            <p><strong>Temperature:</strong> ${Math.round(data.main.temp)}°C</p>
            <p><strong>Description:</strong> ${data.weather[0].description}</p>
            </div>
            `;
        } catch (error) {
            console.error("Error fetching weather data:", error);
        }
    }

    // Reset input field
    document.getElementById("search").value = "";

    // Fetch geolocation data
    const geocodeData = await getLonAndLat();
    if (geocodeData) {
        await getWeatherData(geocodeData.lon, geocodeData.lat);
    }
}
