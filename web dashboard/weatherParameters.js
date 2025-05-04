// Import Firebase functions
import { initializeApp } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";
import { getDatabase, ref, get, child } from "https://www.gstatic.com/firebasejs/9.22.0/firebase-database.js";

let firebaseConfig;
try {
    const configModule = await import('./config.js');
    firebaseConfig = configModule.firebaseConfig;
    if (!firebaseConfig || !firebaseConfig.apiKey) {
        throw new Error("Firebase configuration is invalid or missing apiKey.");
    }
} catch (error) {
    console.error("Failed to import Firebase configuration:", error);
}

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// Function to fetch and display data
function fetchData() {
    // Create a single database reference
    const dbRef = ref(db);

    // Get DOM elements for displaying values
    const tempElement = document.getElementById("tempValue");
    const humidityElement = document.getElementById("humidityValue");
    const soilElement = document.getElementById("soilValue");
    const lightElement = document.getElementById("lightValue");

    // Set loading messages
    tempElement.innerText = "Loading...";
    humidityElement.innerText = "Loading...";
    soilElement.innerText = "Loading...";
    lightElement.innerText = "Loading...";

    // Get data from Firebase
    get(child(dbRef, "weatherParameters"))
        .then((snapshot) => {
            if (snapshot.exists()) {
                const data = snapshot.val();

                // Update DOM elements with the retrieved data
                // FIXED: Simplified the output format and fixed variable references
                tempElement.innerText = `${data.temp}°C`;
                humidityElement.innerText = `${data.humidity}%`;
                soilElement.innerText = `${data.moisture}%`;
                lightElement.innerText = `${data.light} lux`;
            } else {
                // FIXED: Set all elements to show no data message
                tempElement.innerText = "No data found!";
                humidityElement.innerText = "No data found!";
                soilElement.innerText = "No data found!";
                lightElement.innerText = "No data found!";
            }
        })
        .catch((error) => {
            console.error("Error reading data:", error);
            // FIXED: Set all elements to show error message
            tempElement.innerText = `Error: ${error.message}`;
            humidityElement.innerText = `Error: ${error.message}`;
            soilElement.innerText = `Error: ${error.message}`;
            lightElement.innerText = `Error: ${error.message}`;
        });
}

// Call the function immediately when the script loads
fetchData();

// Add auto-refresh functionality (every 60 seconds)
setInterval(fetchData, 60000);