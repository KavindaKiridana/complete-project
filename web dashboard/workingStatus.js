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

    // Get data from Firebase
    get(child(dbRef, "weatherParameters"))
        .then((snapshot) => {
            if (snapshot.exists()) {
                const data = snapshot.val();

                // Check if the "time" key exists in the data
                if (!data.time) {
                    // If "time" key is missing, show error
                    document.getElementById("status").innerHTML = '<i class="fas fa-times-circle me-2"></i> Something went wrong.';
                    document.getElementById("status").className = "status-badge status-danger";
                    console.error("Time key is missing in Firebase data.");
                    return; // Exit the function
                }

                // Update lastUpdatedTime with the value from Firebase
                const lastUpdatedTime = data.time;

                // Log the updated time
                console.log("Last updated time from Firebase:", lastUpdatedTime);

                // Get the current datetime
                const now = new Date();
                // Format date in local time (Sri Lanka)
                const year = now.getFullYear();
                const month = String(now.getMonth() + 1).padStart(2, '0');
                const day = String(now.getDate()).padStart(2, '0');
                const hours = String(now.getHours()).padStart(2, '0');
                const minutes = String(now.getMinutes()).padStart(2, '0');
                const seconds = String(now.getSeconds()).padStart(2, '0');

                const currentTime = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;

                // Convert the strings into Date objects
                const date1 = new Date(lastUpdatedTime);
                const date2 = new Date(currentTime);

                // Subtract the two Date objects (result is in milliseconds)
                const differenceInMilliseconds = date2 - date1;

                // Convert the difference into minutes
                const differenceInMinutes = differenceInMilliseconds / (1000 * 60);

                // Log the differences
                console.log("Difference in minutes:", differenceInMinutes);

                // Update the status based on the difference
                if (differenceInMinutes <= 5) {
                    // If difference is ≤ 5 minutes, show success
                    document.getElementById("status").innerHTML = '<i class="fas fa-check-circle me-2"></i> Working Well';
                    document.getElementById("status").className = "status-badge status-success";
                } else {
                    // If difference is > 5 minutes, show warning
                    document.getElementById("status").innerHTML = '<i class="fas fa-times-circle me-2"></i> The application is not available at this time.';
                    document.getElementById("status").className = "status-badge status-warning";
                }

            } else {
                // If no data is found in Firebase
                console.error("No data found!");
                document.getElementById("status").innerHTML = '<i class="fas fa-times-circle me-2"></i> No data found!';
                document.getElementById("status").className = "status-badge status-danger";
            }
        })
        .catch((error) => {
            // If there's an error fetching data
            console.error("Error reading data:", error);
            document.getElementById("status").innerHTML = '<i class="fas fa-times-circle me-2"></i> Error fetching data!';
            document.getElementById("status").className = "status-badge status-danger";
        });
}

// Call the function immediately when the script loads
fetchData();

// Add auto-refresh functionality (every 60 seconds)
setInterval(fetchData, 60000);