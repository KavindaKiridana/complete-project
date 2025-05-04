//waterTap.js file
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

// Function to fetch and update the water tap status
function updateWaterTapStatus() {
    const dbRef = ref(db, "weatherParameters/tap");
    get(dbRef)
        .then((snapshot) => {
            if (snapshot.exists()) {
                document.getElementById("waterSwitch").checked = snapshot.val() === "true";
                console.log("Water tap state (from Firebase):", snapshot.val());
            } else {
                console.error("No tap data found in Firebase.");
            }
        })
        .catch((error) => console.error("Error reading data:", error));
}

// Fetch water tap status on page load
window.addEventListener("load", updateWaterTapStatus);
