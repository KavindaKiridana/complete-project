#include <Arduino.h>
#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <DHT.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <addons/TokenHelper.h>
#include <addons/RTDBHelper.h>
#include <TimeLib.h>//later added

// --------------------- Firebase Credentials ---------------------
#define API_KEY "AIzaSyBc-v9DXJBOTRBK_ef7kKbx-K1Iuzaqm0w"
#define DATABASE_URL "https://plnat-growth-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define USER_EMAIL "kavindakiridena@gmail.com"
#define USER_PASSWORD "123456"

// --------------------- WiFi Credentials -------------------------
#define WIFI_SSID "Aaa"
#define WIFI_PASSWORD "123@xyzabc"

// --------------------- Pin Definitions --------------------------
#define DHTPIN 4                 // GPIO4
#define DHTTYPE DHT11
#define RELAY_PIN 5             // GPIO5
#define SOIL_MOISTURE_PIN 34    // GPIO34 (Analog)
#define LDR_PIN 35              // GPIO35 (Analog)

// --------------------- Global Objects ---------------------------
DHT dht(DHTPIN, DHTTYPE);
WiFiUDP ntpUDP;
// GMT+5:30 for Sri Lanka (19800 seconds offset)
NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800, 60000);

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// --------------------- Get Formatted Time -----------------------
String getFormattedTime()//later added
{
  timeClient.update();
  time_t rawTime = timeClient.getEpochTime();
  struct tm *timeinfo = localtime(&rawTime);

  char buffer[25];
  sprintf(buffer, "%04d-%02d-%02d %02d:%02d:%02d",
          timeinfo->tm_year + 1900,
          timeinfo->tm_mon + 1,
          timeinfo->tm_mday,
          timeinfo->tm_hour,
          timeinfo->tm_min,
          timeinfo->tm_sec);
  return String(buffer);
}

// --------------------- Setup Function ---------------------------
void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Water tap OFF initially

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\n✅ Successfully connected to WiFi!");

  // Initialize NTP client
  timeClient.begin();
  Serial.println("Waiting for NTP time sync...");
  while (!timeClient.update())
  {
    timeClient.forceUpdate();
    delay(100);
  }
  Serial.println("Time synchronized");

  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

// --------------------- Main Loop --------------------------------
void loop() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  int moisture = analogRead(SOIL_MOISTURE_PIN);
  int light = analogRead(LDR_PIN);
  String currentTime = getFormattedTime();

  Serial.print("Temperature: ");
  Serial.println(temperature);
  Serial.print("Humidity: ");
  Serial.println(humidity);
  Serial.print("Moisture: ");
  Serial.println(moisture);
  Serial.print("Light: ");
  Serial.println(light);
  Serial.print("Time: ");
  Serial.println(currentTime);

  bool tapStatus = false;
  if (temperature > 28.0) {
    digitalWrite(RELAY_PIN, HIGH);
    tapStatus = true;
    Serial.println("Water tap turned ON");
  } else {
    digitalWrite(RELAY_PIN, LOW);
    tapStatus = false;
    Serial.println("Water tap OFF");
  }

  // Send data to Firebase
  if (Firebase.ready()) {
    bool success = true;
    success &= Firebase.RTDB.setFloat(&fbdo, "weatherParameters/temp", temperature);
    success &= Firebase.RTDB.setFloat(&fbdo, "weatherParameters/humidity", humidity);
    success &= Firebase.RTDB.setInt(&fbdo, "weatherParameters/moisture", moisture);
    success &= Firebase.RTDB.setInt(&fbdo, "weatherParameters/light", light);
    success &= Firebase.RTDB.setBool(&fbdo, "weatherParameters/tap", tapStatus);
    success &= Firebase.RTDB.setString(&fbdo, "weatherParameters/time", currentTime);

    if (success) {
      Serial.println("✅ Data sent to Firebase!");
    } else {
      Serial.print("❌ Failed to send data: ");
      Serial.println(fbdo.errorReason());
    }
  }

  delay(10000); // Wait for 10 seconds before next reading
}
