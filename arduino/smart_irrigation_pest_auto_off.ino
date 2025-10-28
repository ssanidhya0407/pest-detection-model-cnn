/*
  smart_irrigation_pest_auto_off.ino
  - Moisture sensor + relay irrigation
  - LCD display (I2C)
  - Piezo buzzer (auto-off after duration)
  Serial commands:
    PEST_ON            -> turn buzzer ON for DEFAULT_PEST_SECONDS
    PEST_ON 15         -> turn buzzer ON for 15 seconds
    PEST_ON:15         -> same as above
    PEST_OFF           -> turn buzzer OFF immediately

  Upload this sketch to your Arduino. Close the Arduino Serial Monitor if the backend needs to open the same COM port.
*/

#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // change to 0x3F if your module uses that

// Pins
const int sensor_pin = A0;
const int relay_pin = 7;    // active LOW relay
const int buzzer_pin = 8;   // piezo buzzer

// States
bool motorState = false;
bool pestDetected = false;

// Auto-off
const unsigned long DEFAULT_PEST_SECONDS = 10; // default seconds to keep buzzer on
unsigned long pestOffAtMs = 0; // millis when buzzer should turn off (0 = none)

// Calibration
const int DRY_VALUE = 900;
const int WET_VALUE = 177;

String serialBuffer = "";

void setup() {
  Serial.begin(9600);
  lcd.init();
  lcd.backlight();

  pinMode(sensor_pin, INPUT);
  pinMode(relay_pin, OUTPUT);
  pinMode(buzzer_pin, OUTPUT);

  digitalWrite(relay_pin, HIGH); // motor OFF
  noTone(buzzer_pin);

  Serial.println("ARDUINO: startup");
  lcd.setCursor(0,0);
  lcd.print("Smart Irrig Init");
  lcd.setCursor(0,1);
  lcd.print("Ready");
  delay(1200);
  lcd.clear();
}

void setPestOnSeconds(unsigned long seconds) {
  pestDetected = true;
  if (seconds == 0) seconds = DEFAULT_PEST_SECONDS;
  pestOffAtMs = millis() + seconds * 1000UL;
  Serial.print("ARDUINO: PEST ON for "); Serial.print(seconds); Serial.println(" s");
  // Immediately show alert on LCD
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("PEST ALERT !!!");
  lcd.setCursor(0,1);
  lcd.print("Buzzer ON");
}

void setPestOff() {
  pestDetected = false;
  pestOffAtMs = 0;
  noTone(buzzer_pin);
  Serial.println("ARDUINO: PEST OFF");
  // Refresh display in next loop
}

void handleSerialLine(const String &line) {
  String cmd = line;
  cmd.trim();
  Serial.print("ARDUINO: CMD_RECEIVED: "); Serial.println(cmd);

  // Support 'PEST_ON', 'PEST_ON 15' or 'PEST_ON:15'
  if (cmd.startsWith("PEST_ON")) {
    // parse parameter
    unsigned long seconds = 0;
    int idxSpace = cmd.indexOf(' ');
    int idxColon = cmd.indexOf(':');
    if (idxSpace > 0) {
      String val = cmd.substring(idxSpace + 1);
      val.trim();
      seconds = (unsigned long) val.toInt();
    } else if (idxColon > 0) {
      String val = cmd.substring(idxColon + 1);
      val.trim();
      seconds = (unsigned long) val.toInt();
    }
    if (seconds == 0) seconds = DEFAULT_PEST_SECONDS;
    setPestOnSeconds(seconds);
  } else if (cmd.equalsIgnoreCase("PEST_OFF")) {
    setPestOff();
  } else {
    Serial.print("ARDUINO: Unknown command: "); Serial.println(cmd);
  }
}

void loop() {
  // read serial non-blocking
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (serialBuffer.length() > 0) {
        handleSerialLine(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 200) serialBuffer = serialBuffer.substring(serialBuffer.length() - 200);
    }
  }

  // sensor read and motor logic
  int sensor_data = analogRead(sensor_pin);
  int moisturePercent = map(sensor_data, DRY_VALUE, WET_VALUE, 0, 100);
  moisturePercent = constrain(moisturePercent, 0, 100);

  if (moisturePercent < 30) motorState = true;
  else if (moisturePercent > 45) motorState = false;

  digitalWrite(relay_pin, motorState ? LOW : HIGH);

  // Buzzer control (on when pestDetected)
  if (pestDetected) {
    tone(buzzer_pin, 2000); // continuous tone
  } else {
    noTone(buzzer_pin);
  }

  // Auto-off logic
  if (pestDetected && pestOffAtMs > 0 && millis() >= pestOffAtMs) {
    Serial.println("ARDUINO: Auto-off timer expired -> turning pest OFF");
    setPestOff();
  }

  // Periodic LCD update when not in pest alert
  static unsigned long lastDisplay = 0;
  if (!pestDetected && millis() - lastDisplay > 500) {
    lastDisplay = millis();
    String condition;
    if (moisturePercent < 30) condition = "Dry";
    else if (moisturePercent < 60) condition = "Medium";
    else condition = "Wet";

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Soil:"); lcd.print(condition);
    lcd.setCursor(0,1);
    lcd.print("M:"); lcd.print(moisturePercent); lcd.print("% ");
    if (motorState) lcd.print("|ON"); else lcd.print("|OFF");
  }

  // Periodic serial debug
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug > 2000) {
    lastDebug = millis();
    Serial.print("ARDUINO: RAW:"); Serial.print(sensor_data);
    Serial.print(" Moist:"); Serial.print(moisturePercent);
    Serial.print("% Motor:"); Serial.print(motorState ? "ON" : "OFF");
    Serial.print(" Pest:"); Serial.println(pestDetected ? "YES" : "NO");
  }

  delay(50);
}
