#include <Servo.h>

// Declare the servo objects
Servo emaxservo_40, emaxservo_50, emaxservo_51, emaxservo_52;
Servo emaxservo_41, emaxservo_42, emaxservo_54, emaxservo_20;
Servo emaxservo_30, emaxservo_22, emaxservo_21, emaxservo_31;
Servo emaxservo_32, emaxservo_12, emaxservo_10, emaxservo_11;

// Array of servo objects
Servo servos[] = {emaxservo_40, emaxservo_50, emaxservo_51, emaxservo_52,
                  emaxservo_41, emaxservo_42, emaxservo_54, emaxservo_20,
                  emaxservo_30, emaxservo_22, emaxservo_21, emaxservo_31,
                  emaxservo_32, emaxservo_12, emaxservo_10, emaxservo_11};

// Corresponding pin numbers
int servoPins[] = {26, 27, 28, 29, 
                   30, 31, 24, 48, 
                   49, 51, 50, 52, 
                   53, 45, 46, 47};

// Default angles (e.g., 90° for neutral position)
int angles[] = {180, 180, 90, 90, 
                180, 90, 90, 0, 
                0, 90, 90, 90, 
                90, 90, 0, 90};

void setup() {
  // Attach servos to their respective pins
  for (int i = 0; i < 16; i++) {
    servos[i].attach(servoPins[i]);
  }

  // Move all servos to their default angles
  for (int i = 0; i < 16; i++) {
    servos[i].write(angles[i]);
  }
}

// Function to set angles for all servos
void setServoAngles(int newAngles[]) {
  for (int i = 0; i < 16; i++) {
    servos[i].write(newAngles[i]);
  }
}

// Example function for a "peace" gesture
void peace() {
  int peaceAngles[] = {0, 0, 90, 90, 
                  90, 90, 90, 0, 
                  0, 90, 90, 90, 
                  90, 90, 180, 90};

  setServoAngles(peaceAngles);
}

void loop() {
  delay(2000);
  peace();  // Move servos to peace sign position
}