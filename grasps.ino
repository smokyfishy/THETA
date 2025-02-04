#include "SBUS.h"
#include "Servo.h"
#include "SCServo.h"

// SBUS object to communicate with the receiver
SBUS x8r(Serial1);

// Servo objects for controlling the hand joints
Servo emaxservo_FON;  // 50 FO N Fore Outwards Normal
Servo emaxservo_FII;  // 51 FI I Fore Inward Inverted
Servo emaxservo_FTN;  // 48 FT N Fore Tendon Non
Servo emaxservo_MON;  // 52 MO N Middle Outwards Normal
Servo emaxservo_MIN;  // 53 MI N Middle Inwards Normal
Servo emaxservo_MTN;  // 49 MT N Middle Tendon Non
Servo emaxservo_ROI;  // 30 RO I Ring Outward Inverted
Servo emaxservo_RII;  // 31 RI I Ring Inward Inverted
Servo emaxservo_RTI;  // 26 RT I Ring Tendon Inverted
Servo emaxservo_LON;  // 28 LO N Little Outward at Knuckle Non-Inverted
Servo emaxservo_LII;  // 29 LI I Little Inward at Knuckle Inverted
Servo emaxservo_LTI;  // 27 LT I Little Tendon Inverted
Servo emaxservo_TON;  // 47 TO N Thumb Outwards Non
Servo emaxservo_TII;  // 45 TI I Thumb Inwards Inverted
Servo emaxservo_TTN;  // 46 TT N Thumb Tendon Non
Servo emaxservo_TRN;  // 24 TR N Thumb Rotation Non-Inverted

// Other variables (to match your existing code)
long default_counter = 1;
int default_stage = 0;
SCServo SERVO;      //Declare a case of SCServo to control the Feetechs
int wrist_pos[] = {512, 512, 512};  // default to center

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  Serial.print("Connected on Serial\n\r");
  Serial3.begin(115200);
  Serial3.print("Connected on Serial3\n\r");
  Serial2.begin(1000000);  // Serial communication for servos
  SERVO.pSerial = &Serial2;

  // Attach servos to their respective pins
  emaxservo_FON.attach(50);
  emaxservo_FII.attach(51);
  emaxservo_FTN.attach(48);
  emaxservo_MON.attach(52);
  emaxservo_MIN.attach(53);
  emaxservo_MTN.attach(49);
  emaxservo_ROI.attach(30);
  emaxservo_RII.attach(31);
  emaxservo_RTI.attach(26);
  emaxservo_LON.attach(28);
  emaxservo_LII.attach(29);
  emaxservo_LTI.attach(27);
  emaxservo_TON.attach(47);
  emaxservo_TII.attach(45);
  emaxservo_TTN.attach(46);
  emaxservo_TRN.attach(24);

  delay(50);  // Short delay for setup to complete
}

void loop() {
  // Call the default pose function at the start
  defaultPose();

  // Call different grasp functions based on input or condition
  if (Serial.available()) {
    int input = Serial.read();
    if (input == '1') {
      powerGrasp();
    } else if (input == '2') {
      precisionGrasp();
    } else if (input == '3') {
      hookGrasp();
    } else if (input == '4') {
      cylindricalGrasp();
    } else if (input == '5') {
      lateralGrasp();
    }
      else if (input == '6') {
      peace();
    }
    }
  }

void defaultPose() {
  // Set all servos to the neutral/default position
  emaxservo_FON.write(90);
  emaxservo_FII.write(90);
  emaxservo_FTN.write(90);
  emaxservo_MON.write(90);
  emaxservo_MIN.write(90);
  emaxservo_MTN.write(90);
  emaxservo_ROI.write(90);
  emaxservo_RII.write(90);
  emaxservo_RTI.write(90);
  emaxservo_LON.write(90);
  emaxservo_LII.write(90);
  emaxservo_LTI.write(90);
  emaxservo_TON.write(90);
  emaxservo_TII.write(90);
  emaxservo_TTN.write(90);
  emaxservo_TRN.write(90);
}

void peace() {
  emaxservo_FON.write(160);
  emaxservo_FII.write(20);
  emaxservo_FTN.write(20);
  emaxservo_MON.write(20);
  emaxservo_MIN.write(160);
  emaxservo_MTN.write(20);
  emaxservo_ROI.write(20);
  emaxservo_RII.write(160);
  emaxservo_RTI.write(160);
  emaxservo_LON.write(20);
  emaxservo_LII.write(160);
  emaxservo_LTI.write(160);
  emaxservo_TON.write(90);
  emaxservo_TII.write(90);
  emaxservo_TTN.write(90);
  emaxservo_TRN.write(90);
}

void powerGrasp() {
  // Set servo positions for power grasp (closed fist)
  emaxservo_FON.write(30);
  emaxservo_FII.write(30);
  emaxservo_FTN.write(30);
  emaxservo_MON.write(30);
  emaxservo_MIN.write(30);
  emaxservo_MTN.write(30);
  emaxservo_ROI.write(30);
  emaxservo_RII.write(30);
  emaxservo_RTI.write(30);
  emaxservo_LON.write(30);
  emaxservo_LII.write(30);
  emaxservo_LTI.write(30);
  emaxservo_TON.write(30);
  emaxservo_TII.write(30);
  emaxservo_TTN.write(30);
  emaxservo_TRN.write(30);
}

void precisionGrasp() {
  // Set servo positions for precision grasp (thumb and index finger pinch)
  emaxservo_FON.write(80);
  emaxservo_FII.write(80);
  emaxservo_FTN.write(80);
  emaxservo_MON.write(80);
  emaxservo_MIN.write(80);
  emaxservo_MTN.write(80);
  emaxservo_ROI.write(30);  // Thumb and index finger
  emaxservo_RII.write(80);
  emaxservo_RTI.write(30);
  emaxservo_LON.write(80);
  emaxservo_LII.write(80);
  emaxservo_LTI.write(80);
  emaxservo_TON.write(80);
  emaxservo_TII.write(80);
  emaxservo_TTN.write(80);
  emaxservo_TRN.write(90);
}

void hookGrasp() {
  // Set servo positions for hook grasp (fingers curled)
  emaxservo_FON.write(40);
  emaxservo_FII.write(40);
  emaxservo_FTN.write(40);
  emaxservo_MON.write(40);
  emaxservo_MIN.write(40);
  emaxservo_MTN.write(40);
  emaxservo_ROI.write(40);
  emaxservo_RII.write(40);
  emaxservo_RTI.write(40);
  emaxservo_LON.write(40);
  emaxservo_LII.write(40);
  emaxservo_LTI.write(40);
  emaxservo_TON.write(40);
  emaxservo_TII.write(40);
  emaxservo_TTN.write(40);
  emaxservo_TRN.write(40);
}

void cylindricalGrasp() {
  // Set servo positions for cylindrical grasp (hand wraps around cylinder)
  emaxservo_FON.write(60);
  emaxservo_FII.write(60);
  emaxservo_FTN.write(60);
  emaxservo_MON.write(60);
  emaxservo_MIN.write(60);
  emaxservo_MTN.write(60);
  emaxservo_ROI.write(60);
  emaxservo_RII.write(60);
  emaxservo_RTI.write(60);
  emaxservo_LON.write(60);
  emaxservo_LII.write(60);
  emaxservo_LTI.write(60);
  emaxservo_TON.write(60);
  emaxservo_TII.write(60);
  emaxservo_TTN.write(60);
  emaxservo_TRN.write(60);
}

void lateralGrasp() {
  // Set servo positions for lateral grasp (thumb and side of index finger)
  emaxservo_FON.write(90);
  emaxservo_FII.write(90);
  emaxservo_FTN.write(90);
  emaxservo_MON.write(90);
  emaxservo_MIN.write(90);
  emaxservo_MTN.write(90);
  emaxservo_ROI.write(90);
  emaxservo_RII.write(90);
  emaxservo_RTI.write(90);
  emaxservo_LON.write(90);
  emaxservo_LII.write(90);
  emaxservo_LTI.write(90);
  emaxservo_TON.write(90);
  emaxservo_TII.write(90);
  emaxservo_TTN.write(90);
  emaxservo_TRN.write(90);
}
