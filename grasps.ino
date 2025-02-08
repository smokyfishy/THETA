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

Servo servos[] = {emaxservo_FON, emaxservo_FII, emaxservo_FTN, 
                  emaxservo_MON, emaxservo_MIN, emaxservo_MTN, 
                  emaxservo_ROI, emaxservo_RII, emaxservo_RTI, 
                  emaxservo_LON, emaxservo_LII, emaxservo_LTI,
                  emaxservo_TON, emaxservo_TII, emaxservo_TTN, emaxservo_TRN};

int servoPins[] = {50, 51, 48,
                   52, 53, 49,
                   30, 31, 26, 
                   28, 29, 27, 
                   47, 45, 46, 24};

int defaultAngle[] = {180, 0, 0,
                      180, 0, 0,
                      0, 180, 180,
                      0, 180, 180,
                      0, 180, 0, 90};

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

  for (int i = 0; i < 16; i++) {
    servos[i].attach(servoPins[i]);
  }

  delay(50);  // Short delay for setup to complete

  defaultPose();
  delay(3000);
  basicGrasp();
}

void loop() {
  // Call different grasp functions based on input or condition
  if (Serial.available()) {
    int input = Serial.read();
    if (input == '0') {
      defaultPose();
    } else if (input == '1') {
      basicGrasp();
    } else if (input == '2') {
      peace();
    } else if (input == '3') {
      okayGrasp();
    } else if (input == '4') {
      cylindricalGrasp();
    } else if (input == '5') {
      lateralGrasp();
    } else if (input == '6') {
      peace();
    }
    }
  }

void setServoAngle(int newAngle[]) {
  for (int i=0; i < 16; i++) {
    servos[i].write(newAngle[i]);
  }
}

void defaultPose() {
  // Set all servos to the neutral/default position
  setServoAngle(defaultAngle);
}

void peace() {
  // Set servo positions for peace sign (extended F and M, contracted T, R, L)
  int peaceAngle[] = {130, 20, 20, 
                      160, 50, 20, 
                      160, 20, 20, 
                      160, 20, 20,
                      160, 160, 160, 90};
  setServoAngle(peaceAngle);
}

void basicGrasp() {
  // Set servo positions for power grasp (closed fist)
  int basicGraspAngle[] = {20, 160, 160, 
                           20, 160, 160, 
                           160, 20, 50, 
                           140, 50, 20,
                           160, 20, 160, 90};
                            //full contraction (tightest pull of string) angles
                            //Servo servos[] = {emaxservo_FON 0  , emaxservo_FII 180, emaxservo_FTN 180, 
                                              //emaxservo_MON 0  , emaxservo_MIN 180, emaxservo_MTN 180, 
                                              //emaxservo_ROI 180, emaxservo_RII 0  , emaxservo_RTI 0, 
                                              //emaxservo_LON 180, emaxservo_LII 0  , emaxservo_LTI 0,
                                              //emaxservo_TON 180, emaxservo_TII 0  , emaxservo_TTN 180, emaxservo_TRN 0};
  setServoAngle(basicGraspAngle);
}

void okayGrasp() {
    // Set servo positions for okay (thumb/forefinger) grasp (half-contracted F, contracted T, extended M, R, L)
  int okayGraspAngle[] = {0, 0, 140, 
                      150, 30, 20, 
                      30, 150, 160, 
                      30, 160, 160,
                      170, 170, 180, 90};
  setServoAngle(okayGraspAngle);
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
