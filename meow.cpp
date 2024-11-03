#include "vex.h"


using namespace vex;


int autonSelection = 2;
//auton selector
void autonUp(void) {
 if (autonSelection == 1) {
   autonSelection = 2;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Right Side");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Right Side");
 } else if (autonSelection == 2) {
   autonSelection = 3;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Left Alliance");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Left Alliance");
 } else if (autonSelection == 3) {
   autonSelection = 4;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Right Alliance");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Right Alliance");
 } else if (autonSelection == 4) {
   autonSelection = 5;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Auton Point");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Auton Point");
 } else if (autonSelection == 5) {
   autonSelection = 6;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Skills");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Skills");
 } else if (autonSelection == 6) {
   autonSelection = 1;
   Brain.Screen.clearScreen();
   Brain.Screen.setCursor(1,1);
   Brain.Screen.print("Left Side");
   Controller1.Screen.clearScreen();
   Controller1.Screen.setCursor(1,1);
   Controller1.Screen.print("Left Side");
 }




}


//Auton Number 1
void leftSide(void) {
 //set 4b and piston
 fourBarA.startRotateFor(forward, 45, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 45, deg, 100, velocityUnits::pct);
 clawPiston.set(false);
 wait(100, msec);
 //move to goal
 leftMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 tilter.startRotateFor(reverse, 460, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //pick up goal
 clawPiston.set(true);
 wait(100, msec);
 //move back with goal
 leftMotorA.startRotateFor(reverse, 800, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 800, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 800, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 800, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //turn
 leftMotorA.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 500, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //drop goal in zone and turn to mid goal
 clawPiston.set(false);
 leftMotorA.startRotateFor(forward, 620, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 620, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 620, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 620, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //go to mid goal
 intake.spin(forward, 100, pct);
 leftMotorA.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 intake.stop();
 //pick up mid goal
 clawPiston.set(true);
 wait(100, msec);
 //go back to zone
 leftMotorA.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 wait(500, msec);






}


//Auton Number 2
void rightSide(void) {
 //set 4b and piston
 fourBarA.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 clawPiston.set(false);
 wait(100, msec);
 //move to goal
 leftMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 tilter.startRotateFor(reverse, 460, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //pick up goal
 clawPiston.set(true);
 wait(100, msec);
 //move back with goal
 leftMotorA.startRotateFor(reverse, 1200, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 1200, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 1200, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 1200, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //drop goal in zone and turn to mid goal
 clawPiston.set(false);
 leftMotorA.startRotateFor(reverse, 200, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 200, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 200, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 200, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //go to mid goal
 intake.spin(forward, 100, pct);
 leftMotorA.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1600, deg, 100, velocityUnits::pct);
 intake.stop();
 //pick up mid goal
 clawPiston.set(true);
 wait(100, msec);
 //go back to zone
 leftMotorA.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 wait(500, msec);
}


//Auton Number 3
void leftAlliance(void) {
 //set 4b and piston
 fourBarA.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 clawPiston.set(false);
 wait(100, msec);
 //move to goal
 leftMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 tilter.startRotateFor(reverse, 460, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //pick up goal
 clawPiston.set(true);
 wait(100, msec);
 //move back with goal
 leftMotorA.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 2000, deg, 100, velocityUnits::pct);
 wait(100, msec);
}


//Auton Number 4
void rightAlliance(void) {
 //set 4b and piston
 fourBarA.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 40, deg, 100, velocityUnits::pct);
 clawPiston.set(false);
 wait(100, msec);
 //move to goal
 leftMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 tilter.startRotateFor(reverse, 460, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 1510, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //pick up goal
 clawPiston.set(true);
 wait(100, msec); 
 fourBarA.startRotateFor(forward, 100, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 100, deg, 100, velocityUnits::pct);
 //move back with goal
 leftMotorA.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 1500, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //turn to alliance goal
 leftMotorA.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 500, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //drive to alliance goal
 leftMotorA.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 500, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //lifts both goals
 tilter.startRotateFor(forward, 200, deg, 100, velocityUnits::pct);


 wait(100, msec);


}
// Auton Number 5
void autonPoint(void) {
 //set 4b and piston
 fourBarA.startRotateFor(forward, 100, deg, 100, velocityUnits::pct);
 fourBarB.startRotateFor(forward, 100, deg, 100, velocityUnits::pct);
 tilter.rotateFor(reverse, 460, deg, 100, velocityUnits::pct);
 clawPiston.set(false);
 wait(100, msec);
 //move to goal
 leftMotorA.startRotateFor(reverse, 600, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(reverse, 600, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(reverse, 600, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(reverse, 600, deg, 100, velocityUnits::pct);
 wait(100, msec);
 tilter.rotateFor(forward, 200, deg, 100, velocityUnits::pct);
 wait(100, msec);
 //move back with goal and score preloads
 intake.spin(forward, 100, velocityUnits::pct);
 leftMotorA.startRotateFor(forward, 500, deg, 100, velocityUnits::pct);
 leftMotorB.startRotateFor(forward, 500, deg, 100, velocityUnits::pct);
 rightMotorA.startRotateFor(forward, 500, deg, 100, velocityUnits::pct);
 rightMotorB.rotateFor(forward, 500, deg, 100, velocityUnits::pct);
 wait(100, msec);
}
// Auton Number 5
void skills(void) {
 //in progress, currently working on a skills program but 2 goals short
 wait(500, msec);
}


void pre_auton() {
 vexcodeInit();
 //auton selector
 LimitSwitchH.pressed(autonUp);
 clawPiston.set(true);
}


void auton (void) {
 if (autonSelection == 1) {
   leftSide();
 } else if (autonSelection == 2) {
   rightSide();
 } else if (autonSelection == 3) {
   leftAlliance();
 } else if (autonSelection == 4) {
   rightAlliance();
 } else if (autonSelection == 5) {
   autonPoint();
 } else if (autonSelection == 6) {
   skills();
 }
}

