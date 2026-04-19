// Speed via ENA/ENB (PWM), direction via IN1/IN3
// IN2 and IN4 are hardwired to GND on the L298N board

#define LEFT_EN  5   // PWM → ENA
#define LEFT_IN1 7   // digital → IN1

#define RIGHT_EN  6  // PWM → ENB
#define RIGHT_IN3 8  // digital → IN3

char cmd = 'x';
unsigned long lastCommandTime = 0;
const int timeout = 2000;

void setMotor(uint8_t en, uint8_t in, int speed) {
  speed = constrain(speed, 0, 255);
  digitalWrite(in, HIGH);   // IN2/IN4 are GND, so HIGH here = forward
  analogWrite(en, speed);
}

void stopMotors() {
  analogWrite(LEFT_EN,  0);
  analogWrite(RIGHT_EN, 0);
}

void setup() {
  Serial.begin(9600);
  pinMode(LEFT_EN,   OUTPUT);
  pinMode(LEFT_IN1,  OUTPUT);
  pinMode(RIGHT_EN,  OUTPUT);
  pinMode(RIGHT_IN3, OUTPUT);
  stopMotors();
}

void loop() {
  if (Serial.available()) {
    cmd = Serial.read();
    lastCommandTime = millis();
  }

  if (millis() - lastCommandTime > timeout) {
    cmd = 'x';
  }

  switch (cmd) {
    case 'w':
      setMotor(LEFT_EN,  LEFT_IN1,  200);
      setMotor(RIGHT_EN, RIGHT_IN3, 200);
      break;
    case 'a':
      setMotor(LEFT_EN,  LEFT_IN1,  80);
      setMotor(RIGHT_EN, RIGHT_IN3, 200);
      break;
    case 'd':
      setMotor(LEFT_EN,  LEFT_IN1,  200);
      setMotor(RIGHT_EN, RIGHT_IN3, 80);
      break;
    case 'x':
    default:
      stopMotors();
      break;
  }
}