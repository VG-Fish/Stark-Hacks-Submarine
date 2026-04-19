// Left motor
#define LEFT_PWM 5   // speed (PWM pin)
#define LEFT_DIR 7   // direction (any digital pin)

// Right motor
#define RIGHT_PWM 6  // speed (PWM pin)
#define RIGHT_DIR 8  // direction (any digital pin)

void setMotor(uint8_t pinPWM, uint8_t pinDir, int val) {
  val = constrain(val, -255, 255);
  if (val >= 0) {
    digitalWrite(pinDir, HIGH);
    analogWrite(pinPWM, val);
  } else {
    digitalWrite(pinDir, LOW);
    analogWrite(pinPWM, -val);
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(LEFT_PWM,  OUTPUT);
  pinMode(LEFT_DIR,  OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);

  setMotor(LEFT_PWM,  LEFT_DIR,  0);
  setMotor(RIGHT_PWM, RIGHT_DIR, 0);
}

void loop() {
  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    int commaIndex = msg.indexOf(',');
    if (commaIndex > 0) {
      int left  = msg.substring(0, commaIndex).toInt();
      int right = msg.substring(commaIndex + 1).toInt();
      setMotor(LEFT_PWM,  LEFT_DIR,  left);
      setMotor(RIGHT_PWM, RIGHT_DIR, right);
    }
  }
}