int motor_pin1  = 7;
int motor_pin2  = 6;
int motor2_pin1 = 5;
int motor2_pin2 = 4;

char cmd = 'x';
unsigned long lastCommandTime = 0;
const int TIMEOUT = 200;

void stopMotors() {
  digitalWrite(motor_pin1,  LOW);
  digitalWrite(motor_pin2,  LOW);
  digitalWrite(motor2_pin1, LOW);
  digitalWrite(motor2_pin2, LOW);
}

void setup() {
  Serial.begin(9600);
  pinMode(motor_pin1,  OUTPUT);
  pinMode(motor_pin2,  OUTPUT);
  pinMode(motor2_pin1, OUTPUT);
  pinMode(motor2_pin2, OUTPUT);
  stopMotors();
}

void loop() {
  if (Serial.available()) {
    cmd = Serial.read();
    lastCommandTime = millis();
  }

  if (millis() - lastCommandTime > TIMEOUT) {
    stopMotors();
    return;
  }

  switch (cmd) {
    case 'w':
      digitalWrite(motor_pin1,  HIGH); digitalWrite(motor_pin2,  LOW);
      digitalWrite(motor2_pin1, HIGH); digitalWrite(motor2_pin2, LOW);
      break;
    case 's':
      digitalWrite(motor_pin1,  LOW);  digitalWrite(motor_pin2,  HIGH);
      digitalWrite(motor2_pin1, LOW);  digitalWrite(motor2_pin2, HIGH);
      break;
    case 'a':
      digitalWrite(motor_pin1,  LOW);  digitalWrite(motor_pin2,  LOW);
      digitalWrite(motor2_pin1, HIGH); digitalWrite(motor2_pin2, LOW);
      break;
    case 'd':
      digitalWrite(motor_pin1,  HIGH); digitalWrite(motor_pin2,  LOW);
      digitalWrite(motor2_pin1, LOW);  digitalWrite(motor2_pin2, LOW);
      break;
    default:
      stopMotors();
      break;
  }
}