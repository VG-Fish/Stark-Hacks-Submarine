int motor_pin1 = 7;
int motor_pin2 = 6;

void setup(){
  pinMode(motor_pin1, OUTPUT);
  pinMode(motor_pin2, OUTPUT);
}

void loop(){
  digitalWrite(motor_pin1, HIGH);
  digitalWrite(motor_pin2, LOW);

  delay(1000);

  digitalWrite(motor_pin1, LOW);
  digitalWrite(motor_pin2, HIGH);

  delay(1000);
}