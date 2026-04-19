#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found");
    while (1);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Adafruit returns m/s² — divide by 9.81 to get g
  float axG = a.acceleration.x / 9.81;
  float ayG = a.acceleration.y / 9.81;
  float azG = a.acceleration.z / 9.81;

  Serial.print(axG, 4);
  Serial.print(",");
  Serial.print(ayG, 4);
  Serial.print(",");
  Serial.println(azG, 4);

  delay(20);
}