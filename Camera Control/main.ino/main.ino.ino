#include <Arduino.h>

// Allocate a buffer large enough for your maximum expected size (e.g., 128x128)
static const uint32_t MAX_IMG_SIZE = 112 * 112;
uint8_t imageBuffer[MAX_IMG_SIZE];

bool readExact(uint8_t *dst, size_t len) {
  size_t got = 0;
  unsigned long start = millis();
  while (got < len) {
    if (Serial.available()) {
      // Serial.readBytes already blocks slightly, read() is safer in a tight
      // loop
      dst[got++] = Serial.read();
      start = millis();
    }
    if (millis() - start > 2000) {
      return false;
    }
  }
  return true;
}

bool waitForHeader() {
  const char target[4] = {'I', 'M', 'G', '0'};
  int matched = 0;

  while (true) {
    if (Serial.available()) {
      char c = Serial.read();
      if (c == target[matched]) {
        matched++;
        if (matched == 4)
          return true;
      } else {
        matched = (c == target[0]) ? 1 : 0;
      }
    }
  }
}

void setup() {
  Serial.begin(921600);
  while (!Serial) {
    delay(10);
  }
  Serial.println("Pico ready");
}

void loop() {
  if (!waitForHeader())
    return;

  uint8_t meta[4];
  if (!readExact(meta, 4)) {
    Serial.println("Meta read timeout");
    return;
  }

  uint16_t w = meta[0] | (meta[1] << 8);
  uint16_t h = meta[2] | (meta[3] << 8);
  uint32_t current_img_size = w * h;

  // Validate that the incoming image fits in our memory
  if (current_img_size > MAX_IMG_SIZE) {
    Serial.println("Error: Image size exceeds MAX_IMG_SIZE");
    return;
  }

  // Read the dynamic size
  if (!readExact(imageBuffer, current_img_size)) {
    Serial.println("Image read timeout");
    return;
  }

  // Example: normalize for model input
  // (You could also dynamically allocate modelInput if RAM allows)
  static float modelInput[MAX_IMG_SIZE];
  for (uint32_t i = 0; i < current_img_size; i++) {
    modelInput[i] = imageBuffer[i] / 255.0f;
  }

  Serial.println("Frame received successfully");
}
