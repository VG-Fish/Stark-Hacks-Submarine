/*
 * NanoCrackSeg Inference on RP2040 (Raspberry Pi Pico)
 *
 * This sketch receives grayscale images from a host PC via serial,
 * runs the 8-bit quantized NanoCrackSeg model for crack segmentation,
 * and sends back the inference results.
 *
 * Serial Protocol:
 *   Host → Pico: [4B: "IMG0"][2B: width][2B: height][W*H bytes: image]
 *   Pico → Host: [4B: "OUT0"][2B: output_size][size bytes: output]
 *
 * Baud Rate: 921600
 * Expected Input: 112×112 grayscale image (uint8)
 * Expected Output: 112×112 segmentation logits (int8)
 */

#include <Arduino.h>
#include "nano_crack_seg_model.h"
#include "pico_crack_seg.h"

/* ========== Configuration ========== */

/** Maximum image size (112×112 = 12,544 bytes) */
static constexpr uint32_t MAX_IMG_SIZE = 112 * 112;

/** Serial communication timeout (milliseconds) */
static constexpr uint32_t SERIAL_TIMEOUT_MS = 2000;

/** Expected image dimensions */
static constexpr uint16_t EXPECTED_WIDTH = 112;
static constexpr uint16_t EXPECTED_HEIGHT = 112;

/* ========== Buffers ========== */

/** Input image buffer (uint8, row-major) */
static uint8_t input_buffer[MAX_IMG_SIZE];

/** Output segmentation buffer (int8, row-major) */
static int8_t output_buffer[MAX_IMG_SIZE];

/** Statistics tracking */
struct
{
  uint32_t frames_received;
  uint32_t successful_inferences;
  uint32_t failed_inferences;
  uint32_t total_inference_time_ms;
  uint32_t max_inference_time_ms;
  uint32_t min_inference_time_ms;
} stats = {0, 0, 0, 0, 0, UINT32_MAX};

/* ========== Utility Functions ========== */

/**
 * Read exact number of bytes from serial with timeout.
 *
 * @param dst: Destination buffer
 * @param len: Number of bytes to read
 * @return: true if successful, false on timeout
 */
bool readExact(uint8_t *dst, size_t len)
{
  size_t got = 0;
  unsigned long start = millis();

  while (got < len)
  {
    if (Serial.available())
    {
      dst[got++] = Serial.read();
      start = millis(); // Reset timeout on each byte
    }

    // Check timeout
    if (millis() - start > SERIAL_TIMEOUT_MS)
    {
      return false;
    }
  }

  return true;
}

/**
 * Wait for "IMG0" header to synchronize with host.
 *
 * @return: true when header is found, false on timeout
 */
bool waitForHeader()
{
  const char target[4] = {'I', 'M', 'G', '0'};
  int matched = 0;
  unsigned long start = millis();

  while (true)
  {
    if (Serial.available())
    {
      char c = Serial.read();

      // Check if character matches expected header
      if (c == target[matched])
      {
        matched++;
        if (matched == 4)
        {
          return true;
        }
      }
      else
      {
        // Reset matching if character doesn't match
        matched = (c == target[0]) ? 1 : 0;
      }

      start = millis(); // Reset timeout on each byte
    }

    // Timeout waiting for header
    if (millis() - start > SERIAL_TIMEOUT_MS)
    {
      return false;
    }
  }
}

/**
 * Send inference results back to host.
 *
 * @param output: Output buffer (int8 array)
 * @param size: Number of output bytes
 * @param inference_time_ms: Inference execution time
 */
void sendResults(const int8_t *output, uint32_t size, uint32_t inference_time_ms)
{
  // Send output header
  Serial.write((const uint8_t *)"OUT0", 4);

  // Send output size (little-endian uint16)
  uint8_t size_bytes[2] = {
      (uint8_t)(size & 0xFF),
      (uint8_t)((size >> 8) & 0xFF)};
  Serial.write(size_bytes, 2);

  // Send output data
  Serial.write((const uint8_t *)output, size);

  // Send status message
  Serial.print("\nOK|");
  Serial.print(inference_time_ms);
  Serial.println("|");
}

/**
 * Print model statistics (debug only).
 */
void printStats()
{
  Serial.println("\n=== Inference Statistics ===");
  Serial.print("Frames received: ");
  Serial.println(stats.frames_received);
  Serial.print("Successful inferences: ");
  Serial.println(stats.successful_inferences);
  Serial.print("Failed inferences: ");
  Serial.println(stats.failed_inferences);

  if (stats.successful_inferences > 0)
  {
    uint32_t avg_time = stats.total_inference_time_ms / stats.successful_inferences;
    Serial.print("Average inference time: ");
    Serial.print(avg_time);
    Serial.println(" ms");
    Serial.print("Min/Max inference time: ");
    Serial.print(stats.min_inference_time_ms);
    Serial.print(" / ");
    Serial.print(stats.max_inference_time_ms);
    Serial.println(" ms");
  }
  Serial.println("============================\n");
}

/* ========== Arduino Setup/Loop ========== */

/**
 * Arduino setup() - called once at startup.
 * Initializes serial communication and the inference model.
 */
void setup()
{
  // Initialize serial communication
  Serial.begin(921600);

  // Wait for serial to be ready
  uint32_t timeout = millis() + 2000;
  while (!Serial && millis() < timeout)
  {
    delay(10);
  }

  Serial.println("\n=== NanoCrackSeg Model on RP2040 ===");
  Serial.print("Model: ");
  Serial.println(model_get_name());
  Serial.print("Model size: ");
  Serial.print(model_get_size() / 1024);
  Serial.println(" KB");
  Serial.print("Input: ");
  Serial.print(MODEL_INPUT_WIDTH);
  Serial.print("×");
  Serial.print(MODEL_INPUT_HEIGHT);
  Serial.println(" (grayscale)");
  Serial.print("Output: ");
  Serial.print(MODEL_OUTPUT_WIDTH);
  Serial.print("×");
  Serial.print(MODEL_OUTPUT_HEIGHT);
  Serial.println(" (segmentation)");

  // Initialize the inference model
  Serial.println("Initializing model...");
  int32_t init_status = model_init();

  if (init_status != 0)
  {
    Serial.println("ERROR: Model initialization failed!");
    Serial.println("Halting execution.");
    while (1)
    {
      delay(1000);
    }
  }

  Serial.println("✓ Model initialized successfully");
  Serial.println("Listening for images on serial...");
  Serial.println("=====================================\n");

  // Initialize statistics
  stats.frames_received = 0;
  stats.successful_inferences = 0;
  stats.failed_inferences = 0;
  stats.total_inference_time_ms = 0;
  stats.max_inference_time_ms = 0;
  stats.min_inference_time_ms = UINT32_MAX;
}

/**
 * Arduino loop() - called repeatedly.
 * Waits for image data, runs inference, and sends results.
 */
void loop()
{
  // Wait for "IMG0" header
  if (!waitForHeader())
  {
    // Timeout, try again
    return;
  }

  // Read metadata (width and height as 2-byte little-endian values)
  uint8_t meta[4];
  if (!readExact(meta, 4))
  {
    Serial.println("ERROR: Metadata read timeout");
    return;
  }

  // Parse dimensions
  uint16_t w = meta[0] | (meta[1] << 8);
  uint16_t h = meta[2] | (meta[3] << 8);
  uint32_t current_img_size = w * h;

  // Validate image size
  if (current_img_size > MAX_IMG_SIZE)
  {
    Serial.print("ERROR: Image size ");
    Serial.print(current_img_size);
    Serial.print(" exceeds MAX_IMG_SIZE ");
    Serial.println(MAX_IMG_SIZE);
    return;
  }

  // Validate dimensions match expected 112×112
  if (w != EXPECTED_WIDTH || h != EXPECTED_HEIGHT)
  {
    Serial.print("WARNING: Expected ");
    Serial.print(EXPECTED_WIDTH);
    Serial.print("×");
    Serial.print(EXPECTED_HEIGHT);
    Serial.print(" but got ");
    Serial.print(w);
    Serial.print("×");
    Serial.println(h);
  }

  // Read image data
  if (!readExact(input_buffer, current_img_size))
  {
    Serial.println("ERROR: Image data read timeout");
    return;
  }

  stats.frames_received++;

  // Run inference
  unsigned long inference_start = millis();
  int32_t invoke_status = model_invoke(input_buffer, output_buffer);
  unsigned long inference_time_ms = millis() - inference_start;

  if (invoke_status != 0)
  {
    Serial.println("ERROR: Model inference failed");
    stats.failed_inferences++;
    return;
  }

  stats.successful_inferences++;
  stats.total_inference_time_ms += inference_time_ms;

  if (inference_time_ms > stats.max_inference_time_ms)
  {
    stats.max_inference_time_ms = inference_time_ms;
  }

  if (inference_time_ms < stats.min_inference_time_ms)
  {
    stats.min_inference_time_ms = inference_time_ms;
  }

  // Send results back to host
  sendResults(output_buffer, current_img_size, inference_time_ms);

  // Print inference time for monitoring
  Serial.print("Frame ");
  Serial.print(stats.frames_received);
  Serial.print(": ");
  Serial.print(inference_time_ms);
  Serial.println(" ms");

  // Print stats every 10 frames
  if (stats.frames_received % 10 == 0)
  {
    printStats();
  }
}
