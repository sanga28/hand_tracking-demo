# Hand_tracking
This project captures hand movements and calculates finger angles using OpenCV and MediaPipe on the PC side, then sends the data to an ESP32 microcontroller for further processing.

---
## Prerequisites
PC Side
- Python 3.10
- OpenCV
- MediaPipe
- NumPy
- PySerial

ESP32 Side
- Arduino IDE
- ESP32 Board Support Package

---
## Troubleshooting
- Ensure the correct COM port is set in the Python script.
- Verify the baud rate matches between the Python script and ESP32 code.
- Check for loose connections.
