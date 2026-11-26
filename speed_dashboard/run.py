"""Start the dashboard:  python run.py   (then open http://localhost:8000)"""
import uvicorn

if __name__ == "__main__":
    # host 0.0.0.0 lets the ESP32-CAM (same WiFi) reach the server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="warning")
