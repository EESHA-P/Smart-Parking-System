# Integrated Smart Parking and Traffic Control System

## Overview
A fully simulated IoT-based smart parking management system that uses Node-RED, MQTT, and a Python prediction module to optimize urban parking and traffic flow. No physical hardware required—ideal for academic prototyping, smart city research, and rapid proof-of-concept development[attached_file:1][file:59].

## Features
- **Real-time slot monitoring:** Simulates vehicle entry, exit, and live occupancy of parking lots.
- **Dynamic traffic signal control:** Adjusts stoplight behavior based on current parking availability.
- **Dashboard visualization:** Interactive Node-RED dashboard for live status, alerts, analytics, and scenario.
- **Smart prediction:** Python module forecasts short-term demand and availability using historical data (30–60 min ahead).
- **Adaptive signboards and rerouting:** Live rerouting and dynamic signs direct drivers to available parking based on real-time analytics.

## Architecture
- **Simulation Engine:** Node-RED flows control car entry/exit, slot updates, reservations, and event triggers.
- **Communication Layer:** MQTT broker enables virtual sensors and system messages.
- **Core Logic:** Handles occupancy, overstay detection, dynamic pricing, and predictive analytics in Python.
- **Visualization:** Node-RED dashboard shows occupancy grid, traffic state, price, gauges, and notifications.

## Installation

1. **Clone the repository**
    ```
    git clone https://github.com/yourusername/smart-parking-iot.git
    cd smart-parking-iot
    ```
2. **Install Node-RED**
    - Download and install Node-RED from [node-red.org](https://nodered.org).
3. **Install Python dependencies**
    ```
    pip install pandas numpy scikit-learn
    ```
4. **Run Node-RED server**
    ```
    node-red
    ```
5. **Import the provided flows**
    - Import `flows.json` and configure paths for inject, exec, and debug nodes.

## Usage

- **Simulation:** Trigger car movement, parking, or reservation events from the dashboard.
- **Prediction:** The Python module (`parking_predictor.py`) analyzes recent history as JSON and provides availability forecasts, confidence scores, and recommendations for the next 30–60 minutes[attached_file:1].
- **Realtime Updates:** Dashboard and signboard update instantly on parking or traffic changes.
- **Scenario Testing:** Emulate congestion, full parking, and dynamic rerouting.

## Example Prediction Output
{
"timestamp": "2025-10-23T10:30:00Z",
"slot": "A3",
"predicted_status": "vacant",
"confidence": 0.88,
"recommendation": "Good time to park"
}

## Performance Highlights
- 15–25% higher prediction accuracy over basic rule-based parking logic[attached_file:1].
- Node-RED dashboard refreshes in ≤ 1 sec latency.
- No physical sensor required – pure software simulation for fast experimentation.

## Contributors
- Stuti Dahal
- Eesha Pedakota

