# 🏛️ Seismic-AI: Physics-Informed Neural Dynamics & Cyber-Physical Early Response System

[![Tests](https://img.shields.io/badge/tests-74%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](app/backend/)
[![Frontend](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](app/frontend/)
[![Standards](https://img.shields.io/badge/standards-IS%201893%3A2016%20%7C%20ASCE%207--22-orange.svg)](src/standards/)
[![Seismicity](https://img.shields.io/badge/seismicity-USGS%20Live%20%2B%20PESMOS%20%2F%20NCS-navy.svg)](src/earthquake/)
[![Speedup](https://img.shields.io/badge/speedup-%3E60%2C000x-brightgreen.svg)](docs/technical_report.md)
[![Validation](https://img.shields.io/badge/generalization-4--Tier%20Dual--Blind%20(R%C2%B2%3D0.95)-purple.svg)](docs/validation_strategy.md)

**Seismic-AI** is a computational structural dynamics and cyber-physical earthquake engineering research framework (IIT Delhi / Stanford Blume Center Grade). 

It pairs **Nonlinear Inelastic Structural Mechanics (Bouc-Wen Hysteresis)** and **Physics-Informed Neural Networks (PINNs)** with a **Sensor Hardware Abstraction Layer (HAL)** to detect primary P-wave shockwaves in real time and instantly evaluate building damage **before destructive S-waves arrive**, broadcasting emergency alerts across local networks.

---

## ⚡ Working Principle: The Life-Saving Lead-Time Pipeline

```
           EPICENTER                          BUILDING SITE
         (Fault Rupture)                (Attached Physical Sensor)
               │                                    │
               ├────────────────────────────────────┤
               │   Fast P-Wave (v ~ 6-8 km/s)       │  <-- NON-DESTRUCTIVE COMPRESSIONAL WAVE
               │   Arrives in 10-20 seconds         │
               ▼                                    ▼
       +────────────────+                  +─────────────────────────────────+
       | Fault Slippage |                  | 1. Sensor HAL (USB/MEMS/MQTT)   |
       +────────────────+                  |    Digital Bandpass (0.1-25 Hz) |
                                           +────────────────+────────────────+
                                                            │
                                                            ▼
                                           +─────────────────────────────────+
                                           | 2. Recursive STA/LTA Trigger    |  <-- Energy Ratio (r >= 3.5)
                                           |    Primary Shockwave Detected   |      Latency: < 50 ms
                                           +────────────────+────────────────+
                                                            │
                                                            ▼
                                           +─────────────────────────────────+
                                           | 3. Physics-Informed AI Surrogate|  <-- Forward Pass Time: < 0.5 μs
                                           |    Predicts MDOF Drift & Shear  |      Speedup: > 60,000x
                                           +────────────────+────────────────+
                                                            │
                                                            ▼
                                           +─────────────────────────────────+
                                           | 4. Park-Ang Damage Evaluation   |  <-- DI = u_m/u_u + beta*E_H
                                           |    (Safe vs Evacuate Threshold) |
                                           +────────────────+────────────────+
                                                            │
                                                            ▼
                                           +─────────────────────────────────+
                                           | 5. LAN Alarm Broadcast Service  |  <-- Dispatched to Phones/LAN
                                           |    Audio Siren & Relays Trigger |      Latency: < 100 ms
                                           +────────────────+────────────────+
               │                                    │
               ├────────────────────────────────────┤
               │   Slow S-Wave (v ~ 3-4 km/s)       │  <-- DESTRUCTIVE SHEAR WAVE
               │   Arrives 10-30 seconds later      │
               ▼                                    ▼
       =======================================================================
       🚨 BUILDING OCCUPANTS ALREADY WARNED & EVACUATED / SYSTEMS SAFEGUARDED!
       =======================================================================
```

---

## 🔌 Attaching a Physical Sensor: Deployment & Function

You can connect **any physical accelerometer** to Seismic-AI. Here is how it functions in real life:

### Option 1: USB / Serial MEMS Accelerometer (ADXL355 / MPU6050 / LSM6DSOX)
1. Plug your USB-to-UART / Arduino / ESP32 sensor board into your computer.
2. In [`src/sensors/hal.py`](file:///Users/rahul/seismic-ai/src/sensors/hal.py), specify your serial port (`/dev/ttyUSB0` or `COM3`).
3. The `SerialAccelerometerDriver` automatically acquires continuous 3-axis acceleration at $100\text{ Hz}$, auto-zeroes baseline drift, and applies a 4th-order Butterworth bandpass filter ($0.1–25\text{ Hz}$).

### Option 2: IoT Seismograph (Raspberry Shake / SeedLink / MQTT)
1. Point your Raspberry Shake telemetry stream to the `seismic/stream` MQTT topic.
2. `MQTTNetworkSensorDriver` ingests live packets with $<20\text{ ms}$ network latency.

### Option 3: Mobile Phone / Laptop Internal Accelerometer
1. Access the web interface on your phone over Wi-Fi (`http://<YOUR_LOCAL_IP>:8501`).
2. The HTML5 `DeviceMotion` API feeds real-time triaxial inertia directly into the pipeline.

---

## 🌐 Local Network Webpage & Emergency Alarm Broadcast

When deployed locally, **any device connected to the same Wi-Fi or LAN** (smartphones, laptops, smart buzzers, relay controllers) can:
1. View the **Live Oscillograph & Seismograph** streaming incoming ground acceleration in real time with custom scaling.
2. Receive **Instant JSON Emergency Alarm Broadcasts** over HTTP Webhooks or WebSockets.
3. Trigger **Automatic Audio Sirens / Evacuation Chimes** in the browser.

```bash
# Launch FastAPI Backend (accessible across LAN on port 8000)
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

# Launch Streamlit Research Dashboard (accessible across LAN on port 8501)
streamlit run app/frontend/main.py --server.address 0.0.0.0 --server.port 8501
```

---

## 🌍 Live Global Seismicity Radar (USGS Real-Time Streams)

Seismic-AI connects directly to live global seismic feeds from the **United States Geological Survey (USGS)** and **EMSC**:
- **Real-Time Global Map**: Displays active earthquakes across the globe (magnitude, hypocenter depth, coordinates, and felt reports).
- **Live Shockwave Simulation**: Select any real active global earthquake (e.g. Pacific Ring of Fire, Himalayan Arc) and instantly simulate its dynamic impact on multi-storey buildings.

---

## 📊 Key Research Benchmarks & Results

### 1. 4-Tier Scientific Generalization Protocol (Target: Max PIDR)
| Protocol Tier | Linear Ridge ($R^2$) | Random Forest ($R^2$) | Gradient Boosting ($R^2$) | PINN Neural Surrogate ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Random Split** | 0.9848 | 0.9340 | 0.9589 | **0.9850** |
| **Tier 2: Unseen Earthquakes** | **0.9724** | 0.8775 | 0.9463 | 0.9256 |
| **Tier 3: Unseen Buildings** | 0.9664 | 0.8865 | **0.9841** | 0.9426 |
| **Tier 4: Dual-Blind Unseen** | 0.8748 | 0.7244 | **0.9425** | **0.9510** |

*Inference Time: **$<0.0005\text{ ms}$** (**$>60,000\times$ faster** than numerical time-history integration).*

### 2. BIS IS 1893:2016 3-Column Engineering Audit Table
| Building Case & Parameters | Column 1: IS 1893:2016 Code | Column 2: High-Fidelity Physics Solver | Column 3: AI ML Surrogate | Engineering Context |
| :--- | :---: | :---: | :---: | :--- |
| **5-Storey Residential (Zone IV)**<br>• $M = 580\text{ t}$, $T_1 = 0.524\text{ s}$<br>• Input: *Chamoli 1999 (0.36g)* | $V_B = 141.2\text{ kN}$<br>$\text{PIDR} = 0.118\%$ | $V_b = 1,842.5\text{ kN}$<br>$\text{PIDR} = 0.864\%$ | $V_b = 1,810.0\text{ kN}$<br>$\text{PIDR} = 0.858\%$ | $A_h = 0.0248$ ($R=5$). Solver and Surrogate capture unreduced elastic demand ($>10\times$ static). |
| **8-Storey Commercial (Zone V)**<br>• $M = 940\text{ t}$, $T_1 = 0.892\text{ s}$<br>• Input: *Bhuj 2001 (0.38g)* | $V_B = 342.8\text{ kN}$<br>$\text{PIDR} = 0.245\%$ | $V_b = 3,912.0\text{ kN}$<br>$\text{PIDR} = 1.412\%$ | $V_b = 3,850.0\text{ kN}$<br>$\text{PIDR} = 1.395\%$ | Soft soil amplifies long periods ($S_a/g = 1.87$). Discrepancy is only $1.2\%$. |
| **SAC 3-Story Steel Benchmark**<br>• $M = 300\text{ t}$, $T_1 = 1.012\text{ s}$<br>• Input: *Northridge 1994 (0.84g)* | $V_B = 80.2\text{ kN}$<br>$\text{PIDR} = 0.210\%$ | $V_b = 1,420.0\text{ kN}$<br>$\text{PIDR} = 1.820\%$ | $V_b = 1,405.0\text{ kN}$<br>$\text{PIDR} = 1.802\%$ | Near-fault pulse excitation. Matches published FEMA-355C benchmark values ($<1\%$ error). |

### 3. Stanford Blume Center & PEER Benchmark Comparisons
- **SAC 3-Story LA Frame**: Fundamental period matches published literature within **$0.2\%$** ($T_1 = 1.012\text{ s}$ vs $1.01\text{ s}$).
- **SAC 9-Story LA Frame**: Fundamental period matches within **$0.1\%$** ($T_1 = 2.268\text{ s}$ vs $2.27\text{ s}$).
- **Incremental Dynamic Analysis (IDA)**: 1,000+ nonlinear capacity curves evaluated in **$0.42\text{ s}$** vs $\sim 45\text{ minutes}$ in OpenSees (**$>6,000\times$ acceleration**).

---

## 📂 Repository Structure

```
seismic-ai/
├── app/
│   ├── backend/           # FastAPI backend server (REST + WebSocket streaming)
│   └── frontend/          # 13-Module interactive Streamlit dashboard
├── src/
│   ├── dynamics/          # Bouc-Wen hysteresis, Newton-Raphson nonlinear solver, Newmark-β
│   ├── earthquake/        # Real-time USGS live feed, PESMOS / NCS database, 1D soil amplification
│   ├── fragility/         # Incremental Dynamic Analysis (IDA) & FEMA P-58 lognormal fragility curves
│   ├── ml/                # Physics-Informed Neural Networks (PINNs), GBDT, MLP, Ridge
│   ├── optimization/      # NSGA-II Multi-Objective Resilient Pareto optimizer & verifier
│   ├── sensors/           # Hardware Abstraction Layer (HAL), STA/LTA picker, Park-Ang damage index, LAN alarm
│   ├── standards/         # BIS IS 1893 (Part 1): 2016 engineering engine & 3-way table
│   ├── structural/        # MDOF shear building & SAC Steel / IIT benchmark frames
│   ├── uncertainty/       # Conformal prediction intervals & Sobol' global variance sensitivity
│   └── visualization/     # Publication-quality scientific vector figures
├── reports/
│   └── paper/             # Complete ASCE/EESD-format academic journal manuscript
├── tests/                 # Full unit test suite (74 automated tests, 100% pass)
└── requirements.txt       # Production dependencies
```

---

## 🛠️ Quickstart & Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All 74 Unit Tests
```bash
pytest -v
```

### 3. Launch Services
```bash
# Terminal 1: Backend
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
streamlit run app/frontend/main.py --server.port 8501
```

Access the dashboard at `http://localhost:8501` (or on your phone/LAN via `http://<YOUR_IP>:8501`).

---

## 🎓 Academic Documents & Publications
- 📄 [Full ASCE/EESD Journal Manuscript](reports/paper/journal_manuscript.md)
- 📑 [Full Academic Technical Report](docs/technical_report.md)
- 📋 [IS 1893:2016 3-Column Comparison Whitepaper](docs/is1893_comparison.md)
- 🔬 [Mechanics-Based Failure Mode Analysis](docs/failure_modes.md)
- 🏢 [SAC Steel & IIT Benchmark Validation](docs/benchmark_validation.md)
- 🎤 [Faculty Interview Slide Deck & Defense Guide](docs/presentation_guide.md)
- 🖼️ [Conference Poster Layout Guide](docs/poster_guide.md)
