# AKI Detection - Coursewrok at Imperial

Real-time **Acute Kidney Injury (AKI)** detection service. It consumes HL7 v2 messages
over an MLLP TCP stream, maintains per-patient creatinine statistics in SQLite, runs a
logistic-regression model on each new blood-test result, and pages clinicians when AKI is
predicted. Runs on Kubernetes with Prometheus metrics and a Grafana dashboard.

## How it works

```
MLLP socket → Message_Parser → SQLite (patient state) → Feature_Processor
            → Inference_Model → (if positive) persist + enqueue alert → ACK
                                          ↓
                    background alert worker → pager POST (with retries)
```

- **Admit (`ADT^A01`)** stores the patient's age and sex.
- **Blood test (`ORU^R01`)** updates the patient's running creatinine features and runs
  inference; a positive prediction is persisted and queued for paging.
- **Discharge (`ADT^A03`)** is logged only.

Per-patient creatinine state is stored as **aggregate features** (running mean/std via
Welford's algorithm, min/max, trend counts), not the full history, so inference is O(1)
per message. Positive alerts are persisted to the DB and drained by a background worker
that retries the pager, so undelivered alerts survive a restart.

## Layout

```
config.py            # Env-var configuration
messaging.py         # MLLP/HL7 parsing + ACK formatting (Message_Parser)
database.py          # SQLite CRUD wrapper (SQLite_Manager)
aki_model.py         # Feature_Processor + Inference_Model
main.py              # System_Engine: socket loop, dispatch, alert worker, reconnection
metrics.py           # Prometheus collectors + metrics server
view_metrics.py      # Local helper to inspect metrics

db/schema.sql        # `patients` table
model/               # training_model.py, training.csv, aki_model.pkl
simulator/           # MLLP simulator + its tests (local only)
tests/               # unittest suites (messaging, database, aki_model, integration)

Dockerfile           # Container image (entrypoint: python -m main)
coursework6.yaml     # Kubernetes PVC + ConfigMap + Deployment + Service
k8s/                 # Prometheus ServiceMonitor + PrometheusRule
grafana/             # Local Prometheus + Grafana stack + dashboard
```

## Running

```bash
# Service (talks to MLLP_ADDRESS / PAGER_ADDRESS)
python -m main

# Simulator: provides the MLLP feed + pager endpoint
python simulator/simulator.py --messages=simulator/messages.mllp --mllp=8440 --pager=8441

# Tests (plain unittest; run as modules from repo root)
python -m tests.test_messaging
python -m tests.test_database
python -m tests.test_aki_model
python -m tests.test_integration      # starts a real simulator subprocess

# Container
docker build -t aki .

# Kubernetes (namespace `sharq`)
kubectl apply -f coursework6.yaml
kubectl apply -f k8s/
```

Prometheus metrics are served on **port 8000** at `/metrics`.

## Configuration (env vars, see `config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `MLLP_ADDRESS` | `localhost:8440` | HL7 message source |
| `PAGER_ADDRESS` | `localhost:8441` | Pager endpoint (`/page`) |
| `MODEL_PATH` | `model/aki_model.pkl` | Pickled `{model, threshold}` |
| `DB_PATH` | `db` | SQLite location (`/state` in k8s) |
| `SCHEMA_PATH` | `db/schema.sql` | Schema applied on first init |
| `HISTORY_PATH` | `/data/history.csv` | Bootstrap history |
| `SYSTEM_LOG_PATH` / `MESSAGE_LOG_PATH` | `/state/logs/*.log` | Logs |
