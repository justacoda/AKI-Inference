# Local Grafana Setup for AKI Detection Metrics

This setup runs Prometheus and Grafana locally to visualize your AKI detection metrics from the Kubernetes pod.

## Prerequisites

- Docker Desktop installed and running
- Port-forward to your Kubernetes pod running on `localhost:8000`

## Quick Start

### 1. Start port-forward to your Kubernetes pod

In one terminal, keep this running:
```bash
kubectl --namespace=sharq port-forward deployment/aki-detection 8000:8000
```

### 2. Start Grafana and Prometheus

In another terminal:
```bash
cd grafana
docker compose up -d
```

**Note:** Use `docker compose` (no hyphen) with modern Docker Desktop.

### 3. Access Grafana

Open your browser and go to:
```
http://localhost:3000
```

**Login credentials:**
- Username: `admin`
- Password: `admin`

(You'll be prompted to change the password on first login - you can skip this)

### 4. Verify Prometheus is scraping

Check that Prometheus is collecting metrics:
```
http://localhost:9090/targets
```

You should see `aki-detection` target as **UP**.

### 5. Create a Dashboard

In Grafana:

1. Click **+** → **Create Dashboard**
2. Click **Add visualization**
3. Select **Prometheus** as the data source
4. In the query builder, try these queries:

**Message Throughput:**
```
rate(aki_messages_received_total[5m])
```

**Total Messages by Type:**
```
aki_messages_received_total
```

**Positive AKI Predictions:**
```
aki_positive_predictions_total
```

**Active Patients:**
```
aki_active_patients
```

**Creatinine Distribution (95th percentile):**
```
histogram_quantile(0.95, rate(aki_creatinine_value_umol_l_bucket[5m]))
```

**Processing Latency (95th percentile):**
```
histogram_quantile(0.95, rate(aki_processing_latency_seconds_bucket[5m]))
```

**MLLP Reconnections Rate:**
```
rate(aki_mllp_reconnections_total[5m])
```

**Model Inference Rate:**
```
rate(aki_model_inferences_total[5m])
```

## Useful Commands

**Stop Grafana/Prometheus:**
```bash
docker compose down
```

**Restart:**
```bash
docker compose restart
```

**View logs:**
```bash
docker compose logs -f
```

**Remove everything (including data):**
```bash
docker compose down -v
```

## Troubleshooting

### Prometheus shows "DOWN" for aki-detection target

1. Make sure port-forward is running: `kubectl --namespace=sharq port-forward deployment/aki-detection 8000:8000`
2. Check you can access metrics locally: `curl http://localhost:8000/metrics`
3. On Mac with Docker Desktop, `host.docker.internal` should work. If not, find your local IP and update `grafana/prometheus.yml`

### Can't access Grafana at localhost:3000

1. Check if containers are running: `docker ps`
2. Check if port 3000 is already in use: `lsof -i :3000`
3. Check logs: `docker compose logs grafana`

### No data showing in Grafana

1. Wait a minute for Prometheus to scrape metrics
2. Check Prometheus targets are UP: http://localhost:9090/targets
3. In Grafana, check the time range picker (top right) - set to "Last 15 minutes"
4. Try a simple query first: `aki_messages_received_total`

## Dashboard Template

Here's a sample dashboard JSON you can import:

1. In Grafana, click **+** → **Import**
2. Paste the JSON from `dashboard-template.json` (if created)
3. Click **Load**

## Alternative: Import Pre-built Dashboard

If you want a quick start, you can import community dashboards:
1. Go to https://grafana.com/grafana/dashboards/
2. Search for "Prometheus" dashboards
3. Import by ID in Grafana

## Cluster Grafana (if available)

Your course might have a cluster-wide Grafana instance. Ask your instructors for:
- Grafana URL
- Login credentials
- Whether your ServiceMonitor is already configured

The cluster Grafana would be better because:
- Already configured with the cluster Prometheus
- No need to run local services
- Persistent dashboards shared across restarts
