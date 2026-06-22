# AKI Detection - Local Monitoring Setup

View your Kubernetes pod metrics locally using Prometheus and Grafana.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `kubectl` configured (you should already have this from CW4)

---

## Step 1: Start Port-Forward

This connects your local machine to the metrics running in the Kubernetes pod.

Open a terminal and run (keep this running the whole time):
```bash
kubectl --namespace=sharq port-forward deployment/aki-detection 8000:8000
```

To verify it's working:
```bash
curl http://localhost:8000/metrics
```
You should see a wall of Prometheus metrics text.

---

## Step 2: Start Grafana & Prometheus

Open a **second terminal**, navigate to the `grafana/` folder and start the containers:
```bash
cd grafana
docker compose up -d
```

Check that both containers are running:
```bash
docker ps
```
You should see `prometheus` and `grafana` listed.

---

## Step 3: Verify Prometheus is Scraping

Open your browser and go to:
```
http://localhost:9090/targets
```

You should see `aki-detection` with status **UP** (green).

> If it shows **DOWN**, wait 30 seconds and refresh. Make sure Step 1 is still running.

---

## Step 4: Open Grafana

Go to:
```
http://localhost:3000
```

Login with:
- **Username:** `admin`
- **Password:** `admin`

Skip the password change prompt if asked.

---

## Step 5: Import the Dashboard

1. Click the **"+"** icon (top right) → **Import dashboard**
2. Click **"Upload dashboard JSON file"**
3. Select `grafana/aki-dashboard.json` from this repo
4. Select **Prometheus** as the data source when prompted
5. Click **Import**

You should now see the AKI Detection dashboard with all panels! 🎉

---

## Stopping

When you're done, stop the containers:
```bash
docker compose down
```

And press `Ctrl+C` in the terminal running port-forward.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker compose` not found | Use `docker compose` not `docker-compose` |
| Docker error about daemon | Open Docker Desktop and wait for it to start |
| Prometheus target is DOWN | Make sure port-forward (Step 1) is still running |
| No data in Grafana | Check Prometheus targets at `http://localhost:9090/targets` |
| Port 3000 already in use | Run `lsof -i :3000` to see what's using it |
