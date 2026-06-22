#!/usr/bin/env python3
"""
Simple script to view Prometheus metrics in a readable format.
Usage: python view_metrics.py [metric_prefix]

Examples:
  python view_metrics.py              # Show all AKI metrics
  python view_metrics.py aki_messages # Show only message metrics
"""

import urllib.request
import sys
from collections import defaultdict

def fetch_metrics(url="http://localhost:8000/metrics"):
    """Fetch metrics from Prometheus endpoint"""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        print("Make sure port-forward is running: kubectl --namespace=sharq port-forward deployment/aki-detection 8000:8000")
        sys.exit(1)

def parse_metrics(content, prefix="aki_"):
    """Parse and organize metrics"""
    metrics = defaultdict(lambda: {"help": "", "type": "", "values": []})
    current_metric = None

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            if line.startswith('# HELP'):
                parts = line.split(' ', 3)
                if len(parts) >= 4:
                    metric_name = parts[2]
                    if metric_name.startswith(prefix):
                        current_metric = metric_name
                        metrics[metric_name]["help"] = parts[3]
            elif line.startswith('# TYPE'):
                parts = line.split(' ', 3)
                if len(parts) >= 4:
                    metric_name = parts[2]
                    if metric_name.startswith(prefix):
                        metrics[metric_name]["type"] = parts[3]
        elif line and current_metric and line.startswith(current_metric):
            # Skip _created metrics for cleaner output
            if "_created{" not in line and not line.startswith(current_metric + "_created "):
                metrics[current_metric]["values"].append(line)

    return metrics

def format_value(line):
    """Extract and format metric value"""
    parts = line.split()
    if len(parts) >= 2:
        metric_with_labels = parts[0]
        value = parts[1]

        # Parse metric name and labels
        if '{' in metric_with_labels:
            metric_name, labels = metric_with_labels.split('{', 1)
            labels = labels.rstrip('}')
            return f"  {labels}: {value}"
        else:
            return f"  {value}"
    return line

def display_metrics(metrics):
    """Display metrics in a readable format"""
    if not metrics:
        print("No metrics found!")
        return

    print("=" * 80)
    print("AKI DETECTION METRICS")
    print("=" * 80)
    print()

    # Sort metrics by name
    for metric_name in sorted(metrics.keys()):
        metric = metrics[metric_name]

        # Skip _created metrics
        if metric_name.endswith('_created'):
            continue

        print(f"📊 {metric_name}")
        print(f"   Type: {metric['type']}")
        if metric['help']:
            print(f"   Description: {metric['help']}")

        if metric['values']:
            print("   Values:")
            for value in metric['values']:
                print(format_value(value))
        else:
            print("   No values recorded yet")

        print()

def main():
    # Get prefix from command line or use default
    prefix = sys.argv[1] if len(sys.argv) > 1 else "aki_"

    print(f"Fetching metrics from http://localhost:8000/metrics...")
    print(f"Filtering for metrics starting with '{prefix}'")
    print()

    content = fetch_metrics()
    metrics = parse_metrics(content, prefix)
    display_metrics(metrics)

    # Show summary
    total_metrics = len([m for m in metrics.keys() if not m.endswith('_created')])
    print("=" * 80)
    print(f"Total metrics: {total_metrics}")
    print("=" * 80)

if __name__ == "__main__":
    main()
