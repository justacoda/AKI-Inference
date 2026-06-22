from prometheus_client import Counter, Gauge, Histogram, start_http_server
import logging

system_up = Gauge(
    'aki_system_up',
    '1 if the system is running and connected to MLLP, 0 otherwise'
)

mllp_up = Gauge(
    'aki_mllp_up',
    '1 if the MLLP connection is currently active, 0 otherwise'
)

pager_up = Gauge(
    'aki_pager_up',
    '1 if the last pager request succeeded, 0 if it failed'
)

messages_received_total = Counter(
    'aki_messages_received_total',
    'Total number of HL7 MLLP messages received',
    ['message_type']
)

blood_tests_received_total = Counter(
    'aki_blood_tests_received_total',
    'Total number of creatinine blood tests processed (ORU^R01 messages)'
)

positive_predictions_total = Counter(
    'aki_positive_predictions_total',
    'Total number of positive AKI predictions made by the model'
)

negative_predictions_total = Counter(
    'aki_negative_predictions_total',
    'Total number of negative AKI predictions made by the model'
)

processing_latency_seconds = Histogram(
    'aki_processing_latency_seconds',
    'Time to fully process a message from receipt to ACK sent',
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, float('inf')]
)

pager_failures_total = Counter(
    'aki_pager_failures_total',
    'Total number of failed pager HTTP requests (non-200 status or exceptions)',
    ['reason']
)

mllp_reconnections_total = Counter(
    'aki_mllp_reconnections_total',
    'Total number of MLLP socket reconnections due to connection failures'
)

creatinine_value_histogram = Histogram(
    'aki_creatinine_value_umol_l',
    'Distribution of creatinine test values in μmol/L',
    buckets=[50, 80, 100, 130, 150, 200, 250, 300, 400, 500, float('inf')]
)

active_patients_gauge = Gauge(
    'aki_active_patients',
    'Current number of active (admitted) patients in database'
)

model_inferences_total = Counter(
    'aki_model_inferences_total',
    'Total number of ML model inferences performed'
)

error_messages_total = Counter(
    'aki_error_messages_total',
    'Total number of error messages logged by the system',
    ['source']
)

class ErrorMessageCounterHandler(logging.Handler):
    """Logging handler that increments error_messages_total for every ERROR-level message."""
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            error_messages_total.labels(source=record.module).inc()

def start_metrics_server(port: int = 8000):
    """
     Metrics available at http://localhost:8000/metrics
    """
    try:
        start_http_server(port)
        logging.getLogger('SYSTEM').info(f"Prometheus metrics server started on port {port}")
    except OSError as e:
        logging.getLogger('SYSTEM').warning(f"Failed to start metrics server on port {port}: {e}")
    except Exception as e:
        logging.getLogger('SYSTEM').error(f"Unexpected error starting metrics server: {e}")
        raise
