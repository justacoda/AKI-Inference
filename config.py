import os

# MLLP
MLLP_ADDRESS = os.getenv('MLLP_ADDRESS', 'localhost:8440')
MLLP_HOST, MLLP_PORT = MLLP_ADDRESS.split(':')
MLLP_PORT = int(MLLP_PORT)

# Pager
PAGER_ADDRESS = os.getenv('PAGER_ADDRESS', 'localhost:8441')
PAGER_URL = f"http://{PAGER_ADDRESS}/page"

# Paths
MODEL_PATH = os.getenv('MODEL_PATH', 'model/aki_model.pkl')
DB_PATH = os.getenv('DB_PATH', 'db')
SCHEMA_PATH = os.getenv('SCHEMA_PATH', 'db/schema.sql')
HISTORY_PATH = os.getenv('HISTORY_PATH', '/data/history.csv')
AKI_GROUND_TRUTH_PATH = os.getenv('AKI_GROUND_TRUTH_PATH', 'data/aki.csv')

# Logs
SYSTEM_LOG_PATH = os.getenv('SYSTEM_LOG_PATH', '/state/logs/system.log')
MESSAGE_LOG_PATH = os.getenv('MESSAGE_LOG_PATH', '/state/logs/message.log')
