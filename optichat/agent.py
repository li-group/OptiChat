import os
import logging

# Disable OpenTelemetry to avoid context management issues
os.environ["OTEL_SDK_DISABLED"] = "true"
# Suppress OpenTelemetry warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

from loguru import logger
logger.add("conversation_logs.txt", rotation="10 MB")

from optichat.sub_agents.root.agent import create_root_agent

root_agent = create_root_agent(workflow="default")