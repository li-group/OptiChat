import os
import logging

# Disable OpenTelemetry to avoid context management issues
os.environ["OTEL_SDK_DISABLED"] = "true"
# Suppress OpenTelemetry warnings
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

from optichat.sub_agents.root.agent import create_root_agent

root_agent = create_root_agent(workflow="default")