#!/bin/bash

# Function to cleanup background processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Start ADK API Server in background
echo "Starting ADK API Server..."
adk api_server &
ADK_PID=$!

# Wait for API server to be ready (simple sleep for now, could be more robust)
echo "Waiting for API Server to initialize..."
sleep 5

# Start Streamlit App
echo "Starting Streamlit App..."
streamlit run app.py

# Wait for background processes
wait $ADK_PID
