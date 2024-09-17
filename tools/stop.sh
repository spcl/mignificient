#!/bin/bash

# Check if a directory parameter was provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a directory as a parameter."
    exit 1
fi

# Get the directory from the first parameter
dir="$1"

# Check if the directory exists
if [ ! -d "$dir" ]; then
    echo "Error: The directory '$dir' does not exist."
    exit 1
fi

# Path to the PID file
pid_file="$dir/roudi.pid"

# Check if the PID file exists
if [ ! -f "$pid_file" ]; then
    echo "Error: PID file '$pid_file' not found."
    exit 1
fi

# Read the PID from the file
pid=$(cat "$pid_file")

# Check if the process exists
if ! kill -0 $pid 2>/dev/null; then
    echo "Error: Process with PID $pid does not exist."
    exit 1
fi

# Attempt to kill the process gently (SIGTERM)
echo "Sending SIGTERM to process $pid..."
kill -15 $pid

# Wait for the process to terminate (up to 10 seconds)
for i in {1..10}; do
    if ! kill -0 $pid 2>/dev/null; then
        echo "Process $pid has been terminated."
        exit 0
    fi
    sleep 1
done

echo "Process $pid did not terminate within 10 seconds."
echo "Force kill with SIGKILL"
kill -9 $pid
