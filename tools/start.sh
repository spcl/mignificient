#!/bin/bash

BUILD_DIR=$1
LOGS_DIR=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p ${LOGS_DIR}

${BUILD_DIR}/iox-roudi -c ${SCRIPT_DIR}/../config/roudi.toml > ${LOGS_DIR}/roudi.out 2>&1 &
ROUDI_PID=$!

while true; do

    if ! kill -0 ${ROUDI_PID} 2>/dev/null; then
        wait ${ROUDI_PID}
        exit_status=$?
        if [ $exit_status -ne 0 ]; then
            echo "Error: Background process iox-roudi (PID ${ROUDI_PID}) has exited with status $exit_status"
            exit $exit_status
        fi
    fi

    if tail -n 1 ${LOGS_DIR}/roudi.out | grep -q "RouDi is ready for clients"; then
        echo "RouDi is ready!"
        break
    fi
    sleep 0.1
done

echo ${ROUDI_PID} > ${LOGS_DIR}/roudi.pid
