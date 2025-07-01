#!/bin/bash
PYTHON_PROGRAM="code_gen_complex.py"

MAX_RETRIES=5

retries=0

TIMEOUT=3600

retries=0
while true; do
    python $PYTHON_PROGRAM \
        --timeout $TIMEOUT \
#### execution body ####
    # Check whether the program was successfully executed
    if [ $? -eq 0 ]; then
        break
    else
        retries=$((retries+1))
        if [ $retries -ge $MAX_RETRIES ]; then
            echo "Maximum number of retries reached. Exiting."
            break
        fi
        echo "Program failed. Retrying in 5 seconds..."
        sleep 5
    fi
done
echo "All operations completed."