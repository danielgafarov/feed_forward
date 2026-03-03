#!/bin/bash
cd $(dirname "${BASH_SOURCE}")
source ./venv/bin/activate
echo "$(python3 feed_forward.py)"