#!/bin/bash

# Glass Box Option Strategist - Startup Script

echo "================================================"
echo "  Glass Box Option Strategist"
echo "================================================"
echo ""
echo "Starting backend server..."
echo ""

cd "$(dirname "$0")/backend"
python app.py
