#!/bin/bash
echo "=== Deepfake Detection Sprint3 Auto Runner ==="

# 1. Python venv
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train (creates .pth, metrics, log)
python ml/training/train_resnet_finetune.py

# 4. Start backend
echo "Starting backend..."
cd backend
npm install
nohup node index.js > ../backend.log 2>&1 &
cd ..

# 5. Start frontend
echo "Starting frontend..."
cd frontend
npm install
npm start