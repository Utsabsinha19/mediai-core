frontend setup:

cd frontend
npm install
cp .env.development .env
npm run dev

ML Training
cd ml-training
pip install -r requirements.txt
python scripts/train_all_models.py


# Production deployment
bash deployment/scripts/deploy.sh

# Start services
bash deployment/scripts/start_backend.sh
bash deployment/scripts/start_frontend.sh































cd ml-training
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/train_all_models.py