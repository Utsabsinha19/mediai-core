#!/bin/bash

# Healthcare AI Platform - Project Structure Creator
# This script creates all folders and files for the AI-powered healthcare application

echo "🏥 Creating Healthcare AI Platform Project Structure..."
echo "========================================================"

# Project root
PROJECT_ROOT="healthcare-ai-platform"
mkdir -p $PROJECT_ROOT
cd $PROJECT_ROOT

# Create backend structure
echo "📁 Creating backend structure..."
mkdir -p backend/{api,core,models,schemas,routes,services,tasks,tests,utils,logs,static/{uploads,reports,temp}}
mkdir -p backend/ml/{models,trainers,data,utils,weights}

# Create backend files
touch backend/api/__init__.py
touch backend/api/dependencies.py
touch backend/api/middleware.py
touch backend/core/__init__.py
touch backend/core/config.py
touch backend/core/security.py
touch backend/core/database.py
touch backend/core/rate_limiter.py
touch backend/models/__init__.py
touch backend/models/user.py
touch backend/models/report.py
touch backend/models/prediction_history.py
touch backend/models/model_metrics.py
touch backend/schemas/__init__.py
touch backend/schemas/auth.py
touch backend/schemas/prediction.py
touch backend/schemas/report.py
touch backend/schemas/admin.py
touch backend/routes/__init__.py
touch backend/routes/auth.py
touch backend/routes/prediction.py
touch backend/routes/reports.py
touch backend/routes/admin.py
touch backend/routes/health.py
touch backend/services/__init__.py
touch backend/services/auth_service.py
touch backend/services/prediction_service.py
touch backend/services/report_service.py
touch backend/services/email_service.py
touch backend/services/storage_service.py
touch backend/tasks/__init__.py
touch backend/tasks/celery_app.py
touch backend/tasks/prediction_tasks.py
touch backend/tasks/email_tasks.py
touch backend/tasks/report_tasks.py
touch backend/tests/__init__.py
touch backend/tests/test_auth.py
touch backend/tests/test_prediction.py
touch backend/tests/test_reports.py
touch backend/tests/test_ml_models.py
touch backend/tests/conftest.py
touch backend/utils/__init__.py
touch backend/utils/pdf_generator.py
touch backend/utils/image_processor.py
touch backend/utils/validators.py
touch backend/utils/logger.py
touch backend/utils/helpers.py
touch backend/utils/recommendations.py
touch backend/utils/constants.py
touch backend/ml/__init__.py
touch backend/ml/models/__init__.py
touch backend/ml/models/resnet50.py
touch backend/ml/models/efficientnet.py
touch backend/ml/models/densenet.py
touch backend/ml/models/xgboost_model.py
touch backend/ml/models/random_forest.py
touch backend/ml/models/lightgbm.py
touch backend/ml/models/ensemble.py
touch backend/ml/trainers/__init__.py
touch backend/ml/trainers/base_trainer.py
touch backend/ml/trainers/cnn_trainer.py
touch backend/ml/trainers/ml_trainer.py
touch backend/ml/trainers/ensemble_trainer.py
touch backend/ml/data/__init__.py
touch backend/ml/data/dataset_loader.py
touch backend/ml/data/data_preprocessor.py
touch backend/ml/data/augmentation.py
touch backend/ml/data/feature_extractor.py
touch backend/ml/utils/__init__.py
touch backend/ml/utils/shap_explainer.py
touch backend/ml/utils/lime_explainer.py
touch backend/ml/utils/grad_cam.py
touch backend/ml/utils/metrics_calculator.py
touch backend/ml/utils/model_selector.py
touch backend/logs/app.log
touch backend/logs/error.log
touch backend/logs/access.log
touch backend/requirements.txt
touch backend/requirements-dev.txt
touch backend/.env.example
touch backend/.env.production
touch backend/.env.staging
touch backend/.gitignore
touch backend/celery_worker.py
touch backend/main.py
touch backend/README.md

# Create frontend structure
echo "📁 Creating frontend structure..."
mkdir -p frontend/public/assets/{images,fonts,icons}
mkdir -p frontend/src/{components,pages,hooks,services,context,utils,styles,assets}
mkdir -p frontend/src/components/{common,prediction,reports,admin,charts}
mkdir -p frontend/src/pages/{auth,dashboard,admin,public}
mkdir -p frontend/src/styles/themes
mkdir -p frontend/tests/{components,pages,services,utils}
mkdir -p frontend/public/assets/{images,fonts,icons}

# Create frontend component files
touch frontend/src/components/common/Navbar.jsx
touch frontend/src/components/common/Footer.jsx
touch frontend/src/components/common/Button.jsx
touch frontend/src/components/common/Input.jsx
touch frontend/src/components/common/Modal.jsx
touch frontend/src/components/common/Card.jsx
touch frontend/src/components/common/Spinner.jsx
touch frontend/src/components/common/Alert.jsx
touch frontend/src/components/common/Breadcrumb.jsx
touch frontend/src/components/common/Tabs.jsx

# Prediction components
touch frontend/src/components/prediction/ImageUploader.jsx
touch frontend/src/components/prediction/PredictionResults.jsx
touch frontend/src/components/prediction/ModelComparisonTable.jsx
touch frontend/src/components/prediction/MetricsChart.jsx
touch frontend/src/components/prediction/ConfidenceScore.jsx
touch frontend/src/components/prediction/SHAPVisualization.jsx
touch frontend/src/components/prediction/GradCAMViewer.jsx

# Reports components
touch frontend/src/components/reports/ReportCard.jsx
touch frontend/src/components/reports/ReportViewer.jsx
touch frontend/src/components/reports/ReportHistory.jsx
touch frontend/src/components/reports/PDFDownloadButton.jsx
touch frontend/src/components/reports/ReportFilters.jsx

# Admin components
touch frontend/src/components/admin/UserManagement.jsx
touch frontend/src/components/admin/SystemAnalytics.jsx
touch frontend/src/components/admin/ModelMonitor.jsx
touch frontend/src/components/admin/ReportAnalytics.jsx
touch frontend/src/components/admin/SettingsPanel.jsx

# Charts components
touch frontend/src/components/charts/LineChart.jsx
touch frontend/src/components/charts/BarChart.jsx
touch frontend/src/components/charts/PieChart.jsx
touch frontend/src/components/charts/RadarChart.jsx
touch frontend/src/components/charts/Heatmap.jsx

# Auth pages
touch frontend/src/pages/auth/Login.jsx
touch frontend/src/pages/auth/Signup.jsx
touch frontend/src/pages/auth/ForgotPassword.jsx
touch frontend/src/pages/auth/ResetPassword.jsx
touch frontend/src/pages/auth/VerifyEmail.jsx

# Dashboard pages
touch frontend/src/pages/dashboard/UserDashboard.jsx
touch frontend/src/pages/dashboard/UploadImage.jsx
touch frontend/src/pages/dashboard/PredictionResults.jsx
touch frontend/src/pages/dashboard/ReportHistory.jsx
touch frontend/src/pages/dashboard/ProfileSettings.jsx
touch frontend/src/pages/dashboard/Notifications.jsx

# Admin pages
touch frontend/src/pages/admin/AdminDashboard.jsx
touch frontend/src/pages/admin/UserManagement.jsx
touch frontend/src/pages/admin/ReportManagement.jsx
touch frontend/src/pages/admin/SystemAnalytics.jsx
touch frontend/src/pages/admin/ModelPerformance.jsx
touch frontend/src/pages/admin/AuditLogs.jsx
touch frontend/src/pages/admin/SystemSettings.jsx

# Public pages
touch frontend/src/pages/public/LandingPage.jsx
touch frontend/src/pages/public/AboutUs.jsx
touch frontend/src/pages/public/ContactUs.jsx
touch frontend/src/pages/public/HowItWorks.jsx
touch frontend/src/pages/public/FAQ.jsx
touch frontend/src/pages/public/PrivacyPolicy.jsx
touch frontend/src/pages/public/TermsOfService.jsx

# Hooks
touch frontend/src/hooks/useAuth.js
touch frontend/src/hooks/usePrediction.js
touch frontend/src/hooks/useReports.js
touch frontend/src/hooks/useLocalStorage.js
touch frontend/src/hooks/useDebounce.js
touch frontend/src/hooks/useMediaQuery.js

# Services
touch frontend/src/services/api.js
touch frontend/src/services/authService.js
touch frontend/src/services/predictionService.js
touch frontend/src/services/reportService.js
touch frontend/src/services/adminService.js
touch frontend/src/services/storageService.js

# Context
touch frontend/src/context/AuthContext.jsx
touch frontend/src/context/ThemeContext.jsx
touch frontend/src/context/NotificationContext.jsx

# Utils
touch frontend/src/utils/constants.js
touch frontend/src/utils/helpers.js
touch frontend/src/utils/validators.js
touch frontend/src/utils/formatters.js
touch frontend/src/utils/errorHandler.js

# Styles
touch frontend/src/styles/globals.css
touch frontend/src/styles/variables.css
touch frontend/src/styles/animations.css
touch frontend/src/styles/themes/light.css
touch frontend/src/styles/themes/dark.css

# Root frontend files
touch frontend/src/App.jsx
touch frontend/src/AppRoutes.jsx
touch frontend/src/main.jsx
touch frontend/src/index.css
touch frontend/src/vite-env.d.ts
touch frontend/package.json
touch frontend/package-lock.json
touch frontend/vite.config.js
touch frontend/vitest.config.js
touch frontend/tailwind.config.js
touch frontend/postcss.config.js
touch frontend/.env.development
touch frontend/.env.production
touch frontend/.env.staging
touch frontend/.eslintrc.js
touch frontend/.prettierrc
touch frontend/index.html
touch frontend/README.md

# Frontend public assets
touch frontend/public/index.html
touch frontend/public/favicon.ico
touch frontend/public/robots.txt
touch frontend/public/manifest.json
touch frontend/public/assets/logo.svg
touch frontend/public/assets/hero-image.png

# Create ML training structure
echo "📁 Creating ML training structure..."
mkdir -p ml-training/notebooks
mkdir -p ml-training/scripts
mkdir -p ml-training/configs
mkdir -p ml-training/data/{raw,processed,augmented,features}

# ML training files
touch ml-training/notebooks/01_data_exploration.ipynb
touch ml-training/notebooks/02_data_preprocessing.ipynb
touch ml-training/notebooks/03_resnet_training.ipynb
touch ml-training/notebooks/04_cnn_models_training.ipynb
touch ml-training/notebooks/05_ml_models_training.ipynb
touch ml-training/notebooks/06_ensemble_creation.ipynb
touch ml-training/notebooks/07_model_evaluation.ipynb
touch ml-training/scripts/download_datasets.py
touch ml-training/scripts/preprocess_data.py
touch ml-training/scripts/train_all_models.py
touch ml-training/scripts/evaluate_models.py
touch ml-training/scripts/optimize_hyperparameters.py
touch ml-training/scripts/create_ensemble.py
touch ml-training/scripts/export_models.py
touch ml-training/scripts/validate_models.py
touch ml-training/configs/resnet_config.yaml
touch ml-training/configs/efficientnet_config.yaml
touch ml-training/configs/xgboost_config.yaml
touch ml-training/configs/ensemble_config.yaml
touch ml-training/configs/training_config.yaml
touch ml-training/requirements.txt
touch ml-training/environment.yaml
touch ml-training/README.md

# Create database structure
echo "📁 Creating database structure..."
mkdir -p database/migrations
mkdir -p database/backups/{daily,weekly,monthly}
mkdir -p database/scripts

# Database files
touch database/migrations/001_create_users.sql
touch database/migrations/002_create_reports.sql
touch database/migrations/003_create_predictions.sql
touch database/migrations/004_create_model_metrics.sql
touch database/migrations/seed_data.sql
touch database/scripts/backup.sh
touch database/scripts/restore.sh
touch database/scripts/migrate.sh
touch database/scripts/seed.sh
touch database/mongodb_init.js
touch database/postgres_init.sql

# Create deployment structure
echo "📁 Creating deployment structure..."
mkdir -p deployment/monitoring
mkdir -p deployment/scripts

# Deployment files
touch deployment/monitoring/prometheus.yml
touch deployment/monitoring/logging.yml
touch deployment/scripts/deploy.sh
touch deployment/scripts/healthcheck.sh
touch deployment/scripts/start_backend.sh
touch deployment/scripts/start_frontend.sh
touch deployment/scripts/start_celery.sh
touch deployment/scripts/cleanup.sh
touch deployment/nginx.conf
touch deployment/supervisor.conf
touch deployment/gunicorn.conf.py
touch deployment/.env.production
touch deployment/README.md

# Create docs structure
echo "📁 Creating documentation structure..."
mkdir -p docs/api
mkdir -p docs/guides
mkdir -p docs/architecture
mkdir -p docs/user
mkdir -p docs/reports

# Documentation files
touch docs/api/openapi.yaml
touch docs/api/postman_collection.json
touch docs/api/api_documentation.md
touch docs/guides/installation.md
touch docs/guides/deployment.md
touch docs/guides/contribution.md
touch docs/guides/troubleshooting.md
touch docs/architecture/system_architecture.md
touch docs/architecture/database_schema.md
touch docs/architecture/ml_pipeline.md
touch docs/architecture/security.md
touch docs/user/user_guide.md
touch docs/user/admin_guide.md
touch docs/reports/model_performance.pdf
touch docs/reports/validation_report.pdf

# Create scripts structure
echo "📁 Creating scripts structure..."
mkdir -p scripts

# Script files
touch scripts/setup.sh
touch scripts/install_deps.sh
touch scripts/run_tests.sh
touch scripts/start_dev.sh
touch scripts/start_prod.sh
touch scripts/backup_data.sh
touch scripts/restore_data.sh
touch scripts/update_models.sh
touch scripts/cleanup.sh
chmod +x scripts/*.sh
chmod +x deployment/scripts/*.sh
chmod +x database/scripts/*.sh

# Create tests structure
echo "📁 Creating tests structure..."
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/performance

# Test files
touch tests/unit/test_auth.py
touch tests/unit/test_prediction.py
touch tests/unit/test_reports.py
touch tests/unit/test_ml_pipeline.py
touch tests/integration/test_api.py
touch tests/integration/test_database.py
touch tests/performance/load_test.py
touch tests/performance/locustfile.py
touch tests/conftest.py
touch tests/pytest.ini
touch tests/coverage.xml

# Create GitHub workflows
echo "📁 Creating GitHub workflows..."
mkdir -p .github/workflows

# Workflow files
touch .github/workflows/ci.yml
touch .github/workflows/test.yml
touch .github/workflows/security-scan.yml

# Root configuration files
echo "📁 Creating root configuration files..."
touch .gitignore
touch .gitattributes
touch .pre-commit-config.yaml
touch .env.example
touch .env.local
touch .eslintrc.json
touch .prettierrc.json
touch .flake8
touch .isort.cfg
touch .black
touch README.md
touch LICENSE
touch CONTRIBUTING.md
touch CODE_OF_CONDUCT.md
touch SECURITY.md
touch CHANGELOG.md
touch Makefile
touch pyproject.toml
touch setup.py
touch setup.cfg
touch requirements.txt
touch requirements-dev.txt
touch requirements-prod.txt
touch package.json
touch package-lock.json
touch vercel.json

# Create .gitignore content
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
*.log
*.pkl
*.pth
*.pt
*.onnx
*.joblib

# Uploads and temp files
backend/static/uploads/*
backend/static/reports/*
backend/static/temp/*
!backend/static/uploads/.gitkeep
!backend/static/reports/.gitkeep
!backend/static/temp/.gitkeep

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Frontend
node_modules/
dist/
dist-ssr/
*.local
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# ML Training
ml-training/data/raw/*
ml-training/data/processed/*
ml-training/data/augmented/*
ml-training/data/features/*
!ml-training/data/raw/.gitkeep
!ml-training/data/processed/.gitkeep
!ml-training/data/augmented/.gitkeep
!ml-training/data/features/.gitkeep

# Database backups
database/backups/*.sql
database/backups/*.dump

# Logs
logs/
*.log

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Jupyter Notebook
.ipynb_checkpoints

# Redis
dump.rdb

# Celery
celerybeat-schedule
celerybeat.pid

# Environment
.env.production
.env.staging
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=healthcare_ai
POSTGRES_URL=postgresql://user:password@localhost:5432/healthcare_db

# Security
SECRET_KEY=your-secret-key-here-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Email (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-password
EMAIL_FROM=noreply@healthcare-ai.com

# File Upload
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.dcm

# Model paths
RESNET_MODEL_PATH=backend/ml/weights/resnet50_best.pth
EFFICIENTNET_MODEL_PATH=backend/ml/weights/efficientnet_best.pth
XGBOOST_MODEL_PATH=backend/ml/weights/xgboost.json
RANDOM_FOREST_PATH=backend/ml/weights/random_forest.pkl
ENSEMBLE_MODEL_PATH=backend/ml/weights/ensemble.pkl

# API Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Frontend URL (CORS)
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Environment
ENVIRONMENT=development
DEBUG=True
EOF

# Create README.md
cat > README.md << 'EOF'
# 🏥 AI-Powered Healthcare Platform

## Medical Image Analysis & Disease Prediction System

### Features
- 🧠 Brain Tumor Detection from MRI scans
- 🫀 Chest X-ray Analysis (Pneumonia, COVID-19, Tuberculosis)
- 🩻 Skin Disease Classification
- 👁️ Eye Disease Detection

### AI Models
- ResNet-50, EfficientNet, DenseNet (Deep Learning)
- XGBoost, Random Forest, LightGBM (Machine Learning)
- Ensemble Learning for Best Performance
- SHAP & Grad-CAM Explainability

### Tech Stack
- **Backend**: FastAPI, Python 3.10+
- **Frontend**: React 18, Tailwind CSS, Vite
- **Database**: MongoDB, PostgreSQL
- **ML**: PyTorch, scikit-learn, XGBoost
- **Deployment**: Nginx, Gunicorn, Supervisor

### Quick Start

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
python main.py