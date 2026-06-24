@echo off
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ============================================
echo   NLP Sentiment Classifier - Startup
echo ============================================
echo.

:: Check Python is installed and in PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Please install Python 3.8+ from https://www.python.org and try again.
    pause
    exit /b 1
)

echo [INFO] Python found:
python --version
echo.

:: Check both datasets are present
if not exist "%PROJECT_DIR%IMDB_Dataset.csv" (
    echo [ERROR] Dataset not found: IMDB_Dataset.csv
    echo         Please download the IMDB dataset and place it in the project folder.
    echo         Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    echo.
    pause
    exit /b 1
)

echo [INFO] IMDB dataset found: IMDB_Dataset.csv

if not exist "%PROJECT_DIR%tweets_dataset_kaggle.csv" (
    echo [ERROR] Dataset not found: tweets_dataset_kaggle.csv
    echo         Please download the tweets dataset and place it in the project folder.
    echo         Source: https://www.kaggle.com/datasets/kazanova/sentiment140
    echo.
    pause
    exit /b 1
)

echo [INFO] Tweets dataset found: tweets_dataset_kaggle.csv
echo.

:: Check if venv exists
if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo [INFO] Found existing virtual environment.
    goto :activate
)

:: venv not found - create it and install dependencies
echo [INFO] No virtual environment found. Creating one...
python -m venv venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo         Make sure Python is installed correctly and try again.
    pause
    exit /b 1
)

echo [INFO] Virtual environment created. Installing dependencies...
echo [INFO] Note: PyTorch and HuggingFace Transformers may take several minutes to install.
echo [INFO]       DistilBERT model weights (~250MB) will download automatically on first run.
echo.
call "%PROJECT_DIR%venv\Scripts\activate.bat"
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

:: Download required NLTK data
echo.
echo [INFO] Downloading required NLTK data packages...
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"

echo.
echo [INFO] Dependencies installed successfully.
goto :run

:activate
echo [INFO] Activating virtual environment...
call "%PROJECT_DIR%venv\Scripts\activate.bat"
echo [INFO] Virtual environment activated.
echo.

:run
echo [INFO] Starting Jupyter Notebook...
echo [INFO] The notebook will open in your browser automatically.
echo [INFO] Press Ctrl+C in this window to stop Jupyter.
echo.
jupyter notebook V3_nlp_sentiment_classifier.ipynb

echo.
echo [INFO] Jupyter stopped.
pause
