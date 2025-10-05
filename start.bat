@echo off
echo Starting ResumeRAG Application...

echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Installing Frontend dependencies...
cd frontend
npm install

echo.
echo Building Frontend...
npm run build
cd ..

echo.
echo Creating uploads directory...
if not exist "uploads" mkdir uploads

echo.
echo Starting FastAPI Server...
echo Server will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:8000/static/
echo API Documentation: http://localhost:8000/docs

python main.py
