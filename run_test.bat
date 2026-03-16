@echo off
set PYTHON_EXE=C:\Program Files\Python310\python.exe
set ENV_DIR=d:\pythonProject\IC Lab\Gait_analysis\pyskl\colab\colab_env_310

if not exist "%ENV_DIR%" (
    "%PYTHON_EXE%" -m venv "%ENV_DIR%"
)

call "%ENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r "d:\pythonProject\IC Lab\Gait_analysis\pyskl\colab\requirements.txt"
python "d:\pythonProject\IC Lab\Gait_analysis\pyskl\colab\run_experiment.py" --config "d:\pythonProject\IC Lab\Gait_analysis\pyskl\colab\experiment_config_fast_test.yaml" > test_output.log 2>&1
echo DONE
