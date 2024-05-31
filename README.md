# Image captioning with encoder cnn & decoder lstm with attention network.

## How to run (Windows)

1. Install miniconda
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```
2. Install python 3.9
- Open Anaconda Prompt (miniconda3)
```bash
cd /path/to/project
conda create --name your_env_name python=3.9
conda activate your_env_name
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run app
```bash
streamlit run main.py
```
