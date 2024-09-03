# Creating Python virtual environment for Jupyter Notebook

## On Linux 

```
python3 -m venv myenv
source myenv/bin/activate
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
pip install -r requirements.txt
jupyter notebook
```
