# GCCR002

Analysis and code for the paper "Recent smell loss is the best predictor of COVID-19: a preregistered, cross-sectional study". 

To reproduce the analysis, we suggest to useing conda to match the dependencies used with the project. Please follow these instructions:

### Using Conda:

1. Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html)
2. Save (and extract the `.zip`) or clone this repository
3. Create a Python environment with all the necessary packages:
   * Open a terminal and navigate to the GCCR002 repository you've just saved
   * Run the following command: `conda env create -f environment.yml`
4. Activate the environment you've just created:  
   Run the following command: `conda activate gccr002`
5. Execute the notebook by either:
  - Running the shell command: `jupyter-notebook` which should open your webbrowser, then in the Jupyter web application run the `main.ipynb` notebook.
  - *Or* running the shell command: `jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=99999 --execute main.ipynb` to bypass the browser and execute the notebook programatically.
