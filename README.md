# GCCR002

Code for the paper "[Recent smell loss is the best predictor of COVID-19: a preregistered, cross-sectional study](https://doi.org/10.1093/chemse/bjaa081)". by Gerkin et al 2020.


## Getting Started (using conda):

To reproduce the analysis, we suggest using conda to match the dependencies used with the project. 

Please follow these instructions:

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


## Citation

Please use the following bibtex entry:

```
@article{10.1093/chemse/bjaa081,
    author = {Gerkin, Richard C and Ohla, Kathrin and Veldhuizen, Maria G and Joseph, Paule V and Kelly, Christine E and Bakke, Alyssa J and Steele, Kimberley E and Farruggia, Michael C and Pellegrino, Robert and Pepino, Marta Y and Bouysset, Cédric and Soler, Graciela M and Pereda-Loth, Veronica and Dibattista, Michele and Cooper, Keiland W and Croijmans, Ilja and Di Pizio, Antonella and Ozdener, M Hakan and Fjaeldstad, Alexander W and Lin, Cailu and Sandell, Mari A and Singh, Preet B and Brindha, V Evelyn and Olsson, Shannon B and Saraiva, Luis R and Ahuja, Gaurav and Alwashahi, Mohammed K and Bhutani, Surabhi and D’Errico, Anna and Fornazieri, Marco A and Golebiowski, Jérôme and Hwang, Liang-Dar and Öztürk, Lina and Roura, Eugeni and Spinelli, Sara and Whitcroft, Katherine L and Faraji, Farhoud and Fischmeister, Florian PhS and Heinbockel, Thomas and Hsieh, Julien W and Huart, Caroline and Konstantinidis, Iordanis and Menini, Anna and Morini, Gabriella and Olofsson, Jonas K and Philpott, Carl M and Pierron, Denis and Shields, Vonnie D C and Voznessenskaya, Vera V and Albayay, Javier and Altundag, Aytug and Bensafi, Moustafa and Bock, María Adelaida and Calcinoni, Orietta and Fredborg, William and Laudamiel, Christophe and Lim, Juyun and Lundström, Johan N and Macchi, Alberto and Meyer, Pablo and Moein, Shima T and Santamaría, Enrique and Sengupta, Debarka and Dominguez, Paloma Rohlfs and Yanik, Hüseyin and Hummel, Thomas and Hayes, John E and Reed, Danielle R and Niv, Masha Y and Munger, Steven D and Parma, Valentina and GCCR Group Author },
    title = "{Recent smell loss is the best predictor of COVID-19 among individuals with recent respiratory symptoms}",
    journal = {Chemical Senses},
    year = {2020},
    month = {12},
    abstract = "{In a preregistered, cross-sectional study we investigated whether olfactory loss is a reliable predictor of COVID-19 using a crowdsourced questionnaire in 23 languages to assess symptoms in individuals self-reporting recent respiratory illness. We quantified changes in chemosensory abilities during the course of the respiratory illness using 0-100 visual analog scales (VAS) for participants reporting a positive (C19+; n=4148) or negative (C19-; n=546) COVID-19 laboratory test outcome. Logistic regression models identified univariate and multivariate predictors of COVID-19 status and post-COVID-19 olfactory recovery. Both C19+ and C19- groups exhibited smell loss, but it was significantly larger in C19+ participants (mean±SD, C19+: -82.5±27.2 points; C19-: -59.8±37.7). Smell loss during illness was the best predictor of COVID-19 in both univariate and multivariate models (ROC AUC=0.72). Additional variables provide negligible model improvement. VAS ratings of smell loss were more predictive than binary chemosensory yes/no-questions or other cardinal symptoms (e.g., fever). Olfactory recovery within 40 days of respiratory symptom onset was reported for ~50\\% of participants and was best predicted by time since respiratory symptom onset. We find that quantified smell loss is the best predictor of COVID-19 amongst those with symptoms of respiratory illness. To aid clinicians and contact tracers in identifying individuals with a high likelihood of having COVID-19, we propose a novel 0-10 scale to screen for recent olfactory loss, the ODoR-19. We find that numeric ratings ≤2 indicate high odds of symptomatic COVID-19 (4\\&lt;OR\\&lt;10). Once independently validated, this tool could be deployed when viral lab tests are impractical or unavailable.}",
    issn = {1464-3553},
    doi = {10.1093/chemse/bjaa081},
    url = {https://doi.org/10.1093/chemse/bjaa081},
    note = {bjaa081},
    eprint = {https://academic.oup.com/chemse/advance-article-pdf/doi/10.1093/chemse/bjaa081/35137502/bjaa081.pdf},
}


```


Thank you! Please don't hesitate to contact us if you have any questions  


