# Muscle electromyographic activity normalized to maximal muscle activity, not to M<sub>max</sub>, better represents voluntary activation

Joanna Diong<sup>1,2</sup>
Kenzo Kishimoto<sup>3</sup>
Jane E Butler<sup>2,4</sup>
Martin E Héroux<sup>2,4</sup>

<sup>1</sup>School of Medical Sciences, Faculty of Medicine and Health, The University of Sydney, New South Wales, Australia \
<sup>2</sup>Neuroscience Research Australia (NeuRA), Sydney, New South Wales, Australia \
<sup>3</sup>School of Health Sciences, Faculty of Medicine and Health, The University of Sydney, New South Wales, Australia \
<sup>4</sup>School of Medical Sciences, University of New South Wales, New South Wales, Australia  


## Suggested citation

Diong J, Kishimoto K, Butler JE, Héroux ME (2022) Muscle electromyographic activity normalized to maximal muscle activity,
not to M<sub>max</sub>, better represents voluntary activation. PLOS One (in press).

## Protocol registration

The protocol for this study was registered on the Open Science Framework (OSF): [https://osf.io/7f3nk][rego]

This study is a secondary analysis of data from a related study, which is under the same registered protocol. \
The description, data and code of the related study are available from the GitHub repository:
[https://github.com/joannadiong/Kishimoto_et_al_2021_JAP][Kishimoto]

## Data

Raw data to generate Fig 1 are available from the **data -> raw** folder in the zipped folder "sub22.zip" in this GitHub repository in these formats:
* Spike2 .smr 
* Matlab .mat 
* Text .txt

Unzip the subject data folder into this location. 

Processed data to generate Figs 2 and 3 are available from the **data -> proc** folder as the files: 
* subjects_data.csv
* subjects_data.json
* subjects_times_mvc.csv

## Code

Python code files (Python v3.8) were written by Joanna Diong with input from Martin Héroux.

### Python files

`script_proc`: Main script to run analysis.

`fig-1`: Script to generate PNG and SVG files of single participant trial data (Fig 2), saved in **data -> proc -> sub22**

`process`, `utilities`, `trials_key`: Modules containing functions used to clean data and plot figures. 

### External dependency

`process` calls the deprecated Python package `spike2py` written by Martin Héroux.
The deprecated package is bundled with this repo as "spike2py.zip"
Download the package, save it in a location outside of this project folder,
and point the Python interpreter (i.e. add the root) towards that location.

(For the new and revised packaged, see the [spike2py GitHub page][spike2py].)

### Running Python code

A reliable way to reproduce the analysis would be to run the code in an integrated development environment for Python (e.g. [PyCharm][pycharm]). 

Create a virtual environment and install dependencies. Using the Terminal (Mac or Linux, or PyCharm Terminal), 

```bash 
python -m venv env
```
Next, activate the virtual environment. 

For Mac or Linux, 

```bash
source env/bin/activate
```

For Windows, 

```bash
.\env\Scripts\activate
```

Then, install dependencies,

```bash
pip install -r requirements.txt
```

Download all code files and data.zip into a single folder.
Unzip the data file into the same location.
Download the `spike2py` package. Point the Python interpreter to the location of `spike2py`.

Run each file separately: 

1. `script_proc.py`
2. `fig-1.py`

## Output

Table 1. is generated using data from "results.txt", saved in **data -> proc**

Table 2. is generated using data from "subjects_data_describe.csv", saved in **data -> proc**

Fig 1. is generated using data from sub22, and saved in **data -> proc -> sub22**

Fig 2. is generated using data from "subjects_data.csv", saved as PNG and SVG files labelled "emg_torque" in **data -> proc**

Fig 3. is generated using data from "subjects_data.csv", saved as PNG and SVG files labelled "emg_normalised" in **data -> proc**


[rego]: https://osf.io/7f3nk
[Kishimoto]: https://github.com/joannadiong/Kishimoto_et_al_2021_JAP
[osf]: https://osf.io/wt7z8/ 
[spike2py]: https://github.com/MartinHeroux/spike2py
[pycharm]: https://www.jetbrains.com/pycharm/promo/?gclid=Cj0KCQiAtqL-BRC0ARIsAF4K3WFahh-pzcvf6kmWnmuONEZxi544-Ty-UUqKa4EelnOxa5pAC9C4_d4aAisxEALw_wcB 
