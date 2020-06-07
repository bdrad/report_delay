# Algorithmic Prediction of Delayed Radiology Turn-Around-Time during Non-Business Hours

### Repository accompanying manuscript


# How to use this Repo
## Installation Instructions

1. Clone this repository

  `git clone https://github.com/bdrad/report_delay.git`


2. Install dependencies (install via pip or conda)

 * numpy
 * pandas
 * sklearn
 * matplotlib
 * xgboost
 * argparse


3. Install Rad Classify (utilized for preprocessing steps & fastText wrapper): https://github.com/bdrad/rad_classify

## Data
Users will need to provide their own training data in 2 parts.

### Main Data
The following column headers should be used for the input data and saved as a .xlsx file.
 - "Minimum of Exam Ordered to Prelim/First Com": the interpretation time as seconds (float)
 - "Minimum of Exam Completed to Prelim/First Com": the total time as seconds (float)
 - "Report Text": raw clinical history (str)
 - "Patient Status": inpatient or emergency (str)
 - "Patient Status numerical": inpatient or emergency (int)
 - "Time of Day Label": evening, morning, afternoon, late night (str)
 - "Time of Day Label numerical": time of day (int)
 - "Body Part Label numerical": body part examined (int)
 - "Preliminary Report By": Trainee performing the report (str)
 - "Preliminary Report By numerical": integer
 - "Preliminary Report Date": datetime
 - "Point of Care": the hospital campus/facility (str)
 - "Exam Code": study description (str)

### Secondary Data for PGY levels
With the following columns, create a key 'trainees.xlsx'. __Place this file in the root directory.__

 - "Preliminary Report By": string
 - "PGY": integer



## Training and evaluation script
`train_evaluation.py` contain the training and test set evaluation scripts for their respective classifier.

To run the script:

  `python train_evaluation.py --data_path PATH_TO_DATA --time_delay CHOOSE_TIME_DELAY`

  - `CHOOSE_TIME_DELAY` must be `interpretation_time` or  `total_time`
  - `PATH_TO_DATA` must be a relative path to the the main data .xlsx file
