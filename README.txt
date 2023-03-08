========== DEPENDENCIES ==========

Install dependencies by executing creating an new conda environment from environment.yml file:

$ conda env create -n ENVNAME --file environment.yml

Or create a new environment first then install dependencies using the requirements.txt file:

$ pip install -r requirements.txt

To train, use, or evaluate the weighted model (train_weighted_model.py) an environment should be created using environment2.yml or install dependencies from requirements2.txt

============ EXECUTION ===========

To test and evaluate the model on the test set provided, run "python evaluation.py" in an environment with all the dependencies installed.

To run entire project from scratch, the order of execution is:

1. nist_scraper.py
2. sdbs_scraper.py
3. preprocessing.py
4. split_data.py
5. data_augmentation.py
6. hyperparameter_optimization.py
7. train_model.py
8. train_weighted_model.py (switch environment2.yml)
9. optimal_threshold.py (switch back to environment.yml)
10. evaluation.py

============= SCRIPTS ============

Brief description of scripts.

DATA COLLECTION:

nist_scraper.py - scrapes training dataset from NIST.
sdbs_scraper.py - scrapes training dataset from SDBS.
sdbs_ids.py - scrapes IDs of molecules to be used for sdbs_scraper.py.

DATA PROCESSING:

preprocessing.py - performs preprocessing of the dataset to be used for model training.
data_augmentation.py - augments the preprocessed dataset (use to create augmented dataset).
split_data.py - splits dataset into training, validation, and test sets.

HYPERPARAMETER OPTIMIZATION:

hyperparameter_optimization.py - finds the optimal hyperparameters for CNN model.

MODEL TRAINING:

train_model.py - trains the model without any weights applied on the loss function.
train_weighted_model.py - trains the model with weighted loss function to balance the unbalanced data.

EVALUATION:

evaluation.py - evaluates the trained model.

OTHER:

smarts.py - shows the defined SMARTS strings for the functional groups considered.
optimal_thresholding.py - calculates the optimal probability threshold for the classification of each functional group.

=========== OTHER INFO ===========

All augmented models are based on the extended list of functional groups.

The augmentation was carried out in 25%, 50%, 75%, 100% of the initial dataset.

The performance analysis of the model was calculated using 0_extended.h5 model.