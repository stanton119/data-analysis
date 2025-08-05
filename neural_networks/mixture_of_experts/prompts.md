# Prompts

Read the README in README.md. Look at the module src/models.py implement the MMoE model.

Read the README in README.md. Look at the module src/train.py. I'm getting two mlflow logs for each training run. What is the preferred option to log to mlflow?

Read the README in README.md. Look at the module src/models.py. Do the models need to take in task names? 

Read the README in README.md. Look at the module src/torch_datasets.py. One hot encoded features are not common across train and test sets as the data loaders are created separately.

Read the README in README.md. Look at the module src/torch_datasets.py. If we do one hot encoding etc in the dataloader setup, what do we do at inference time?

I'm building a pytorch model. I have a train and test datasets which need some stateful processing (one hot encoding etc.). How do I create data loaders ready for the model and keep the stateful processing consistent across train and test dataframes.

Read the paper in /Users/stantoon/Documents/VariousProjects/github/data-analysis/neural_networks/mixture_of_experts/mmoe.pdf by converting to markdown. How do they use the uci dataset as a multi task dataset?

Read the README in README.md. I want to measure AUC and maybe others on the test dataloader for a trained model. Should this happen in train.py or a separate module which loads a trained model and runs evaluation?

Read the README in README.md. Give me a command to run each model on the CSI census dataset

Read the README in README.md and src/evaluate.py. Can the evaluated metrics log values (even graphs) to mlflow? The models are trained with mlflow. Should we log evaluation metrics there as well, on the same run?

Read the README in README.md. The uci census dataset has a column `relationship` which has values like married/husband/wife etc. This will explain away the married status target. Should we remove the column somewhere?
