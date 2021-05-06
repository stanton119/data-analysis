# AzureML Opensource
2021-05-06

----
Outline:

* AzureML
* Pipeline libraries
* ML centric packages
* Kedro

---
## AzureML

----
### Overview

* Every node needs a python script
* Every pipeline needs a script
* Experiment tracking
* Dataset tracking
* Some DAG based pipelines
* Remote compute

----
### Pros
* Fairly flexible
* Container orchestration is handled
  * Parallel nodes
* Caches outputs for reuse
* Integrated into Azure

----
### Cons
* Slow to develop
  * Lots of boilerplate
* Hard to debug
  * Can't run a pipeline locally
  * Cannot run single node
* Every node runs in a separate container
  * Slow in a simple pipeline
* No real support for DAGs without other Azure pipeline tech (ADF/AzureDevOps)
* Stuck on Azure

----
### Example
```
pipeline - build_training_pipeline.py
node - data_prep_step.py
```

---
## Pipeline libraries

Lots of well supported packages for running DAGs

* Everything is based around Kubernetes
* SDKs python mostly
* Decorators commonly used
* Support for Spark

----
### Pipeline libraries

* [Airflow](https://airflow.apache.org)
  * [Example](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
* [Luigi](https://github.com/spotify/luigi)
* [Argo](https://github.com/argoproj/argo-workflows)
* [Prefect](https://github.com/prefecthq/prefect)
  * [Tutorial](https://docs.prefect.io/core/tutorial/02-etl-flow.html)
* [Dagster](https://github.com/dagster-io/dagster)
  * [Tutorial](https://docs.dagster.io/tutorial/intro-tutorial/connecting-solids)

[Comparisons](https://medium.com/@will_flwrs/python-data-engineering-tools-the-next-generation-354e00f2f060)

---
## ML centric packages

* Versioning datasets
* Serving models
* etc.

----
### ML centric packages

* [Kubeflow](https://www.kubeflow.org)
  * Based on Argo
  * ML pipelines run on kubernetes, much larger projects?
  * Similar to AzureML [pipelines](https://github.com/kubeflow/examples/tree/master/pipelines/azurepipeline/code)
* [Metaflow](https://github.com/Netflix/metaflow)
  * AWS based ML stack, notebook emphasis?
* [DVC](https://github.com/iterative/dvc)
  * dataset versioning, pipelines
* [MLflow](https://github.com/mlflow/mlflow)
  * [Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)
  * experiment tracking, dataset versioning
  * Very similar to AzureML
* Kedro...

---
## Kedro

More likely to be of immediate use

----
### Kedro - Overview
[Github](https://github.com/quantumblacklabs/kedro)
* Opinionated data science project template
* Data catalogue
  * *easy* to move between platforms and local
  * Dataset versioning
* Handles IO
  * Separates business logic from IO
* Self documenting data layers
* Docker to productionise
* Plugins for extras
  * Argo, Kubeflow, Airflow, MLflow

----
### Demo...

[Github](https://github.com/stanton119/data-analysis/tree/master/NBA/nba-analysis)

...comparison back to AzureML...

----
### Questions

* Can we productionise in Azure easily?
  * Serverless compute possible
  * Kubernetes based, so hopefully easy
* Use pyspark in the middle?