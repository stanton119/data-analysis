# Kedro

* Takes care of loading xlsx, csv etc. to dataframes and then inputs to functions
  * How do we adjust those loading functions, e.g. xlsx sheet
* How to manage with conda environments properly?
  * How does the install operation work?
* Can we deploy the whole pipeline in a docker container
  * https://junglescouteng.medium.com/jungle-scout-case-study-kedro-airflow-and-mlflow-use-on-production-code-150d7231d42e
  * https://kedro.readthedocs.io/en/latest/07_extend_kedro/04_plugins.html#community-developed-plugins
* Plugins to run in kubeflow etc that allow pipelines to be split across VMs
* Can we run pyspark in the middle?

* Dataset versioning - uses latest version by default

Seems to cover pipelines (DAGs), dataset creation/IO, dataset versioning, project template

Template:
* data layers
* docs
* tests

Comparisons:
* Airflow - DAGs, no experiment tracking, no ML focus
* https://www.kubeflow.org - ML pipelines run on kubernetes, much larger project
* Mlflow - experiment tracking, dataset versioning
* https://github.com/Netflix/metaflow - AWS based ML stack, notebook emphasis?
* Prefect - airflow 2?
  * Scalable, easy interface, run locally, azure connection
  * https://docs.prefect.io/core/tutorial/01-etl-before-prefect.html
* Dagster
* DVC - dataset versioning, pipelines

* https://medium.com/@will_flwrs/python-data-engineering-tools-the-next-generation-354e00f2f060
