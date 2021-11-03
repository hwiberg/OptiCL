# OptiCL
OptiCL is an end-to-end framework for mixed-integer optimization (MIO) with data-driven learned constraints. We address a problem setting in which a practitioner wishes to optimize decisions according to some objective and constraints, but that we have no known functions relating our decisions to the outcomes of interest. We propose to *learn* predictive models for these outcomes using machine learning, and to subsequently *optimize* decisions by embedding the learned models in a larger MIO formulation.  

The framework and full methodology are detailed in our manuscript, Mixed-Integer Optimization with Constraint Learning.

## How to use OptiCL
You can install the OptiCL package locally by cloning the repository and running ```pip install .``` within the home directory of the repo. This will allow you to load `opticl` in Python; see the example notebooks for specific usage of the functions.

## The OptiCL pipeline
Our pipeline requires two inputs from a user:
- Training data, with features classified as contextual variables, decisions, and outcomes.
- An initial conceptual model, which is defined by specifying the decision variables and any domain-driven fixed constraints or deterministic objective terms. 

Given these inputs, we implement a pipeline that:
1. Learns predictive models for the outcomes of interest by using a moel training and selection pipeline with cross-validation. 
2. Efficiently charactertizes the feasible decision space, or "trust region," using the convex hull of the observed data.
3. Embeds the learned models and trust region into a MIO formulation.

OptiCL requires no manual specification of a trained ML model, although the end-user can optionally restrict to a subset of model types to be considered in the selection pipeline. Furthermore, we expose the underlying trained models within the pipeline, providing transparency and allowing for the predictive models to be externally evaluated.

### Examples
We illustrate the full OptiCL pipeline in three notebooks:
- A case study on food basket optimization for the World Food Programme (**notebooks/WFP/The Palatable Diet Problem.ipynb**): This notebook presents a simplified version of the case study in the manuscript. It shows how to train and select models for a single learned outcome, define a conceptual model with a known objective and constraints, and solve the MIO with an additional learned constraint. 
- A general pipeline overview (**notebooks/Pipeline/Model_embedding.ipynb**): This notebook demonstrates the general features of the pipleine, including the procedure for training and embedding models for multiple outcomes, the specification of each outcome as either a constraint or objective term, and the incorporation of contextual features and domain-driven constraints. 
- Model verification (**notebooks/Pipeline/Model_Verification.ipynb**): This notebook shows the training and embedding of a single model and compares the sklearn predictions to the MIO predictions to verify the MIO embeddings.

The package currently fully supports model training and embedding for continuous outcomes across all ML methods, as demonstrated in the example notebooks. Binary classification and multi-class classification support are in development and will be available soon.  

## Citation
Our software can be cited as:
````
  @misc{OptiCL,
    author = "Donato Maragno and Holly Wiberg",
    title = "OptiCL: Mixed-integer optimization with constraint learning",
    year = 2021,
    url = "https://github.com/hwiberg/OptiCL/"
  }
````

## Get in touch!
Our package is under active development. We welcome any questions or suggestions. Please submit an issue on Github, or reach us at d.maragno@uva.nl and hwiberg@mit.edu. 
