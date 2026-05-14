# Roadmap
Before you start any of the tasks below, read the README.md. Furthermore,
the following steps are already done. Use this checklist by yourself to
track the progress of what is already done:
- [x] Step 1 — Recreate the Baseline
- [x] Step 2 — Specify the experimental design
- [ ] Step 3 — Run the experiments
- [ ] Step 4 — Try to improve upon the F2 score.

For recreating the baseline and running the experiments the results
(i.e. the model pipeline code should go into results/baseline,
results/experiment1, results/experiment2, etc.). When the experients
are done, please create a jupityer notebook for the respective
baseline or experiment that covers all the necessary steps to build
the respective model pipeline. It is important that when you run the
experiments, you have to safe the models (e.g. models/baseline or
models/experiment1) and then import the models later on in the
notebooks you create so we don't waste compute and time. In the
notebook of the experiments you create, please also add references
to the papers where the pipeline came from.


# Step 1 - Recreate the Baseline
First, check out the baseline that the authors of the PathMNIST dataset
have reached an recreate it. You should achieve similar results than
the authors did.


## Step 1 — Specify the Experimental Design

Before coding, define which experiments you will actually run. Base those experiments on the research
that was already done in data/research.md. If this research is not sufficient, research yourself more
and document the sources. Create a md report in data/experiment-design.md. Also specify for each
experiment which combination of ML models/aproaches is used. And also cite from which paper which
approach is coming from

## Step 3 - Run the Experiments
For each experiment you defined in the step before, create a notebook in notebooks/ and run the experiment to
see if you can replicate the results from the different research papers on our dataset. Only stop
if you have successfully replicated the results from the respective paper. 

## Step 4 — Try to improve upon the f2 score
Try a wide range of different techniques to improve upon the best f2 score that resulted from the
different experiments. Do so by try out different techniques of ML approaches. If you managed to
achieve an improvement, create also a notebook in notebooks/ called improvement.
