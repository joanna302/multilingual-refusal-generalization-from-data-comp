# Does adding languages fails to improve robustness ? An empirical study of multilingual safety training. 

This repository contains the codes and results presented in the paper "Does adding languages fails to improve robustness ? An empirical study of multilingual safety training." 

## Setup 
```
# Clone the repo
git clone https://github.com/joanna302/multilingual-refusal-generalization-from-data-comp.git
cd multilingual-refusal-generalization-from-data-comp

# Create the environment 
python -m venv myenv 
source myenv/bin/activate 
pip install -r requirements.txt
```
## Repo architecture 

* The folder "train" are used to process the data, fine-tune the models, carry out the inferences and evaluate the results.

The folder ``` train/process_data``` was used the create the training and eval datasets. 
We use the script ```train/train.py``` to fine-tune the models, the script ```train/inference.py``` to run the inference on the evaluation datasets. We then evaluate the outputs (detect the refusals) with the script ```train/eval/eval_results.py```. We use the notebook ```translate_output_llm.ipynb``` to translate the outputs for evaluate only output in high-resource languages. 
We then re-evaluate the translated outputs thanks to the scripts ```eval_results.py```

Train a model : 
```

```

Inference : 
```

```
Evaluation : 
```

```


* The folder "res_analysis" presentes the results of the multilingual fine-tuning on various models. 

* The codes presented in the folder "latent_space_viz" has been adapted from: https://github.com/andyrdt/refusal_direction. This folder is used for the study of the latent space (part 6 of the paper). \
Run the script : 
```
python -m plot_latent_space --model_path [model_path]
```
The notebook ```latent_space_viz/results_viz_latent_space.ipynb``` presents the results we obtained. 