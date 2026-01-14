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

Train a model : 
```

```

Inference : 
```

```
Evaluation : 
```

```
Translate output for evaluate only output in high-resource languages : 
```

```


* The folder "res_analysis" presentes the results of the multilingual fine-tuning on various models. 

* The codes presented in the folder "latent_space_viz" has been adapted from: https://github.com/andyrdt/refusal_direction. This folder is used for the study of the latent space (part 6 of the paper). \
Run the script : 
```
python -m plot_latent_space --model_path joanna303/gemma-3-4B-Base_en_alpaca_1_part_SFT_8e-05
```