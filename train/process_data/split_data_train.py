import os 
import argparse
import pandas as pd

from datasets import load_dataset, Dataset, concatenate_datasets
from google.cloud import translate_v2 as translate

from utils_data import load_and_sample_data 

rs=42 

template_prompt = """Below is an instruction:
### Instruction :
{}
### Response:
{} """

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--instruction_dataset_name', 
        type=str,
        default="HuggingFaceH4/no_robots")
    parser.add_argument(
        '--safety_dataset_name', 
        type=str, 
        default="allenai/wildjailbreak")
    parser.add_argument(
        '--N_safety_cat', 
        type=int, 
        default=750)
    parser.add_argument(
        '--N_instruc', 
        type=int, 
        default=9_000)
    parser.add_argument(
        '--proportion', 
        type=int, 
        default=0.1)
    parser.add_argument(
        '--path', 
        type=str, 
        default="datasets_train/split")
    return parser.parse_args()

def formatting_prompts_safety(examples):
    if "adversarial" in examples["data_type"]:
        instructions = examples["adversarial"]
    else:
        instructions = examples["vanilla"]
    outputs = examples["completion"]
    text = template_prompt.format(instructions, outputs) 
    return { "text" : text, "instruction" : instructions, "chosen_response" : outputs, }

def formatting_prompts_instruc(examples):
    instructions=list(filter(lambda x: x["role"]=="user", examples["messages"]))[0]["content"]
    outputs=list(filter(lambda x: x["role"]=="assistant", examples["messages"]))[0]["content"]
    text = template_prompt.format(instructions, outputs) 
    return { "text" : text, "instruction" : instructions, "chosen_response" : outputs, }

def clean_dataset(ds, safety=False): 
    if safety==True:
        ds=ds.rename_column("data_type", "category")
    cols_to_remove = ds.column_names 
    cols_to_remove.remove("text")
    cols_to_remove.remove("safety")
    cols_to_remove.remove("category")
    cols_to_remove.remove("instruction")
    cols_to_remove.remove("chosen_response")
    return ds.remove_columns(cols_to_remove)

def create_dataset(data_instruc, data_safety): 
    ds_instruc = data_instruc.map(formatting_prompts_instruc)
    ds_safety = data_safety.map(formatting_prompts_safety)
    # add column 
    ds_instruc=ds_instruc.add_column("safety", [0]*len(ds_instruc))
    ds_safety=ds_safety.add_column("safety", [1]*len(ds_safety))
    return concatenate_datasets([clean_dataset(ds_instruc), clean_dataset(ds_safety, safety=True)])


def split_data(ds): 

    # list of lg for each subset 
    list_lg_subset_1 = ['fr', 'zh', 'lo', 'pag', 'en']
    list_lg_subset_2 = ['pt', 'ar', 'tt', 'mt', 'en']
    list_lg_subset_3 = ['en']

    # select subsets to be translate 
    df = Dataset.to_pandas(ds) 
    list_subsets=[]
    nb_translation_cat = int(args.proportion * args.N_safety_cat)
    nb_instr_cat = df[df['safety']==0].groupby('category').count()['text'] * args.proportion
    
    # select randomly prompts to be translated in lg1 
    subset_lg1_safety=df[df['safety']==1].groupby('category').apply(lambda x: x.sample(n=nb_translation_cat, replace=False, random_state=rs)).reset_index(drop=True)

    list_df_lg1_instr = []
    for cat, nb in nb_instr_cat.items(): 
        list_df_lg1_instr.append(df[(df['safety']==0) & (df['category']==cat)].apply(lambda x: x.sample(n=int(nb), replace=False, random_state=rs).reset_index(drop=True)))

    subset_lg1_instr = pd.concat(list_df_lg1_instr)
    subset_lg1 = pd.concat([subset_lg1_instr, subset_lg1_safety]) 

    # remove them for the initial dataset 
    df = df[~df['text'].isin(subset_lg1['text'])]

    # add new col with list of lg 
    subset_lg1["lg_subset"]=[list_lg_subset_1] * len(subset_lg1)

    list_subsets.append(subset_lg1)
  
    ### second subset 
    subset_lg2_safety=df[df['safety']==1].groupby('category').apply(lambda x: x.sample(n=nb_translation_cat, replace=False, random_state=rs)).reset_index(drop=True)

    list_df_lg2_instr = []
    for cat, nb in nb_instr_cat.items(): 
        list_df_lg2_instr.append(df[(df['safety']==0) & (df['category']==cat)].apply(lambda x: x.sample(n=int(nb), replace=False, random_state=rs).reset_index(drop=True)) )

    subset_lg2_instr = pd.concat(list_df_lg2_instr)
    subset_lg2 = pd.concat([subset_lg2_instr, subset_lg2_safety]) 

    # remove them for the initial dataset 
    df = df[~df['text'].isin(subset_lg2['text'])]

    # add new col with list of lg 
    subset_lg2["lg_subset"]=[list_lg_subset_2] * len(subset_lg2)
    list_subsets.append(subset_lg2)

    # add new col for df 
    df["lg_subset"] = [list_lg_subset_3]*len(df)
    list_subsets.append(df)

    return pd.concat(list_subsets)



def save_data_train(dataset, path):
    dataset.to_csv(f"{path}/split_data.csv")


if __name__ == "__main__":

    args = parse_arguments()
          
    data_instruc = load_and_sample_data(args.instruction_dataset_name, "train", "instruction", args.N_instruc)
    data_safety = load_and_sample_data(args.safety_dataset_name, "train", "safety", args.N_safety_cat)
    ds = create_dataset(data_instruc, data_safety)
    ds_splitted = split_data(ds)
    save_data_train(ds_splitted, args.path)




