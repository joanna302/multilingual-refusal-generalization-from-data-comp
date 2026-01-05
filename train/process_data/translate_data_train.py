import os 
import json 
import argparse
import pandas as pd
import numpy as np 

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
        '--lg1', 
        type=str, 
        default="pag")
    parser.add_argument(
        '--lg2', 
        type=str, 
        required=False,
        default="mt")
    parser.add_argument(
        '--path', 
        type=str, 
        default="datasets_train")
    return parser.parse_args()

# Function to get value for a specific key
def get_value_from_dict(json_string, key):
    dictionary = json.loads(json_string)
    return dictionary.get(key)

def reshape_data(df): 
    df[f'language']= df['lg_subset'].apply(lambda x: args.lg1 if args.lg1 in x else None)
    if args.lg2!=None: 
        df[f'language']= df.apply(lambda row : args.lg2 if args.lg2 in row['lg_subset'] else row['language'], axis=1)
    df['language']=df['language'].fillna('en')
    return df 

def translate_data(df, already_translated=True): 

    # translation of the columns instruction and chosen_response if the language specified in the col language 
    # translatation 
    list_subsets_trans = []

    df_en = df[df["language"]=="en"]

    # API google translate
    translate_client = translate.Client()

    instructions_translated = []
    responses_translated = []
    texts_translated = []
    list_lg = [args.lg1]
    if args.lg2!=None: 
        list_lg.append(args.lg2)

    for lg in list_lg : 
        df_lg = df[df["language"]==lg]

        # check if data already translated and saved
        path = f"{args.path}/trad"
        if os.path.isdir(path): 
            list_files = os.listdir(path)
            already_translated = False
            for file in list_files: 
                if lg in file.split("_"): 
                    df_file = pd.read_csv(f"{path}/{file}")
                    already_translated=True
                
        # keep the same translation 
        if already_translated == True: 
            print(f"Subset already translated in {lg}")
            df_file_lg = df_file[df_file["language"]==lg]
            list_subsets_trans.append(df_file_lg)

        else : 
            print(f"Translation in {lg}")
            instructions_translated = []
            responses_translated = []
            texts_translated = []

            for instruction, chosen_response in zip(df_lg["instruction"], df_lg["chosen_response"]):
                instruction_translated = translate_client.translate(instruction, target_language=lg)['translatedText']
                chosen_response_translated = translate_client.translate(chosen_response, target_language=lg)['translatedText']
                
                #instruction_translated = instruction
                #chosen_response_translated = chosen_response

                instructions_translated.append(instruction_translated)
                responses_translated.append(chosen_response_translated) 

                texts_translated.append(template_prompt.format(instruction_translated, chosen_response_translated))
            
            # replace non translated data by translated data 
            df_lg["instruction"] = instructions_translated
            df_lg["chosen_response"] = responses_translated
            df_lg["text"] = texts_translated

            list_subsets_trans.append(df_lg.copy())
    
    list_subsets_trans.append(df_en)

    return pd.concat(list_subsets_trans)


def save_data(dataset):
    if args.lg2!=None: 
        dataset.to_csv(f"{args.path}/trad/data_{args.lg1}_{args.lg2}_.csv")
    else: 
        dataset.to_csv(f"{args.path}/trad/data_{args.lg1}_.csv")

def load_dataset(): 
    dataset = pd.read_csv(f"{args.path}/split/split_data.csv")
    return dataset 

if __name__ == "__main__":

    args = parse_arguments()
    df = load_dataset()
    df_reshaped = reshape_data(df)
    df_translated = translate_data(df_reshaped)
    save_data(df_translated)
    



