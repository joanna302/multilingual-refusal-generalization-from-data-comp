import argparse
import pandas as pd
import random 
import langcodes

from iso639 import Lang

from datasets import load_dataset, Dataset, concatenate_datasets
from google.cloud import translate_v2 as translate

from utils_data import load_and_sample_data 

"""
Creation of the eval dataset for the 1st phase (language selection)
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--safety_dataset_name', 
        type=str, 
        default="allenai/wildjailbreak")
    parser.add_argument(
        '--nb_sampled', 
        type=int, 
        default=200)
    parser.add_argument(
        '--prop_benign', 
        type=int, 
        default=0.1)
    parser.add_argument(
        '--path_list_cat', 
        type=str, 
        default="lang2tax.txt")
    parser.add_argument(
        '--path_list_lg_qwen', 
        type=str, 
        default="lg_qwen.txt")
    parser.add_argument(
        '--path_list_lg_google', 
        type=str, 
        default="lg_google.txt")
    parser.add_argument(
        '--lg_used', 
        type=dict, 
        default={ 
            "fr":"high", 
            "pt":"high", 
            "zh":"high", 
            "ar":"high", 
            "lo":"low", 
            "tt":"low", 
            "pag":"low",
            "mt":"low"}
            )
    return parser.parse_args()

def select_lg(): 
    with open(args.path_list_lg_google, 'r') as file:
        lg_google = file.readlines()
    with open(args.path_list_lg_qwen, 'r') as file:
        lg_qwen = file.readlines()
    lg_google_qwen = [lg.strip('\n') for lg in lg_google if lg in lg_qwen]

    # languages by category 
    df_lg_cat = pd.read_fwf(args.path_list_cat, header=None)
    df_lg_cat["lg"] = df_lg_cat[0].apply(lambda x : x.split(",")[0])
    df_lg_cat["cat"] = df_lg_cat[0].apply(lambda x : x.split(",")[-1])
    df_lg_cat = df_lg_cat.drop([0], axis=1)

    # group into 3 cat instead of 5 
    lg_low = df_lg_cat[(df_lg_cat["cat"]=="0") | (df_lg_cat["cat"]=="1") | (df_lg_cat["cat"]=="2") ]
    lg_middle = df_lg_cat[(df_lg_cat["cat"]=="3") | (df_lg_cat["cat"]=="4") ]
    lg_high = df_lg_cat[(df_lg_cat["cat"]=="5")]

    # check if lg in google and supported by qwen 
    list_google_qwen_low = [lg for lg in lg_google_qwen if lg in list(lg_low["lg"])]
    list_google_qwen_middle = [lg for lg in lg_google_qwen if lg in list(lg_middle["lg"])]
    list_google_qwen_high = [lg for lg in lg_google_qwen if lg in list(lg_high["lg"])]

    # dictionnary with lg 
    dict_lg = args.lg_used 
    dict_lg["en"]="high"

    list_lg = [Lang(x).name.lower() for x in list(dict_lg.keys())]

    # sample 3 languages by category 
    random.seed(20)
    selected_lg_low = random.sample([elem for elem in list_google_qwen_low if elem not in list_lg], 3)
    print("Low languages selected:", selected_lg_low)
    selected_lg_low = [langcodes.find(x).language for x in selected_lg_low]
    selected_lg_middle = random.sample([elem for elem in list_google_qwen_middle if elem not in list_lg], 3)
    print("Medium languages selected:", selected_lg_middle)
    selected_lg_middle = [langcodes.find(x).language for x in selected_lg_middle]
    selected_lg_high = random.sample([elem for elem in list_google_qwen_high if elem not in list_lg], 3)
    print("High languages selected:", selected_lg_high)
    selected_lg_high = [langcodes.find(x).language for x in selected_lg_high]

    # add to the dict 
    for lg in selected_lg_low: 
        dict_lg[lg]="low"
    for lg in selected_lg_middle: 
        dict_lg[lg]="middle"
    for lg in selected_lg_high: 
        dict_lg[lg]="high"
    return dict_lg


def translate_eval_data(data_safety, dict_lg_selected): 

    translated_datasets=[]

    # API google translate
    translate_client = translate.Client()   

    for language, category in dict_lg_selected.items():
        adversarials_translated = []
        df_translated = pd.DataFrame(columns=["adversarial", "label","data_type", "language", "language_category"])
        if language!='en': 
            print(f"Translation in {language}")
            for row in data_safety: 
                adversarial_translated = translate_client.translate(row["adversarial"], target_language=language)["translatedText"]
                #adversarial_translated = row["adversarial"]
                print(adversarial_translated)
                adversarials_translated.append(adversarial_translated)
            df_translated["adversarial"]=adversarials_translated
        else : 
            df_translated["adversarial"]=data_safety["adversarial"]
        
        df_translated["label"]=data_safety["label"]
        df_translated["data_type"]=data_safety["data_type"]
        df_translated["language"]=[language]*len(data_safety)
        df_translated["language_category"]=[category]*len(data_safety)
        translated_datasets.append(df_translated.copy())

    return pd.concat(translated_datasets)

def complete_dataset_harmless_examples(nb_new_ex=180): 
    """
    Add more adversarial harmless examples. 
    """
    rs = 42
    # select data not in the base dataset 
    base_data = pd.read_csv("datasets_eval/adv/data_eval.csv")
    df=pd.read_csv(f"hf://datasets/{args.safety_dataset_name}/eval/eval.tsv", sep="\t")
    df_harmless = df[df["label"]==0]
    base_data_en = base_data[base_data["language"]=="en"]
    base_data_harmless = base_data_en[base_data_en["label"]==0]
    df_harmless_without_base = df_harmless[~df_harmless["adversarial"].isin(base_data_harmless["adversarial"])]
    df_selected=df_harmless_without_base.apply(lambda x: x.sample(nb_new_ex, replace=False, random_state=rs)).reset_index(drop=True)
    
    # translate new data
    dict_lg_selected = select_lg()
    print(dict_lg_selected)
    ds_eval_new = translate_eval_data(Dataset.from_pandas(df_selected), dict_lg_selected)
    # concat with previous dataset 
    ds_concat = pd.concat([base_data, ds_eval_new])
    # save 
    save_data(ds_concat, "datasets_eval/adv", complete=True)
    



def save_data(dataset, path, complete=False):
    if complete==False : 
        dataset.to_csv(f"{path}/data_eval.csv")
    else : 
        dataset.to_csv(f"{path}/data_eval_completed.csv")

def create_dataset_eval(): 
    data_safety = load_and_sample_data(args.safety_dataset_name, "eval", "safety", N=args.nb_sampled, prop_benign=args.prop_benign)
    dict_lg_selected = select_lg()
    ds_eval = translate_eval_data(data_safety, dict_lg_selected)
    save_data(ds_eval, "datasets_eval/adv")

if __name__ == "__main__":

    args = parse_arguments()
    
    #create_dataset_eval()
    complete_dataset_harmless_examples()
