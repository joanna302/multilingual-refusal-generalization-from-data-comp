
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets

rs=42 

def load_and_sample_data(dataset_name, split, type, N=200, prop_benign=False): 
    if type=='instruction': 
        if dataset_name=="HuggingFaceH4/ultrafeedback_binarized": 
            split="train_prefs"
        ds=load_dataset(dataset_name, split=split)
        if split=="train": 
            ds=ds.shuffle(seed=rs)
            return ds.select(range(N))
        else: 
            return ds
    else: 
        df=pd.read_csv(f"hf://datasets/{dataset_name}/{split}/{split}.tsv", sep="\t")
        if split=="train": 
            df=df.groupby('data_type').apply(lambda x: x.sample(n=N, replace=False, random_state=rs)).reset_index(drop=True)
        elif split=="eval": 
            df=df.groupby('data_type').apply(lambda x: x.sample(int(N) if x.name == "adversarial_harmful" else  int(N * prop_benign), replace=False, random_state=rs)).reset_index(drop=True)
        return Dataset.from_pandas(df)
