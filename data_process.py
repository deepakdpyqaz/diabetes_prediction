import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
import pathlib
import sklearn
from absl import logging
from sklearn.preprocessing import StandardScaler
logging.set_verbosity(logging.INFO)  
import time
import pickle
DATA_DIR = os.getenv("PREPROCESS_DATA_DIR") or "data"
OUTPUT_DATA_DIR = os.getenv("PREPROCESS_OUTPUT_DATA_DIR") or "processed_data"

if not os.path.exists(OUTPUT_DATA_DIR):
    os.mkdir(OUTPUT_DATA_DIR)
path = pathlib.Path(DATA_DIR)
files = list(map(lambda x:str(x),path.glob("*.csv")))
logging.info(f" {len(files)} csv files found")

for f in files:
    try:
        fname = f.split("/")[-1]
        logging.info(f"Processing {fname}")
        start = time.time()
        df = pd.read_csv(f)
        outcome = df["Outcome"]
        df = df.drop("Outcome",axis=1)
        columns = df.columns
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        df = pd.DataFrame(df,columns=columns)
        df["Outcome"] = outcome
        df.to_csv(os.path.join(OUTPUT_DATA_DIR,fname),index=False)
        end = time.time()
        logging.info(f"Processed {fname} -------> {round(end-start,2)} ms")
        if not os.path.exists("scalers"):
            os.mkdir("scalers")
        with open("scalers/scaler.pickle","wb") as pf:
            pickle.dump(scaler,pf)
        os.remove(f)
    except Exception as e:
        logging.error(str(e))


logging.info("Preprocessing Finished")