import os
import sys

import pandas as pd
from dotenv import load_dotenv

from logger import log

logger = log.setup_logging()


load_dotenv()


def process_raw_data(url):
    # Load the data
    df = pd.read_csv(url)

    # Replace the labels list with actual label names in your dataset
    labels = eval(os.getenv("labels"))
    label2id = dict(enumerate(labels))
    id2label = {v: k for k, v in label2id.items()}

    # Create integer labels
    df.rename(columns={"data": "article", "labels": "category"}, inplace=True)
    df["label"] = df["category"].map(id2label)

    dir_path = str(os.path.join(*url.split("/")[:2])) + "/processed/bbc_data.csv"

    logger.info(f"Saving the Datasets in {dir_path}")

    # Save the datasets to CSV
    df.to_csv(str(dir_path), index=False)
    return str(dir_path)


if __name__ == "__main__":
    try:
        rawdata_url = sys.argv[1]
        logger.info(
            f"Executing command:  python {__file__.split('/')[-1]} {rawdata_url}"
        )
        processed_url = process_raw_data(rawdata_url)

    except Exception as e:
        logger.error(str(e))
