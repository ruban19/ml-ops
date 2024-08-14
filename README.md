# ml-ops
Repository designed for MLOps activity.

## Environment Setup

### Prerequisites

1. Ensure Python 3.10 is installed on your system.
2. Clone the repository.

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

3. Create a virtual environment using Python 3.10 and activate it:

    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

4. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

5. Edit `.env` file in the root directory of the project to store your Hugging Face and MLflow tokens:

    Add your tokens to the `.env` file:

    ```plaintext
    HF_TOKEN=your_hugging_face_token
    MLFLOW_URI=your_mlflow_tracking_uri
    MLFLOW_TRACKING_PASSWORD=your_mlflow_token_or_password
    ```


### Formatting Codebase

To format the codebase, you can use the `make` command:

```bash
make format
```


## Pre-Processing the Data:

`To Run your pre processing layer :`

``` bash
python src/preprocessing/process_data.py "./data/raw/bbc_data.csv"
```


## Model Training:

`To Run your model training :`

``` bash
python src/classification/model_training.py "./data/processed/bbc_data.csv"
```
