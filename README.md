# Topic Classification Repository (SRIP)
## Repository Structure

* `src/`: The core source code for all implemented models.
  * `tfidf_lr/`: Classical baseline using Scikit-Learn (TF-IDF + Logistic Regression).
  * `fasttext/`: FastText neural baseline built dynamically via PyTorch `nn.EmbeddingBag`.
  * `word_embedding/`: Neural baseline averaging global word contexts explicitly.
  * `lstm/`: Bidirectional LSTM capturing forward/backward sequential context.
  * `transformer/`: The Final Model. A Custom Mini-Transformer Encoder utilizing parallel multi-head self-attention.
* `experiments/`: Organized directory housing all training logs, JSON configurations, saved model parameters (`.pt` / `.joblib`), and classification metrics (`metrics.txt`) for each respective architecture.
* `tools/`: Utility scripts leveraged for raw Parquet file schema analysis and memory-safe dataset sub-sampling.

---

## Setup & Installation

Before running the models, install the required PyTorch and Data Science dependencies:

```bash
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preparation & Sampling

The primary dataset scales to 10 million text snippets horizontally compressed inside a Parquet configuration. Loading this unconditionally into Pandas will immediately trigger Out-Of-Memory (OOM) crashing. 

### 1. Download the Dataset
Download the original massive dataset from this URL link:
[dataset_10M.parquet Download](https://drive.google.com/file/d/1iib_mYxLcN6pNVHMpANeUCpe2qum0n-K/view)

Place the resulting `dataset_10M.parquet` loosely into the root directory of this repository.

### 2. Extract a Working Sample
To encourage rapid iteration and safely evade arbitrary OOM issues, we utilize the sequence of tools contained inside the `tools/` folder:

* **`tools/data_analyse.py`**: Runs analytical diagnostics on the massive master file. It loads PyArrow Parquet File Metadata to strictly determine schemas and isolate row group capacities without pushing the data into physical memory arrays.
* **`tools/extract_sample.py`**: Executes an extraction process by mapping to specific structural row groups inside the 10M file, and extracting a slice of sequential index values sequentially. By default, running this constructs a locally manageable `dataset_sample_200k.parquet`.

**Run the extraction natively:**
```bash
python tools/extract_sample.py
```

## Running the Baselines

Once your sub-sampled parquet resides locally, you can iteratively trigger training loops for each progressive architectural experiment. By default, these scripts target the `dataset_sample_200k.parquet` configuration, compute performance against its 10% stratified hold-out split, and pipe metrics down into `experiments/`.

```bash
python src/tfidf_lr/train.py

python src/fasttext/train.py
python src/word_embedding/train.py
python src/lstm/train.py
```

## The Final Model: Custom Transformer

The final modal of our investigation is the Custom Transformer structurally located within `src/transformer/`. It captures deep syntactic properties while processing long sequences heavily in parallel, effectively nullifying LSTM/RNN gating bottlenecks while maintaining a parameter profile strictly scaled beneath 10M (dramatically smaller than the 5-Billion scale limit context).

```bash
python src/transformer/train.py --data_path dataset_sample_200k.parquet --save_dir experiments/transformer/


```
optional args 
--epochs 
--batch_size 
--lr 
### (Inference Mode)
You can directly loop into the architecture via inference mapping and feed custom unstructured strings natively to predict 24 distinctive classifications.

```bash
python src/transformer/inference.py --model_dir experiments/transformer/
```
**Example Standard Output:**
```text
Enter text: ____________
Predicted Topic: __________
```
