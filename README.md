# Mag2Edge
Official implementation of Mag2Edge for directed edge embedding.

This repository contains the implementation of the algorithm **Mag2Edge**.

Requirements

Install the required dependencies:

pip install -r requirements.txt


Data Preparation

Ensure the dataset files are placed in the dataset/ directory. The project expects the following structure:

Reddit: train_edges.csv, val_edges.csv, test_edges.csv, and *_embeds_labels.pkl

WikiConflict: wikiconflict.csv

MOOC: mooc_actions.tsv, mooc_action_labels.tsv

Epinions: epinions_data.txt

Amazon: The script will automatically download movies.txt.gz if missing.

Usage

Run the experiments using main.py. The script automatically runs 5 random seeds and reports the Mean ± Std.

Arguments

--dataset: Target dataset (reddit, wikiconflict, amazon, mooc, epinions)

--gpu: GPU ID to use (default: 0)

--data_dir: Path to dataset directory (default: dataset)

Run Experiments

1. Reddit

python main.py --dataset reddit --gpu 0


2. WikiConflict

python main.py --dataset wikiconflict --gpu 0


3. Amazon Movies

python main.py --dataset amazon --gpu 0


4. MOOC

python main.py --dataset mooc --gpu 0


5. Epinions

python main.py --dataset epinions --gpu 0
