# TransNAS_TSAD: A NAS-Enhanced Transformer for Time-Series Anomaly Detection
TransNAS_TSAD innovatively combines transformer models with Neural Architecture Search (NAS) to set a new standard in time-series anomaly detection. Building upon the foundation laid by TranAD, this project pushes the boundaries by optimizing both performance and model complexity across a variety of datasets.

# Getting Started with the Demo
The provided Jupyter Notebook (demo_TransNAS_TSAD.ipynb) is the perfect starting point for experiencing TransNAS_TSAD on Google Colab. It's designed to guide you through the process of setting up, running the model, and interpreting results.

# Prerequisites
Before diving into the demo, please ensure:
All necessary files are placed according to the data path structure mentioned.

The entire repository is uploaded to your Colab environment to maintain the required directory paths.
Libraries needed for execution are listed within the demo.ipynb. Please install them as instructed in the notebook.
Data Preprocessing
The preprocess_TransNAS_TSAD script is crucial for preparing your dataset for the model. This preprocessing functionality, inspired by the TranAD project, is tailored for optimal compatibility with TransNAS_TSAD. Adjust the paths in your Colab environment to align with the structure provided in the repository for seamless operation.

# Running the Demo
Adjust Data Paths: Ensure the DATA_PATH, OUTPUT_FOLDER, and other relevant paths in the notebook match your Colab directory structure.
Install Dependencies: Follow the instructions in demo.ipynb to install required libraries.
Run the Notebook: Execute the cells sequentially to train and evaluate the model.
Acknowledgments

This project owes its inspiration to the TranAD paper, from which certain preprocessing and evaluation functions have been adapted with modifications to better suit our NAS-based approach.

# Results

![Training-testing](https://github.com/ejokhan/TransNAS_TSAD/assets/19641451/5d4c2f29-396b-47bb-a4fb-293abe756d18)
![Testing result](https://github.com/ejokhan/TransNAS_TSAD/assets/19641451/fa9f817b-1bf9-4441-b6c5-091ebca8c776)
![Pareto results](https://github.com/ejokhan/TransNAS_TSAD/assets/19641451/97d3b736-4fdd-4909-b814-dfbe207d016d)

# Cite this work as (Looking for Journal to sumbit)

@misc{haq2023transnastsad,
      title={TransNAS-TSAD: Harnessing Transformers for Multi-Objective Neural Architecture Search in Time Series Anomaly Detection}, 
      author={Ijaz Ul Haq and Byung Suk Lee},
      year={2023},
      eprint={2311.18061},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

