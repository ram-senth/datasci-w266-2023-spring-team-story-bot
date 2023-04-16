# StoryBot: Interactive Story Generation For Kids
### Group: Ghiwa Lamah, Ram Senthamarai, Nicolas Loffreda
### Section Instructor: Natalie Ahn
  
<br>

Repository for the final project for MIDS W266: Natural Language Processing with Deep Learning, Spring 2023 course from the UCB MIDS program.

The goal for this project is to develop an NLP sentence continuation model that will enable an eventual interactive story development experience for children, in order to make the activity more accessible while reducing the burden on childcare providers. This system takes in the last few sentences of a story as an input, and generates the next sentence while staying on topic.

<br>

#### Repo Folders:
###### -   Data: 
    - Raw data file for the <a ref="https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus">Children Stories Text Corpus</a>
    - The data processing Jupyter notebook
    - The processed S1, S2 and S3 datasets (inside a 7zip given its large size)
##### -   Fine-tuning: 
    - model_finetuning: Jupyter notebook used to train OPT and T5
    - bert2bert_train_eval.ipynb: Jupyter notebook used to train and evlauate Bert2Bert (B2B)
    - inferencing.ipynb: Jupyter notebook used for initial inference of model outputs
##### -   Evaluation: 
###### - Auto:
        - Auto-Evaluation.ipynb: Jupyter notebook used to perform auto-evaluation
        - final_scores.csv: csv of summary of scores for all models
        - Csv's of auto-evaluation results for each fine-tuned model
###### - Manual:
        - annotation.ipynb: Jupyter notebook used to perform manual annotation
        - prompts_for_manual_evaluation: List of 30 prompts used for manual evaluation process
        - manual_eval_generated_output.csv: Generated output for all models for the list of 30 prompts. Five sequences generated for each prompt/model combination.
        - manual_eval_generated_output_with_bib.csv: Annotated version of the csv above, with flags indicating which returned sequence the model chose as best, and which returned sequence the annotator chose as best.
        - Csv's of the resulting annotations made by each annotator, scoring the returned sequences on the five chosen metrics: Relevance, Readability, Grammar, Non-Redundancy, and Kid-Friendly Language.