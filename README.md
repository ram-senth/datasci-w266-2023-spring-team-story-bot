# StoryBot: Interactive Story Generation For Kids
### Group: Ghiwa Lamah, Ram Senthamarai, Nicolas Loffreda
### Section Instructor: Natalie Ahn
  
<br>

Repository for the final project for MIDS W266: Natural Language Processing with Deep Learning, Spring 2023 course from the UCB MIDS program.

The goal for this project is to develop an NLP sentence continuation model that will enable an eventual interactive story development experience for children, in order to make the activity more accessible while reducing the burden on childcare providers. This system takes in the last few sentences of a story as an input, and generates the next sentence while staying on topic.

<a href="https://docs.google.com/presentation/d/1omxUU91ZXW47QygvzZoxBZUsqPzz1QJyix97pT2D8Bo/edit#slide=id.g2189013d091_0_233">View Final Presentation Slides</a>

<br>

## Repo Folders:
### - Code:
- data_processing.ipynb: The data processing Jupyter notebook
- model_finetuning.ipynb: Jupyter notebook used to train OPT and T5
- inferencing.ipynb: Jupyter notebook used for initial inference of model outputs
- Auto-Evaluation.ipynb: Jupyter notebook used to perform auto-evaluation
- bert2bert_train_eval.ipynb: Jupyter notebook used to train and evlauate Bert2Bert (B2B)
- annotation.ipynb: Jupyter notebook used to perform manual annotation
- common.py: Setup and configurations imported by other notebooks in this folder 

### -   Data: 
- Raw data file for the <a href="https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus">Children Stories Text Corpus</a>
- The processed S1, S2 and S3 datasets (inside a 7zip given its large size)
### -   Evaluation: 
#### - Auto:
- final_scores.csv: csv of summary of scores for all models
- Csv's of auto-evaluation results for each fine-tuned model
#### - Manual:
- prompts_for_manual_evaluation: List of 30 prompts used for manual evaluation process
- manual_eval_generated_output.csv: Generated output for all models for the list of 30 prompts. Five sequences generated for each prompt/model combination.
- manual_eval_generated_output_with_bib.csv: Annotated version of the csv above, with flags indicating which returned sequence the model chose as best, and which returned sequence the annotator chose as best.
- Csv's of the resulting annotations made by each annotator, scoring the returned sequences on the five chosen metrics: Relevance, Readability, Grammar, Non-Redundancy, and Kid-Friendly Language.