__version__='1.3'
SEED = 42
MAIN_DATA_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}.csv' #needs base path and dataset
TRAIN_VAL_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}_train_val.csv' #needs base path and dataset
TEST_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}_test.csv' #needs basepath and dataset
SAVED_MODEL_PATH_FORMAT = '{}saved_models/final/{}-{}-finetuned' # needs base path, model name and dataset
VALID_MODEL_FAMILIES = set(['t5', 'opt'])
VALID_DATASETS = set(['s1', 's2', 's3'])
T5_PROMPT = 'Generate next line: '
OPT_PROMPT = ''
CHECKPOINTS_TO_SAVE = 1
TRAINING_SAMPLES = -1
VAL_SAMPLES = 10000

def create_configs(project_base_path, t5_trainer_provider, t5_datasets_provider, opt_trainer_provider, opt_datasets_provider):
  return {
      't5_s1': TuningConfig('t5', 'google/t5-v1_1-base', 
                  dataset='s1', max_len=65, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=128,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path),
      't5_s2': TuningConfig('t5', 'google/t5-v1_1-base', 
                  dataset='s2', max_len=110, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path),
      't5_s3': TuningConfig('t5', 'google/t5-v1_1-base', 
                  dataset='s3', max_len=150, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path),
      'opt_s2': TuningConfig('opt', 'facebook/opt-350m', 
                  dataset='s2', max_len=110, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=opt_trainer_provider,
                  datasets_provider=opt_datasets_provider,
                  prompt=OPT_PROMPT, 
                  project_base_path=project_base_path)
  }

class TuningConfig:
  def __init__(self, model_family, model_name, dataset, max_len, epochs,
               training_samples, val_samples, batch_size,
               trainer_provider, datasets_provider, prompt, 
               project_base_path):

    self.model_family = model_family
    self.model_name = model_name
    self.dataset = dataset
    self.max_len = max_len
    self.epochs = epochs
    self.training_samples = training_samples
    self.val_samples = val_samples
    self.train_batch_size = batch_size
    self.val_batch_size = 8
    self.project_base_path = project_base_path
    self.main_data_file = MAIN_DATA_FILE_FORMAT.format(project_base_path, dataset)
    self.train_val_data_file = TRAIN_VAL_FILE_FORMAT.format(project_base_path, dataset)
    self.test_data_file = TEST_FILE_FORMAT.format(project_base_path, dataset)
    self.tuned_model_path = SAVED_MODEL_PATH_FORMAT.format(project_base_path, model_name, dataset)
    self.trainer_provider = trainer_provider
    self.datasets_provider = datasets_provider
    self.prompt = prompt
    