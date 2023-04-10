__version__='1.4'
SEED = 42
MAIN_DATA_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}.csv' #needs base path and dataset
TRAIN_VAL_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}_train_val.csv' #needs base path and dataset
TEST_FILE_FORMAT = '{}data/posptproc_corpus_spacy_{}_test.csv' #needs basepath and dataset
SAVED_MODEL_PATH_FORMAT = '{}saved_models/final/{}' # needs base path, model name
VALID_MODEL_FAMILIES = set(['t5', 'opt'])
VALID_DATASETS = set(['s1', 's2', 's3'])
T5_PROMPT = 'Generate next line: '
OPT_PROMPT = ''
B2B_PROMPT = ''
CHECKPOINTS_TO_SAVE = 1
TRAINING_SAMPLES = -1
VAL_SAMPLES = 10000

def create_configs(project_base_path, 
    t5_trainer_provider=None, t5_datasets_provider=None, 
    opt_trainer_provider=None, opt_datasets_provider=None, 
    b2b_trainer_provider=None, b2b_datasets_provider=None):
  return {'t5_s1': TuningConfig('t5_s1', 't5', 
                  'google/t5-v1_1-base',
                  'google/t5-v1_1-base-s1-finetuned', 
                  dataset='s1', max_len=65, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=128,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      't5_s2': TuningConfig('t5_s2', 't5', 
                  'google/t5-v1_1-base',
                  'google/t5-v1_1-base-s2-finetuned', 
                  dataset='s2', max_len=110, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      't5_s3': TuningConfig('t5_s3', 't5', 
                  'google/t5-v1_1-base',
                  'google/t5-v1_1-base-s3-finetuned', 
                  dataset='s3', max_len=150, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=t5_trainer_provider,
                  datasets_provider=t5_datasets_provider,
                  prompt=T5_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'opt_s2': TuningConfig('opt_s2', 'opt', 
                  'facebook/opt-350m',
                  'facebook/opt-350m-s2-finetuned', 
                  dataset='s2', max_len=110, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=opt_trainer_provider,
                  datasets_provider=opt_datasets_provider,
                  prompt=OPT_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'opt_s3': TuningConfig('opt_s3', 'opt', 
                  'facebook/opt-350m',
                  'facebook/opt-350m-s3-finetuned', 
                  dataset='s3', max_len=150, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=opt_trainer_provider,
                  datasets_provider=opt_datasets_provider,
                  prompt=OPT_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'b2b_s1': TuningConfig('b2b_s1', 'bert', 
                  'bert-base-cased',
                  'bert2bert/bert2bert_s1_e3_mxlen65', 
                  dataset='s1', max_len=65, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=128,
                  trainer_provider=b2b_trainer_provider,
                  datasets_provider=b2b_datasets_provider,
                  prompt=B2B_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'b2b_s2': TuningConfig('b2b_s2', 'bert', 
                  'bert-base-cased',
                  'bert2bert/bert2bert_s2_e3_mxlen110', 
                  dataset='s2', max_len=110, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=b2b_trainer_provider,
                  datasets_provider=b2b_datasets_provider,
                  prompt=B2B_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'b2b_s3': TuningConfig('b2b_s3', 'bert', 
                  'bert-base-cased',
                  'bert2bert/bert2bert_s3_e3_mxlen150', 
                  dataset='s3', max_len=150, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=b2b_trainer_provider,
                  datasets_provider=b2b_datasets_provider,
                  prompt=B2B_PROMPT, 
                  project_base_path=project_base_path,
                  tuned=True),
      'baseline': TuningConfig('baseline', 'opt', 
                  'facebook/opt-350m',
                  'facebook/opt-350m', 
                  dataset=None, max_len=150, epochs=3, 
                  training_samples=TRAINING_SAMPLES,
                  val_samples=VAL_SAMPLES, batch_size=64,
                  trainer_provider=opt_trainer_provider,
                  datasets_provider=opt_datasets_provider,
                  prompt='Continue the next sentence of the story making the language appropriate for kids between 6 and 12 years old: ', 
                  project_base_path=project_base_path,
                  tuned=False),
  }

class TuningConfig:
  def __init__(self, name, model_family, base_model, model_name, dataset, 
               max_len, epochs,
               training_samples, val_samples, batch_size,
               trainer_provider, datasets_provider, prompt, 
               project_base_path, tuned):
    self.name = name
    self.model_family = model_family
    self.base_model = base_model
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
    self.tuned = tuned
    
class T5Inferencer:
  def __init__(self, device, model, tokenizer, prompt='', max_new_tokens=100, tensor_type='pt', num_beams=3):
    self.model = model
    self.tokenizer = tokenizer
    self.prompt = prompt
    self.max_new_tokens = max_new_tokens
    self.tensor_type = tensor_type
    self.num_beams = num_beams
    self.device = device

  def __call__(self, context_lines):
    test_inputs = self.tokenizer([self.prompt + ' '.join(context_lines)], return_tensors=self.tensor_type)
    test_output_ids = self.model.generate(
        test_inputs['input_ids'].to(self.device),
        num_beams=self.num_beams,
        no_repeat_ngram_size=2,
        num_return_sequences=self.num_beams,
        max_new_tokens=self.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=0)
    decoded = [self.tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for out_ids in test_output_ids]
    return decoded

class OptInferencer:
  def __init__(self, device, model, tokenizer, max_new_tokens=50, tensor_type='pt', num_beams=3):
    self.model = model
    self.tokenizer = tokenizer
    self.max_new_tokens = max_new_tokens
    self.tensor_type = tensor_type
    self.num_beams = num_beams
    self.device = device

  def __call__(self, context_lines):
    test_inputs = self.tokenizer([' '.join(context_lines)], return_tensors=self.tensor_type)

    test_output_ids = self.model.generate(
      test_inputs['input_ids'].to(self.device),
      num_beams=self.num_beams,
      no_repeat_ngram_size=2,
      num_return_sequences=self.num_beams,
      max_length = self.max_new_tokens,
      do_sample=True,
      top_k=0,
      early_stopping=True
    )
    decoded = [self.tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace('\n', ' ') for out_ids in test_output_ids]
    return decoded

class B2BInferencer:
  def __init__(self, device, model, tokenizer, max_new_tokens=50, tensor_type='pt', num_beams=3):
    self.model = model
    self.tokenizer = tokenizer
    self.max_new_tokens = max_new_tokens
    self.tensor_type = tensor_type
    self.num_beams = num_beams
    self.device = device

  def __call__(self, context_lines):
    test_inputs = self.tokenizer([' '.join(context_lines)], return_tensors=self.tensor_type)
    test_output_ids = self.model.generate(
        test_inputs['input_ids'].to(self.device), 
        # attention_mask=attention_mask,
        max_new_tokens=self.max_new_tokens,
        num_beams=self.num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=self.num_beams)

    decoded = [self.tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).replace('\n', ' ') for out_ids in test_output_ids]
    return decoded