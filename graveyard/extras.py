class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class StoryDataIterator:    
    def __init__(self,
                 tokenizer,
                 n_examples,
                 max_load_at_once,
                 data_filename,
                 src_max_length=512,
                 target_max_length=128,
                 shuffle=True):
        
        self.tokenizer = tokenizer
        self.n_examples = n_examples
        self.max_load_at_once = max_load_at_once
        self.data_filename = data_filename
        self.src_max_length = src_max_length
        self.target_max_length = target_max_length
        self.shuffle = shuffle
        
        # Initialize row order, call on_epoch_end to shuffle row indices
        self.row_order = np.arange(1, self.n_examples+1)
        self.on_epoch_end()

        # Load first chunk of max_load_at_once examples
        self.df_curr_loaded = self._load_next_chunk(0)
        self.curr_idx_in_load = 0

    def preprocess_data(self, text_pair):
        orig_text, target_text = text_pair
        orig_encoded = self.tokenizer.batch_encode_plus(
            [orig_text],
            max_length=self.src_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        orig_input_ids = orig_encoded['input_ids'][0]
        orig_attention_mask = orig_encoded['attention_mask'][0]
        
        target_encoded = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        label_ids = target_encoded['input_ids'][0]
        
        return {'input_ids': orig_input_ids,
                'attention_mask': orig_attention_mask,
                'labels': label_ids}

    def _load_next_chunk(self, idx):
        load_start = idx
        load_end = idx + self.max_load_at_once

        # Indices to skip are the ones in the shuffled row_order before and
        # after the chunk we'll use for this chunk
        load_idx_skip = self.row_order[:load_start] + self.row_order[load_end:]
        self.df_curr_loaded = pd.read_csv(self.data_filename, skiprows=load_idx_skip)
        self.df_curr_loaded = self.df_curr_loaded.sample(frac=1)
    
    def __len__(self):
        return self.n_examples
    
    def __getitem__(self, idx):
        if self.df_curr_loaded is None or self.curr_idx_in_load >= len(self.df_curr_loaded):
            self._load_next_chunk(idx)
            self.curr_idx_in_load = 0
        
        text_pair = self.df_curr_loaded[['variable', 'label']].values.astype(str)[self.curr_idx_in_load]
        self.curr_idx_in_load += 1
        
        item_data = self.preprocess_data(text_pair)        
        return item_data
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.row_order = list(np.random.permutation(self.row_order))


# train_data_iterator = StoryDataIterator(
#     tokenizer=t5_tokenizer,
#     n_examples=NUM_TRAIN_SAMPLES,
#     max_load_at_once=MAX_LOAD_AT_ONCE,
#     data_filename=TRAIN_DATA_FILE,
#     src_max_length=SRC_MAX_LENGTH,
#     target_max_length=TARGET_MAX_LENGTH
# )

# val_data_iterator = StoryDataIterator(
#     tokenizer=t5_tokenizer,
#     n_examples=NUM_VAL_SAMPLES,
#     max_load_at_once=MAX_LOAD_AT_ONCE,
#     data_filename=VAL_DATA_FILE,
#     src_max_length=SRC_MAX_LENGTH,
#     target_max_length=TARGET_MAX_LENGTH
# )
