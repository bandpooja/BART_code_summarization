import os
import os.path as osp
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from preprocessing.data_generation import TokeinzerRawTrainingData


class SummarizerWithPretrainedTokenizer:
    """
        A class to train the model using a prertained tokenizer
    """
    def __init__(self,  tokenizer, bart_model, model_loc: str):
        self.model_loc = model_loc
        self.logs_loc = osp.join(model_loc, 'logs')
        os.makedirs(self.logs_loc, exist_ok=True)

        self.tokenizer = tokenizer
        self.bart_model = bart_model

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.bart_model
        )
        self.trainer = None

    def train(self, train_dataset, val_dataset, compute_metrics,
              n_epochs: int = 10, batch_sz: int = 16, label_smoothing: float = 0.1,
              logging_steps: int = 500):

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.model_loc,
            num_train_epochs=n_epochs,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=batch_sz,
            per_device_eval_batch_size=batch_sz,
            predict_with_generate=True,
            label_smoothing_factor=label_smoothing,
            logging_dir=self.logs_loc,
            gradient_accumulation_steps=batch_sz,
            gradient_checkpointing=True,
            fp16=True,
            # logging_steps=logging_steps,
            logging_strategy = "epoch",
            save_steps=500000,
            eval_accumulation_steps=100000
        )

        self.trainer = Seq2SeqTrainer(
            model=self.bart_model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        self.trainer.train()


class SummarizerWithCustomTokenizer:
    """
        A class to train the model and tokenizer
    """
    def __init__(self,  tokenizer, bart_model, model_loc: str, tokenizer_loc: str):
        self.model_loc = model_loc
        self.tokenizer_loc = tokenizer_loc
        self.logs_loc = osp.join(model_loc, 'logs')
        
        os.makedirs(self.model_loc, exist_ok=True)
        os.makedirs(self.tokenizer_loc, exist_ok=True)
        os.makedirs(self.logs_loc, exist_ok=True)

        self.tokenizer = tokenizer
        self.bart_model = bart_model

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.bart_model
        )
        self.trainer = None

    def train_tokenizer(self, tokenizer_data: TokeinzerRawTrainingData, tokenizer_fname: str = "one4all-tokenizer"):
        # making sure that it's a fast tokenizer because only they can be trained
        assert self.tokenizer.is_fast

        self.tokenizer = self.tokenizer.train_new_from_iterator(tokenizer_data.batch_iterator(),
                                                                vocab_size=30000)

        # save the toke
        # saving the tokenizer where we the model is to be saved
        self.tokenizer.save_pretrained(osp.join(self.tokenizer_loc, tokenizer_fname))

    def train_model(self, train_dataset, val_dataset, compute_metrics,
                    n_epochs: int = 10, batch_sz: int = 16, label_smoothing: float = 0.1,
                    logging_steps: int = 500, gpus: int = 4):
        
        if gpus > 0:
            gpus = gpus
        else:
            # using CPU but the batch-size per device will get a zero division error if this is set to 0
            gpus = 1

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.model_loc,
            num_train_epochs=n_epochs,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=int(batch_sz/gpus),
            per_device_eval_batch_size=int(batch_sz/gpus),
            predict_with_generate=True,
            label_smoothing_factor=label_smoothing,
            logging_dir=self.logs_loc,
            gradient_accumulation_steps=batch_sz,
            gradient_checkpointing=True,
            fp16=True,
            # logging_steps=logging_steps,
            logging_strategy = "epoch",
            save_steps=500000,
            eval_accumulation_steps=100000,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.bart_model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()

    def return_tokenizer(self):
        return self.tokenizer
