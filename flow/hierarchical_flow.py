import os
import os.path as osp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
import torch


from flow.metrics import HuggingFaceMetricsComputer
from models.summarization.bart_model import SummarizerWithCustomTokenizer
from models.classifier.bert import CodeSearchNetClassifier
from preprocessing.datamodule import CodeSearchNetBERTModule
from preprocessing.dataset import CodeSearchNetBARTDataset
from preprocessing.data_generation import TokeinzerRawTrainingData, return_CodeSearchNet_dataframe


class HierarchicalCodeSummarizationModel:
    """
        A class with all the functionalities of a Hierarchical Summarization model.
        Objects of this class provide functionality like train, prediction and evalution,
         making it worthy of being called a model.
    """
    def __init__(self, pretrained_bert_model_name: str = "bert-base-uncased",
                 pretrained_bart_model_name: str = "ncoop57/summarization-base-code-summarizer-java-v0",
                 languages: list = ['python', 'java', 'javascript', 'php'],
                 bert_model_dir: str = "",
                 bart_model_dirs: list = ["", "", "", ""],
                 bart_tokenizer_dirs: list = ["", "", "", ""],
                 continue_training: bool = False,
                 initial_wts_dir: str = None):
        """
            Constructor

            :param pretrained_bert_model_name: hugging face model name
            :param pretrained_bart_model_name: hugging face model name
            :param languages: languages to train on from the CodeSearchNet dataset
            :param bert_model_dir: path to save the classification model in
            :param bart_model_dirs: list of paths to save the summarization models in
            :param bart_tokenizer_dirs: list of paths to save the tokenizer for each summarization model (programming-language)
        """

        self.df_code = None
        self.languages = languages
        self.initial_wts_dir = initial_wts_dir
        self.BERT_MODEL_NAME = pretrained_bert_model_name
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        if len(available_gpus) > 1:
            self.backend = "dp"  # ddp -> accelerator by default with multiple GPUs times out

        if initial_wts_dir:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(osp.join(initial_wts_dir, 'bert_tokenizer'))
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL_NAME)
        self.bert_model_path = bert_model_dir

        self.BART_MODEL_NAME = pretrained_bart_model_name
        self.bart_models = {}
        self.bart_tokenizers = {}
        self.bart_model_dirs = bart_model_dirs
        self.bart_tokenizer_dirs = bart_tokenizer_dirs

        if initial_wts_dir:
            self.bart_tokenizer = AutoTokenizer.from_pretrained(osp.join(initial_wts_dir, 'bart_tokenizer'))
        else:
            self.bart_tokenizer = AutoTokenizer.from_pretrained(self.BART_MODEL_NAME)

        for model_dir, tokenizer_dir, lang in zip(bart_model_dirs, bart_tokenizer_dirs, languages):
            if os.path.exists(model_dir) and continue_training:
                self.bart_models[lang] = BartForConditionalGeneration.from_pretrained(model_dir)
            elif self.initial_wts_dir:
                self.bart_models[lang] = BartForConditionalGeneration.from_pretrained(osp.join(initial_wts_dir, 'bart'))
            else:
                self.bart_models[lang] = BartForConditionalGeneration.from_pretrained(self.BART_MODEL_NAME)

            if os.path.exists(tokenizer_dir) and continue_training:
                self.bart_tokenizers[lang] = AutoTokenizer.from_pretrained(tokenizer_dir)
            else:
                self.bart_tokenizers[lang] = self.bart_tokenizer

        # predefining self variables
        self.bert_data_module = None
        self.classifier_model = CodeSearchNetClassifier(
            n_classes=len(self.languages),
            steps_per_epoch=1,
            n_epochs=0,
            bert_model_name=self.BERT_MODEL_NAME,
            initial_wts_dir=self.initial_wts_dir
        )

    def prepare_classifier_dataset(self, cache_dir: str = "", BATCH_SIZE: int = 32):
        """
            A function to prepare and load the dataset in memory for training

            :param BATCH_SIZE: batch-size for the dataset
            :return: None
        """
        self.df_code = return_CodeSearchNet_dataframe(languages=self.languages, cache_dir=cache_dir)
        self.BATCH_SIZE = BATCH_SIZE

        self.bert_data_module = CodeSearchNetBERTModule(self.df_code[self.df_code['set'] == 'train'],
                                                        self.df_code[self.df_code['set'] == 'validation'],
                                                        self.df_code[self.df_code['set'] == 'test'],
                                                        self.bert_tokenizer,
                                                        batch_size=self.BATCH_SIZE)
        self.bert_data_module.setup()

    def train_classifier(self, N_EPOCHS: int = 10, gpus: int = 4):
        """
            A function to train the classification head of the two level model

            :param N_EPOCHS: epochs to train for
            :param gpus: Number of GPUs to use for training (must be less than available)
            :return: None
        """
        model = CodeSearchNetClassifier(
            n_classes=len(self.languages),
            steps_per_epoch=len(self.df_code[self.df_code['set'] == 'train']) // self.BATCH_SIZE,
            n_epochs=N_EPOCHS,
            bert_model_name=self.BERT_MODEL_NAME,
            initial_wts_dir=self.initial_wts_dir
        )

        trainer = pl.Trainer(num_sanity_val_steps=0, default_root_dir=self.bert_model_path, 
                             accelerator=self.backend, max_epochs=N_EPOCHS,
                             gpus=gpus, progress_bar_refresh_rate=30,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                                        ModelCheckpoint(
                                            dirpath=self.bert_model_path,
                                            filename="bert",
                                            monitor="val_loss",
                                            mode="min",
                                            save_last=True,
                                            save_top_k=1
                                        )])

        trainer.fit(model, self.bert_data_module)
        self.classifier_model = model

    def train_summarizers(self, N_EPOCHS: int = 10):
        """
             A function to train the summarizers for each language

            :param N_EPOCHS: epochs to train each model for
            :return: None
        """
        for idx, lang in enumerate(self.languages):
            df_lang = self.df_code[self.df_code["language"] == lang]

            df_train = df_lang[df_lang["set"] == "train"]
            df_val = df_lang[df_lang["set"] == "validation"]

            tokenizer_data = TokeinzerRawTrainingData(dataset='code_search_net', names=[lang])
            tokenizer_data.raw_training_data()

            train_dataset = CodeSearchNetBARTDataset(df_train, self.bart_tokenizers[lang])
            val_dataset = CodeSearchNetBARTDataset(df_val, self.bart_tokenizers[lang])

            summarizer = SummarizerWithCustomTokenizer(tokenizer=self.bart_tokenizers[lang],
                                                       bart_model=self.bart_models[lang],
                                                       model_loc=self.bart_model_dirs[idx],
                                                       tokenizer_loc=self.bart_tokenizer_dirs[idx])
            summarizer.train_tokenizer(tokenizer_data=tokenizer_data,
                                       tokenizer_fname=f"hierarchical-tokenizer-{lang}")

            metrics = HuggingFaceMetricsComputer(summarizer.return_tokenizer())
            summarizer.train_model(train_dataset=train_dataset, val_dataset=val_dataset,
                                   compute_metrics=metrics.compute_metrics, n_epochs=N_EPOCHS)

    def predict(self):
        """
            A function to predict the summary using the train model

            :return:
        """
        pass

    def load(self, bert_tokenizer_dir, bert_model_dir, bart_tokenizers_dir: dict, bart_models_dir: dict):
        """
            A function to load the saved model from its path

        """
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_dir)
        self.classifier_model.load(bert_model_dir)

        for lang in self.languages:
            self.bart_tokenizers[lang] = AutoTokenizer.from_pretrained(bart_tokenizers_dir[lang])
            self.bart_models[lang] = BartForConditionalGeneration.from_pretrained(bart_models_dir[lang])

    def evaluate(self):
        """
            A function to perform evaluation on the set

            :return:
        """
        pass
