import os.path as osp
import random

from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm import tqdm

from flow.metrics import HuggingFaceMetricsComputer
from models.summarization.bart_model import SummarizerWithCustomTokenizer
from preprocessing.dataset import CodeSearchNetBARTDataset
from preprocessing.data_generation import return_CodeSearchNet_dataframe, TokeinzerRawTrainingData


class One4AllCodeSummarizationModel:
    """
            A class with all the functionalities of a One4All Summarization model.
            Objects of this class provide functionality like train, prediction and evalution,
             making it worthy of being called a model.
    """
    def __init__(self, bart_model_name: str = "ncoop57/summarization-base-code-summarizer-java-v0",
                 languages: list = ['python', 'java', 'javascript', 'php'],
                 bart_model_dir: str = "", bart_tokenizer_dir: str = "", continue_training: bool = False,
                 initial_wts_dir: str = None):
        """
             Constructor

            :param bart_model_name: hugging face model name
            :param languages: languages to train on from CodeSearchNet
            :param bart_model_dir: path to save the model in
            :param bart_tokenizer_dir: path to save the tokenizer in
        """

        self.BART_MODEL_NAME = bart_model_name
        self.bart_tokenizer_dir = bart_tokenizer_dir
        self.bart_model_dir = bart_model_dir
        self.languages = languages
        
        if initial_wts_dir:
            self.bart_model = BartForConditionalGeneration.from_pretrained(osp.join(initial_wts_dir, 'bart'))        
        elif osp.exists(self.bart_model_dir) and continue_training:
            self.bart_model = BartForConditionalGeneration.from_pretrained(self.bart_model_dir)
        else:
            self.bart_model = BartForConditionalGeneration.from_pretrained(self.BART_MODEL_NAME)

        if initial_wts_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(osp.join(initial_wts_dir, 'bart_tokenizer'))
        elif osp.exists(bart_tokenizer_dir) and continue_training:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bart_tokenizer_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BART_MODEL_NAME)
        
        self.df_code = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.tokenizer_data = None
        self.train_dataset = None
        self.val_dataset = None

        self.summarizer = SummarizerWithCustomTokenizer(tokenizer=self.tokenizer,
                                                        bart_model=self.bart_model,
                                                        model_loc=self.bart_model_dir,
                                                        tokenizer_loc=self.bart_tokenizer_dir)

    def prepare_dataset(self, cache_dir: str = None):
        """
            A function to prepare and load the datasets in memory for training

            :return: None
        """
        self.df_code = return_CodeSearchNet_dataframe(cache_dir=cache_dir)
        # self.df_train = self.df_code[self.df_code["set"] == "train"]
        # self.df_val = self.df_code[self.df_code["set"] == "validation"]
        # self.tokenizer_data = TokeinzerRawTrainingData(dataset='code_search_net',
        #                                                names=self.languages)
        # self.tokenizer_data.raw_training_data()
        #
        # self.train_dataset = CodeSearchNetBARTDataset(self.df_train, self.tokenizer)
        # self.val_dataset = CodeSearchNetBARTDataset(self.df_val, self.tokenizer)
        self.df_test = self.df_code[self.df_code["set"] == "test"]

    def train_tokenizer(self):
        """
            A function to train the tokenizers
        
            :return: None
        """
        # self.summarizer.train_tokenizer(tokenizer_data=self.tokenizer_data, tokenizer_fname="one4all-tokenizer")

    def train_model(self, N_EPOCHS: int = 10, gpus: int = 4, batch_sz: int = 16):
        """
            A function to train the summarizer model
        """
        metrics = HuggingFaceMetricsComputer(self.tokenizer)
        self.summarizer.train_model(train_dataset=self.train_dataset,
                                    val_dataset=self.val_dataset, compute_metrics=metrics.compute_metrics,
                                    n_epochs=N_EPOCHS, gpus=gpus, batch_sz=batch_sz)

    def predict(self, txt):
        """
            A function to predict the summary using the train model

            :return: predicted summary of the code
        """
        return self.summarizer.predict(txt)

    def load(self, tokenizer_dir, model_dir):
        """
            A function to load the saved model from its path
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_dir)
        self.summarizer.load(self.bart_model, self.tokenizer)

    def evaluate(self):
        """
            A function to perform evaluation on the set

            :return:
        """
        trues = []
        preds = []

        self.df_test = self.df_test.sample(n=100, replace=True, random_state=random.randint(0, 1000))
        for idx in tqdm(range(len(self.df_test)), desc="making predictions"):
            try:
                row_ = self.df_test.iloc[idx]
                preds.append(self.summarizer.predict(row_['code']))
                trues.append(row_['summary'])
            except:
                pass
        computer = HuggingFaceMetricsComputer(self.tokenizer)
        res = computer.compute_metrics((preds, trues))

        print(res)
        return res
