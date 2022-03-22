import datasets
import numpy as np


class HuggingFaceMetricsComputer:
    """
        A class to evalute metrics for logging during the model-training in hugging-face
    """
    def __init__(self, tokenizer):
        """
            Constructor

            :param tokenizer: tokenizer
        """
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_preds):
        """
            A function computing ROGUE and BLEU scores
            
            :param eval_preds: predictions of the model
            :return: {metric: score}
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # define the metrics
        metrics_rogue = datasets.load_metric('rouge')
        metrics_bleu = datasets.load_metric('bleu')

        result_bleu = metrics_bleu.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )["bleu"]

        # rogue computes the following scores ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        result_rogue = metrics_rogue.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        result = {
            "bleu": result_bleu,
            "rogue1": result_rogue["rogue1"],
            "rogue2": result_rogue["rogue2"],
            "rogueL": result_rogue["rogueL"],
            "rogueLsum": result_rogue["rogueLsum"],
        }
        return result
