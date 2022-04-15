import argparse
import numpy as np
import os.path as osp
import random
import torch

from flow.hierarchical_flow import HierarchicalCodeSummarizationModel


def arg_parse():
    parser = argparse.ArgumentParser(description='Input Variables')
    parser.add_argument(
        "--model_loc", "-o",
        required=False,
        default="./One4All",
        help="directory to save the output in"
    )
    parser.add_argument(
        "--bart_model_name",
        required=False,
        default="ncoop57/bart-base-code-summarizer-java-v0",
        help="required to load model"
    )
    parser.add_argument(
        "--bert_model_name",
        required=False,
        default="bert-base-uncased",
        help="required to load model"
    )

    args = parser.parse_args()
    return (
        args.model_loc,
        args.bart_model_name,
        args.bert_model_name
    )


if __name__ == "__main__":
    model_loc, bart_model_name, bert_model_name = arg_parse()
    """
        A wrapper to use the pipeline to train the Hierarchical Code Summarization Model
        
        Flow
            - set the random set
            - prepare the dataset
            - initiate the model
            - train the classification tail
            - train the summarization head
    """

    # region set random seeds
    RANDOM_SEED = 2022
    torch.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)
    # endregion

    # region set parameters for training
    # model_location = './Hierarchical/'
    model_location = model_loc
    BATCH_SZ = 8
    N_EPOCHS_CLS = 10
    N_EPOCHS_SUM = 10
    gpus = 4  # 1
    # endregion

    # region initialize the model
    model = HierarchicalCodeSummarizationModel(
        pretrained_bert_model_name=bert_model_name,  # "bert-base-uncased",
        pretrained_bart_model_name=bart_model_name,  # "ncoop57/bart-base-code-summarizer-java-v0",
        languages=['python', 'java', 'javascript', 'php'],
        bert_model_dir=osp.join(model_location, 'BERT'),
        bart_model_dirs=[osp.join(model_location, 'BART', 'python'), osp.join(model_location, 'BART', 'java'),
                         osp.join(model_location, 'BART', 'javascript'), osp.join(model_location, 'BART', 'php')],
        bart_tokenizer_dirs=[osp.join(model_location, 'BART', 'python'), osp.join(model_location, 'BART', 'java'),
                             osp.join(model_location, 'BART', 'javascript'), osp.join(model_location, 'BART', 'php')],
        continue_training=False
    )
    # endregion

    # region prepare and load the dataset
    model.prepare_classifier_dataset(BATCH_SIZE=32)
    # endregion
    # region train the language classification tail
    model.train_classifier(N_EPOCHS=N_EPOCHS_CLS, gpus=gpus)
    # endregion
    # region train the summarizer head
    model.train_summarizers(N_EPOCHS=N_EPOCHS_SUM)
    # endregion

    # todo: add prediction and loading functions
