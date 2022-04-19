import argparse
import numpy as np
import random
import torch

from flow.one4all_flow import One4AllCodeSummarizationModel


def arg_parse():
    parser = argparse.ArgumentParser(description='Input Variables')
    parser.add_argument(
        "--model_loc", "-o",
        required=False,
        default="./One4All",
        help="directory to save the output in"
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        default=None,
        help="required to load local data"
    )
    parser.add_argument(
        "--bart_model_name",
        required=False,
        default="ncoop57/bart-base-code-summarizer-java-v0",
        help="required to save model"
    )
    parser.add_argument(
        "--initial_wts_dir",
        required=False,
        default=None,
        help="required to load model"
    )

    args = parser.parse_args()
    return (
        args.model_loc,
        args.cache_dir,
        args.bart_model_name,
        args.initial_wts_dir
    )


if __name__ == "__main__":
    model_loc, cache_dir, bart_model_name, initial_wts_dir = arg_parse()
    """
        A wrapper to use the pipeline to train the Hierarchical Code Summarization Model

        Flow
            - set the random set
            - prepare the dataset
            - initiate the model
            - train the summarization head
    """

    # region set random seeds
    RANDOM_SEED = 2022
    torch.manual_seed(2022)
    np.random.seed(2022)
    random.seed(2022)
    # endregion

    # region set parameters for training
    # model_location = './One4All/'
    model_location = model_loc
    BATCH_SZ = 128
    N_EPOCHS = 10
    gpus = 4
    # endregion
    print('defining model')
    # region initialize the model
    model = One4AllCodeSummarizationModel(
        bart_model_name=bart_model_name,  # "ncoop57/bart-base-code-summarizer-java-v0",
        languages=['python', 'java', 'javascript', 'php'],
        bart_model_dir=model_location, bart_tokenizer_dir=model_location,
        continue_training=False,
        initial_wts_dir=initial_wts_dir
    )
    # endregion
    print('preparing dataset')
    # region prepare dataset for training
    model.prepare_dataset(cache_dir=cache_dir)
    # endregion
    # region train summarizer
    print('try to train tokenizer')
    model.train_tokenizer()
    print('train model')
    model.train_model(N_EPOCHS=N_EPOCHS, gpus=gpus)
    # endregion

    # todo: add prediction, evaluation and model-loading
