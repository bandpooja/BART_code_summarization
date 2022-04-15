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
        "--bart_model_name",
        required=False,
        default="ncoop57/bart-base-code-summarizer-java-v0",
        help="required to load model"
    )

    args = parser.parse_args()
    return (
        args.model_loc,
        args.bart_model_name
    )


if __name__ == "__main__":
    model_loc, bart_model_name = arg_parse()
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
    BATCH_SZ = 8
    N_EPOCHS = 10
    gpus = 4
    # endregion

    # region initialize the model
    model = One4AllCodeSummarizationModel(
        bart_model_name=bart_model_name,  # "ncoop57/bart-base-code-summarizer-java-v0",
        languages=['python', 'java', 'javascript', 'php'],
        bart_model_dir=model_location, bart_tokenizer_dir=model_location,
        continue_training=False
    )
    # endregion

    # region prepare dataset for training
    model.prepare_dataset()
    # endregion
    # region train summarizer
    model.train_tokenizer()
    model.train_model(N_EPOCHS=N_EPOCHS)
    # endregion

    # todo: add prediction, evaluation and model-loading
