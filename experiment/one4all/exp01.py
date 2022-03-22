import numpy as np
import random
import torch

from flow.one4all_flow import One4AllCodeSummarizationModel

if __name__ == "__main__":
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
    model_location = './One4All/'
    BATCH_SZ = 32
    N_EPOCHS = 10
    gpus = 1
    # endregion

    # region initialize the model
    model = One4AllCodeSummarizationModel(
        bart_model_name="ncoop57/bart-base-code-summarizer-java-v0",
        languages=['python', 'java', 'javascript', 'php'],
        bart_model_dir=model_location, bart_tokenizer_dir=model_location
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
