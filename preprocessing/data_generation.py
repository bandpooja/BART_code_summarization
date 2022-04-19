from datasets import load_dataset, concatenate_datasets, load_from_disk
import os.path as osp
import pandas as pd


def return_CodeSearchNet_dataframe(languages: list = ['python', 'java', 'javascript', 'php'], cache_dir: str = None):
    """
        A function to load the CodeSearchNet dataset
    """
    # making sure that we have the all the languages in the set
    assert 'python' in languages, Exception("dataset only defined for python, java, js and php")
    assert 'java' in languages, Exception("dataset only defined for python, java, js and php")
    assert 'javascript' in languages, Exception("dataset only defined for python, java, js and php")
    assert 'php' in languages, Exception("dataset only defined for python, java, js and php")
    
    if cache_dir:
        dataset_java = load_from_disk(osp.join(cache_dir, 'java'))
        dataset_python = load_from_disk(osp.join(cache_dir, 'java'))
        dataset_javascript = load_from_disk(osp.join(cache_dir, 'javascript'))
        dataset_php = load_from_disk(osp.join(cache_dir, 'php'))        
    else:
        dataset_java = load_dataset('code_search_net', 'java')
        dataset_python = load_dataset('code_search_net', 'python')
        dataset_javascript = load_dataset('code_search_net', 'javascript')
        dataset_php = load_dataset('code_search_net', 'php')

    df_code = pd.DataFrame()
    codes = []
    languages = []
    set_ = []
    summaries = []

    for data, lang in zip([dataset_java, dataset_python, dataset_javascript, dataset_php],
                          ['java', 'python', 'javascript', 'php']):
        for split in ['train', 'validation', 'test']:
            for code, summary in zip(data[split]['whole_func_string'], data[split]['func_documentation_string']):
                codes.append(code)
                languages.append(lang)
                set_.append(split)
                summaries.append(summary)

    df_code['code'] = codes
    df_code['language'] = languages
    df_code['set'] = set_
    df_code['summary'] = summaries

    return df_code


class TokeinzerRawTrainingData:
    """
        A class to generate dataset for tokenizer training
    """
    def __init__(self, dataset: str = 'code_search_net', names: list = ['python', 'java', 'javascript', 'php']):
        self.dataset = dataset
        self.names = names
        self.batch_size = 1000
        self.dataset_ = None

    def raw_training_data(self):
        """
            Returns raw training data useful for training the tokenizer.
        """
        datasets = []
        for name in self.names:
            datasets.append(load_dataset(self.dataset, name=name, split="train"))
        self.dataset_ = concatenate_datasets(datasets)

    def batch_iterator(self):
        for i in range(0, len(self.dataset_), self.batch_size):
            yield self.dataset_[i:(i+self.batch_size)]['whole_func_string']
