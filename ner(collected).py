# import packages
import pandas as pd
import string
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from apollo_util import Util
from itertools import combinations, product
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import tokenize

import nltk
nltk.download('punkt')

# is cuda available?
torch.cuda.is_available()

ut = Util()


class NER:

    def __init__(self, path_to_data):
        ''' Initialization function for named entity recognition parts

            :param path_to_data: Path to news content
        '''
        self.tagger = SequenceTagger.load('ner')
        self.data = pd.read_csv(path_to_data)
        self.df_contraptions=pd.DataFrame()
    
    def remove_s(self, entity, entity_series):
        ''' Function to remove plurals from word

            :param entity: dataframe containing news data
            :param entity series: name of file to be saved
            :return: returns an entity
        '''
        if (entity[-1] == 's') & (entity[:-1] in entity_series):
            return entity[:-1]
        else:
            return entity
        
    
    def print_to_csv(self, data, outname, outdir="data"):
        ''' Function to print news variable to csv file
        
            :param news: dataframe containing news data
            :param outname: name of file to be saved
            :param outdir: target directory(will be created if does not exist)
        '''
        if os.name == "nt":
            outdir = ".\\"+outdir
        if os.name == "posix":
            outdir = "./"+outdir
            
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            
        fullname = os.path.join(outdir, outname)
        print("Printing to "+fullname)
        data.to_csv(fullname)
        print("done")
    
    def get_ner_data(self,df_row):
        '''Function to extract named entities from a paragraph
           
           :param df_row: Row within the data frame
           :returns two data frames:
            - the first is a dataframe of all unique entities (orgs)
            - the second is the links between the entities
        '''
        paragraph=df_row.content
        # changed above row
        # remove newlines and odd characters
        paragraph = re.sub('\r', '', paragraph)
        paragraph = re.sub('\n', ' ', paragraph)
        paragraph = re.sub("’s", '', paragraph)
        paragraph = re.sub("“", '', paragraph)
        paragraph = re.sub("”", '', paragraph)

        
        # tokenise sentences
        sentences = tokenize.sent_tokenize(paragraph)
        sentences = [Sentence(sent) for sent in sentences]
        
        # predict named entities
        for sent in sentences:
            self.tagger.predict(sent)
        
        # collect sentence NER's to list of dictionaries
        sent_dicts = [sentence.to_dict(tag_type='ner') for sentence in sentences]
        
        # collect entities and types
        entities = []
        types = []
        for sent_dict in sent_dicts:
            entities.extend([entity['text'] for entity in sent_dict['entities']])
            types.extend([str(entity['labels'])[1:4] for entity in sent_dict['entities']])

        # create dataframe of entities (nodes)
        df_ner = pd.DataFrame(data={'entity': entities, 'type': types})
        df_ner = df_ner[df_ner['type'].isin(['ORG'])]
        df_ner = df_ner[df_ner['entity'].map(lambda x: isinstance(x, str))]
        df_ner = df_ner[~df_ner['entity'].isin(self.df_contraptions['contraption'].values)]
        df_ner['entity'] = df_ner['entity'].map(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        df_ner = df_ner.drop_duplicates().sort_values('entity')
        
        # get entity combinations
        combs = list(combinations(df_ner['entity'], 2))
        
        # create dataframe of relationships (edges)
        df_links = pd.DataFrame(data=combs, columns=['from', 'to'])
        
        # Adding information to links for data tracking and visualization- use one section OR the other depending on data source
        df_links['title']=df_row.title
        df_links['date']=df_row.date
        
        return df_ner, df_links
    
    
    def processfile(self):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (orgs)
                - the second is the links between the entities
        '''
        pronouns = ['I', 'You', 'It', 'He', 'She', 'We', 'They']
        df_ner = pd.DataFrame()
        df_links = pd.DataFrame()

        # remove pronouns
        suffixes = ["", "’m", "’re", "’s", "’ve", "’d", "'m",
            "'re", "'s", "'ve", "'d", "m", "re", "s", "ve", "d"]
        contraptions = [(p, s) for p in pronouns for s in suffixes]
        self.df_contraptions = pd.DataFrame(contraptions, columns=['pronoun', 'suffix'])
        self.df_contraptions['contraption'] = self.df_contraptions.apply(lambda x: x['pronoun'] + x['suffix'], axis=1)
        contraptions = self.df_contraptions.contraption.values

        # Count the number of content extracted in each media article
        df_domain = self.data.groupby('media').agg({'content': 'count'}).reset_index()
        df_domain.columns = ['media', 'count']
        df_domain = df_domain.sort_values('count', ascending=False)
        dfd_small = df_domain.iloc[1:21, :]
        # apply function
        for row in tqdm(self.data.iloc[0:20, :].itertuples(index=False)):
        # changed above row
            try:
                df_ner_temp, df_links_temp = self.get_ner_data(row)
                df_ner = df_ner.append(df_ner_temp)
                df_links = df_links.append(df_links_temp)
            except:
                continue

        # remove plurals and possessives
        df_links['to'] = df_links['to'].map(lambda x: self.remove_s(x, df_ner['entity'].values))
        df_links['from'] = df_links['from'].map(lambda x: self.remove_s(x, df_ner['entity'].values))
        df_ner['entity_cl'] = df_ner['entity'].map(lambda x: self.remove_s(x, df_ner['entity'].values))
        df_links[df_links['to'].str.contains('They')]
        return df_ner, df_links
    
    
if __name__ == "__main__":
    n = NER('.\\data\\news.csv')
    
    [ner,links]=n.processfile()
  
    ut.print_to_csv(links,"df_links.csv")
    ut.print_to_csv(ner,"df_ner.csv")