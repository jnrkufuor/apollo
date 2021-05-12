# import packages
import pandas as pd
import string
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from util import Util
from itertools import combinations, product
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
from datetime import datetime
from nltk import tokenize

import nltk
nltk.download('punkt')

# is cuda available?
torch.cuda.is_available()

ut = Util()


class Processor:

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
            - the first is a dataframe of all unique entities (persons and orgs)
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
    
    
    def process_file(self):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
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

        # apply function
        df_domain = self.data.groupby('media').agg({'content': 'count'}).reset_index()
        df_domain.columns = ['media', 'count']
        df_domain = df_domain.sort_values('count', ascending=False)
        dfd_small = df_domain.iloc[1:21, :]

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
    
    def create_comention_matrix(self,df_links,draw=False):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        #Build Co-mention Matrix
        df_links[['from', 'to', 'weight']].sort_values('weight', ascending=False)
        col=[]

        #Extract Unique Columns
        for row in df_links.iterrows():
            if row[1]['from'] not in col:
                col.append(row[1]['from'])
            if row[1]['to'] not in col:
                col.append(row[1]['to'])

        df_matrix = pd.DataFrame(0,columns =col,index=col)

        for row in df_links.iterrows():
            df_matrix[row[1]['from']][row[1]['to']] = row[1]['weight']
            df_matrix[row[1]['to']][row[1]['from']] = row[1]['weight']
            df_matrix[row[1]['from']][row[1]['from']] = 1
            df_matrix[row[1]['to']][row[1]['to']] = 1
        
        if draw:
            sns.set(rc={'figure.figsize':(20,15)})
            sns.heatmap(df_matrix).set_title("Frequency heatmap for Comention Matrix")
        return df_matrix
    
    def correlation_matrix_to_dataframe(self, df_data):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        df_prices = df_data.drop(df_data.columns[0], axis=1)
        df_prices.index = df_prices.columns

        # Get correlation pairs for Price and Volume
        df_corr = df_data[abs(df_data) >= 0.0001].stack().reset_index()

        #Take out lower triangle 
        df_corr  = df_corr[df_corr['level_0'].astype(str)!=df_corr['level_1'].astype(str)]
        df_corr['ordered-cols'] = df_corr.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)

        #Remove duplicates and exclude self-correlated values
        df_data = df_corr.drop_duplicates(['ordered-cols'])
        df_data.reset_index(drop=True, inplace=True)
        df_data.drop(['ordered-cols'], axis=1, inplace=True)

        #rename columns
        df_data.columns = ["from","to","correlation"]
        return df_corr
    
    def subset_financial_data(self,df_data_list,upper=0.8,lower=0.5):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        df_finance_nds = pd.DataFrame(columns = ["from", "to", "weight"])
        df_corr_price = df_data_list[2]
        df_corr_vol = df_data_list[1]
        for i in range(1,len(df_corr_price)):
            if(abs(df_corr_price["correlation"][i]) > upper or abs(df_corr_vol["correlation"][i]) > upper):
                df_finance_nds= df_finance_nds.append({"from" : df_corr_vol["from"][i], "to" : df_corr_vol["to"][i], "weight" : ((abs(df_corr_price["correlation"][i])+abs(df_corr_price["correlation"][i]))/2)},ignore_index=True)
            elif (abs(df_corr_price["correlation"][i]) < upper and abs(df_corr_vol["correlation"][i]) < upper):
                if (abs(df_corr_price["correlation"][i]) >= lower and abs(df_corr_vol["correlation"][i]) >= lower):
                    df_finance_nds = df_finance_nds.append({"from" : df_corr_vol["from"][i], "to" : df_corr_vol["to"][i], "weight" : ((abs(df_corr_price["correlation"][i])+abs(df_corr_price["correlation"][i]))/2)},ignore_index=True)


    def subset_news_data(self,df_links,criteria=4):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        df_links = df_links[df_links['weight'] > criteria]
        return df_links
    
    def interval_price_by_quarter(self,df_price,start_dates,end_dates,i):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
                - the third is 
        '''
        df_price_pres = df_price[(df_price.index >= start_dates[3*i]) & (df_price.index <= end_dates[3*i+2])]
        df_price_next = df_price[(df_price.index >= start_dates[3*(i+1)]) & (df_price.index <= end_dates[3*(i+1)+2])]
        df_price_pct_pres = df_price_pres.pct_change().dropna(how='all')
        df_price_pct_next = df_price_next.pct_change().dropna(how='all')
        price_corr = df_price_pct_pres.corr()
        return [df_price_pres,df_price_next,price_corr]
    
    def generate_month_labels(self,df_price_pres,df_price_next):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        month_pct_chg=df_price_next.mean(axis=0) - df_price_pres.mean(axis=0)
        month_chg_label=pd.Series(np.zeros(len(month_pct_chg)))
        for index in range(0, len(month_pct_chg),1):
            if month_pct_chg[index] > 0:
                month_chg_label[index]=1
            else:
                month_chg_label[index]=0

        month_chg_label.index=month_pct_chg.index
        return month_chg_label

    def interval_news_by_quarter(self,df_links,start_dates,i):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        df_links = self.create_new_dates(df_links)
        df_links_pres = df_links[(df_links['date'] >= start_dates[3*i]) & (df_links['date'] <= start_dates[3*i+2])]
        df_links_pres = df_links_pres.groupby(['from', 'to']).size().reset_index()
        df_links_pres.rename(columns={0: 'weight'}, inplace=True)
        df_links_pres.reset_index(drop=True, inplace=True)
        df_links_pres = df_links_pres[df_links_pres['weight'] > 1]

        return df_links
    
    def create_new_dates(self,df_links):
        ''' Function to process news data and return organisation links in the format of from to and count

            :returns two data frames:
                - the first is a dataframe of all unique entities (persons and orgs)
                - the second is the links between the entities
        '''
        new_dates=[]
        for date in df_links['date']:
            reg_date=re.sub("^.*?([A-Z])", "\\1", date)
            temp_date=datetime.strptime(reg_date,"%b %d, %Y")
            new_dates.append(pd.to_datetime(datetime.strftime(temp_date, "%Y-%m-%d")))

        df_links['date']=new_dates
        return df_links
    
    
if __name__ == "__main__":
    n = Processor('.\\data\\news.csv')
    
    [ner,links]=n.process_file()
  
    ut.print_to_csv(links,"df_links.csv")
    ut.print_to_csv(ner,"df_ner.csv")