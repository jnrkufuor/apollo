#import packages
import numpy as np
import pandas as pd
import os
from util import Util
import seaborn as sns
ut = Util()

class Manipulator(object):

    def __init__(self):
        ''' Initialization function for data manipulator
        '''
        pass
    
    def read_csv(self,path_to_data):
        ''' Reads data from a file and returns a panda DataFrame

            :param path_to_data: path to data file
            :return dataframe: data frame of read in data
        '''
        dataframe =  pd.read_csv(path_to_data)
        return dataframe

    def convert_matrix_to_links(self,correlation_matrix=[pd.DataFrame()]):
        ''' Reads data from a file and returns a panda DataFrame

            :param path_to_data: path to data file
            :return dataframe: data frame of read in data
        '''
        #find unique pairs from correlation coeffecient
        for i in range(correlation_matrix.size()):
            correlation_matrix[i] = correlation_matrix[i].drop(correlation_matrix[i].columns[0], axis=1)
            correlation_matrix[i].index = correlation_matrix[i].columns
            matrix = correlation_matrix[i][abs(correlation_matrix[i]) >= 0.0001].stack().reset_index()
            
            matrix  = matrix[matrix['level_0'].astype(str)!=matrix['level_1'].astype(str)]
            matrix['ordered-cols'] = matrix.apply(lambda x: '-'.join(sorted([x['level_0'],x['level_1']])),axis=1)
            
            #Remove duplicates and exclude self-correlated values
            #for price
            matrix = matrix.drop_duplicates(['ordered-cols'])
            matrix.reset_index(drop=True, inplace=True)
            matrix.drop(['ordered-cols'], axis=1, inplace=True)
            
            #rename columns
            matrix.columns = ["from","to","correlation"]
            correlation_matrix[i] = matrix
        
        return correlation_matrix
    
    def news_to_correlation_matrix(self,news_data,draw=False):
        ''' Returns a correlation matrix of the news data

            :param news_data: from_to data frame of news data generated from NER.py
            :param draw: construt a heatmap of the data
            :return df_matrix: returns a correlation matrix of news co-mentions
        '''
        #Subset mews data. Count all links and store under weight column
        news_data = news_data.groupby(['from', 'to']).size().reset_index()
        news_data.rename(columns={0: 'weight'}, inplace=True)
        news_data.reset_index(drop=True, inplace=True)
        
        #Build Co-mention Matrix
        news_data[['from', 'to', 'weight']].sort_values('weight', ascending=False)
        col=[]

        #Extract Unique Columns
        for row in news_data.iterrows():
            if row[1]['from'] not in col:
                col.append(row[1]['from'])
            if row[1]['to'] not in col:
                col.append(row[1]['to'])

        df_matrix = pd.DataFrame(0,columns =col,index=col)

        for row in news_data.iterrows():
            df_matrix[row[1]['from']][row[1]['to']] = row[1]['weight']
            df_matrix[row[1]['to']][row[1]['from']] = row[1]['weight']
            df_matrix[row[1]['from']][row[1]['from']] = 1
            df_matrix[row[1]['to']][row[1]['to']] = 1
        
        if draw:
            sns.heatmap(df_matrix).set_title("Frequency heatmap for Comention Matrix")
        return df_matrix
            
    def subset_news_data(self,news_data,criteria):
        ''' Subsets news data and returns dataframe

            :param news_data: from_to data frame of news data generated from NER.py
            :param criteria: cut off criteria(A count variable greater than 1)
            :return df_matrix: returns a correlation matrix of news co-mentions
        '''
        #Subset mews data. Count all links and store under weight column
        news_data = news_data.groupby(['from', 'to']).size().reset_index()
        news_data.rename(columns={0: 'weight'}, inplace=True)
        news_data.reset_index(drop=True, inplace=True)
        
        #normalize values and create co-mention matrix - Use Z-score normmalization?
        #df_links['weight'] =(df_links['weight']-df_links['weight'].min())/(df_links['weight'].max()-df_links['weight'].min())

        #Use Hyper parameter for now
        news_data = news_data[news_data['weight'] > criteria]
       
        return news_data
        
        
    


if __name__ == "__main__":
    m = Manipulator()

    ut.print_to_csv(news,"news.csv")