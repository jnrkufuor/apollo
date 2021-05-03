#import packages
import numpy as np
import pandas as pd
import os
from apollo_util import Util
from GoogleNews import GoogleNews
from goose3 import Goose
from goose3.configuration import Configuration


class Scraper:

    def __init__(self, company_list, period="None", date_range="None"):
        ''' Initialization function for data scraper 

            :param period: Optional choose period (period and custom day range should not set together)
            :param date_range: A list of the start and end date you want to fetch news on.  (mm/dd/yyyy) [Start, End] ~ ["02-12-2002","02-12-2020"] 
            :param company_list: A list of the companies you want to filter the news on. Example: ["MSFT","APPL","VX"]
        '''
        if period != "None":
            self.period_exists=True
            self.period = period
            self.gns = GoogleNews(period='7d')
        elif date_range != "None":
            self.date_range_exists=True
            self.date_range = date_range
            self.gns = GoogleNews(start=date_range[0], end=date_range[1])
        else:
            self.period=period
            self.date_range=date_range
            self.gns = GoogleNews()
        self.company_list = company_list

    def set_date_range(self, date_range):
        ''' Function to set the date range for the scraper object

            :param date_range: A list of the start and end date you want to fetch news on.  (mm/dd/yyyy) [Start, End] ~ ["02-12-2002","02-12-2020"] 
        '''
        self.gns.clear()
        self.period="None"
        self.date_range = date_range
        self.gns.set_time_range(start=date_range[0], end=date_range[1])

    def set_period(self, period):
        ''' Function to set the period

            :param period: Optional choose period (period and custom day range should not set together)
        '''
        self.gns.clear()
        self.date_range="None"
        self.period = period
        self.gns.set_period(period)

    def set_company_list(self, company_list):
        ''' Function to set the list of companies to filter on

            :param company_list: A list of the companies you want to filter the news on. Example: ["MSFT","APPL","VX"]
        '''
        self.company_list = company_list

    def fetch_news_data(self,num_of_articles):
        ''' Function to fetch news based on set parameters

            :param num_of_articles: number of articles to fetch
            :return newsframe: dataframe with news content
        '''
        newsframe = pd.DataFrame()
        for company in self.company_list:
            self.gns.search(company)
            df_comp = pd.DataFrame(self.gns.result()).iloc[0:10, ]
            newsframe = newsframe.append(df_comp)
            self.gns.clear()
        
        newsframe["content"]=np.zeros(len(newsframe.iloc[:,1]))
        g = Goose()
        with Goose({'http_timeout': 5.0}) as g:
            pass
        for i in range(0,num_of_articles):
            try:
                newsframe.iloc[i,7]=(g.extract(url=newsframe.iloc[i,5])).cleaned_text
            except:
                newsframe.iloc[i,7]="Missing_Article"
        return newsframe

    def check_status(self):
        ''' Function to print instance variables
        '''
        print(self.company_list)
        print(self.gns)
        if self.period != "None":
            print(self.period)
        if self.date_range != "None":
            print(self.date_range)



if __name__ == "__main__":
    s = Scraper(["AAPL", "MSFT"])
    ut = Util()
    s.set_period("1m")
    #s.set_date_range(["02-12-2002","02-12-2020"])

    #s.check_status()
    news = s.fetch_news_data(10)
    #Util.print_to_csv(news,"news.csv","data")
    ut.print_to_csv(news,"news.csv")