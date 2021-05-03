#import packages
import numpy as np
import pandas as pd
import os
from GoogleNews import GoogleNews


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

    def fetch_news_data(self):
        ''' Function to fetch news based on set parameters

            :return newsframe: dataframe with news content
        '''
        newsframe = pd.DataFrame()
        for company in self.company_list:
            self.gns.search(company)
            df_comp = pd.DataFrame(self.gns.result()).iloc[0:10, ]
            newsframe = newsframe.append(df_comp)
            self.gns.clear()
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
    
    def print_to_csv(self, news, outname="news.csv", outdir="data"):
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
        print(fullname)
        print("Printing to file")
        news.to_csv(fullname)
        print("done")


if __name__ == "__main__":
    s = Scraper(["AAPL", "MSFT"])
    
    #s.set_period("1m")
    #s.set_date_range(["02-12-2002","02-12-2020"])

    #s.check_status()
    #news = s.fetch_news_data()
    #s.print_to_csv(news)
