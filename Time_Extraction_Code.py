#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:15:24 2020

@author: vaibhavishah
"""

from datetime import datetime, time, date
import csv
import __main__
import pandas as pd
import datetime
import datefinder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.datasets import *

impr = 'Report Text'
fc = []
path = '.csv'
iandpname = ['Report Text', 'Preliminary Report Date']


def extractCommunication():
    oncalldf = pd.read_csv(path, skipinitialspace=True, usecols=iandpname)
    for index, row in oncalldf.iterrows():
        impression = str(row[impr])
        
        rel = impression.index('IMPRESSION:')
        reltext = impression[rel:]
        
        try:
            actrel = reltext.index('discussed')
            actreltext = reltext[actrel:]

            dates = datefinder.find_dates(actreltext)
            datelist = list(dates)
        
        except:
            try:
                actrel = reltext.index('communicated')
                actreltext = reltext[actrel:]

                dates = datefinder.find_dates(actreltext)
                datelist = list(dates)
            except:
                dates = datefinder.find_dates(reltext)
                datelist = list(dates)
            
        today = date.today()
        
        for i in datelist.copy():
            if (i.hour == 0 and i.minute == 0) :
                datelist.remove(i)
            elif (i.day == 0 or i.month == 0):
                datelist.remove(i)
            elif (i.month == today.month and i.day == today.day):
                datelist.remove(i)

        #if no communication exists, write NA, else use first date and time found (represents first communication)
        if datelist == []:
            fc.append('N/A')
        else:
            fc.append(datelist[0])