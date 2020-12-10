#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:42:26 2020

@author: simon
"""


"""
This project aims to scrape and analyze customer reviews from Bestbuy.com.

Use monitors as an example.
However, the code works with any other BestBuy products too

Notice A:
The code may generate duplicated records if the "total verified purchases review is less than 21 (less than one page)"
Therefore, please select popular products to run the code or perform data cleaning afterward.

Notice B:
Bestbuy has the technology to DYNAMICLY change the elements on its website; therefore, the code cannot automatically scrape data for multiple products.

However, the code still works well with easy human intervention to solve the problem in Notice B:
    
Solution 1:
Scroll down the website manually and click "REVIEWS" when open a product page for the FIRST TIME

Solution 2:
Use the keyword "user-generated-content-ratings-and-reviews" to search the web script
and update the dynamic path in line #150

EXAMPLE:
    dynamic_path = "//div[@id='user-generated-content-ratings-and-reviews-5a4fb66c-c665-4f28-93b8-0e11db4ee25c']"

"""
# %%%% Preliminaries and library loading
import os
import pandas as pd
import numpy as np
import re
import shelve
import time
from datetime import datetime
import operator

import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.datasets                import load_iris
from sklearn.model_selection         import cross_val_score, cross_validate, ShuffleSplit, train_test_split, GridSearchCV
from sklearn.naive_bayes             import GaussianNB
from sklearn                         import tree
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import plot_tree
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes             import GaussianNB
from sklearn import tree
from sklearn import linear_model


from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
import statsmodels.formula.api as sm


# libraries to crawl websites
from bs4 import BeautifulSoup
from selenium import webdriver
from pynput.mouse import Button, Controller


#%%
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 0)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#%%
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)

# Please update the path before running the code
os.chdir("")
# Please update the path before running the code
path_to_driver = ""

#%% 
driver = webdriver.Chrome(executable_path=path_to_driver)


#%%
"This function scrape the reviews on a given webpage"
def search_current_page():
    time.sleep(4 + abs(np.random.normal(0,1))*2)
    
    reviews = driver.find_elements_by_xpath("//div[@class='review-item-content col-xs-12 col-md-8 col-lg-9']")
    
    for r in range(len(reviews)):
            one_review                   = {}
            one_review['scrapping_date'] = datetime.datetime.now()
            one_review['url']            = driver.current_url
            soup                         = BeautifulSoup(reviews[r].get_attribute('innerHTML'))  

            one_review['product_id'] = product_id_current
   
            # Get the raw review
            try:
                one_review_raw = reviews[r].get_attribute('innerHTML')
            except:
                one_review_raw = ""
                one_review['review_raw'] = one_review_raw
        
            # Get Posted Time
            try:
                review_posted_time = re.findall('[A-Z][a-z][a-z][a-z]* [0-9][0-9]*, [0-9][0-9][0-9][0-9] [0-9][0-9]*:[0-9][0-9] [A-Z][A-Z]',reviews[r].get_attribute('innerHTML'))[0]
            except:
                review_posted_time = ""
            one_review['review_posted_time'] = review_posted_time  
        
            # Get Review Title
            try:
                review_title = soup.find('div', attrs={'class':'review-heading'}).text[37:]
            except:
                review_title = ""
            one_review['review_title'] = review_title
        
            # Get Review Content
            try:
                review_text = soup.find('div', attrs={'class':'ugc-review-body body-copy-lg'}).text
            except:
                review_text = ""
            one_review['review_text'] = review_text

            # Get number of stars
            try:
                N_stars = re.findall('[0-5] out of [0-5] [Ss]tar',reviews[r].get_attribute('innerHTML'))[0][0]
            except:
                N_stars = ""
            one_review['N_stars'] = N_stars
        
            reviews_one_monitor.append(one_review)
            
    return reviews_one_monitor
        

#reviews = driver.find_elements_by_xpath("//div[@class='review-item-content col-xs-12 col-md-8 col-lg-9']")
#reviews[0].get_attribute('innerHTML')
#re.findall('[A-Z][a-z][a-z][a-z]* [0-9][0-9], [0-9][0-9][0-9][0-9] [0-9][0-9]*:[0-9][0-9] [A-Z][A-Z]',reviews[0].get_attribute('innerHTML'))[0]  


#%% 1. Data scraping
product_id          = []
reviews_one_monitor = []

#%%
"""
Products included:
    [A - Dell]
    A1. Dell - S2319NX 23" IPS LED FHD Monitor - Black/Silver
    A2. Dell - S2719DGF 27" LED QHD FreeSync Monitor (DisplayPort, HDMI) - Black
    A3. Dell - 27" IPS LED FHD FreeSync Monitor - Piano Black
    A4. Dell - 32" LED Curved QHD FreeSync Monitor with HDR (DisplayPort, HDMI, USB)
    
    [B - LG]
    B1. LG - 24" IPS LED FHD FreeSync Monitor - Black
    B2. LG - 27UL600-W 27" IPS LED 4K UHD FreeSync Monitor with HDR (DisplayPort, HDMI) - Silver/White
    B3. LG - UltraGear 27" IPS LED QHD FreeSync Monitor with HDR (HDMI) - Black
    B4. LG - 34WL500-B 34" IPS LED UltraWide FHD FreeSync Monitor with HDR (HDMI) - Black
    
    [C - HP]
    C1. HP - 24f 23.8" IPS LED FHD FreeSync Monitor - Natural Silver
    C2. HP - 25x 24.5" LED FHD Monitor (HDMI) - Gray/Green
    C3. HP - 27f 27" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Natural Silver
    C4. HP - 31.5" IPS LED FHD Monitor (HDMI, VGA) - Black
    
    
    [D - Samsung]
    D1. Samsung - 390 Series 24" LED Curved FHD FreeSync Monitor (DVI, HDMI, VGA) - High glossy black
    D2. Samsung - T55 Series 27" LED 1000R Curved FHD FreeSync Monitor (DisplayPort, HDMI, VGA)
    D3. Samsung - UR55 Series 28" IPS 4K UHD Monitor - Dark Gray/Blue
    D4. Samsung - UJ59 Series U32J590UQN 32" LED 4K UHD FreeSync Monitor (DisplayPort, HDMI) - Dark Gray/Blue
    
    
    [E - ASUS]
    E1. ASUS - 23.8" IPS LCD FHD FreeSync Gaming Monitor (DisplayPort, DVI, HDMI) - Black
    E2. ASUS - VG245H 24” FHD 1ms FreeSync Console Gaming Monitor (Dual HDMI, VGA) - Black
    E3. ASUS - 27" IPS LCD FHD FreeSync Gaming Monitor (DisplayPort, DVI, HDMI) - Black
    E4. ASUS - ZenScreen 15.6” Portable Monitor (USB) - Dark Gray
    
    [F - Acer]
    F1. Acer - 23.6" LED FHD Monitor (DVI, HDMI, VGA) - Black
    F2. Acer - 23.8" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Black
    F3. Acer - 27" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Black
    F4. Acer - Predator XB272 27" LED FHD G-SYNC Monitor (DisplayPort, HDMI, USB) - Black
    
    Total = 24
    
"""


# Creating the list of links.
links_to_scrape = [ 
                    # A - Dell
                   'https://www.bestbuy.com/site/dell-s2319nx-23-ips-led-fhd-monitor-black-silver/6237640.p?skuId=6237640', # Missing
                   'https://www.bestbuy.com/site/dell-s2719dgf-27-led-qhd-freesync-monitor-displayport-hdmi-black/6293714.p?skuId=6293714',
                   'https://www.bestbuy.com/site/dell-27-ips-led-fhd-freesync-monitor-piano-black/6394138.p?skuId=6394138',
                   'https://www.bestbuy.com/site/dell-32-led-curved-qhd-freesync-monitor-with-hdr-displayport-hdmi-usb/6375331.p?skuId=6375331',
                   
                    # B - LG
                   'https://www.bestbuy.com/site/lg-24-ips-led-fhd-freesync-monitor-black/6362423.p?skuId=6362423', # Dell
                   'https://www.bestbuy.com/site/lg-27ul600-w-27-ips-led-4k-uhd-freesync-monitor-with-hdr-displayport-hdmi-silver-white/6329956.p?skuId=6329956',
                   'https://www.bestbuy.com/site/lg-ultragear-27-ips-led-qhd-freesync-monitor-with-hdr-hdmi-black/6358119.p?skuId=6358119',
                   'https://www.bestbuy.com/site/lg-34wl500-b-34-ips-led-ultrawide-fhd-freesync-monitor-with-hdr-hdmi-black/6329954.p?skuId=6329954',
                   
                   # C - HP
                   'https://www.bestbuy.com/site/hp-24f-23-8-ips-led-fhd-freesync-monitor-natural-silver/6317590.p?skuId=6317590',
                   'https://www.bestbuy.com/site/hp-25x-24-5-led-fhd-monitor-hdmi-gray-green/6280605.p?skuId=6280605', #LG
                   'https://www.bestbuy.com/site/hp-27f-27-ips-led-fhd-freesync-monitor-hdmi-vga-natural-silver/6219205.p?skuId=6219205',
                   'https://www.bestbuy.com/site/hp-31-5-ips-led-fhd-monitor-hdmi-vga-black/6361917.p?skuId=6361917',
                   
                   # D - Samsung
                   'https://www.bestbuy.com/site/samsung-390-series-24-led-curved-fhd-freesync-monitor-dvi-hdmi-vga-high-glossy-black/5044601.p?skuId=5044601',
                   'https://www.bestbuy.com/site/samsung-t55-series-27-led-1000r-curved-fhd-freesync-monitor-displayport-hdmi-vga/6402202.p?skuId=6402202',
                   'https://www.bestbuy.com/site/samsung-ur55-series-28-ips-4k-uhd-monitor-dark-gray-blue/6386391.p?skuId=6386391',
                   'https://www.bestbuy.com/site/samsung-uj59-series-u32j590uqn-32-led-4k-uhd-freesync-monitor-displayport-hdmi-dark-gray-blue/6293716.p?skuId=6293716',
                   
                   # E - ASUS
                   'https://www.bestbuy.com/site/asus-23-8-ips-lcd-fhd-freesync-gaming-monitor-displayport-dvi-hdmi-black/6395359.p?skuId=6395359',
                   'https://www.bestbuy.com/site/asus-vg245h-24-fhd-1ms-freesync-console-gaming-monitor-dual-hdmi-vga-black/5591926.p?skuId=5591926',
                   'https://www.bestbuy.com/site/asus-27-ips-lcd-fhd-freesync-gaming-monitor-displayport-dvi-hdmi-black/6336778.p?skuId=6336778',
                   'https://www.bestbuy.com/site/asus-zenscreen-15-6-portable-monitor-usb-dark-gray/6403999.p?skuId=6403999',
                   
                   # F - Acer
                   'https://www.bestbuy.com/site/acer-23-6-led-fhd-monitor-dvi-hdmi-vga-black/6404005.p?skuId=6404005',
                   'https://www.bestbuy.com/site/acer-23-8-ips-led-fhd-freesync-monitor-hdmi-vga-black/6401005.p?skuId=6401005',
                   'https://www.bestbuy.com/site/acer-27-ips-led-fhd-freesync-monitor-hdmi-vga-black/6401007.p?skuId=6401007',
                   'https://www.bestbuy.com/site/acer-predator-xb272-27-led-fhd-g-sync-monitor-displayport-hdmi-usb-black/6238705.p?skuId=6238705'
                   ]
                   


l        = 0
one_link = links_to_scrape[l]
driver.get(one_link)


#%%
# dynamic_path = "//div[@id='user-generated-content-ratings-and-reviews-86dda784-c3d4-484a-9f0a-85c24dfe94b8']"


# %%% 
# Expand the reviews
# driver.find_element_by_xpath(dynamic_path).click()
# time.sleep(2)

# See all reviews
driver.find_element_by_xpath("//a[@class='btn btn-secondary v-medium see-all-reviews']").click()
time.sleep(2.3)

# Show only Verified Purchases
driver.find_element_by_xpath("//div[@class='switch']").click()
time.sleep(2.8)


#%%

product_infomration_current = driver.find_elements_by_xpath("//h2[@class='heading-6 product-title']")

# Get the product skuId
try:
    product_id_current = re.findall('skuId=[0-9]{7}',product_infomration_current[0].get_attribute('innerHTML'))[0][6:]
except:
    product_id_current= "Unknown"

# Append the current product to total product list
product_id.append(product_id_current)

search_finished = 0

while (search_finished != 2):
    search_current_page()
    
    # Test if there is any more review. If not, stop the while loop
    # This is a trick part: Only the first and last review page has the element (//a[@aria-disabled='true'), so I could use this as a switch, but the contion to break the loop should be search_finished = 2, becasue search_finished jump from 0 to 1 on the first page
    # The code will work IF the product only have one page of review. However, the code may generate duplicated records becasue it copies the first page for two times
    try:
        driver.find_element_by_xpath("//a[@aria-disabled='true']").get_attribute('innerHTML')
        driver.find_element_by_xpath("//a[@data-track='Page next']").click()
        search_finished += 1
    except:
        driver.find_element_by_xpath("//a[@data-track='Page next']").click()
        
        

#%%
"Only output the result when everything is compeleted"

# review_df = pd.DataFrame.from_dict(reviews_one_monitor)

# review_df.to_excel('Review_Data.xlsx')


#%% 2. Data Cleaning

review_df = pd.read_excel('Data/Review_Data.xlsx').dropna().reset_index()

# Create brand lists. Elements in the lists are product ID at BestBuy
Dell_list       = [6237640, 6293714, 6394138, 6375331]
LG_list         = [6362423, 6329956, 6358119, 6329954]
HP_list         = [6317590, 6280605, 6219205, 6361917]
Samsung_list    = [5044601, 6402202, 6386391, 6293716]
ASUS_list       = [6395359, 5591926, 6336778, 6403999]
Acer_list       = [6404005, 6401005, 6401007, 6238705]

review_df['brand'] = 0

for i in range(review_df.shape[0]):
    try:
        review_df['review_posted_time'][i] = datetime.strptime(str(review_df['review_posted_time'][i]), '%b %d, %Y %I:%M %p')
    except ValueError:
        review_df['review_posted_time'][i] = datetime.strptime(str(review_df['review_posted_time'][i]), '%Y-%m-%d %H:%M:%S')  
    if review_df['product_id'][i] in Dell_list:
        review_df['brand'][i] = 'Dell'
    elif review_df['product_id'][i] in LG_list:
        review_df['brand'][i] = 'LG'
    elif review_df['product_id'][i] in HP_list:
        review_df['brand'][i] = 'HP'
    elif review_df['product_id'][i] in Samsung_list:
        review_df['brand'][i] = 'Samsung'
    elif review_df['product_id'][i] in ASUS_list:
        review_df['brand'][i] = 'ASUS'
    elif review_df['product_id'][i] in Acer_list:
        review_df['brand'][i] = 'Acer'

# Delete unuseful columns
review_df = review_df.drop(['Unnamed: 0','Unnamed: 0.1','index','scrapping_date'],axis = 1)

num_company = 6

company_list = ["ASUS","Acer","Dell","HP","LG","Samsung"]

#%% 3. Data Visualization
# 3.1 Reviews star distribution

star_by_brand = review_df.groupby(['brand','N_stars']).size()
star_by_brand_reshape = star_by_brand.values.reshape(num_company,5)

labels = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
ASUS_star       = star_by_brand_reshape[0]
Acer_star       = star_by_brand_reshape[1]
Dell_star       = star_by_brand_reshape[2]
HP_star         = star_by_brand_reshape[3]
LG_star         = star_by_brand_reshape[4]
Samsung_star    = star_by_brand_reshape[5]

#%% Without Sacling

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars
distance_factor = 3

fig, ax = plt.subplots(figsize=(10,5))
rects1_1_1 = ax.bar(x - width * distance_factor*3/3, ASUS_star, width, label='ASUS')
rects1_1_2 = ax.bar(x - width * distance_factor*2/3, Acer_star, width, label='Acer')
rects1_1_3 = ax.bar(x - width * distance_factor*1/3, Dell_star, width, label='Dell')
rects1_1_4 = ax.bar(x + width * distance_factor*0/3, HP_star, width, label='HP')
rects1_1_5 = ax.bar(x + width * distance_factor*1/3, LG_star, width, label='LG')
rects1_1_6 = ax.bar(x + width * distance_factor*2/3, Samsung_star, width, label='Samsung')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('1-1 Count by Brand and Star')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(alpha = 0.5)

autolabel(rects1_1_1)
autolabel(rects1_1_2)
autolabel(rects1_1_3)
autolabel(rects1_1_4)
autolabel(rects1_1_5)
autolabel(rects1_1_6)

fig.tight_layout()

plt.show()

#%% With Sacling
ASUS_star_percent       = star_by_brand_reshape[0] / star_by_brand_reshape[0].sum() * 100
Acer_star_percent       = star_by_brand_reshape[1] / star_by_brand_reshape[1].sum() * 100
Dell_star_percent       = star_by_brand_reshape[2] / star_by_brand_reshape[2].sum() * 100
HP_star_percent         = star_by_brand_reshape[3] / star_by_brand_reshape[3].sum() * 100
LG_star_percent         = star_by_brand_reshape[4] / star_by_brand_reshape[4].sum() * 100
Samsung_star_percent    = star_by_brand_reshape[5] / star_by_brand_reshape[5].sum() * 100

fig, ax = plt.subplots(figsize=(10,5))
rects1_2_1 = ax.bar(x - width * distance_factor*3/3, ASUS_star_percent, width, label='ASUS')
rects1_2_2 = ax.bar(x - width * distance_factor*2/3, Acer_star_percent, width, label='Acer')
rects1_2_3 = ax.bar(x - width * distance_factor*1/3, Dell_star_percent, width, label='Dell')
rects1_2_4 = ax.bar(x + width * distance_factor*0/3, HP_star_percent, width, label='HP')
rects1_2_5 = ax.bar(x + width * distance_factor*1/3, LG_star_percent, width, label='LG')
rects1_2_6 = ax.bar(x + width * distance_factor*2/3, Samsung_star_percent, width, label='Samsung')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage')
ax.set_title('1-2 Count by Brand and Star (Percentage)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(alpha = 0.5)

autolabel(rects1_2_1)
autolabel(rects1_2_2)
autolabel(rects1_2_3)
autolabel(rects1_2_4)
autolabel(rects1_2_5)
autolabel(rects1_2_6)

fig.tight_layout()

#%% 3.2 Review By Time
review_date_df = review_df['review_posted_time']
review_date_df = review_date_df.dt.date

date_count_df = review_date_df.value_counts().sort_index()

date_count_df.plot(
    kind = 'area',
    title = '2 - Review Count Over Time',
    figsize = (10,5),
    grid = True
    )

#%% 4. Data Spiltting
review_df['ML_group']  = np.random.randint(100, size = review_df.shape[0])
review_df              = review_df.sort_values(by='ML_group')

# Putting structure in the text
#
corpus_review = review_df.review_text.to_list()
corpus_title = review_df.review_title.to_list()

corpus_review = [str (item) for item in corpus_review]
corpus_title = [str (item) for item in corpus_title]

vectorizer = CountVectorizer(lowercase   = True,
                             ngram_range = (1,1),
                             max_df      = 0.85,
                             min_df      = 0.01)

X_review  = vectorizer.fit_transform(corpus_review)
X_title  = vectorizer.fit_transform(corpus_title)

print(vectorizer.get_feature_names())
print(X_review.toarray())
print(X_title.toarray())

print('Nunber of keywords selelcted for review analysis is {}'.format(len(X_review.toarray()[0])))
print('Nunber of keywords selelcted for title analysis is {}'.format(len(X_title.toarray()[0])))


# Split the data

Training_size = 80
Validation_size = 10
Testing_size = Training_size + Validation_size

df_train = review_df.ML_group < Training_size        
df_valid  = (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)
df_test = review_df.ML_group >= Testing_size

y_train  = review_df.N_stars[df_train].to_list()
y_valid  = review_df.N_stars[df_valid].to_list()
y_test   = review_df.N_stars[df_test].to_list()

X_review_train = X_review[np.where(df_train)[0],:]
X_review_valid = X_review[np.where(df_valid)[0],:]
X_review_test  = X_review[np.where(df_test)[0],:]

X_title_train = X_title[np.where(df_train)[0],:]
X_title_valid = X_title[np.where(df_valid)[0],:]
X_title_test  = X_title[np.where(df_test)[0],:]

#%% 5.Building model for review analysis

#%% 5.1 Linear Regression
model                  = LinearRegression()
clf_review_linear      = model.fit(X_review_train, y_train)
y_review_pred          = clf_review_linear.predict(X_review_valid)

review_df['N_star_review_reg'] = np.concatenate(
         [
                 clf_review_linear.predict(X_review_train),
                 clf_review_linear.predict(X_review_valid),
                 clf_review_linear.predict(X_review_test)
         ]
         ).round().astype(int)

review_df.loc[review_df['N_star_review_reg']>5,'N_star_review_reg'] = 5
review_df.loc[review_df['N_star_review_reg']<1,'N_star_review_reg'] = 1

confusion_matrix_review_linear_train = np.zeros((5,5))
confusion_matrix_review_linear_train = pd.DataFrame(confusion_matrix_review_linear_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_review_linear_valid = confusion_matrix_review_linear_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_linear_train.iloc[i][j] = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_reg == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_review_linear_valid.iloc[i][j] = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_reg == j+1) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))]).shape[0]

print('Confusion matrix for linear regression on training data (review analysis)')
print(confusion_matrix_review_linear_train)

print('Confusion matrix for linear regression on validation data (review analysis)')
print(confusion_matrix_review_linear_valid)

prediction_score_review_linear_train = review_df[(review_df.N_stars == review_df.N_star_review_reg) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The linear model has a prediction score of {:.2f} on training set".format(prediction_score_review_linear_train))

prediction_score_review_linear_valid = review_df[(review_df.N_stars == review_df.N_star_review_reg) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The linear model has a prediction score of {:.2f} on validation set".format(prediction_score_review_linear_valid))


#%% 5.2 Lasso Regression
model            = linear_model.Lasso(alpha=0.1)
clf_review_Lasso = model.fit(X_review_train, y_train)

review_df['N_star_review_lasso'] = np.concatenate(
        [
                clf_review_Lasso.predict(X_review_train),
                clf_review_Lasso.predict(X_review_valid),
                clf_review_Lasso.predict(X_review_test)
        ]
        ).round().astype(int)

review_df.loc[review_df['N_star_review_lasso']>5,'N_star_review_lasso'] = 5
review_df.loc[review_df['N_star_review_lasso']<1,'N_star_review_lasso'] = 1

# Now build the confusion matrix for Lasso Regression
confusion_matrix_review_Lasso_train = np.zeros((5,5))
confusion_matrix_review_Lasso_train = pd.DataFrame(confusion_matrix_review_Lasso_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_review_Lasso_valid = confusion_matrix_review_Lasso_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_Lasso_train.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_lasso == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_review_Lasso_valid.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_lasso == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Lasso Regression on training data (review analysis)')
print(confusion_matrix_review_Lasso_train)

print('Confusion matrix for Lasso Regression on validation data (review analysis)')
print(confusion_matrix_review_Lasso_valid)

prediction_score_review_Lasso_train = review_df[(review_df.N_stars == review_df.N_star_review_lasso) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Lasso Regression model has a prediction score of {:.2f} on training set".format(prediction_score_review_Lasso_train))

prediction_score_review_Lasso_valid = review_df[(review_df.N_stars == review_df.N_star_review_lasso) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Lasso Regression model has a prediction score of {:.2f} on validation set".format(prediction_score_review_Lasso_valid))


#%% 5.3 KNN
k                = 1;
results_list_knn = [];
max_k_nn         = 50
for k in range(1,max_k_nn + 1):
    clf_review_knn      = KNeighborsClassifier(n_neighbors=k).fit(X_review_train, y_train)
    results_list_knn.append(
            np.concatenate(
                    [
                            clf_review_knn.predict(X_review_train),
                            clf_review_knn.predict(X_review_valid),
                            clf_review_knn.predict(X_review_test )
                    ])
    )
    print('K = {} is done'.format(k))
    
    
review_results_knn              = pd.DataFrame(results_list_knn).transpose()
review_results_knn['df_train'] = df_train.to_list()
review_results_knn['df_valid']  = df_valid.to_list()
review_results_knn['df_test'] = df_valid.to_list()

review_results_knn['N_stars'] = review_df['N_stars'].copy()

#%% Build a dictionary that stores the validation accuracy of each K.
knn_review_dict = {}
for i in range(1,max_k_nn):
    knn_review_dict[i] = review_results_knn[(review_results_knn.N_stars == review_results_knn[i-1]) & (review_results_knn.df_valid == True)].shape[0]/review_df[(review_results_knn.df_valid == True)].shape[0]
    
# Rank the testing accuracy and get the best parameter setting for K
best_k_review = max(knn_review_dict.items(), key=operator.itemgetter(1))[0] + 1

print("The best parameter for k is",best_k_review,"and the best validation accuracy score is {:.2f}".format(knn_review_dict.get(best_k_review - 1)))

# Append the optimal knn result to review_df
try:
    review_df.drop(['N_star_review_knn'], axis = 1)
except:
    review_df.loc[:,'N_star_review_knn'] = review_results_knn.iloc[:, best_k_review -1]

# Now build the confusion matrix for the best parameter
confusion_matrix_review_knn_train = np.zeros((5,5))
confusion_matrix_review_knn_train = pd.DataFrame(confusion_matrix_review_knn_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_review_knn_valid = confusion_matrix_review_knn_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_knn_train.iloc[i][j]   = (review_results_knn[(review_results_knn.N_stars == i + 1) & (review_results_knn[best_k_review - 1] == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_review_knn_valid.iloc[i][j]   = (review_results_knn[(review_results_knn.N_stars == i + 1) & (review_results_knn[best_k_review - 1] == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for KNN (k = {}) on training data (review analysis)'.format(best_k_review))
print(confusion_matrix_review_knn_train)

print('Confusion matrix for KNN (k = {}) on validation data (review analysis)'.format(best_k_review))
print(confusion_matrix_review_knn_valid)

prediction_score_review_knn_train = review_df[(review_df.N_stars == review_df.N_star_review_knn) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The KNN has a prediction score of {:.2f} on training set".format(prediction_score_review_knn_train))

prediction_score_review_knn_valid = review_df[(review_df.N_stars == review_df.N_star_review_knn) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The KNN has a prediction score of {:.2f} on validation set".format(prediction_score_review_knn_valid))



#%% 5.4 Naive Bayes Classification
clf_review_NB                  = GaussianNB().fit(X_review_train.toarray(), y_train)
review_df['N_star_review_NB']     = np.concatenate(
        [
                clf_review_NB.predict(X_review_train.toarray()),
                clf_review_NB.predict(X_review_valid.toarray()),
                clf_review_NB.predict(X_review_test.toarray( ))
        ]).round().astype(int)
review_df.loc[review_df['N_star_review_NB']>5,'N_star_review_NB'] = 5
review_df.loc[review_df['N_star_review_NB']<1,'N_star_review_NB'] = 1

# Now build the confusion matrix for Naive Bayes Classification
confusion_matrix_review_NB_train = np.zeros((5,5))
confusion_matrix_review_NB_train = pd.DataFrame(confusion_matrix_review_NB_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_review_NB_valid = confusion_matrix_review_NB_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_NB_train.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_NB == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_review_NB_valid.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_NB == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Naive Bayes Classification on training data (review analysis)')
print(confusion_matrix_review_NB_train)

print('Confusion matrix for Naive Bayes Classification on validation data (review analysis)')
print(confusion_matrix_review_NB_valid)

prediction_score_review_NB_train = review_df[(review_df.N_stars == review_df.N_star_review_NB) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Naive Bayes Classification model has a prediction score of {:.2f} on training set".format(prediction_score_review_NB_train))

prediction_score_review_NB_valid = review_df[(review_df.N_stars == review_df.N_star_review_NB) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Naive Bayes Classification model has a prediction score of {:.2f} on validation set".format(prediction_score_review_NB_valid))


#%% 5.5 Decision Trees
criterion_chosen         = ['entropy','gini'][1]
random_state             = 96
max_depth                = 10
results_list_review_tree = []
for depth in range(2,max_depth + 1):
    clf_review_tree      = tree.DecisionTreeClassifier(
            criterion    = criterion_chosen, 
            max_depth    = depth,
            random_state = 96).fit(X_review_train.toarray(), y_train)

    results_list_review_tree.append(
            np.concatenate(
                    [
                            clf_review_tree.predict(X_review_train.toarray()),
                            clf_review_tree.predict(X_review_valid.toarray()),
                            clf_review_tree.predict(X_review_test.toarray())
                    ]).round().astype(int)
            )
    
tree.plot_tree(clf_review_tree) 

results_review_tree              = pd.DataFrame(results_list_review_tree).transpose()
results_review_tree['df_train'] = df_train.to_list()
results_review_tree['df_valid'] = df_valid.to_list()
results_review_tree['df_test']  = df_test.to_list()

results_review_tree['N_stars'] = review_df['N_stars'].copy()

#%%
# Build a dictionary that stores the testing accuracy of max_depth.
tree_review_dict = {}
for i in range(max_depth-2):
    tree_review_dict[i] = results_review_tree[(results_review_tree.N_stars == results_review_tree[i]) & (results_review_tree.df_test == True)].shape[0]/review_df[(results_review_tree.df_test == True)].shape[0]
    
# Rank the testing accuracy and get the best parameter setting for max_depth
best_max_depth_review = max(tree_review_dict.items(), key=operator.itemgetter(1))[0] + 2

print("The best parameter for max_depth is",best_max_depth_review,"and the best testing accuracy score is {:.2f}".format(tree_review_dict.get(best_max_depth_review)))

#%%
# Append the optimal tree result to review_df
try:
    review_df.drop(['N_star_review_tree'], axis = 1)
except:
    review_df.loc[:,'N_star_review_tree'] = results_review_tree.iloc[:, best_max_depth_review - 2]

# Now build the confusion matrix for the best parameter
confusion_matrix_review_tree_train = np.zeros((5,5))
confusion_matrix_review_tree_train = pd.DataFrame(confusion_matrix_review_tree_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_review_tree_valid = confusion_matrix_review_tree_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_tree_train.iloc[i][j]   = (results_review_tree[(results_review_tree.N_stars == i + 1) & (results_review_tree[best_max_depth_review - 2] == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_review_tree_valid.iloc[i][j]   = (results_review_tree[(results_review_tree.N_stars == i + 1) & (results_review_tree[best_max_depth_review - 2] == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Decision Tree (max depth = {}) on training data (review analysis)'.format(best_max_depth_review))
print(confusion_matrix_review_tree_train)

print('Confusion matrix for Decision Tree (max depth = {}) on validation data (review analysis)'.format(best_max_depth_review))
print(confusion_matrix_review_tree_valid)

prediction_score_review_tree_train = review_df[(review_df.N_stars == review_df.N_star_review_tree) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Decision Tree has a prediction score of {:.2f} on training set".format(prediction_score_review_tree_train))

prediction_score_review_tree_valid = review_df[(review_df.N_stars == review_df.N_star_review_tree) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Decision Tree has a prediction score of {:.2f} on validation set".format(prediction_score_review_tree_valid))


#%% 5.6 Find optimal model: Not just based on accuracy, but confusion matrix
print("The linear model has a prediction score of {:.2f} on validation set".format(prediction_score_review_linear_valid))
print("The Lasso Regression model has a prediction score of {:.2f} on validation set".format(prediction_score_review_Lasso_valid))
print("The KNN has a prediction score of {:.2f} on validation set".format(prediction_score_review_knn_valid))
print("The Naive Bayes Classification model has a prediction score of {:.2f} on validation set".format(prediction_score_review_NB_valid))
print("The Decision Tree has a prediction score of {:.2f} on validation set".format(prediction_score_review_tree_valid))


print(confusion_matrix_review_linear_valid)
print(confusion_matrix_review_Lasso_valid)
print(confusion_matrix_review_knn_valid)
print(confusion_matrix_review_NB_valid)
print(confusion_matrix_review_tree_valid)


# Optimal Model: Lasso
prediction_score_review_Lasso_test = review_df[(review_df.N_stars == review_df.N_star_review_lasso) & (review_df.ML_group > Testing_size)].shape[0]/review_df[(review_df.ML_group > Testing_size)].shape[0]
print("The Lasso Regression model has a prediction score of {:.2f} on testing set".format(prediction_score_review_Lasso_test))

confusion_matrix_review_Lasso_test = np.zeros((5,5))
confusion_matrix_review_Lasso_test = pd.DataFrame(confusion_matrix_review_tree_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_review_Lasso_test.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_review_lasso == j+1) & (review_df.ML_group > Testing_size)]).shape[0]

print(confusion_matrix_review_Lasso_test)



#%% 6.Building model for title analysis

#%% 6.1 Linear Regression
model                  = LinearRegression()
clf_title_linear      = model.fit(X_title_train, y_train)
y_title_pred          = clf_title_linear.predict(X_title_valid)

review_df['N_star_title_reg'] = np.concatenate(
         [
                 clf_title_linear.predict(X_title_train),
                 clf_title_linear.predict(X_title_valid),
                 clf_title_linear.predict(X_title_test)
         ]
         ).round().astype(int)

review_df.loc[review_df['N_star_title_reg']>5,'N_star_title_reg'] = 5
review_df.loc[review_df['N_star_title_reg']<1,'N_star_title_reg'] = 1

confusion_matrix_title_linear_train = np.zeros((5,5))
confusion_matrix_title_linear_train = pd.DataFrame(confusion_matrix_title_linear_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_title_linear_valid = confusion_matrix_title_linear_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_linear_train.iloc[i][j] = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_reg == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_title_linear_valid.iloc[i][j] = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_reg == j+1) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))]).shape[0]

print('Confusion matrix for linear regression on training data (title analysis)')
print(confusion_matrix_title_linear_train)

print('Confusion matrix for linear regression on validation data (title analysis)')
print(confusion_matrix_title_linear_valid)

prediction_score_title_linear_train = review_df[(review_df.N_stars == review_df.N_star_title_reg) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The linear model has a prediction score of {:.2f} on training set".format(prediction_score_title_linear_train))

prediction_score_title_linear_valid = review_df[(review_df.N_stars == review_df.N_star_title_reg) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The linear model has a prediction score of {:.2f} on validation set".format(prediction_score_title_linear_valid))


#%% 6.2 Lasso Regression
model            = linear_model.Lasso(alpha=0.1)
clf_title_Lasso = model.fit(X_title_train, y_train)

review_df['N_star_title_lasso'] = np.concatenate(
        [
                clf_title_Lasso.predict(X_title_train),
                clf_title_Lasso.predict(X_title_valid),
                clf_title_Lasso.predict(X_title_test)
        ]
        ).round().astype(int)

review_df.loc[review_df['N_star_title_lasso']>5,'N_star_title_lasso'] = 5
review_df.loc[review_df['N_star_title_lasso']<1,'N_star_title_lasso'] = 1

# Now build the confusion matrix for Lasso Regression
confusion_matrix_title_Lasso_train = np.zeros((5,5))
confusion_matrix_title_Lasso_train = pd.DataFrame(confusion_matrix_title_Lasso_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_title_Lasso_valid = confusion_matrix_title_Lasso_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_Lasso_train.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_lasso == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_title_Lasso_valid.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_lasso == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Lasso Regression on training data (title analysis)')
print(confusion_matrix_title_Lasso_train)

print('Confusion matrix for Lasso Regression on validation data (title analysis)')
print(confusion_matrix_title_Lasso_valid)

prediction_score_title_Lasso_train = review_df[(review_df.N_stars == review_df.N_star_title_lasso) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Lasso Regression model has a prediction score of {:.2f} on training set".format(prediction_score_title_Lasso_train))

prediction_score_title_Lasso_valid = review_df[(review_df.N_stars == review_df.N_star_title_lasso) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Lasso Regression model has a prediction score of {:.2f} on validation set".format(prediction_score_title_Lasso_valid))


#%% 6.3 KNN
k                = 1;
results_list_knn = [];
max_k_nn         = 50
for k in range(1,max_k_nn + 1):
    clf_title_knn      = KNeighborsClassifier(n_neighbors=k).fit(X_title_train, y_train)
    results_list_knn.append(
            np.concatenate(
                    [
                            clf_title_knn.predict(X_title_train),
                            clf_title_knn.predict(X_title_valid),
                            clf_title_knn.predict(X_title_test )
                    ])
    )
    print('K = {} is done'.format(k))
    
    
title_results_knn              = pd.DataFrame(results_list_knn).transpose()
title_results_knn['df_train'] = df_train.to_list()
title_results_knn['df_valid']  = df_valid.to_list()
title_results_knn['df_test'] = df_valid.to_list()

title_results_knn['N_stars'] = review_df['N_stars'].copy()

#%% Build a dictionary that stores the validation accuracy of each K.
knn_title_dict = {}
for i in range(1,max_k_nn):
    knn_title_dict[i] = title_results_knn[(title_results_knn.N_stars == title_results_knn[i-1]) & (title_results_knn.df_valid == True)].shape[0]/review_df[(title_results_knn.df_valid == True)].shape[0]
    
# Rank the testing accuracy and get the best parameter setting for K
best_k_title = max(knn_title_dict.items(), key=operator.itemgetter(1))[0] + 1

print("The best parameter for k is",best_k_title,"and the best validation accuracy score is {:.2f}".format(knn_title_dict.get(best_k_title - 1)))

# Append the optimal knn result to review_df
try:
    review_df.drop(['N_star_title_knn'], axis = 1)
except:
    review_df.loc[:,'N_star_title_knn'] = title_results_knn.iloc[:, best_k_title -1]

# Now build the confusion matrix for the best parameter
confusion_matrix_title_knn_train = np.zeros((5,5))
confusion_matrix_title_knn_train = pd.DataFrame(confusion_matrix_title_knn_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_title_knn_valid = confusion_matrix_title_knn_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_knn_train.iloc[i][j]   = (title_results_knn[(title_results_knn.N_stars == i + 1) & (title_results_knn[best_k_title - 1] == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_title_knn_valid.iloc[i][j]   = (title_results_knn[(title_results_knn.N_stars == i + 1) & (title_results_knn[best_k_title - 1] == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for KNN (k = {}) on training data (title analysis)'.format(best_k_title))
print(confusion_matrix_title_knn_train)

print('Confusion matrix for KNN (k = {}) on validation data (title analysis)'.format(best_k_title))
print(confusion_matrix_title_knn_valid)

prediction_score_title_knn_train = review_df[(review_df.N_stars == review_df.N_star_title_knn) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The KNN has a prediction score of {:.2f} on training set".format(prediction_score_title_knn_train))

prediction_score_title_knn_valid = review_df[(review_df.N_stars == review_df.N_star_title_knn) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The KNN has a prediction score of {:.2f} on validation set".format(prediction_score_title_knn_valid))



#%% 6.4 Naive Bayes Classification
clf_title_NB                  = GaussianNB().fit(X_title_train.toarray(), y_train)
review_df['N_star_title_NB']     = np.concatenate(
        [
                clf_title_NB.predict(X_title_train.toarray()),
                clf_title_NB.predict(X_title_valid.toarray()),
                clf_title_NB.predict(X_title_test.toarray( ))
        ]).round().astype(int)
review_df.loc[review_df['N_star_title_NB']>5,'N_star_title_NB'] = 5
review_df.loc[review_df['N_star_title_NB']<1,'N_star_title_NB'] = 1

# Now build the confusion matrix for Naive Bayes Classification
confusion_matrix_title_NB_train = np.zeros((5,5))
confusion_matrix_title_NB_train = pd.DataFrame(confusion_matrix_title_NB_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_title_NB_valid = confusion_matrix_title_NB_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_NB_train.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_NB == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_title_NB_valid.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_NB == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Naive Bayes Classification on training data (title analysis)')
print(confusion_matrix_title_NB_train)

print('Confusion matrix for Naive Bayes Classification on validation data (title analysis)')
print(confusion_matrix_title_NB_valid)

prediction_score_title_NB_train = review_df[(review_df.N_stars == review_df.N_star_title_NB) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Naive Bayes Classification model has a prediction score of {:.2f} on training set".format(prediction_score_title_NB_train))

prediction_score_title_NB_valid = review_df[(review_df.N_stars == review_df.N_star_title_NB) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Naive Bayes Classification model has a prediction score of {:.2f} on validation set".format(prediction_score_title_NB_valid))


#%% 6.5 Decision Trees
criterion_chosen         = ['entropy','gini'][1]
random_state             = 96
max_depth                = 10
results_list_title_tree = []
for depth in range(2,max_depth + 1):
    clf_title_tree      = tree.DecisionTreeClassifier(
            criterion    = criterion_chosen, 
            max_depth    = depth,
            random_state = 96).fit(X_title_train.toarray(), y_train)

    results_list_title_tree.append(
            np.concatenate(
                    [
                            clf_title_tree.predict(X_title_train.toarray()),
                            clf_title_tree.predict(X_title_valid.toarray()),
                            clf_title_tree.predict(X_title_test.toarray())
                    ]).round().astype(int)
            )
    
tree.plot_tree(clf_title_tree) 

results_title_tree              = pd.DataFrame(results_list_title_tree).transpose()
results_title_tree['df_train'] = df_train.to_list()
results_title_tree['df_valid'] = df_valid.to_list()
results_title_tree['df_test']  = df_test.to_list()

results_title_tree['N_stars'] = review_df['N_stars'].copy()

#%%
# Build a dictionary that stores the testing accuracy of max_depth.
tree_title_dict = {}
for i in range(max_depth-2):
    tree_title_dict[i] = results_title_tree[(results_title_tree.N_stars == results_title_tree[i]) & (results_title_tree.df_test == True)].shape[0]/review_df[(results_title_tree.df_test == True)].shape[0]
    
# Rank the testing accuracy and get the best parameter setting for max_depth
best_max_depth_title = max(tree_title_dict.items(), key=operator.itemgetter(1))[0] + 2

print("The best parameter for max_depth is",best_max_depth_title,"and the best testing accuracy score is {:.2f}".format(tree_title_dict.get(best_max_depth_title)))

#%%
# Append the optimal tree result to review_df
try:
    review_df.drop(['N_star_title_tree'], axis = 1)
except:
    review_df.loc[:,'N_star_title_tree'] = results_title_tree.iloc[:, best_max_depth_title - 2]

# Now build the confusion matrix for the best parameter
confusion_matrix_title_tree_train = np.zeros((5,5))
confusion_matrix_title_tree_train = pd.DataFrame(confusion_matrix_title_tree_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])
confusion_matrix_title_tree_valid = confusion_matrix_title_tree_train.copy()

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_tree_train.iloc[i][j]   = (results_title_tree[(results_title_tree.N_stars == i + 1) & (results_title_tree[best_max_depth_title - 2] == j+1) & (review_df.ML_group < Training_size)]).shape[0]
        confusion_matrix_title_tree_valid.iloc[i][j]   = (results_title_tree[(results_title_tree.N_stars == i + 1) & (results_title_tree[best_max_depth_title - 2] == j+1) & (review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size)]).shape[0]
        
print('Confusion matrix for Decision Tree (max depth = {}) on training data (title analysis)'.format(best_max_depth_title))
print(confusion_matrix_title_tree_train)

print('Confusion matrix for Decision Tree (max depth = {}) on validation data (title analysis)'.format(best_max_depth_title))
print(confusion_matrix_title_tree_valid)

prediction_score_title_tree_train = review_df[(review_df.N_stars == review_df.N_star_title_tree) & (review_df.ML_group < Training_size)].shape[0]/review_df[(review_df.ML_group < Training_size)].shape[0]
print("The Decision Tree has a prediction score of {:.2f} on training set".format(prediction_score_title_tree_train))

prediction_score_title_tree_valid = review_df[(review_df.N_stars == review_df.N_star_title_tree) & ((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]/review_df[((review_df.ML_group < Testing_size)&(review_df.ML_group >= Training_size))].shape[0]
print("The Decision Tree has a prediction score of {:.2f} on validation set".format(prediction_score_title_tree_valid))

#%% 6.6 Find optimal model: Not just based on accuracy, but confusion matrix
print("The linear model has a prediction score of {:.2f} on validation set".format(prediction_score_title_linear_valid))
print("The Lasso Regression model has a prediction score of {:.2f} on validation set".format(prediction_score_title_Lasso_valid))
print("The KNN has a prediction score of {:.2f} on validation set".format(prediction_score_title_knn_valid))
print("The Naive Bayes Classification model has a prediction score of {:.2f} on validation set".format(prediction_score_title_NB_valid))
print("The Decision Tree has a prediction score of {:.2f} on validation set".format(prediction_score_title_tree_valid))


print(confusion_matrix_title_linear_valid)
print(confusion_matrix_title_Lasso_valid)
print(confusion_matrix_title_knn_valid)
print(confusion_matrix_title_NB_valid)
print(confusion_matrix_title_tree_valid)


# Optimal Model: Linear
prediction_score_title_reg_test = review_df[(review_df.N_stars == review_df.N_star_title_reg) & (review_df.ML_group > Testing_size)].shape[0]/review_df[(review_df.ML_group > Testing_size)].shape[0]
print("The linear model has a prediction score of {:.2f} on testing set".format(prediction_score_title_reg_test))

confusion_matrix_title_reg_test = np.zeros((5,5))
confusion_matrix_title_reg_test = pd.DataFrame(confusion_matrix_title_tree_train, columns=['1 (Prediction)','2','3','4','5'],index = ['1 (Actual)','2','3','4','5'])

for i in range(0,5):
    for j in range(0,5):
        confusion_matrix_title_reg_test.iloc[i][j]   = (review_df[(review_df.N_stars == i + 1) & (review_df.N_star_title_reg == j+1) & (review_df.ML_group > Testing_size)]).shape[0]

print(confusion_matrix_title_reg_test)
