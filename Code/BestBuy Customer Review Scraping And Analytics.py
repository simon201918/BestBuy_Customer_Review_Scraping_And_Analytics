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
import datetime
import os
import pandas as pd
import numpy as np
import re
import shelve
import time
import datetime

# libraries to crawl websites
from bs4 import BeautifulSoup
from selenium import webdriver
from pynput.mouse import Button, Controller


pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.width',800)

# Please update the path before running the code
os.chdir('UPDATE NEEDED')

# Please update the path before running the code
path_to_driver ='UPDATE NEEDED/chromedriver'


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
                one_review_posted_time = re.findall('[A-Z][a-z][a-z][a-z]* [0-9][0-9]*, [0-9][0-9][0-9][0-9] [0-9][0-9]*:[0-9][0-9] [A-Z][A-Z]',reviews[r].get_attribute('innerHTML'))[0]
            except:
                one_review_posted_time = ""
            one_review['one_review_posted_time'] = one_review_posted_time  
        
            # Get Review Title
            try:
                one_review_title = soup.find('div', attrs={'class':'review-heading'}).text[37:]
            except:
                one_review_title = ""
            one_review['one_review_title'] = one_review_title
        
            # Get Review Content
            try:
                one_review_text = soup.find('div', attrs={'class':'ugc-review-body body-copy-lg'}).text
            except:
                one_review_text = ""
            one_review['one_review_text'] = one_review_text

            # Get number of stars
            try:
                one_review_stars = re.findall('[0-5] out of [0-5] [Ss]tar',reviews[r].get_attribute('innerHTML'))[0][0]
            except:
                one_review_stars = ""
            one_review['one_review_stars'] = one_review_stars
        
            reviews_one_monitor.append(one_review)
            
    return reviews_one_monitor
        

#reviews = driver.find_elements_by_xpath("//div[@class='review-item-content col-xs-12 col-md-8 col-lg-9']")
#reviews[0].get_attribute('innerHTML')
#re.findall('[A-Z][a-z][a-z][a-z]* [0-9][0-9], [0-9][0-9][0-9][0-9] [0-9][0-9]*:[0-9][0-9] [A-Z][A-Z]',reviews[0].get_attribute('innerHTML'))[0]  


#%%
product_id          = []
reviews_one_monitor = []

#%%
"""
Products included:
    LG - 24" IPS LED FHD FreeSync Monitor - Black
    Dell - S2319NX 23" IPS LED FHD Monitor - Black/Silver
    Dell - 27" IPS LED FHD FreeSync Monitor - Piano Black
    HP - 24f 23.8" IPS LED FHD FreeSync Monitor - Natural Silver
    Samsung - UR55 Series 28" IPS 4K UHD Monitor - Dark Gray/Blue
"""


# Creating the list of links.
links_to_scrape = ['https://www.bestbuy.com/site/lg-24-ips-led-fhd-freesync-monitor-black/6362423.p?skuId=6362423',
                   'https://www.bestbuy.com/site/dell-s2319nx-23-ips-led-fhd-monitor-black-silver/6237640.p?skuId=6237640',
                   'https://www.bestbuy.com/site/dell-27-ips-led-fhd-freesync-monitor-piano-black/6394138.p?skuId=6394138',
                   'https://www.bestbuy.com/site/hp-24f-23-8-ips-led-fhd-freesync-monitor-natural-silver/6317590.p?skuId=6317590',
                   'https://www.bestbuy.com/site/samsung-ur55-series-28-ips-4k-uhd-monitor-dark-gray-blue/6386391.p?skuId=6386391']
                   
l        = 4
one_link = links_to_scrape[l]
driver.get(one_link)


#%%
dynamic_path = "//div[@id='user-generated-content-ratings-and-reviews-86dda784-c3d4-484a-9f0a-85c24dfe94b8']"


# %%% 
# Expand the reviews
driver.find_element_by_xpath(dynamic_path).click()
time.sleep(1.5)

# See all reviews
driver.find_element_by_xpath("//a[@class='btn btn-secondary v-medium see-all-reviews']").click()
time.sleep(1)

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

result = pd.DataFrame.from_dict(reviews_one_monitor)

result.to_csv('Reviews.csv')

















