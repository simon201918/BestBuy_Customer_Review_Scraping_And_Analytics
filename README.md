
# BestBuy Customer Review Scraping and Analytics

This project aims to scrape and analyze customer reviews data from [**Bestbuy.com**](https://www.bestbuy.com). The research will focus on monitors, but the code works with any other BestBuy products too.

**Currently, the project is still under development, but the code used to scrape the data is shared here.**

## Scraping the Customer Review
To use the code, please pay attention to the notices below.

### Notice A:
Please make sure to download "[**chromedriver** ](https://chromedriver.chromium.org)" and update the file path before running the code.

    # Please update the path before running the code
    os.chdir('UPDATE NEEDED')
    
    # Please update the path before running the code
    path_to_driver ='UPDATE NEEDED/chromedriver'

### Notice B:
The code may generate duplicated records if the ***total verified purchases review is less than 21 (less than one page)***
Therefore, please select popular products to run the code or perform data cleaning afterward.

### Notice C:
BestBuy has the technology to **DYNAMICLY** change its website elements; consequently, the code cannot automatically scrape data for multiple products.

However, the code still works well with easy human intervention to solve the problem in Notice B:
    
#### Solution 1:

![enter image description here](https://github.com/simon201918/BestBuy_Customer_Review_Scraping_And_Analytics/blob/main/Pictures/Solution_1%20Click%20%22Review%22.png?raw=true)

Scroll down the website manually and click ***"Reviews"*** when opening a product page for the ***FIRST TIME***. Then run the rest of the code after the line below (itself included):

    driver.find_element_by_xpath("//a[@class='btn btn-secondary v-medium see-all-reviews']").click()



#### Solution 2:
Use the keyword ***"user-generated-content-ratings-and-reviews"*** to search the web script and update the dynamic XPath in the code:

    dynamic_path = "//div[@id='user-generated-content-ratings-and-reviews-86dda784-c3d4-484a-9f0a-85c24dfe94b8']"
