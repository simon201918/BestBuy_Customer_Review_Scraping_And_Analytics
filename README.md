
# BestBuy Customer Review Scraping and Analytics

This project scrapes and analyzes monitor customer reviews data from [Bestbuy.com](https://www.bestbuy.com). Specifically, the research uses customer review content to forecast the review score (star).

This code also could be easily applied to investigate other BestBuy products. For the detailed instruction of review scraping, please click [here](https://github.com/simon201918/BestBuy_Customer_Review_Scraping_And_Analytics/blob/main/Data%20Scraping%20Instruction.md).

## 1 - Results Highlight
Most candidate models have high prediction accuracy because the review data is highly is unbalanced. Therefore, this study also utilizes confusion matrix to evaluate model performance.

Lasso regression is the optimal model for review analysis, while linear regression is the optimal model for title analysis.

## 2 - Data Visualization
**2.1 Review Count by Brand and Star**

  ![say sth](https://github.com/simon201918/BestBuy_Customer_Review_Scraping_And_Analytics/blob/main/Pictures/1.1%20Count%20by%20Brand%20and%20Star.png?raw=true)

This graph shows the review count by brand and stars of all candidate monitors. HP has the most customer review among all the brands, and five and four stars are the most popular review. However, we can't tell whether all the brands have a similar review distribution.

**2.2 Review Count by Brand and Star (Percentage)**

![say sth](https://github.com/simon201918/BestBuy_Customer_Review_Scraping_And_Analytics/blob/main/Pictures/1.2%20Count%20by%20Brand%20and%20Star%20(Percentage).png?raw=true)

This graph shows the review percentage distribution by brand. All companies have similar distribution. ASUS, Dell and HP enjoy a higher five-star review percentage than the rest of brands, though the difference is not materially significant.

**2.3 Review Count Over Time by Date**

![say sth](https://github.com/simon201918/BestBuy_Customer_Review_Scraping_And_Analytics/blob/main/Pictures/2.%20Review%20Count%20Over%20Time.png?raw=true)

Finally, the majority of the reviews were written in 2019 and 2020. This makes sense for monitors because electronic products update fast. It is also surprising to see that some of the candidate monitors have review record since 2016, which means it haven't been updated for five years.

The data also have seasonality, with the fourth quarter been the busiest season of the year. Note that the review data is scarped on Dec.8 2020.

## 3 - Modeling

There are two types of reviews data at BestBuy - the title of the review, and the main content. This study uses machine learning models and sentiment analysis technics to analyze both reviews and test whether the results are materially different.

The sample size of this project is 25,368.

The training, validation and testing set are 80%, 10%, and 10% respectively.

After converting all the reviews into vectors with CountVectorizer, this project utilizes five machine learning models to predict customer review score (star). The candidate models are linear regression, lasso regression, k-nearest neighbor, naive bayes classification and decision trees. The validation accuracy of the models is summarized in the table below.

|  |Main Content|Review Title	|
|--|--|--|
|Linear Regression  |75%  |71%	 |
|Lasso Regression  |78%  |79%	 |
|K-Nearest Neighbor  |74% |77% |
|Naive Bayes Classification  |29%  |18%	 |
|Decision Trees |79%|78%	 |

There are three quick findings from this table. 


**(1). For each model, there is no big prediction power differences in main content analysis and review title analysis.**

**(2). Naive bayes classification is terribly worse than other models.**

**(3). All other models have very similar prediction power (range from 71% to 79%.**

The first finding is exciting because that means the models could predict the score only with title information. The main body of the review is usually much monger than title and takes more storage and computation power to deal with.

The poor naive bayes classification result may be explained by the violation of naive bayes in this dataset. Naive Bayes classifier assume that the effect of the value of a predictor (_x_) on a given class (_c_) is independent of the values of other predictors. This is clearly not suitable for text analysis where the predictors have high correlation. After all, your sentence must be grammatically correct, so the sentence cannot be a random mix of vocabularies!

As for the third finding (similar prediction model), cautious are needed to interpret the result. Since the dataset is highly unbalanced towards high scores. The confusion matrix of the decision tree model with main content on validation data is printed below.

|  |1 (Prediction)  |2 | 3 | 4 | 5 |
|--|--|--|--|--|--|
|**1 (Actual)**   |**0**  | 0 |0  | 0 | 37 |
|**2**  | 0 | **0** | 0 |0  | 23 |
|**3**  | 0 | 0 | **0** | 0 | 74 |
|**4** |0  | 0 | 0 | **0** | 403 |
|**5** | 0 | 0 | 0 | 0 |**1977**  |


Indeed, by classifying all reviews as 5-star review, the model achieves an accuracy of 79%, but this doesn't really help.

After carefully comparing the accuracy score and confusion matrix of all models on both training and validation set, lasso regression seems to be the optimal model for main content analysis while linear regression works the best for title analysis.

Their confusion matrix on TESTING DATA are printed below.

**Lasso regression on main content analysis**
||1 (Prediction)|2 | 3 | 4 | 5 |
|--|--|--|--|--|--|
|**1  (Actual)**|**0**  | 0 |0  | 2 | 39 |
|**2** 			 | 0 | **0** | 0 |2  | 26 |
|**3**			 | 0 | 0 | **0** | 2 | 52 |
|**4**			 |0  | 0 | 0 | **5** | 388 |
|**5**			 | 0 | 0 | 0 | 14 |**1811**  |

**Linear regression on title content analysis**
||1 (Prediction)|2 | 3 | 4 | 5 |
|--|--|--|--|--|--|
|**1  (Actual)**|**0**  | 0 |5  | 25 | 11 |
|**2**			 | 0 | **0** | 4 |15  | 9 |
|**3** 			 | 0 | 0 | **11** | 21 | 22 |
|**4** 			 |1  | 0 | 7 | **111** | 274 |
|**5** 			 | 0 | 0 | 5 | 260 |**1560**  |

## 4 - Conclusion
Linear model and lasso regression are the recommended models for sentiment analysis with customer review. When both review main content and title are available, choosing title as model input could improve model performance and lower computation difficulties. However, cautions are need before applying such conclusion to other review categories.

## 5 - Appendix
### Monitor  Information
* A - Dell

[Dell - S2319NX 23" IPS LED FHD Monitor - Black/Silver](https://www.bestbuy.com/site/dell-s2319nx-23-ips-led-fhd-monitor-black-silver/6237640.p?skuId=6237640)
[Dell - S2719DGF 27" LED QHD FreeSync Monitor (DisplayPort, HDMI) - Black](https://www.bestbuy.com/site/dell-s2719dgf-27-led-qhd-freesync-monitor-displayport-hdmi-black/6293714.p?skuId=6293714)
[Dell - 27" IPS LED FHD FreeSync Monitor - Piano Black](https://www.bestbuy.com/site/dell-27-ips-led-fhd-freesync-monitor-piano-black/6394138.p?skuId=6394138)
[Dell - 32" LED Curved QHD FreeSync Monitor with HDR (DisplayPort, HDMI, USB)](https://www.bestbuy.com/site/dell-32-led-curved-qhd-freesync-monitor-with-hdr-displayport-hdmi-usb/6375331.p?skuId=6375331)

* B - LG

[LG - 24" IPS LED FHD FreeSync Monitor - Black](https://www.bestbuy.com/site/lg-24-ips-led-fhd-freesync-monitor-black/6362423.p?skuId=6362423)
[LG - 27UL600-W 27" IPS LED 4K UHD FreeSync Monitor with HDR (DisplayPort, HDMI) - Silver/White](https://www.bestbuy.com/site/lg-27ul600-w-27-ips-led-4k-uhd-freesync-monitor-with-hdr-displayport-hdmi-silver-white/6329956.p?skuId=6329956)
[LG - UltraGear 27" IPS LED QHD FreeSync Monitor with HDR (HDMI) - Black](https://www.bestbuy.com/site/lg-ultragear-27-ips-led-qhd-freesync-monitor-with-hdr-hdmi-black/6358119.p?skuId=6358119)
[LG - 34WL500-B 34" IPS LED UltraWide FHD FreeSync Monitor with HDR (HDMI) - Black](https://www.bestbuy.com/site/lg-34wl500-b-34-ips-led-ultrawide-fhd-freesync-monitor-with-hdr-hdmi-black/6329954.p?skuId=6329954)

* C - HP

[HP - 24f 23.8" IPS LED FHD FreeSync Monitor - Natural Silver](https://www.bestbuy.com/site/hp-24f-23-8-ips-led-fhd-freesync-monitor-natural-silver/6317590.p?skuId=6317590)
[HP - 25x 24.5" LED FHD Monitor (HDMI) - Gray/Green](https://www.bestbuy.com/site/hp-25x-24-5-led-fhd-monitor-hdmi-gray-green/6280605.p?skuId=6280605)
[HP - 27f 27" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Natural Silver](https://www.bestbuy.com/site/hp-27f-27-ips-led-fhd-freesync-monitor-hdmi-vga-natural-silver/6219205.p?skuId=6219205)
[HP - 31.5" IPS LED FHD Monitor (HDMI, VGA) - Black](https://www.bestbuy.com/site/hp-31-5-ips-led-fhd-monitor-hdmi-vga-black/6361917.p?skuId=6361917)

* D - Samsung

[Samsung - 390 Series 24" LED Curved FHD FreeSync Monitor (DVI, HDMI, VGA) - High glossy black](https://www.bestbuy.com/site/samsung-390-series-24-led-curved-fhd-freesync-monitor-dvi-hdmi-vga-high-glossy-black/5044601.p?skuId=5044601)
[Samsung - T55 Series 27" LED 1000R Curved FHD FreeSync Monitor (DisplayPort, HDMI, VGA)](https://www.bestbuy.com/site/samsung-t55-series-27-led-1000r-curved-fhd-freesync-monitor-displayport-hdmi-vga/6402202.p?skuId=6402202)
[Samsung - UR55 Series 28" IPS 4K UHD Monitor - Dark Gray/Blue](https://www.bestbuy.com/site/samsung-ur55-series-28-ips-4k-uhd-monitor-dark-gray-blue/6386391.p?skuId=6386391)
[Samsung - UJ59 Series U32J590UQN 32" LED 4K UHD FreeSync Monitor (DisplayPort, HDMI) - Dark Gray/Blue](https://www.bestbuy.com/site/samsung-uj59-series-u32j590uqn-32-led-4k-uhd-freesync-monitor-displayport-hdmi-dark-gray-blue/6293716.p?skuId=6293716)

* E - ASUS

[ASUS - 23.8" IPS LCD FHD FreeSync Gaming Monitor (DisplayPort, DVI, HDMI) - Black](https://www.bestbuy.com/site/asus-23-8-ips-lcd-fhd-freesync-gaming-monitor-displayport-dvi-hdmi-black/6395359.p?skuId=6395359)
[ASUS - VG245H 24” FHD 1ms FreeSync Console Gaming Monitor (Dual HDMI, VGA) - Black](https://www.bestbuy.com/site/asus-vg245h-24-fhd-1ms-freesync-console-gaming-monitor-dual-hdmi-vga-black/5591926.p?skuId=5591926)
[ASUS - 27" IPS LCD FHD FreeSync Gaming Monitor (DisplayPort, DVI, HDMI) - Black](https://www.bestbuy.com/site/asus-27-ips-lcd-fhd-freesync-gaming-monitor-displayport-dvi-hdmi-black/6336778.p?skuId=6336778)
[ASUS - ZenScreen 15.6” Portable Monitor (USB) - Dark Gray](https://www.bestbuy.com/site/asus-zenscreen-15-6-portable-monitor-usb-dark-gray/6403999.p?skuId=6403999)

* F - Acer

[Acer - 23.6" LED FHD Monitor (DVI, HDMI, VGA) - Black](https://www.bestbuy.com/site/acer-23-6-led-fhd-monitor-dvi-hdmi-vga-black/6404005.p?skuId=6404005)
[Acer - 23.8" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Black](https://www.bestbuy.com/site/acer-23-8-ips-led-fhd-freesync-monitor-hdmi-vga-black/6401005.p?skuId=6401005)
[Acer - 27" IPS LED FHD FreeSync Monitor (HDMI, VGA) - Black](https://www.bestbuy.com/site/acer-27-ips-led-fhd-freesync-monitor-hdmi-vga-black/6401007.p?skuId=6401007)
[Acer - Predator XB272 27" LED FHD G-SYNC Monitor (DisplayPort, HDMI, USB) - Black](https://www.bestbuy.com/site/acer-predator-xb272-27-led-fhd-g-sync-monitor-displayport-hdmi-usb-black/6238705.p?skuId=6238705)

