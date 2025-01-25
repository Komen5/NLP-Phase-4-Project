# Twitter Sentiment Analysis Project
<img src="images/image2.jpg" alt="Negative, Neutral and Positive Emotions" width="800" height="400">

This project utilizes sentiment analysis of Twitter data to understand customer views and preferences regarding iPhone and Google products. By examining user sentiments, Best Buy intends to make stocking decisions that match customer demand and boost overall satisfaction.


## Overview

Best Buy, a leading retailer of iPhone and Google products, aims to improve its inventory choices by analyzing user opinions on these items. This project utilizes sentiment analysis of Twitter data to understand customer views and preferences regarding iPhone and Google products. By examining user sentiments, Best Buy intends to make stocking decisions that match customer demand and boost overall satisfaction. The project involves gathering Twitter data on discussions, reviews, and mentions of iPhone and Google products, followed by data preprocessing and sentiment analysis. The insights gained will help Best Buy optimize its product selection and align it with customer preferences.

## Problem statement

Best Buy, a top reseller of iPhone and Google products, encounters difficulties in aligning its inventory with customer preferences. The lack of a structured approach to analyzing user sentiments on Twitter hinders data-driven stocking decisions. This project aims to utilize Twitter sentiment analysis to better understand customer opinions. By gaining insights into user sentiments, Best Buy aims to enhance its stocking decisions, ensuring product availability that matches customer preferences and improves overall customer satisfaction and loyalty.

## Business Value

By leveraging Twitter sentiment analysis to understand customer opinions, Best Buy will be able to make more informed stocking decisions leading to better allocation of resources, improved product assortment, enhanced customer satisfaction and profitability. By staying ahead of trends and understanding customer sentiments better than competitors, Best Buy can position itself as a preferred destination for purchasing iPhone and Google products.

## Objectives

The goal of this project is to analyse customer sentiments on Twitter to inform Best Buy's stocking decisions for iPhone and Google products. By analyzing customer opinions, the aim is to improve resource allocation, optimize product assortment, enhance customer satisfaction, and ultimately increase profitability.

## Research Questions

1. What are the predominant sentiments expressed by customers on Twitter regarding iPhone and Google products?

2. What factors influence the polarity of tweets related to iPhone and Google products on Twitter?

3. Which specific features of iPhone and Google products are most frequently praised or criticized by users on Twitter?

4. Which machine learning model is most effective in sentiment analysis?



## Data Understanding

The Present dataset contains a series of 9093 Tweets, the dataset has been pre-labeled by human raters. Raters judged if the tweet's text expressed a positive, negative or no emotion towards a brand and/or product, any time an emotion was expressed the rater was then asked to identify the brand or product that was the target of that emotion. All of this data was compiled into a CSV file labeled "judge-1377884607_tweet_product_company.csv" that can be found in the root of this repository.

The Tweets were in large part centered on Apple and Google products during/after the 2011 South by Southwest (SXSW) Conference. The resulting data file contains three columns per row, one for the tweet's text, one for the emotion expressed and one for the target product/brand of that emotion, when identifiable.

Data was sourced from CrowdFlower via data.world, added by Kent Cavender-Bares on August 30, 2013.

## Conclusions

1. The sentiment analysis revealed distinct trends in customer opinions about iPhone and Google products. Positive sentiments dominated, reflecting strong customer satisfaction, while negative sentiments provided valuable insights into areas of improvement.
2. Among the models tested, the Fine-Tuned Neural Networks model performed the best with an accuracy of 90%. This demonstrates its suitability for text classification tasks like sentiment analysis.
3. Frequent mentions of specific features, such as "store," "link," and "ipad," highlighted customer priorities and areas of focus for both brands. These terms provided actionable insights into customer needs and preferences.
4. Tweets were effectively categorized into ambiguous, negative, neutral, and positive emotions. This categorization provided a nuanced understanding of customer sentiment, enabling more targeted decision-making.

## Recommendations

1. Regularly monitor and analyze customer feedback on social media platforms to stay updated on changing preferences and emerging trends.
2. Investigate and address the root causes of negative sentiments. For instance, frequent mentions of "store" in a negative context could indicate issues with in-store experiences or product availability.
3. Pay attention to commonly mentioned product features and ensure they align with customer expectations. For example, improving features like "ipad" or "link" based on customer discussions can boost satisfaction.
4. Adopt and integrate sentiment analysis tools into business processes to automate and scale this analysis for other products and brands.
5. Utilize the best-performing model (Neural Networks (Fine-Tuned) for ongoing sentiment analysis tasks. Regularly retrain the model with new data to maintain its accuracy and relevance.
6. Expand the sentiment analysis framework to include competitors and other product categories. This can provide a comprehensive view of market dynamics and inform strategic decisions.
