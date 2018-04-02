# Wining and Mining

This data set has been downloaded from [Kaggle](https://www.kaggle.com/zynicide/wine-reviews) and contains data of wine reviews from [WineEnthusiast](https://www.winemag.com/).

## Value for Money 
### [notebooks/1.0-jb-value-for-money.ipynb](notebooks/1.0-jb-value-for-money.ipynb)

One of the many important roles data analytics can play in ecommerce is in deciding how to display search results to customers. Imagine you are searching for a flight from Cape Town to Amsterdam. There are potentially hundreds of different flight options that could be returned. How should the the online travel agent decide what to show you first? The interests of the company and you, the customer, are aligned here - You want to see results that are most relevant to you nearer the top because its convenient, aleviating the need for you to wade through less relevant results. And from the online travel agent's perspective, they want to show you flight option that you are most likely to book nearer to the top, since you will be less likely to leave the site before finding an option that you'd like to book. But how do companies decide what results are most 'relevant' - options that you are most likely to book? One common category of results are ones that are good "value for money". This is intuitive, since people generally want to get the most "bang for buck", nomatter what their price-range. In this notebook, I explore how we can think of value for money by exploring a dataset of wines, which were scraped from the [WineEnthusiast](https://www.winemag.com/) website and which you can find [here](https://www.kaggle.com/zynicide/wine-reviews).

Consider this: you are a data scientist at an online wine store. You are experimenting with a new feature - highlighting the best value for money wine on the top of the search results page. Someone could search for wines from South Africa and get back a bunch of potential results.

You might have some way of sorting by default (maybe by popularity or price) but at the top of the results you would highlight the best value for money wine. The question is, how do you determine which wine that is?

I will be exploring 3 different potential ways of doing this, adding layers of sophistication in each subsequent method:
1. Rank wines from highest to lowest score within their respective price range. The best value for money wine could be the one with the highest rank out of the wines returned by the search.
2. Create a value for money index. The wine with the highest index would be the best value for money.
3. Regress price on available predictors and measure how far above the predicted price each wine is. The best value wine would be the one farthest below its predicted price.

This notebook proceeds in 3 parts - first, I am going to do some data wrangling in order to get the data into a form that will be easier to analyse later. Then I will explain and develop the 3 methods for working out which wines are the best value for money. Finally, I will wrap up with some potential issues and next steps for putting such a feature into production, before concluding with a summary.
