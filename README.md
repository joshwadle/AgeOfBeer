# THE AGE OF BEER #

![Beer glasses graph](Images/Accumulation_Clarity_4Topics.png)


## BACKGROUND ##
We are living in the Age of Beer, new beers and breweries are being created all of the time. This leads me to think about the question "How old is my Beer?" and that is what this project is about. Fortunately, New Belgium was looking for a way to classify the expiration date or the age that a particular beer is no longer "True to Brand" or TTB. So with New Belgiums help and Data I created this project. My Project is an extension from the work of [Jason Olsons](https://github.com/Jason-Olson/Scoring-Beer-For-New-Belgium) and [Jan Van Zeghbroeck](https://github.com/janvanzeghbroeck/Seeing-Taste). They laid the ground work to clean and structure the comments to use in my analysis.

## GOAL ##
The Goal is to create visualizations that for how a beer changes over time. This will be put into a Flask app that will integrate with what New Belgiums intranet.

## DATA ##
The Data was a set of Comments from New Belgium Beer Testers about over 50 beers at different ages. These testers are all rated based on how they have rated beers in the past and how accurate they are with the tests that New Belgium gives them. The projects linked above show us how these ratings were created. A quick summary would be that each tester is rated on how well they are able to identify specific flavors based on taste, aroma, clarity, and body of the beer based on how the master tasters had alter the beer.

The data was very unbalanced with most of the beers only having comments for ages 1 to 5 months old. There were around a quarter of Beers with comments for 9-12 months and then only a couple Beers had ages all the way out to 25 months.

 ![Screenshot of Accumulation with only Clarity brocken into 4 topics by LDA](Images/Age_Distribution.png)

##### Figure 1: The Age distribution for all of the Beers #####

Figure 1 is a histogram that shows how the Age is broken up over the whole dataset. You can see that the majority of comments are about beers that are 1 to 5 months old. What this graph doesn't show is a distribution of comments about each individual beers. This is important because if a

## Methods ##

#### Tokenizing and Lemmatizing #####
The first thing I did was use the dictionary that New Belgium uses to change the comments from natural language into the corresponding flavors that their expert tasters have identified as being important to beer. Then I used Spacy to lemmatize and tokenize the string comment with the corresponding flavors. I used these new final comments to create TF-IDF vectors using the Sklearn library TfidfVectorizer. I used the TFIDF vectors for the rest of the project.

#### LDA ####

My next step for this project was to put my TF-IDF vectors for each comment into topics using a Latent Dirichlet allocation model. An LDA model simultaneously makes topics and groups documents into topics. I broke the vectors up depending on what beer they were talking about and then what question they were referring to (clarity, body, taste, aroma, final). The LDA model with the best separation for the majority of the beers I was looking at was with 5 topics for each beer. The topics for each beer are going to be different but in general below in Figure 2 is what a plot of the LDA module looks like.

 ![PLDAvis](Images/ScreenShot.png)
##### Figure 2: The Visualization of the topics found in Beer B #####

How these topics change over time is what is really interesting. The below plot shows the percentage that each comment on average fits into the above topics.

 ![Screenshot of Accumulation with only Clarity brocken into 4 topics by LDA](Images/B_Taste_Topics_percent.png)

##### Figure 3: The Precent of each topic based on how comments for the taste comments for Beer B #####

Figure 3 shows shows the how the topics can change over time for a specific beer. This graph was actually created from one batch of beer with the cumulative comments from months 9, 12 and 15.

#### MODELS ####
In the course of this project I used a variety of models to see what would work the best. I used Naive Bayes, Random Forest, XGBoost, MLP, and then a Lasso Regression. The most consistent results came from a Lasso Regression on the precent of each topic present in each comment from my LDA model for each Beer and Question. Some beers preformed a lot better than others just based on how the data was structured.

#### Lasso Regression ####
The Lasso Regression was the model that gave the best results. A Lasso regression is a way to make a linear regression that will penalize features that are large. This penalty is used so that if one feature is was larger than al of the rest it will not dominate. Below figure 4 is a plot of the residuals for a test set versus the predictions for that test set.

![Residuals](Images/Risidual_plot.png)
##### Figure 4: Plot of the Residuals from a Lasso Regression #####

These residuals are the errors that this Lasso Regression doesn't account for.

## Final Product ##
The final product is an app that is apart of New Belgiums intranet it can update automatically when given new information about a specific type of beer.

## FUTURE ##
I believe that there are trends in this data but with the lack of time difference to accurately see the. I think we would need to collect data for around 2 years to start to see really predictive results. The models I have created just need more data and then they will be highly predictive.

## Thanks##
Thanks to New Belgium Brewery for providing the tasting comments. Thanks to Jan and Matt for making this project possible.

