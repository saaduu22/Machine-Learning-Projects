# Soirts Betting Project

Footodd.io Primer
By Saad Shahid

Footodd.io is a personal project of mine to help me explore sports betting, sportsbooks and machine learning. All things im interested in. 

tl;dr: A Machine Learning Algorithim in app.py tries to predict the outcome of football matches, compares that to a sportsbook aggregator data to generate the best betting strategy. DOES NOT AUTOMATICALLY PLACE ANY BETS WHATSOEVER

I created the basis of the machine learning algorithm a few months ago and have tweaked it a bit but still need to put in a lot of time on it (which im finding more difficult to put in then I thought I would). Even though the algorithm can be improved I always think it’s a good idea to show people the product and get feedback during the process, so here it is!

This primer is a very quick summary of how the product works, some of the constraints and my plans for the future of this work. 

How it works:

To explain how it all comes together, I will break it down into three main areas. Data Gathering, Machine Learning and User Interface. I have a rough sketch of the system flow posted below in case it helps

The site itself is hosted on Heroku because of the ease of scaling dynos and cost purposes

Data Gathering:

There are 2 main sources of data that are used. 

1.	Football Data: I built a web scraper to download and make machine readable data from the premier league from https://football-data.co.uk/ - Raw Data db
2.	Sportsbook Aggregator: For this I use the odds api to get historical data and store that in a database that I can query – Odds db

The data once gathered gets stored into a database that is then queried and used in the algorithm using a custom flask endpoint

Machine Learning:

Currently the model uses a random forest regression model to create predictions over a training set that are then back tested on a testing set. This generates predictions and precision scores.

In most cases like this, a gradient boost would be better but because the model is so new and I don’t feel like I’ve spent enough time exploring the data, for now I want to stick with random forest regression. In cases where the relationship between the dependent outcome and independent outcome aren’t extremely well established, I’ve found random forest generations can work slightly better. Predictions once made are stored in the prediction database so the model only needs to run once 

Emperically, we see a roughly 3-5% increase in precision by using random forest models on the current dataset and strategy. This is probably due to overfitting and noise modeling by gradient boosts. I used Scikit learn as the machine learning module of choice 

User Interface:

For the user interface I use Webflow as a UI library and some small amounts of custom JS, HTML and CSS. Because it’s such a small application I didn’t bother with react but it could be an interesting option to explore. All in all I would highly recommend Webflow to everyone and I’m really happy with how quickly I could put a UI together


Winning Strategy:

1.	Small Bets approach: Unless we find a slam dunk oppurtunity, I recommend using small bets to win over the long run. In fact most of our betting strategies default to 70:30 split to hedge the outcomes. The conservative approach is due to the lack of confidence in the algorithim but also it is extremely difficult to consistently predict upsets. This ensures there is no one make or break event. As long as we can generate returns higher then most people then that’s a win, remember, machine learning is part discovering oppurtunities, but it’s also a lot of automation, so small wins over times add up

2.	EV as winning metric: The winning strategy is based on a simple expected value theorum. Basically if our EV is positive then we stick with our predicted winner and place the majority of our money on what the model asked us to, but if its negative then we flip our decision. Doing the inverse of what our model said. Sounds counter intuitive…..Why go through this trouble to not listen to the algorithim. But in cases where the EV is negative, its likely that sportsbook have a significnatly more negative opinion of one of the teams then we do whereby because we are hedging as part of our small bets approach, our hedging strategy could cause us to actually lose. Since we are likely to only be negative by small amounts it is still a relatively small bet. Emperically its also been noted that our model performs best in closer matches then in one-sided ones. Its likely that sentiment over indexes the teams performance once it gets above or under a certain threashold 

Current Limitations:

1.	Overly-indexing on historical data: Since there is a time lag on our test data to the next predictable event, our model might miss out on important changes in that time. Think about clubs being bought by billionaires (Newcastle), big signings (Mbappe) etc. All of these things will drastically change a teams performance but will likely be missed by our algorithm for now

2.	Form Data: Teams that are in form tend to stay in form. Currently the algorithm can’t account for that. I want to implement a moving averages approach where our algorithm considers a difficulty weighted win/loss ratio over an n number of games to boost or penalise a team

3.	Dataset Size: Currently splitting our dataset into training and testing sets limits our ability to predict, but also our precision. I don’t want to go too far back in time as teams evolve too quickly, instead I will be looking to source data from other leagues. (la liga, Serie A etc.) under the assumptions that the fundamentals of winning a game will translate over

4.	Lack of categorical variables: We don’t currently index any significant categorical data which I want to start to do. Areas I’ve been exploring are twitter sentiment, manager performance etc. Its likely that most objective data is already being accounted for by sportsbook so it would be difficult to do something to beat them there

5.	Computational Cost: Machine learning algorithms can quickly get expensive so for now I have decided to keep it light on the data storage and computational side of things, which results in poorer outcomes

All that being said, it’s a side project….So there might be bugs, it might not be perfect but I’m very happy with the amount of exposure its gotten me to sportsbook, sports betting and machine learning…..Which was the main purpose behind it

I hope that adequately explains this project, if you have any more questions, feel free to reach out to me at saad22shahid@gmail.com!

Thanks for reading!
