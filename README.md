# Natural Language Processing & Reddit


## Can We Determine Whether a Reddit Post Belongs to r/science or r/todayilearned?

[Reddit](https://www.reddit.com/) likes to call itself the frontpage of the internet. The site is broken out into subreddits that host a specific topic. It can range from a sports team like the [Miami Dolphins](https://www.reddit.com/r/miamidolphins/) to [Dungeons & Dragons](https://www.reddit.com/r/DungeonsAndDragons/) and any obscure subculture that branches off of those. Moderators try their best to keep everything in the subreddit on topic and relevant, but that can be hard, especially on larger subreddits that cover things like world news or sports that can be active 24 hours a day. This raises the question of whether or not it is possible to figure out where a post comes from if you do not know the original subreddit. Most of the time, there is only a title and then a link. Sometimes there is what's called selftext, text in the post outside of the title, but not always. Can the limited text available be used to determine where a post came from? The short answer is yes, you can predict post's subreddit use Natural Language Processing. 


## How Can We Determine a Post's Original Subreddit?

Every subreddit is different, and as such, each subreddit has a certain lingo or common words that pop up again and again. If you can learn what those words are, you can start to use them to determine a post's origin using a model. By breaking a post down to its base elements, the words, using a Count Vectorizer or a Term Frequency-Inverse Document Frequency Vectorizer, you can start create a matrix of all the words and numbers used in a collection of posts. Once you have that matrix, you can then use it in a model, such as K-Nearest Neighbors or Logistic Regression and predict where the post came from. 


## A Sample Case: r/science or r/todayilearned?

To show how this can be done, let's look at two popular subreddits: [r/science](https://www.reddit.com/r/science/) and [r/todayilearned](https://www.reddit.com/r/todayilearned/). The r/science subreddit focuses on all things science and technology, and most posts contain a headline and a link to the article with said headline. The moderators do a good job at keeping posts mostly on topic and not straying to far from science. The r/todayilearned subreddit is much more diverse in topics. Redditers post random facts or tidbits that they come across to share with others. Some of the posts are random and very unrelated, but other posts are heavily influenced by world events. I will show an example of that later on. Both subreddits can be considered educational, albeit in very different ways. The challenge is to see if it is possible to determine whether a post came from r/science or r/todayilearned. 


### Gathering and Cleaning the Data

The first step was to gather posts from each subreddit. To do this, an API was used in the [01_Data_Gathering](./code/01_Data_Gathering.ipynb) notebook to gather the posts. [Pushshift's](https://github.com/pushshift/api) API was tasked with collecting 1,500 posts from each subreddit. Unfortunately the API stopped working right at the end an only 1,447 posts from r/todayilearned were collected, but it was more than enough for my purposes. All of the posts were collected into one dataframe and saved as a single csv in [raw](./data/raw). The posts were then imported into the [02_Cleaning_and_EDA](./code/02_Cleaning_and_EDA.ipynb) notebook and cleaned. There were 4 initial features collected for each post: 'title', 'selftext', 'subreddit', and 'created_utc'. You can learn more about these features in the data dictionary section below.


The first step in the data cleaning was to fill in any null values, of which there were a lot. Both subreddits typically only included a title and then a link to an article or website. As such, the selftext was often blank, resulting in a null. Of 2,997 posts, 2,971 had no selftext. That meant that some posts did have selftext which could be used to help make a more accurate model. Thus the null values were filled in with a blank string. Additionally, any selftext that contained only '\[removed]' was changed to a blank string. When a redditer or moderator deletes the body of the post, the title may remain but the selftext says '\[removed]'. This is consistent across all subreddits and would not be useful for determining a post's origin, so replacing it with a blank string saves time down the road when analyzing the most common words. The next step was to create a new column called 'all_text' that combined 'title' and 'selftext' into one string to be run through the model. The last step was to then convert one subreddit into the positive outcome and the other into the negative outcome. Determined solely by alphabetically order, r/science became the positive and r/todayilearned became the negative.


### What Can We Learn About the Collected Posts?

The rest of the [02_Cleaning_and_EDA](./code/02_Cleaning_and_EDA.ipynb) notebook was devoted to exploratory data analysis (EDA). Most of the EDA was focused on finding out what words are most common to each subreddit. The thing to keep in mind while doing this type of EDA is that what helps in exploring the data does not necessarily work well in the model. 


To breakdown each post into a list of words, hereafter known as tokens, a count vectorizer was used and the different parameters were adjusted to see if a more distinct list of words for each subreddit could be determined. To start, the count vectorizer was run as is to get a baseline list. It was obvious pretty quickly that common words such as 'the' and 'of' needed to be removed. That led to stop words being used. Stop words are common words that are omitted from a model, making it use a narrower, more specialized matrix of words. Utilizing the stop_words parameter made the lists much more unique. However, there was a word that was not in the built-in stop words that needed to be omitted. The vast majority of posts (77% in the training dataset) in r/todayilearned start with 'TIL'. If a post had this at the beginning, it was very, very likely from the r/todayilearned subreddit. Including it could both make classifying the posts too easy and/or mess up the model. Since the percentage is so high, the model may severely overweight 'TIL' and think that every post that doesn't have 'TIL' would belong to r/science, but that would be false. That would conversely make classifying the majority of posts much easier. As such, it was decided to add it to the stop words list to avoid overfitting the model and possibly breaking it. 


Now that the common words were gone, a much more unique word list for each subreddit was defined. Looking at the top words for r/todayilearned, the most common words were relating to the war in Ukraine that broke out during the course of this project. This shows that r/todayilearned can be heavily influenced by world events. On the other hand, r/science was more academic, with words such as 'study', 'new', and 'covid' being in the most common words. Trying to narrow down the lists even more, the min_df and max_df parameters were explored. Min_df removes words that don't occur under a certain number while max_df removes words that occur too often. Min_df did not reduce the most common words, but it did reduce the total number of words in the matrix. When set to 5, min_df dropped the number of words in the matrix from 9,457 to 1,239. Max_df did not affect the data until it was given a percentage of .05, and all that did was remove 'study' and 'new' from r/science's word list. 


The next parameter to be explored was n-gram. N-gram looks at a specified amount of consecutive words, usually 2, and adds them to the matrix. This did change the top words lists, but not a whole lot. The top words were still present, but they were now combined with other words. Not a whole lot could be gained from this in EDA, but it would be useful for modeling. The custom tokenizer parameter was explored, but it did not provide anything new at this time. Given more time, it could be explored more and maybe help more accurately breakdown the posts. The same could be said for token pattern. Custom tokens were explored using [Regex101](https://regex101.com/), but the best token created did not do anything new. In fact, it generated the same amount of words for the matrix as the default setting did. This was the last count vectorizer explored. 


The last two sections in the EDA notebook created two new columns, 'word_count' and 'sentiment'. The 'word_count' column counted the number of words in a post by breaking the string up using spaces. It is not perfect, but it got the job done for the purposes of this project. When plotted, it became clear that r/science tended to have shorter posts than r/todayilearned, which had a more even spread of post lengths. This feature would be used in the later modeling. The 'sentiment' column was created using a sentiment analyzer to see if there was a patten to the posts in terms of being positive or negative. Both subreddits ran the gambit from very negative to very positive, with median being purely neutral. As such, it was determined that this feature would not be used. 


Please look at that notebook to see more notes, graphs, and charts that expand on the analysis and help provide some more context on what was done. 


### Data Dictionary

|Feature|Type|Description|
|---|---|---|
|title|object|The title of the post. Every post that isn't removed has a title that is either a dicussion topic or the headline of an article shared in the post|
|selftext|object|This is the body of the post. Any text other than the title is included here. For most of the posts gathered, there was not selftext since the posts provided links to articles instead.|
|created_utc|int|The time the post was created in Universal time. This column was only used to generate posts and ensure there weren't any duplicates.|
|all_text|object|This is the combined title and selftext feature. It is what the models will use to make sure that all of the possible text is analyzed for the model.|
|word_count|int|Total number of words in the post. Generated by breaking up the all_text string on spaces.|
|sentiment|float|Generated using a sentiment analyzer. The closer the number is to 1, the more positive it is. The closer to -1, the more negative it is. 0 is neutral.|


### Using the Data to Generate Predictions

After EDA was completed, it was time to create some models in the [03_Modeling](./code/03_Modeling.ipynb) notebook. There were two models used: K-Nearest Neighbors (KNN) and Logistic Regression (LogReg). KNN is not always the best model in terms of accuracy, but it is fast. It determines which subreddit a post is from based on the closest known posts. When put into a pipeline with a randomized search or grid search that can tweak the parameters, a decent model can be generated. LogReg is a better model that is also fast. It takes in features and assigns a coefficient to them. It then looks at the post and uses an internal equation to predict what subreddit the post belongs to. It is usually more accurate than KNN because KNN is limited to looking at nearby posts while LogReg focuses more on the post itself to determine it's subreddit.


There where two vectorizers used in conjunction with each model: count vectorizer and term frequency-inverse document frequency vectorizer (TF-IDF). Both of these vectorizers generate word matrices for KNN and LogReg to use, but they do it in different ways. Count vecotrizer creates a column for each word in every post and counts the occurances of the word in the corpus. TF-IDF also counts each word, but takes into account how common the word is across all documents. These vectorizers were put into a randomized search and grid search, along with models. Grid search was dropped after the first model because it did not perform better on the test data than the randomized search, the scores were close together, and randomized search was faster. The randomized search function allowed a variety of parameters to be tried out together to determine the best combination for the models and vectorizers. Each model in the 03 notebook shows the best scores and best parameters. 


The word count feature was added in at the end of the notebook to see if it would improve the models. It actually made the KNN model worse, but it did slightly improve the LogReg model, making it the best model overall. The best vectorizer was the Term Frequency-Inverse Document Frequency Vectorizer (TfidfVectorizer) with the following settings: stop_words of 'til', ngram_range of (1,2), min_df of 1, max_df of 0.5, and binary of False. The Logistic Regression model with a solver of 'saga' and a logreg_penalty of 'none', and included the word_count feature, was the best model.


Please look at the [03_Modeling](./code/03_Modeling.ipynb) notebook for more details on each model that was run and scores were like for each.


## What Can We Conclude?

Now that the best model was determined, it was time to evaluate it and see just how good it was at predicting data. In the [04_Best_Model_and_Final_Thoughts](./code/04_Best_Model_and_Final_Thoughts.ipynb) notebook, I provide present the best model. It has an accuracy score of 88.67% when trying to figure out what subreddit a post belongs to. That is pretty accurate given that 'TIL' has been omitted. This model, with a few tweaks, can be used sort posts from other subreddits so long as it is trained on them. Another positive note is that false positives and false negatives were not heavily skewed in either direction. Overall, this is a good model that proves that you can create a model that can predict where a post comes from with nearly 90% accuracy. At least when it comes to comparing r/science and r/todayilearned. Please see the 04 notebook for some more final notes.


## What are the next steps?

There are couple of things that could be done to expand on this project. There are a ton of different models that I was taught towards the end of the week that I could use to get predictions that may be even more accurate. I also tried working with regular expressions and custom tokenizers to no success, so if I had more time I could explore those more.

If I wanted to keep the same model, I would wait a couple of months and try the same subreddits again to see how it would perform. I think r/science would be about the same but I'm curious as to what the top words would be like for r/todayilearned after the Ukraine war is over. I would think the model accuracy would go down, but the nature of r/todayilearned is such that any major world event could easily generate a lot of posts there, making something like 'pineapple' (maybe a rainbow pineapple is grown or something?) take over the top spot for a week or so as redditers learn more about pineapples.

All in all, I am happy with how the project turned out, but I would like to have had more time to try newer models and maybe gather even more posts to better train the models with.