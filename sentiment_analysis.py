import pandas as pd
import numpy as np
import nltk
import psycopg2
import psycopg2.extras
import sys

df = pd.read_csv(r'C:\Users\001am\Downloads\news\guardian_headlines.csv')
#print(df.head())


df1 = df.dropna()
#print(df1.isnull().sum())

df1.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
df1['removing_https'] = df1['Headlines'].str.replace(r'http.*$', '', case=False)
df1['lowercase'] = df1['removing_https'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df1['punctuation'] = df1['lowercase'].str.replace(r'[^\w\s]', '')


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
df1['review_nopunc_nostop'] = df1['punctuation'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

from textblob import Word
# Lemmatize final review format
df1['cleaned_tweet'] = df1['review_nopunc_nostop']\
.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#text = df1['cleaned_tweet']
'''
cleaned_tweets = []
for i in text:
        cleaned_tweets.append(i)
'''


#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
'''for tweet in cleaned_tweets[:10]:
    print(tweet)
    s = sia.polarity_scores(tweet)
    for k in sorted(s):
        print('{0}: {1}, '.format(k, s[k]), end='')
        print()

'''
df1['sentiment'] = df1.cleaned_tweet.apply(lambda x: sia.polarity_scores(x)['compound'])
#pd.set_option('display.max_colwidth',-1)
#print(df1[['Headlines', 'sentiment']][0:20])
#df1.to_excel('./output.xls')
#print(df1.head())

df2 = df1[['Headlines','sentiment']]

#################################################CODE--OVER##################################################


param_dic = {
    "host"      : "localhost",
    "database"  : "sentiment_analysis",
    "user"      : "postgres",
    "password"  : "Kumar@711"
}

#setting up the connection
def connect(params_dic):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1) 
    print("Connection successful")
    return conn
conn = connect(param_dic)

#creating table
def create_table(cursor):
    cursor.execute("""
        DROP TABLE IF EXISTS data;
        CREATE TABLE data (
            headlines         TEXT not null,
            sentiment         float not null,
            
        );""")

#inserting into table
def fcn(df,table,cur):

    if len(df) > 0:
        df_columns = list(df)
        # create (col1,col2,...)
        columns = ",".join(df_columns)

        # create VALUES('%s', '%s",...) one '%s' per column
        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 

        #create INSERT INTO table (columns) VALUES('%s',...)
        insert_stmt = "INSERT INTO {} ({}) {}".format(table,columns,values)
        cur = conn.cursor()
        psycopg2.extras.execute_batch(cur, insert_stmt, df.values)
    conn.commit()

cur = conn.cursor()
fcn(df2
,'data',cur)

#printing the table
cur.execute("select * from data limit 10")
output = cur.fetchall()

for i in output:
    
    print("Headlines :", i[0])
    print("Sentiments :", i[1])


#################################################extra#######################################################

'''
def findpolarity(data):
    sid = SentimentIntensityAnalyzer()
    polarity = sid.polarity_scores(data)
    if(polarity['compound'] >= 0.5):  
        sentiment = 1
    if(polarity['compound'] <= -0.5):
        sentiment = -1 
    if(polarity['compound'] < 0.5 and polarity['compound'] >-0.5):
        sentiment = 0     
    return (sentiment)



findpolarity(cleaned_tweets[0])
sentiments = []
for i in range(0, len(cleaned_tweets)):
    s = findpolarity(cleaned_tweets[i])
    sentiments.append(s)

print(len(sentiment))
#print(len(cleaned_tweets))

'''