import pickle
import re

import matplotlib
import regex
import pandas as pd
import numpy as np
# import emoji
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from PIL import Image
import stopwords
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from urlextract import URLExtract
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams.update({'font.size': 13})

def rawToDf(file, key):
    '''Converts raw .txt file into a Data Frame'''

    split_formats = {
        '12hr': '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr': '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom': ''
    }
    datetime_formats = {
        '12hr': '%d/%m/%Y, %I:%M %p - ',
        '24hr': '%d/%m/%Y, %H:%M - ',
        'custom': ''
    }

    with open(file, 'r', encoding='utf-8') as raw_data:
        # print(raw_data.read())
        raw_string = ' '.join(raw_data.read().split(
            '\n'))  # converting the list split by newline char. as one whole string as there can be multi-line messages
        user_msg = re.split(split_formats[key], raw_string)[
                   1:]  # splits at all the date-time pattern, resulting in list of all the messages with user names
        date_time = re.findall(split_formats[key], raw_string)  # finds all the date-time patterns

        df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})  # exporting it to a df

    # converting date-time pattern which is of type String to type datetime,
    # format is to be specified for the whole string where the placeholders are extracted by the method
    df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])

    # split user and msg
    usernames = []
    msgs = []
    for i in df['user_msg']:
        a = re.split('([\w\W]+?):\s',
                     i)  # lazy pattern match to first {user_name}: pattern and spliting it aka each msg from a user
        if (a[1:]):  # user typed messages
            usernames.append(a[1])
            msgs.append(a[2])
        else:  # other notifications in the group(eg: someone was added, some left ...)
            usernames.append("group_notification")
            msgs.append(a[0])

    # creating new columns
    df['user'] = usernames
    df['message'] = msgs

    # dropping the old user_msg col.
    df.drop('user_msg', axis=1, inplace=True)

    return df

# df = rawToDf('sample_chat.txt', '12hr')
# df['day'] = df['date_time'].dt.strftime('%a')
# df['month'] = df['date_time'].dt.strftime('%b')
# df['year'] = df['date_time'].dt.year
# df['date'] = df['date_time'].apply(lambda x: x.date())
# print(df.head())

##### TOP STATISTICS. ##########
extract = URLExtract()
# # total users.

def users(df):
    users = df["user"].unique().tolist()
    users.remove("group_notification")
    active_users = users
    return active_users


## stats.  (total msg, total words, media msgs, link shared)
def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages,len(words), num_media_messages, len(links)

###### ACTIVITY #######

# 1) DAILY ACTIVITY OVER PERIOD OF TIME.
def daily_activity(df):
    df1 = df.copy()
    df1['message_count'] = [1] * df1.shape[0]
    df1.drop(columns='year', inplace=True)
    df1 = df1.groupby('date').sum().reset_index()


    # Improving Default Styles using Seaborn

    # For better readablity;

    df1.plot(x="date", y="message_count", figsize=(15, 15))

    # A basic plot

    # sns.barplot(df1.date, df1.message_count)
    plt.title('Daily activity in the group')
    plt.ylabel("Number of messages")
    plt.xlabel("Day")

    # Saving the plots
    plt.savefig('static/images/daily_activity.png')

    ## most active days.
    top10days = df1.sort_values(by="message_count", ascending=False).head(
        10)  # Sort values according to the number of messages per day.
    top10days.reset_index(inplace=True)  # reset index in order.
    top10days.drop(columns="index", inplace=True)  # dropping original indices.
    # Improving Default Styles using Seaborn

    top10days.plot(x="date", y="message_count", kind="bar", figsize=(15,6))
    plt.title("Most active days")
    plt.xlabel("Day")
    plt.ylabel("Message count")
    # Saving the plots
    plt.savefig('static/images/top10_days.png')


# 2) HOURLY ACTIVITY.
def hourly_activity(df):
    df3 = df.copy()
    df3['message_count'] = [1] * df.shape[0]  # helper column to keep a count.

    df3['hour'] = df3['date_time'].apply(lambda x: x.hour)

    grouped_by_time = df3.groupby('hour').sum().reset_index().sort_values(by='hour')


    # Beautifying Default Styles using Seaborn


    grouped_by_time.plot(x="hour", y="message_count", kind="bar", figsize=(15,8))
    plt.title('Hourly activity')
    plt.xlabel("hour")
    plt.ylabel("Number of messages")
    # Saving the plots;
    plt.savefig('static/images/hourly_activity.png')

# hourly_activity(df)

# 3 MONTHLY ACTIVITY.

def monthly_timeline(selected_user, df):
    df['month_num'] = df['date_time'].dt.month
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    # sns.barplot(timeline.month, timeline.message)
    # plt.plot((timeline.month), timeline.message )
    timeline.plot(x="month", y="message", kind="bar")
    plt.title('Monthly activity in the group')
    plt.ylabel("Number of messages")
    plt.xlabel("Month")
    # Saving the plots
    plt.savefig('static/images/monthly_activity.png')
    return timeline


# monthly_timeline("Overall", df)
import dataframe_image as dfi

###### ANALYSISNG ACTIVITY OF USERS IN THE GROUP ######33
def most_active_users(df):
    users = df["user"].unique().tolist()
    users.remove("group_notification")
    active_users = len(users)
    # non_active_users = 237 - len(active_users)
    top = 15
    if active_users == 2:
        top = 2
    elif 2 < active_users < 15:
        top = active_users/3



    ## top 15 users.
    df2 = df.copy()
    df2 = df2[df2.user != "group_notification"]
    topdf = df2.groupby("user")["message"].count().sort_values(ascending=False)

    # Final Data Frame
    topdf = topdf.head(top).reset_index()
    dfi.export(topdf, "static/images/top_users_df.png")
    ## adding a col of shortnames for betting visibiiyt in plot.
    topdf["order"] = [i for i in range(top)]

    # Improving Default Styles using Seaborn
    # sns.set_style("darkgrid")

    # Increasing the figure size
    # plt.figure(figsize=(10, 6))
    topdf.plot(x="order", y="message", kind="bar", figsize=(10, 6))
    # plt.bar(topdf.order, topdf.message)  # basic bar chart
    plt.xlabel("Top users")
    plt.ylabel("Corresponding number of messages")
    plt.savefig("static/images/most_active_users.png")
    topdf.drop('order', inplace=True, axis=1)
    return topdf


##### TOP USED WORDS (WORDCLOUD) ######

def word_cld(df):
    df3 = df.copy()
    df3['message_count'] = [1] * df.shape[0]  # helper column to keep a count.

    df3['hour'] = df3['date_time'].apply(lambda x: x.hour)

    grouped_by_time = df3.groupby('hour').sum().reset_index().sort_values(by='hour')
    comment_words = ' '
    # stopwords --> Words to be avoided while forming the WordCloud,
    # removed group_notifications like 'joined', 'deleted';
    # removed really common words like "yeah" and "okay".
    stopwords = STOPWORDS.update(
        ['group', 'remove', 'yes', 'hai', 'ok', 'okay', 'yeah', 'group', 'no', 'nahi', 'hi', 'joined using', 'na',
         'tha', 'deleted', 'message', 'oh', 'tha', 'haa', 'ha', 'haan', 'omitted', 'link', 'notification'])

    for val in df3.message.values:
        # typecaste each val to string.
        val = str(val)
        # split the value.
        tokens = val.split()
        # Converts each token into lowercase.
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        for words in tokens:
            comment_words = comment_words + words + ' '
    wordcloud = WordCloud(width=500, height=500,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=8).generate(comment_words)
    image = wordcloud.to_image()
    image.save("static/images/word_cloud.png")


# word_cld(df)
#### sentiment analysis now. ######

def extract_words(words):
    return dict([(word, True) for word in words])

def sentiment(df):
    user_list = df.user
    message_list = df.message
    sentiment = {}
    for i in range(len(user_list)):
        if (user_list[i] != 'group_notification'):
            sentiment[user_list[i]] = 0

    f = open('classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()

    for i in range(len(user_list)):
        if (user_list[i] != 'group_notification'):
            message = extract_words(message_list[i])
            s = classifier.classify(message)
            if (s == 'pos'):
                sentiment[user_list[i]] += 1
            else:
                sentiment[user_list[i]] -= 1

    df_sentiment = pd.DataFrame(columns=['User_Name', 'Sentiment'])
    names_list = list()
    sentiment_list = list()
    pos, neg = 0, 0

    for name in sentiment:
        sent = ''
        if sentiment[name] > 0:
            sent = 'positive'
            sentiment[name] = 'positive'
            pos += 1
        elif sentiment[name] == 0:
            sent = 'neutral'
            sentiment[name] = 'neutral'
            neg += -1
        else:
            sent = 'negative'
            sentiment[name] = 'negative'

        names_list.append(name)
        sentiment_list.append(sent)

    df_sentiment['User_Name'] = names_list
    df_sentiment['Sentiment'] = sentiment_list

    labels = ['Positive chat', 'Negative chat']
    sizes = [pos, abs(neg)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Whatsapp Chat Sentiment Analysis')
    plt.savefig('static/images/pie_chart.png')
    return df_sentiment

