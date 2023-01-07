from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

import helper

import re
import regex

import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

import stopwords
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates


# Create flask app
flask_app = Flask(__name__)
@flask_app.route("/")
def Home():
    return render_template("index.html")

# @flask_app.route("/Analysis", methods = ["POST"])
# def predict():
#     print("Complete analysis")


@flask_app.route('/analysis', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        format = str(request.form.get('formats'))
        df = helper.rawToDf(f.filename, format)
        df['day'] = df['date_time'].dt.strftime('%a')
        df['month'] = df['date_time'].dt.strftime('%b')
        df['year'] = df['date_time'].dt.year
        df['date'] = df['date_time'].apply(lambda x: x.date())

        # top statistics.
        total_messages, total_words, media_messages, links = helper.fetch_stats("Overall", df)
        helper.daily_activity(df) ## saving daily activity plot.
        helper.hourly_activity(df)
        helper.monthly_timeline("Overall", df)
        top_users = helper.most_active_users(df)
        helper.word_cld(df)
        print(top_users)
        # top_users.drop('order', inplace=True, axis=1)
        df_sentiment = helper.sentiment(df)


        return render_template("analysis.html", chat_name=f.filename, total_msg=total_messages, words=total_words,
                               media=len(helper.users(df)), links=links,
                               column_names=df_sentiment.columns.values, row_data=list(df_sentiment.values.tolist()),
                               zip=zip,
                               column_names1 = top_users.columns.values, row_data1=list(top_users.values.tolist()))



if __name__ == "__main__":
    flask_app.run(debug=True)