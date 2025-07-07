import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def dist_plot(y):
    sns.distplot(y)
    plt.show()

def bar_plot(df):
    df['company'].value_counts().plot(kind='bar')

def bar_plot_2(X,y):
    sns.barplot(x=X['company'], y=y)
    plt.xticks(rotation='vertical')
    plt.show()

def bar_plot_3(X,y):
    sns.barplot(x=X['type'], y=y)
    plt.xticks(rotation='vertical')
    plt.show()

def bar_plot_4(X):
    X['cpu'].value_counts().plot(kind='bar')

def bar_plot_5(X,y):
    sns.barplot(x=X['touchscreen'], y=y)
    plt.xticks(rotation='vertical')
    plt.show()

def heatmap_plot(df):
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.get_cmap('RdYlGn'), square=True, fmt='.2f')