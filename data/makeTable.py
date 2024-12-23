import os
import pandas as pd
import numpy as np



base_src =  os.path.dirname( os.path.abspath(__file__))

def user() : 
	u_user_src = os.path.join(base_src,'u.user')
	u_col = ['user_id','age','sex','occupation','zip_code']
	users = pd.read_csv(u_user_src, sep = "|", names = u_col, encoding = 'latin-1')
	users = users.set_index('user_id')
	return users


def movies(): 
	u_user_src = os.path.join(base_src,'u.item')
	
	i_cols = ['movie_id','title','release date','video release date','IMDB URL','unknown','Action','Adventure','Animat ion',
	'Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','FilmNoir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
	movies = pd.read_csv(u_user_src, sep = '|', names = i_cols,encoding='latin-1')
	movies = movies.set_index('movie_id')
	return movies


def ratings():
    u_user_src = os.path.join(base_src,'u.data')
    u_col = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_csv(u_user_src, sep = "\t", names = u_col, encoding = 'latin-1')
    ratings = ratings.set_index('user_id')
    ratings.drop('timestamp',axis=1, inplace=True)
    return ratings
