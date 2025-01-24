import pandas as pd
from gensim.models import Word2Vec
import nltk
import numpy as np
import os

df = pd.read_csv('Projet2/imdb_top_1000.csv')

user_interactions =[
    {"user_id": 1,"serie":"Interstellar" ,"genre":["Adventure","Drama","Sci-fi"],"view_time":169,"rating":9} ,
    {"user_id": 1,"serie":"The Dark Knight" ,"genre":["Action","Adventure"],"view_time":152,"rating":7} , 
    {"user_id": 1,"serie":"Inception" ,"genre":["Action","Adventure","Sci-fi"],"view_time":60,"rating":None} ,
    {"user_id": 1,"serie":"Deadpool 2" ,"genre":["Action","Adventure","Comedy"],"view_time":119,"rating":7} ,
    {"user_id": 1,"serie":"Captain America: The Winter Soldier" ,"genre":["Adventure","Drama","Sci-fi"],"view_time":40,"rating":None} ,
    {"user_id": 1,"serie":"Avengers: Endgame" ,"genre":["Action","Adventure","Drama"],"view_time":181,"rating":8} ,
    {"user_id": 1,"serie":"Rear Window" ,"genre":["Mystery","Thriller"],"view_time":20,"rating":None} ,
]


## nettoyage du dataset 

## remplir les meta scores manquant de certains film par la moyenne des meta scores 

mean_meta_score = df['Meta_score'].mean()

df['Meta_score'] = df['Meta_score'].fillna(mean_meta_score)

## suppression des lignes dupliquees 

df = df.drop_duplicates()




## tri des preferences du user en fonction des infos des interactions 

## trouver les films que le user aime 

films_liked = [film['serie'] for film in user_interactions if film['rating'] is not None and film['rating'] >= 7]


## criteres a evaluer ( ordre decroissant ) : genre ,  star1 , star2 , star3 , star4 ,, imdb_rating , meta_score , director 

infos_films_liked = {
    film: df[df['Series_Title'] == film].iloc[0].drop('Poster_Link').to_dict() 
    for film in films_liked
}

##print(infos_films_liked)

genres = []
imdb_rating = 7
stars = []
directors = []
meta_score = 70

for film in infos_films_liked:
    
    liste = infos_films_liked[film]['Genre'].split(",")
    for mot in liste : 
        if mot not in genres:
            genres.append(mot)
            

    directors.append(infos_films_liked[film]['Director'])
    
    if infos_films_liked[film]['Star1'] not in stars:
        stars.append(infos_films_liked[film]['Star1'])
    if infos_films_liked[film]['Star2'] not in stars:
        stars.append(infos_films_liked[film]['Star2'])
    if infos_films_liked[film]['Star3'] not in stars:
        stars.append(infos_films_liked[film]['Star3'])
    if infos_films_liked[film]['Star4'] not in stars:
        stars.append(infos_films_liked[film]['Star4'])
    

genres = [mot.strip() for mot in genres]

##print(genres)

filtered_df = df[df['Genre'].apply(lambda x: any(genre in x for genre in genres))]

##supprimer les films que le user a vu : 

filtered_df = filtered_df[~filtered_df['Series_Title'].apply(lambda x: x in films_liked)]



filtered_df = filtered_df[
    filtered_df[['Star1', 'Star2', 'Star3', 'Star4']].apply(lambda row: any(x in stars for x in row), axis=1)
]

filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= imdb_rating]

filtered_df = filtered_df[filtered_df['Meta_score'] >= meta_score]

##print(filtered_df['Series_Title'].tolist())

filtered_df = filtered_df[filtered_df['Director'].apply(lambda x: x in directors)]


##print(filtered_df['Series_Title'].tolist())

films_similaires_metadonnees = filtered_df['Series_Title'].tolist()









phrases = df['Overview'].tolist()

## tokeniser les phrases ( resumes )

phrases_tokenized = [nltk.word_tokenize(phrase) for phrase in phrases]

## nettoyer les phrases ( enlever juste la ponctuation et mettre tout en minuscule )

phrases_tokenized = [[word.lower() for word in phrase if word.isalpha()] for phrase in phrases_tokenized]


df['Tokenized'] = phrases_tokenized


## charger le modele

model = Word2Vec.load('Projet2/model.model')

## representation vectorielles des resumes des films

df['moy'] = [[] for _ in range(len(df))]

for idx in range(len(df)):
    
    # Tokeniser le texte
    tokens = df.at[idx, 'Tokenized']
    

    # Initialiser la liste pour stocker les vecteurs des mots présents dans le texte
    tab = []
    
    for mot in tokens:
        if mot in model.wv:
            tab.append(model.wv[mot])
    
    # Calculer la moyenne des vecteurs des mots
    if len(tab) > 0:
        moy = np.mean(tab, axis=0)
    else:
        moy = np.zeros(model.vector_size) 
    
    # Sauvegarder la moyenne calculée dans la colonne 'moy' du DataFrame
    df.at[idx, 'moy'] = moy


## representation vectorielle des preferences du user

user_vector = np.zeros(model.vector_size)

for film in films_liked:
    idx = df[df['Series_Title'] == film].index[0]
    user_vector += df.at[idx, 'moy']

user_vector /= len(films_liked)

## calculer la similarite cosinus entre le vecteur du user et les vecteurs des films

df['similarity'] = df['moy'].apply(lambda x: np.dot(x, user_vector) / (np.linalg.norm(x) * np.linalg.norm(user_vector)))

## trier les films en fonction de la similarite

df = df.sort_values(by='similarity', ascending=False)

##print(df['Series_Title'].tolist()[:10])

films_similaires_resumes = df['Series_Title'].tolist()[:10]


print('films similaires en fonctions du casting  : ',films_similaires_metadonnees)

print('les 10 films recommandes en fonction des synopsis : ',films_similaires_resumes)