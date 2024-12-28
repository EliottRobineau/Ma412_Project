# -*- coding: utf-8 -*-
"""

Ma412 lib

ABDELMALEK Enzo
ROBINEAU Eliott 
Sialelli Janelle

This file contains every functions used for the project.

"""

import matplotlib.pyplot as plt 
import re
import unicodedata
import os
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

global stop_words
global stopwords_2 

stop_words = set(stopwords.words('english'))

stopword_2 = ["present","place","take","taking","use","using","already","al","also","obtain","obtaining","obtained",
                      "suggested","suggestion","suggest","suggesting","report","make","made","makes","first","work",
                      "recent","evidence","discovery","known", "show", "shows", "showed", "significant", "results", "different",
                      "however", "despite"]
stop_words.update(stopword_2)

#%% Function that displays the N most frequent labels of the database

def Display_labels(labels,N):
    """
    Parameters
    ----------
    labels : Dict
        Dictionary that contains every unique label and their frequency in the abstracts
    N : Int
        Integer that indicates the number of most frequent labels we want to display

    Returns
    -------
    Display the N most frequent labels of the database on a histogram

    """
    most_words = list(labels.values())[0:N] # We recover the frequency of the N most labels in the database
    legend = list(labels.keys())[0:N] # We recover the labels associated with their frequency
    
    # We create a histogram to display the labels
    plt.bar(legend, most_words, color = 'skyblue', edgecolor='black')

    # Tuning of the histogram ; including title and labels for the axes
    plt.title('Histogram of most used labels')
    plt.xlabel('labels')
    plt.ylabel('Frequency')
    
    # Displaying the labels on the x-axis
    plt.xticks(rotation=45, ha='right')  # Rotation to avoid overlapping
    plt.grid()
    plt.ylim(min(most_words) - 15, max(most_words) + 5) # Adjust of the boundaries on the y-axis to have a better display
    
    # Display of the histogram
    plt.tight_layout()  # To avoid labels being cut off
    plt.show()


#%% Those functions allow us to normalize the texts before embedding them for the first model

def strip_accents(text):
    """
    Parameters
    ----------
    text : Str
        Text from which we want to remove the accents

    Returns
    -------
    Str
        Text without accents
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    """
    Parameters
    ----------
    text : Str
        Text we want to normalize for the MLPClassifier

    Returns
    -------
    detokenized_text : Str
        Norlalized text
    """
    # Removing the accents of the text 
    text = strip_accents(text.lower())
    text = re.sub('[_]+', ' ', text)
    text = re.sub('[^0-9a-zA-Z ]', '', text)
    
    # Tokenize the text into words (split by whitespace)
    words = text.split()

    # Filter out stopwords
    text_filtered = [word for word in words if word not in stop_words]
    
    # Detokenize (join the words back into a single string)
    detokenized_text = ' '.join(text_filtered)
    return detokenized_text

#%% Those functions allow us to normalize the texts before embedding them for the secon model

def format_labels(labels):
    """
    Parameters
    ----------
    labels : List
        List of labels taht we want to put in the adapted syntax for fasttext.train_supervised

    Returns
    -------
    TYPE
        String of labels with the adapted syntax for fasttetx.train_supervised

    """
    return " ".join([f"__label__{label}" for label in labels])

def prepare_fasttext_data(titles, abstracts, labels, output_file,N):
    """
    Parameters
    ----------
    titles : List
        List of titles of the articles
    abstracts : List
        List of abstracts of the articles
    labels : List
        List of labels of the articles
    output_file : Str
        Name of the output file that we want to create
    N : Int
        Number of articles considered for the training set

    Returns
    -------
    Creates the file.txt 

    """
    # Remove the file if it already exists, managing possible errors
    if os.path.exists(output_file):
        os.remove(output_file)
        
    if abstracts is None:
        abstracts = [""]*N
    if labels is None:
        labels = [""]*N
    if titles is None:
        titles = [""]*N
        
    with open(output_file, "w", encoding="utf-8") as f_out:
        for title, abstract, article_labels in zip(titles, abstracts, labels):
            # Normalization of the labels
            article_labels = format_labels(article_labels)
            
            # Combining the texts and writting them in the file
            combined_text = f"{article_labels} {title} {abstract}"
            f_out.write(combined_text + "\n")

def fasttext_preprocessed(output_file,input_file):
    """
    Parameters
    ----------
    output_file : Str
        String indicating the output file to write in
    input_file : Str
        String indicating the input file to fetch texts from

    Returns
    -------
    Puts the output file in the adapted syntaxt for fasttext.train_supervised

    """
    # Remove the file if it already exists, managing possible errors
    if os.path.exists(output_file):
        os.remove(output_file)
        
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            # Adding spaces around ponctuation characters 
            line = re.sub(r"([.\!?,'/()])", r" \1 ", line)
            # Lower_case the text
            line = line.lower()
            # Write the line in the output file
            f_out.write(line)
