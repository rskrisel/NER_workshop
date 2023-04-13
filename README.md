# Named Entity Recognition

In this workshop, we are going to learn how to transform large amounts of text into a database using Named Entity Recognition (NER). NER can computationally identify people, places, laws, events, dates, and other elements in a text or collection of texts.

## What is Named Entity Recognition?
*Explanation borrowed from Melanie Walsh's [Introduction to Cultural Analytics & Python](https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/12-Named-Entity-Recognition.html)*
</br>
</br>
Named Entity Recognition is a fundamental task in the field of natural language processing (NLP). NLP is an interdisciplinary field that blends linguistics, statistics, and computer science. The heart of NLP is to understand human language with statistics and computers. Applications of NLP are all around us. Have you ever heard of a little thing called spellcheck? How about autocomplete, Google translate, chat bots, or Siri? These are all examples of NLP in action!

Thanks to recent advances in machine learning and to increasing amounts of available text data on the web, NLP has grown by leaps and bounds in the last decade. NLP models that generate texts and images are now getting eerily good.

Open-source NLP tools are getting very good, too. We’re going to use one of these open-source tools, the Python library spaCy, for our Named Entity Recognition tasks in this lesson.

## What is spaCy?
In this workshop, we are using the spaCy library to run the NER. SpaCy relies on machine learning models that were trained on a large amount of carefully-labeled texts. These texts were, in fact, often labeled and corrected by hand. The English-language spaCy model that we’re going to use in this lesson was trained on an annotated corpus called “OntoNotes”: 2 million+ words drawn from “news, broadcast, talk shows, weblogs, usenet newsgroups, and conversational telephone speech,” which were meticulously tagged by a group of researchers and professionals for people’s names and places, for nouns and verbs, for subjects and objects, and much more. Like a lot of other major machine learning projects, OntoNotes was also sponsored by the Defense Advaced Research Projects Agency (DARPA), the branch of the Defense Department that develops technology for the U.S. military.

When spaCy identifies people and places in a text or collection of text, the NLP model is actually making predictions about the text based on what it has learned about how people and places function in English-language sentences.

### spaCy Named Entities
Below is a Named Entities chart for English-language spaCy taken from [its website](https://spacy.io/api/annotation#named-entities). This chart shows the different named entities that spaCy can identify as well as their corresponding type labels.

|Type Label|Description|
|:---:|:---:|
|PERSON|People, including fictional.|
|NORP|Nationalities or religious or political groups.|
|FAC|Buildings, airports, highways, bridges, etc.|
|ORG|Companies, agencies, institutions, etc.|
|GPE|Countries, cities, states.|
|LOC|Non-GPE locations, mountain ranges, bodies of water.|
|PRODUCT|Objects, vehicles, foods, etc. (Not services.)|
|EVENT|Named hurricanes, battles, wars, sports events, etc.|
|WORK_OF_ART|Titles of books, songs, etc.|
|LAW|Named documents made into laws.|
|LANGUAGE|Any named language.|
|DATE|Absolute or relative dates or periods.|
|TIME|Times smaller than a day.|
|PERCENT|Percentage, including ”%“.|
|MONEY|Monetary values, including unit.|
|QUANTITY|Measurements, as of weight or distance.|
|ORDINAL|“first”, “second”, etc.|
|CARDINAL|Numerals that do not fall under another type.|


### Install spaCy:


```python
# !pip install -U spacy
```

### Download the spaCy Language Model
Next we need to download the English-language model (en_core_web_sm), which will be processing and making predictions about our texts. This is the model that was trained on the annotated “OntoNotes” corpus. You can download the en_core_web_sm model by running the cell below:


```python
# !python -m spacy download en_core_web_sm
```

*Note: spaCy offers models for other languages including Chinese, German, French, Spanish, Portuguese, Russian, Italian, Dutch, Greek, Norwegian, and Lithuanian.*

*spaCy offers language and tokenization support for other language via external dependencies — such as PyviKonlpy for Korean*

## Import all relevant libraries for collecting data and processing the NER

We will import:
- Spacy and displacy to run the NER and visualize our results
- en_core_web_sm to import the spaCy language model
- Pandas library for organizing and displaying data (we’re also changing the pandas default max row and column width display setting)
- Glob and pathlib to connect to folders on our operating system
- Requests to get data from an API and also to web scrape
- PPrint to make our JSON results readable
- Beautiful Soup to make our HTML results readable



```python
import spacy
from spacy import displacy
import en_core_web_sm
import pandas as pd
pd.options.display.max_rows = 600
pd.options.display.max_colwidth = 400
import glob
from pathlib import Path
import requests
import pprint
from bs4 import BeautifulSoup
```

## Load the spaCy language model


```python
nlp = en_core_web_sm.load()
```

## Collect your Data: Combining APIs and Web Scraping

In this workshop, we are going to collect data from news articles in two ways. First, by using connect to the NewsAPI and gathering a collection of URLs related to a specific news topic. Next, by web scraping those URLs to save the articles as text files. For detailed instructions on working with the NewsAPI, please refer to this ["Working with APIs" tutorial](https://gist.github.com/rskrisel/4ff9629df9f9d6bf5a638b8ba6c13a68) and for detailed instructions on how to web scrape a list of URLs please refer to the ["Web Scraping Media URLs in Python"](https://github.com/rskrisel/web_scraping_workshop) tutorial. 

### Install the News API


```python
# !pip install newsapi-python
```

### Store your secret key


```python
secret= '123456789'
```

### Define your endpoint


```python
url = 'https://newsapi.org/v2/everything?'
```

### Define your query parameters


```python
parameters = {
    'q': 'Ukraine',
    'searchIn':'title',
    'pageSize': 20, 
    'language' : 'en',
    'apiKey': secret 
    }
```

### Make your data request


```python
response = requests.get(url, params=parameters)
```

### Visualize your JSON results


```python
response_json = response.json()
pprint.pprint(response_json)
```

### Check what keys exist in your JSON data


```python
response_json.keys()
```

### See the data stored in each key


```python
print(response_json['status'])
print(response_json['totalResults'])
print(response_json['articles'])
```

### Check the datatype for each key


```python
print(type(response_json['status']))
print(type(response_json['totalResults']))
print(type(response_json['articles']))
```

### Make sure the list reads as a dictionary


```python
type(response_json['articles'][0])
```

### Convert the JSON key into a Pandas Dataframe


```python
df_articles = pd.DataFrame(response_json['articles'])
df_articles
```

### Define a function to web scrape text from the list of URLs in the Dataframe


```python
def scrape_article(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    html_string = response.text
    return html_string
```

### Apply the function to the Dataframe and store the results in a new column


```python
df_articles['scraped_text'] = df_articles['url'].apply(scrape_article)
```


```python
df_articles
```

### Use the Beautiful Soup library to make the scraped html text legible and save each article in a text file
*Note: make sure you create a folder named "files" before running this step*


```python
id = 0
for text in df_articles['scraped_text']:
    soup = BeautifulSoup(text)
    article = soup.get_text()
    
    id += 1
    with open(f"files/{id}.txt", "w") as file:
        file.write(str(article))
```

### Use glob to connect to the file directory where your articles are saved and store it in a variable


```python
directory = "files"
articles = glob.glob(f"{directory}/*.txt")
```

### Make sure you have data stores in your files variable


```python
articles
```

### Let's run the NER on a single article first


```python
filepath = "files/1.txt"
text = open(filepath, encoding='utf-8').read()
doc = nlp(text)
```

### Let's use displacy to visualize our results


```python
displacy.render(doc, style="ent")
```

### Let's see a list of the identified entities


```python
doc.ents
```

### Let's add the entity label next to each entity: 


```python
for named_entity in doc.ents:
    print(named_entity, named_entity.label_)
```

### Let's filter the results to see all entities labelled as "PERSON":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "PERSON":
        print(named_entity)
```

### Let's filter the results to see all entities labelled as "NORP":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "NORP":
        print(named_entity)
```

### Let's filter the results to see all entities labelled as "GPE":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "GPE":
        print(named_entity)
```

### Let's filter the results to see all entities labelled as "LOC":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "LOC":
        print(named_entity)
```

### Let's filter the results to see all entities labelled as "FAC":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "FAC":
        print(named_entity)
```

### Let's filter the results to see all entities labelled as "ORG":


```python
for named_entity in doc.ents:
    if named_entity.label_ == "ORG":
        print(named_entity)
```

### Let's define a function that will entify all the entities in our document and save the output as a dictionary:


```python
entities=[]
entity_type = [] 
entity_identified = []
for named_entity in doc.ents:
    entity_type.append(named_entity.label_)
    entity_identified.append(named_entity.text)
    entity_dict = {'Entity_type': entity_type, 'Entity_identified': entity_identified}
    entities.append(entity_dict)
print(entities)
```

### Let's build on this function to run this process across our entire collection of texts:


```python
all_entities = []
for filepath in articles:
    text = open(filepath, encoding='utf-8').read()
    doc = nlp(text)
    entity_type = [] 
    for ent in doc.ents:
        entity_type.append(ent.label_)
    entity_identified = [] 
    for ent in doc.ents:
        entity_identified.append(ent.text)
    ent_dict = {'File_name': filepath, 'Entity_type': entity_type, 'Entity_identified': entity_identified}
    all_entities.append(ent_dict)
print(all_entities)
```

### Let's visualize our results in a Pandas Dataframe sorted by the file name


```python
df_NER = pd.DataFrame(all_entities)
df_NER = df_NER.sort_values(by='File_name', ascending=True)
df_NER 
```

### Let's explode our Dataframe so we have just one entity value per row pegged to the file name


```python
df_NER = df_NER.set_index(['File_name'])
df_NER = df_NER.apply(pd.Series.explode).reset_index()
df_NER[:25]
```

### Let's filter our results by GPE


```python
df_NER[df_NER['Entity_type'] == 'GPE'][:15]
```

### Let's filter our results by LAW


```python
df_NER[df_NER['Entity_type'] == 'LAW'][:15]
```

### Let's filter our results by Money


```python
df_NER[df_NER['Entity_type'] == 'MONEY'][:15]
```


