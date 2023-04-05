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
from collections import Counter
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
secret= '571e874fe6674690a5ea658e5937d47c'
```

### Define your endpoint


```python
url = 'https://newsapi.org/v2/everything?'
```

### Define your query parameters


```python
parameters = {
    'q': 'CHIPS Act', 
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

    {'articles': [{'author': 'Igor Bonifacic',
                   'content': 'Chipmakers hoping to tap into the Biden '
                              'administrations $39 billion semiconductor '
                              'manufacturing subsidy program will need to sign '
                              'agreements promising they wont expand production '
                              'capacity in China. T… [+1800 chars]',
                   'description': 'Chipmakers hoping to tap into the Biden '
                                  'administration’s $39 billion semiconductor '
                                  'manufacturing subsidy program will need to sign '
                                  'agreements promising they won’t expand '
                                  'production capacity in China. The requirement '
                                  'was among a handful of funding conditions t…',
                   'publishedAt': '2023-03-04T17:26:37Z',
                   'source': {'id': 'engadget', 'name': 'Engadget'},
                   'title': 'Biden administration bars CHIPS Act funding '
                            'recipients from expanding in China',
                   'url': 'https://www.engadget.com/biden-administration-bars-chips-act-funding-recipients-from-expanding-in-china-172637590.html',
                   'urlToImage': 'https://s.yimg.com/uu/api/res/1.2/O0kRn2KWxWw2Gij4lWICow--~B/Zmk9ZmlsbDtoPTYzMDtweW9mZj0wO3c9MTIwMDthcHBpZD15dGFjaHlvbg--/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2023-02/f7fa4d60-bab0-11ed-afae-99a6c04fc149.cf.jpg'},
                  {'author': None,
                   'content': 'We use cookies and data to<ul><li>Deliver and '
                              'maintain Google services</li><li>Track outages and '
                              'protect against spam, fraud, and '
                              'abuse</li><li>Measure audience engagement and site '
                              'statistics to unde… [+1131 chars]',
                   'description': "S. Korea's trade minister to raise concerns "
                                  'over Chips Act in US ...\xa0\xa0Reuters',
                   'publishedAt': '2023-03-08T03:55:00Z',
                   'source': {'id': 'google-news', 'name': 'Google News'},
                   'title': "S. Korea's trade minister to raise concerns over "
                            'Chips Act in US ... - Reuters',
                   'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMicGh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3RlY2hub2xvZ3kvcy1rb3JlYXMtdHJhZGUtbWluaXN0ZXItcmFpc2UtY29uY2VybnMtb3Zlci1jaGlwcy1hY3QtdXMtbWVldGluZ3MtMjAyMy0wMy0wOC_SAQA?oc%3D5&gl=FR&hl=en-US&pc=n&src=1',
                   'urlToImage': None},
                  {'author': 'https://www.facebook.com/bbcnews',
                   'content': 'The Dutch government says before the summer it will '
                              'put restrictions on the country\'s "most advanced" '
                              'chip exports to protect its national security, '
                              'following a similar move by the US.\r\n'
                              'It will inclu… [+2395 chars]',
                   'description': 'The measures will affect Dutch firm ASML, which '
                                  'is a key part of the global microchip supply '
                                  'chain.',
                   'publishedAt': '2023-03-09T03:46:28Z',
                   'source': {'id': 'bbc-news', 'name': 'BBC News'},
                   'title': 'US-China chip war: Netherlands moves to restrict some '
                            'exports',
                   'url': 'https://www.bbc.co.uk/news/business-64897794',
                   'urlToImage': 'https://ichef.bbci.co.uk/news/1024/branded_news/898D/production/_128931253_gettyimages-1354885833.png'},
                  {'author': 'EditorDavid',
                   'content': 'The final season of Star Trek: Picardfeatures the '
                              'return of the Klingon Worf, reports Polygon, '
                              'calling it "the chance to give one of sci-fi\'s '
                              "most beloved supporting characters something that's "
                              'usual… [+3499 chars]',
                   'description': 'The final season of Star Trek: Picard features '
                                  'the return of the Klingon Worf, reports '
                                  'Polygon, calling it "the chance to give one of '
                                  "sci-fi's most beloved supporting characters "
                                  "something that's usually reserved only for "
                                  'Captains and Admirals: a glorious thir…',
                   'publishedAt': '2023-03-05T09:04:00Z',
                   'source': {'id': None, 'name': 'Slashdot.org'},
                   'title': "Worf's Final Act: a 'Star Trek' Legend Looks Back",
                   'url': 'https://entertainment.slashdot.org/story/23/03/05/0313230/worfs-final-act-a-star-trek-legend-looks-back',
                   'urlToImage': 'https://a.fsdn.com/sd/topics/tv_64.png'},
                  {'author': 'Andrea Hsu',
                   'content': "President Biden's ambitious proposals to address "
                              'the high cost and short supply of child care '
                              "haven't garnered enough support in Congress, so now "
                              'his administration has come up with a '
                              'workaround.\r\n'
                              'Ge… [+6055 chars]',
                   'description': 'The administration is turning to semiconductors '
                                  'in the hopes of expanding affordable child '
                                  'care.',
                   'publishedAt': '2023-03-17T09:00:36Z',
                   'source': {'id': None, 'name': 'NPR'},
                   'title': 'Biden has big ideas for fixing child care. For now a '
                            'small workaround will have to do',
                   'url': 'https://www.npr.org/2023/03/17/1162869162/child-care-chips-semiconductors-manufacturing-raimondo-subsidies',
                   'urlToImage': 'https://media.npr.org/assets/img/2023/03/16/gettyimages-161145124_wide-1f062d47e8a8d91d840a9c29aba33d5c52586cde-s1400-c100.jpg'},
                  {'author': 'Laura Dobberstein',
                   'content': 'The US Commerce Department proposed rules on '
                              'Tuesday that would limit the amount of CHIPS Act '
                              'recipients can invest to expand semiconductor '
                              'manufacturing in countries the US considers '
                              'adversarial.\r\n'
                              'T… [+2712 chars]',
                   'description': 'Allows a little expansion of output from '
                                  'existing Middle Kingdom facilities\n'
                                  'The US Commerce Department proposed rules on '
                                  'Tuesday that would limit the amount of CHIPS '
                                  'Act recipients can invest to expand '
                                  'semiconductor manufacturing in countries the US '
                                  'considers…',
                   'publishedAt': '2023-03-22T06:32:06Z',
                   'source': {'id': None, 'name': 'Theregister.com'},
                   'title': 'US details CHIPS Act rules that give China and South '
                            'Korea some comfort',
                   'url': 'https://www.theregister.com/2023/03/22/chips_act_rules_floated/',
                   'urlToImage': 'https://regmedia.co.uk/2023/03/22/shutterstock_us_korea_taiwan_china_motherboard.jpg'},
                  {'author': 'Brendan Bordelon and Caitlin Oprysko',
                   'content': 'But the new law has also attracted an armada of '
                              'lobbyists repping a wide range of industries. Some '
                              'may be making long-shot bids lobbyists for Snap, '
                              'for example, plan to ask Washington to subsidize '
                              'pl… [+11802 chars]',
                   'description': 'Lobbyists are descending on Washington, all '
                                  'hungry for a piece of the CHIPS and Science Act '
                                  '— even if some bids have little chance of '
                                  'making the cut.',
                   'publishedAt': '2023-03-17T08:30:00Z',
                   'source': {'id': 'politico', 'name': 'Politico'},
                   'title': 'Everybody in Washington wants a byte of the CHIPS law',
                   'url': 'https://www.politico.com/news/2023/03/17/chips-law-companies-washington-lobbying-00086687',
                   'urlToImage': 'https://static.politico.com/a5/e9/ed7ea593463aac5b690046668670/https-delivery-gettyimages.com/downloads/586113576'},
                  {'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas '
                             'Rossolillo, and Billy Duberstein)',
                   'content': 'Suzanne Frey, an executive at Alphabet, is a member '
                              'of The Motley Fool’s board of directors. John '
                              'Mackey, former CEO of Whole Foods Market, an Amazon '
                              'subsidiary, is a member of The Motley Fool’s boar… '
                              '[+1048 chars]',
                   'description': 'More regulations and requirements are being '
                                  'released for the CHIPS Act that semiconductor '
                                  'investors might want to know about.',
                   'publishedAt': '2023-03-24T20:01:00Z',
                   'source': {'id': None, 'name': 'Motley Fool'},
                   'title': 'U.S. CHIPS Act Recipients Face Expansion Hurdles in '
                            'China',
                   'url': 'https://www.fool.com/investing/2023/03/24/us-chips-act-recipients-face-expansion-hurdles-in/',
                   'urlToImage': 'https://g.foolcdn.com/editorial/images/725864/copy-of-jose-najarro-2023-03-24t153421346.png'},
                  {'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas '
                             'Rossolillo, and Billy Duberstein)',
                   'content': 'Check out this short video to learn what '
                              'semiconductor investors Jose Najarro, Nicholas '
                              'Rossolillo, and Billy Duberstein had to say about '
                              'the CHIPS Act and how companies like Intel\xa0(INTC '
                              '0.37%) can b… [+268 chars]',
                   'description': 'Applications for the CHIPS Act are now open. It '
                                  'comes with some rules that semiconductor '
                                  'companies must follow, from profit sharing to '
                                  'buyback and international expansion '
                                  'regulations.',
                   'publishedAt': '2023-03-08T14:56:40Z',
                   'source': {'id': None, 'name': 'Motley Fool'},
                   'title': 'The CHIPS Act Is Open for Business -- Which Stocks '
                            'Will Benefit Most?',
                   'url': 'https://www.fool.com/investing/2023/03/08/the-chips-act-is-open-for-business-which-stocks-wi/',
                   'urlToImage': 'https://g.foolcdn.com/editorial/images/723864/copy-of-jose-najarro-2023-03-08t090543667.png'},
                  {'author': 'cbsnews.com',
                   'content': 'You already know that there are computer chips in '
                              'your computer and your phone. But you may not '
                              'realize just how many other things in your life '
                              "rely on chips. They're also in your clocks, toys, "
                              'therm… [+297 chars]',
                   'description': 'You already know that there are computer chips '
                                  'in your computer and your phone. But you may '
                                  'not realize just how many other things in your '
                                  "life rely on chips. They're also in your "
                                  'clocks, toys, thermostats, and every single '
                                  'thing in your kitchen. "Our demand …',
                   'publishedAt': '2023-03-05T14:24:04Z',
                   'source': {'id': None, 'name': 'Biztoc.com'},
                   'title': "The CHIPS Act: Rebuilding America's technological "
                            'infrastructure',
                   'url': 'https://biztoc.com/x/582cdb2369a83030',
                   'urlToImage': 'https://c.biztoc.com/p/582cdb2369a83030/og.webp'},
                  {'author': 'newsfeedback@fool.com (Nicholas Rossolillo)',
                   'content': 'The bear market of 2022 bludgeoned technology '
                              'stocks, but it also brought an end to the chip '
                              'shortage for many markets -- at exactly the wrong '
                              'time for investors. Supply of high-end chips for '
                              'smartph… [+5097 chars]',
                   'description': 'Mature chip manufacturing could get a big boost '
                                  'from government-led investment in the coming '
                                  'years.',
                   'publishedAt': '2023-03-23T12:45:00Z',
                   'source': {'id': None, 'name': 'Motley Fool'},
                   'title': 'The CHIPS Act Is Accepting Applications -- Does That '
                            'Make GlobalFoundries Stock a Buy?',
                   'url': 'https://www.fool.com/investing/2023/03/23/the-chips-act-is-accepting-applications-does-that/',
                   'urlToImage': 'https://g.foolcdn.com/editorial/images/725359/semiconductor-technician-with-wafer-in-manufacturing-plant.jpg'},
                  {'author': None,
                   'content': 'Roboticists have been using a technique similar to '
                              'the ancient art of paper folding to develop '
                              'autonomous machines out of thin, flexible sheets. '
                              'These lightweight robots are simpler and cheaper to '
                              'ma… [+4224 chars]',
                   'description': 'A multidisciplinary team has created a new '
                                  'fabrication technique for fully foldable robots '
                                  'that can perform a variety of complex tasks '
                                  'without relying on semiconductors.',
                   'publishedAt': '2023-04-03T20:26:32Z',
                   'source': {'id': None, 'name': 'Science Daily'},
                   'title': 'Origami-inspired robots can sense, analyze and act in '
                            'challenging environments',
                   'url': 'https://www.sciencedaily.com/releases/2023/04/230403162632.htm',
                   'urlToImage': 'https://www.sciencedaily.com/images/scidaily-icon.png'},
                  {'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas '
                             'Rossolillo, and Billy Duberstein)',
                   'content': 'While numerous U.S.-based chip design companies '
                              'exist, most of the chip manufacturing is done by '
                              'international players. Unfortunately, some of these '
                              'global players believe requirements for the CHIPS … '
                              '[+340 chars]',
                   'description': 'The CHIPS Act is meant to improve semiconductor '
                                  'manufacturing and design in the U.S. Still, the '
                                  'requirements for these subsidies might be too '
                                  'much for certain international semiconductor '
                                  'companies.',
                   'publishedAt': '2023-03-13T10:16:00Z',
                   'source': {'id': None, 'name': 'Motley Fool'},
                   'title': "The CHIPS Act's International Hurdles -- What It "
                            'Means for Semiconductor Investors',
                   'url': 'https://www.fool.com/investing/2023/03/13/the-chips-acts-international-hurdles-what-it-means/',
                   'urlToImage': 'https://g.foolcdn.com/editorial/images/724405/copy-of-jose-najarro-2023-03-12t201709621.png'},
                  {'author': 'Reuters',
                   'content': 'MEXICO CITY Tesla could begin producing its first '
                              'cars in Mexico next year, with the electric vehicle '
                              'maker close to receiving its final permits allowing '
                              'factory construction to begin in Nuevo Leon n… '
                              '[+1628 chars]',
                   'description': 'Filed under:\n'
                                  ' Green,Plants/Manufacturing,Tesla,Electric\n'
                                  ' Continue reading Tesla could begin producing '
                                  'electric cars in Mexico next year\n'
                                  'Tesla could begin producing electric cars in '
                                  'Mexico next year originally appeared on '
                                  'Autoblog on Tue, 7 Mar 2023 08:43:00 E…',
                   'publishedAt': '2023-03-07T13:43:00Z',
                   'source': {'id': None, 'name': 'Autoblog'},
                   'title': 'Tesla could begin producing electric cars in Mexico '
                            'next year',
                   'url': 'https://www.autoblog.com/2023/03/07/tesla-mexico-production-2024/',
                   'urlToImage': 'https://o.aolcdn.com/images/dims3/GLOB/crop/4368x2457+0+0/resize/800x450!/format/jpg/quality/85/https://s.yimg.com/os/creatr-uploaded-images/2023-02/ec46bca0-bc14-11ed-9c5d-895f539777f5'},
                  {'author': None,
                   'content': 'We use cookies and data to<ul><li>Deliver and '
                              'maintain Google services</li><li>Track outages and '
                              'protect against spam, fraud, and '
                              'abuse</li><li>Measure audience engagement and site '
                              'statistics to unde… [+1131 chars]',
                   'description': 'CHIPS Act: Biden increases government to '
                                  "increase control over everyone's "
                                  'business\xa0\xa0Washington Examiner',
                   'publishedAt': '2023-03-07T23:38:00Z',
                   'source': {'id': 'google-news', 'name': 'Google News'},
                   'title': 'CHIPS Act: Biden increases government to increase '
                            "control over everyone's business - Washington "
                            'Examiner',
                   'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMie2h0dHBzOi8vd3d3Lndhc2hpbmd0b25leGFtaW5lci5jb20vb3Bpbmlvbi9jaGlwcy1hY3QtYmlkZW4taW5jcmVhc2VzLWdvdmVybm1lbnQtdG8taW5jcmVhc2UtY29udHJvbC1vdmVyLWV2ZXJ5b25lcy1idXNpbmVzc9IBAA?oc%3D5%26hl%3Den-CA%26gl%3DCA%26ceid%3DCA:en&gl=FR&hl=en-CA&pc=n&src=1',
                   'urlToImage': None},
                  {'author': 'Amy Chew',
                   'content': 'Kuala Lumpur, Malaysia Malaysias Anwar Ibrahim is '
                              'expected to navigate between deepening economic '
                              'ties with his countrys biggest trading partner and '
                              'tackling thorny issues such as the South China Sea… '
                              '[+6866 chars]',
                   'description': 'Malaysian leader expected to navigate between '
                                  'deepening economic ties and tackling '
                                  'differences in talks with Xi Jinping.',
                   'publishedAt': '2023-03-30T01:09:48Z',
                   'source': {'id': 'al-jazeera-english',
                              'name': 'Al Jazeera English'},
                   'title': 'Malaysia’s Anwar faces balancing act on first China '
                            'trip',
                   'url': 'https://www.aljazeera.com/economy/2023/3/30/malaysias-anwar-faces-balancing-act-on-first-china-trip',
                   'urlToImage': 'https://www.aljazeera.com/wp-content/uploads/2023/03/AP23061118140422-1.jpg?resize=1920%2C1440'},
                  {'author': 'Digg Editors',
                   'content': 'Officials in Washington have been sounding the '
                              'security alarm about TikTok for years, but now the '
                              'United States is closer than ever to blocking the '
                              'social app from operating within its borders. '
                              'TikTo… [+8017 chars]',
                   'description': 'With the RESTRICT Act, Sen. Warner wants to '
                                  "reset the way the US treats China's tech "
                                  'industry.',
                   'publishedAt': '2023-03-29T19:18:13Z',
                   'source': {'id': None, 'name': 'Restofworld.org'},
                   'title': 'US Senator Mark Warner On TikTok, China And The End '
                            'Of Techno-Optimism',
                   'url': 'https://restofworld.org/2023/mark-warner-interview-restrict-act/',
                   'urlToImage': 'https://149346090.v2.pressablecdn.com/wp-content/uploads/2023/03/cropped-230329_GK_MarkWarner057-copy.jpg'},
                  {'author': 'fortune.com',
                   'content': 'Aaron Nichols walked past rows of kale growing on '
                              'his farm, his knee-high brown rubber boots speckled '
                              'with some of the richest soil on earth, and gazed '
                              'with concern toward fields in the distance. Jus… '
                              '[+306 chars]',
                   'description': 'Aaron Nichols walked past rows of kale growing '
                                  'on his farm, his knee-high brown rubber boots '
                                  'speckled with some of the richest soil on '
                                  'earth, and gazed with concern toward fields in '
                                  'the distance. Just over the horizon loomed a '
                                  'gigantic building of the semicon…',
                   'publishedAt': '2023-03-26T14:36:04Z',
                   'source': {'id': None, 'name': 'Biztoc.com'},
                   'title': 'Chance to host semiconductor factories under CHIPS '
                            'Act has Oregon reconsidering rules against urban '
                            'sprawl',
                   'url': 'https://biztoc.com/x/3511e057c4d9f5c2',
                   'urlToImage': 'https://c.biztoc.com/p/3511e057c4d9f5c2/og.webp'},
                  {'author': 'Elliot Ackerman',
                   'content': 'Lockheed Martin builds its advanced mobile rocket '
                              'launchers in a converted diaper factory, of all '
                              'places. When I visited the plant in southern '
                              'Arkansas at the end of February, I found it humming '
                              'with… [+9948 chars]',
                   'description': 'But after decades of decay, will it happen '
                                  'quickly enough to save Ukraine?',
                   'publishedAt': '2023-03-09T18:04:56Z',
                   'source': {'id': None, 'name': 'The Atlantic'},
                   'title': 'The Arsenal of Democracy Is Reopening for Business',
                   'url': 'https://www.theatlantic.com/ideas/archive/2023/03/american-defense-manufacturing-ukraine-aid-arkansas/673327/?utm_source=feed',
                   'urlToImage': None},
                  {'author': None,
                   'content': 'We use cookies and data to<ul><li>Deliver and '
                              'maintain Google services</li><li>Track outages and '
                              'protect against spam, fraud, and '
                              'abuse</li><li>Measure audience engagement and site '
                              'statistics to unde… [+1131 chars]',
                   'description': "The CHIPS Act: Rebuilding America's "
                                  'technological infrastructure\xa0\xa0CBS News',
                   'publishedAt': '2023-03-05T14:11:07Z',
                   'source': {'id': 'google-news', 'name': 'Google News'},
                   'title': "The CHIPS Act: Rebuilding America's technological "
                            'infrastructure - CBS News',
                   'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMiXGh0dHBzOi8vd3d3LmNic25ld3MuY29tL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv0gFgaHR0cHM6Ly93d3cuY2JzbmV3cy5jb20vYW1wL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv?oc%3D5&gl=FR&hl=en-US&pc=n&src=1',
                   'urlToImage': None}],
     'status': 'ok',
     'totalResults': 993}


### Check what keys exist in your JSON data


```python
response_json.keys()
```




    dict_keys(['status', 'totalResults', 'articles'])



### See the data stored in each key


```python
print(response_json['status'])
print(response_json['totalResults'])
print(response_json['articles'])
```

    ok
    993
    [{'source': {'id': 'engadget', 'name': 'Engadget'}, 'author': 'Igor Bonifacic', 'title': 'Biden administration bars CHIPS Act funding recipients from expanding in China', 'description': 'Chipmakers hoping to tap into the Biden administration’s $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they won’t expand production capacity in China. The requirement was among a handful of funding conditions t…', 'url': 'https://www.engadget.com/biden-administration-bars-chips-act-funding-recipients-from-expanding-in-china-172637590.html', 'urlToImage': 'https://s.yimg.com/uu/api/res/1.2/O0kRn2KWxWw2Gij4lWICow--~B/Zmk9ZmlsbDtoPTYzMDtweW9mZj0wO3c9MTIwMDthcHBpZD15dGFjaHlvbg--/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2023-02/f7fa4d60-bab0-11ed-afae-99a6c04fc149.cf.jpg', 'publishedAt': '2023-03-04T17:26:37Z', 'content': 'Chipmakers hoping to tap into the Biden administrations $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they wont expand production capacity in China. T… [+1800 chars]'}, {'source': {'id': 'google-news', 'name': 'Google News'}, 'author': None, 'title': "S. Korea's trade minister to raise concerns over Chips Act in US ... - Reuters", 'description': "S. Korea's trade minister to raise concerns over Chips Act in US ...\xa0\xa0Reuters", 'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMicGh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3RlY2hub2xvZ3kvcy1rb3JlYXMtdHJhZGUtbWluaXN0ZXItcmFpc2UtY29uY2VybnMtb3Zlci1jaGlwcy1hY3QtdXMtbWVldGluZ3MtMjAyMy0wMy0wOC_SAQA?oc%3D5&gl=FR&hl=en-US&pc=n&src=1', 'urlToImage': None, 'publishedAt': '2023-03-08T03:55:00Z', 'content': 'We use cookies and data to<ul><li>Deliver and maintain Google services</li><li>Track outages and protect against spam, fraud, and abuse</li><li>Measure audience engagement and site statistics to unde… [+1131 chars]'}, {'source': {'id': 'bbc-news', 'name': 'BBC News'}, 'author': 'https://www.facebook.com/bbcnews', 'title': 'US-China chip war: Netherlands moves to restrict some exports', 'description': 'The measures will affect Dutch firm ASML, which is a key part of the global microchip supply chain.', 'url': 'https://www.bbc.co.uk/news/business-64897794', 'urlToImage': 'https://ichef.bbci.co.uk/news/1024/branded_news/898D/production/_128931253_gettyimages-1354885833.png', 'publishedAt': '2023-03-09T03:46:28Z', 'content': 'The Dutch government says before the summer it will put restrictions on the country\'s "most advanced" chip exports to protect its national security, following a similar move by the US.\r\nIt will inclu… [+2395 chars]'}, {'source': {'id': None, 'name': 'Slashdot.org'}, 'author': 'EditorDavid', 'title': "Worf's Final Act: a 'Star Trek' Legend Looks Back", 'description': 'The final season of Star Trek: Picard features the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi\'s most beloved supporting characters something that\'s usually reserved only for Captains and Admirals: a glorious thir…', 'url': 'https://entertainment.slashdot.org/story/23/03/05/0313230/worfs-final-act-a-star-trek-legend-looks-back', 'urlToImage': 'https://a.fsdn.com/sd/topics/tv_64.png', 'publishedAt': '2023-03-05T09:04:00Z', 'content': 'The final season of Star Trek: Picardfeatures the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi\'s most beloved supporting characters something that\'s usual… [+3499 chars]'}, {'source': {'id': None, 'name': 'NPR'}, 'author': 'Andrea Hsu', 'title': 'Biden has big ideas for fixing child care. For now a small workaround will have to do', 'description': 'The administration is turning to semiconductors in the hopes of expanding affordable child care.', 'url': 'https://www.npr.org/2023/03/17/1162869162/child-care-chips-semiconductors-manufacturing-raimondo-subsidies', 'urlToImage': 'https://media.npr.org/assets/img/2023/03/16/gettyimages-161145124_wide-1f062d47e8a8d91d840a9c29aba33d5c52586cde-s1400-c100.jpg', 'publishedAt': '2023-03-17T09:00:36Z', 'content': "President Biden's ambitious proposals to address the high cost and short supply of child care haven't garnered enough support in Congress, so now his administration has come up with a workaround.\r\nGe… [+6055 chars]"}, {'source': {'id': None, 'name': 'Theregister.com'}, 'author': 'Laura Dobberstein', 'title': 'US details CHIPS Act rules that give China and South Korea some comfort', 'description': 'Allows a little expansion of output from existing Middle Kingdom facilities\nThe US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers…', 'url': 'https://www.theregister.com/2023/03/22/chips_act_rules_floated/', 'urlToImage': 'https://regmedia.co.uk/2023/03/22/shutterstock_us_korea_taiwan_china_motherboard.jpg', 'publishedAt': '2023-03-22T06:32:06Z', 'content': 'The US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers adversarial.\r\nT… [+2712 chars]'}, {'source': {'id': 'politico', 'name': 'Politico'}, 'author': 'Brendan Bordelon and Caitlin Oprysko', 'title': 'Everybody in Washington wants a byte of the CHIPS law', 'description': 'Lobbyists are descending on Washington, all hungry for a piece of the CHIPS and Science Act — even if some bids have little chance of making the cut.', 'url': 'https://www.politico.com/news/2023/03/17/chips-law-companies-washington-lobbying-00086687', 'urlToImage': 'https://static.politico.com/a5/e9/ed7ea593463aac5b690046668670/https-delivery-gettyimages.com/downloads/586113576', 'publishedAt': '2023-03-17T08:30:00Z', 'content': 'But the new law has also attracted an armada of lobbyists repping a wide range of industries. Some may be making long-shot bids lobbyists for Snap, for example, plan to ask Washington to subsidize pl… [+11802 chars]'}, {'source': {'id': None, 'name': 'Motley Fool'}, 'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)', 'title': 'U.S. CHIPS Act Recipients Face Expansion Hurdles in China', 'description': 'More regulations and requirements are being released for the CHIPS Act that semiconductor investors might want to know about.', 'url': 'https://www.fool.com/investing/2023/03/24/us-chips-act-recipients-face-expansion-hurdles-in/', 'urlToImage': 'https://g.foolcdn.com/editorial/images/725864/copy-of-jose-najarro-2023-03-24t153421346.png', 'publishedAt': '2023-03-24T20:01:00Z', 'content': 'Suzanne Frey, an executive at Alphabet, is a member of The Motley Fool’s board of directors. John Mackey, former CEO of Whole Foods Market, an Amazon subsidiary, is a member of The Motley Fool’s boar… [+1048 chars]'}, {'source': {'id': None, 'name': 'Motley Fool'}, 'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)', 'title': 'The CHIPS Act Is Open for Business -- Which Stocks Will Benefit Most?', 'description': 'Applications for the CHIPS Act are now open. It comes with some rules that semiconductor companies must follow, from profit sharing to buyback and international expansion regulations.', 'url': 'https://www.fool.com/investing/2023/03/08/the-chips-act-is-open-for-business-which-stocks-wi/', 'urlToImage': 'https://g.foolcdn.com/editorial/images/723864/copy-of-jose-najarro-2023-03-08t090543667.png', 'publishedAt': '2023-03-08T14:56:40Z', 'content': 'Check out this short video to learn what semiconductor investors Jose Najarro, Nicholas Rossolillo, and Billy Duberstein had to say about the CHIPS Act and how companies like Intel\xa0(INTC 0.37%) can b… [+268 chars]'}, {'source': {'id': None, 'name': 'Biztoc.com'}, 'author': 'cbsnews.com', 'title': "The CHIPS Act: Rebuilding America's technological infrastructure", 'description': 'You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They\'re also in your clocks, toys, thermostats, and every single thing in your kitchen. "Our demand …', 'url': 'https://biztoc.com/x/582cdb2369a83030', 'urlToImage': 'https://c.biztoc.com/p/582cdb2369a83030/og.webp', 'publishedAt': '2023-03-05T14:24:04Z', 'content': "You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They're also in your clocks, toys, therm… [+297 chars]"}, {'source': {'id': None, 'name': 'Motley Fool'}, 'author': 'newsfeedback@fool.com (Nicholas Rossolillo)', 'title': 'The CHIPS Act Is Accepting Applications -- Does That Make GlobalFoundries Stock a Buy?', 'description': 'Mature chip manufacturing could get a big boost from government-led investment in the coming years.', 'url': 'https://www.fool.com/investing/2023/03/23/the-chips-act-is-accepting-applications-does-that/', 'urlToImage': 'https://g.foolcdn.com/editorial/images/725359/semiconductor-technician-with-wafer-in-manufacturing-plant.jpg', 'publishedAt': '2023-03-23T12:45:00Z', 'content': 'The bear market of 2022 bludgeoned technology stocks, but it also brought an end to the chip shortage for many markets -- at exactly the wrong time for investors. Supply of high-end chips for smartph… [+5097 chars]'}, {'source': {'id': None, 'name': 'Science Daily'}, 'author': None, 'title': 'Origami-inspired robots can sense, analyze and act in challenging environments', 'description': 'A multidisciplinary team has created a new fabrication technique for fully foldable robots that can perform a variety of complex tasks without relying on semiconductors.', 'url': 'https://www.sciencedaily.com/releases/2023/04/230403162632.htm', 'urlToImage': 'https://www.sciencedaily.com/images/scidaily-icon.png', 'publishedAt': '2023-04-03T20:26:32Z', 'content': 'Roboticists have been using a technique similar to the ancient art of paper folding to develop autonomous machines out of thin, flexible sheets. These lightweight robots are simpler and cheaper to ma… [+4224 chars]'}, {'source': {'id': None, 'name': 'Motley Fool'}, 'author': 'newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)', 'title': "The CHIPS Act's International Hurdles -- What It Means for Semiconductor Investors", 'description': 'The CHIPS Act is meant to improve semiconductor manufacturing and design in the U.S. Still, the requirements for these subsidies might be too much for certain international semiconductor companies.', 'url': 'https://www.fool.com/investing/2023/03/13/the-chips-acts-international-hurdles-what-it-means/', 'urlToImage': 'https://g.foolcdn.com/editorial/images/724405/copy-of-jose-najarro-2023-03-12t201709621.png', 'publishedAt': '2023-03-13T10:16:00Z', 'content': 'While numerous U.S.-based chip design companies exist, most of the chip manufacturing is done by international players. Unfortunately, some of these global players believe requirements for the CHIPS … [+340 chars]'}, {'source': {'id': None, 'name': 'Autoblog'}, 'author': 'Reuters', 'title': 'Tesla could begin producing electric cars in Mexico next year', 'description': 'Filed under:\n Green,Plants/Manufacturing,Tesla,Electric\n Continue reading Tesla could begin producing electric cars in Mexico next year\nTesla could begin producing electric cars in Mexico next year originally appeared on Autoblog on Tue, 7 Mar 2023 08:43:00 E…', 'url': 'https://www.autoblog.com/2023/03/07/tesla-mexico-production-2024/', 'urlToImage': 'https://o.aolcdn.com/images/dims3/GLOB/crop/4368x2457+0+0/resize/800x450!/format/jpg/quality/85/https://s.yimg.com/os/creatr-uploaded-images/2023-02/ec46bca0-bc14-11ed-9c5d-895f539777f5', 'publishedAt': '2023-03-07T13:43:00Z', 'content': 'MEXICO CITY Tesla could begin producing its first cars in Mexico next year, with the electric vehicle maker close to receiving its final permits allowing factory construction to begin in Nuevo Leon n… [+1628 chars]'}, {'source': {'id': 'google-news', 'name': 'Google News'}, 'author': None, 'title': "CHIPS Act: Biden increases government to increase control over everyone's business - Washington Examiner", 'description': "CHIPS Act: Biden increases government to increase control over everyone's business\xa0\xa0Washington Examiner", 'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMie2h0dHBzOi8vd3d3Lndhc2hpbmd0b25leGFtaW5lci5jb20vb3Bpbmlvbi9jaGlwcy1hY3QtYmlkZW4taW5jcmVhc2VzLWdvdmVybm1lbnQtdG8taW5jcmVhc2UtY29udHJvbC1vdmVyLWV2ZXJ5b25lcy1idXNpbmVzc9IBAA?oc%3D5%26hl%3Den-CA%26gl%3DCA%26ceid%3DCA:en&gl=FR&hl=en-CA&pc=n&src=1', 'urlToImage': None, 'publishedAt': '2023-03-07T23:38:00Z', 'content': 'We use cookies and data to<ul><li>Deliver and maintain Google services</li><li>Track outages and protect against spam, fraud, and abuse</li><li>Measure audience engagement and site statistics to unde… [+1131 chars]'}, {'source': {'id': 'al-jazeera-english', 'name': 'Al Jazeera English'}, 'author': 'Amy Chew', 'title': 'Malaysia’s Anwar faces balancing act on first China trip', 'description': 'Malaysian leader expected to navigate between deepening economic ties and tackling differences in talks with Xi Jinping.', 'url': 'https://www.aljazeera.com/economy/2023/3/30/malaysias-anwar-faces-balancing-act-on-first-china-trip', 'urlToImage': 'https://www.aljazeera.com/wp-content/uploads/2023/03/AP23061118140422-1.jpg?resize=1920%2C1440', 'publishedAt': '2023-03-30T01:09:48Z', 'content': 'Kuala Lumpur, Malaysia Malaysias Anwar Ibrahim is expected to navigate between deepening economic ties with his countrys biggest trading partner and tackling thorny issues such as the South China Sea… [+6866 chars]'}, {'source': {'id': None, 'name': 'Restofworld.org'}, 'author': 'Digg Editors', 'title': 'US Senator Mark Warner On TikTok, China And The End Of Techno-Optimism', 'description': "With the RESTRICT Act, Sen. Warner wants to reset the way the US treats China's tech industry.", 'url': 'https://restofworld.org/2023/mark-warner-interview-restrict-act/', 'urlToImage': 'https://149346090.v2.pressablecdn.com/wp-content/uploads/2023/03/cropped-230329_GK_MarkWarner057-copy.jpg', 'publishedAt': '2023-03-29T19:18:13Z', 'content': 'Officials in Washington have been sounding the security alarm about TikTok for years, but now the United States is closer than ever to blocking the social app from operating within its borders. TikTo… [+8017 chars]'}, {'source': {'id': None, 'name': 'Biztoc.com'}, 'author': 'fortune.com', 'title': 'Chance to host semiconductor factories under CHIPS Act has Oregon reconsidering rules against urban sprawl', 'description': 'Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Just over the horizon loomed a gigantic building of the semicon…', 'url': 'https://biztoc.com/x/3511e057c4d9f5c2', 'urlToImage': 'https://c.biztoc.com/p/3511e057c4d9f5c2/og.webp', 'publishedAt': '2023-03-26T14:36:04Z', 'content': 'Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Jus… [+306 chars]'}, {'source': {'id': None, 'name': 'The Atlantic'}, 'author': 'Elliot Ackerman', 'title': 'The Arsenal of Democracy Is Reopening for Business', 'description': 'But after decades of decay, will it happen quickly enough to save Ukraine?', 'url': 'https://www.theatlantic.com/ideas/archive/2023/03/american-defense-manufacturing-ukraine-aid-arkansas/673327/?utm_source=feed', 'urlToImage': None, 'publishedAt': '2023-03-09T18:04:56Z', 'content': 'Lockheed Martin builds its advanced mobile rocket launchers in a converted diaper factory, of all places. When I visited the plant in southern Arkansas at the end of February, I found it humming with… [+9948 chars]'}, {'source': {'id': 'google-news', 'name': 'Google News'}, 'author': None, 'title': "The CHIPS Act: Rebuilding America's technological infrastructure - CBS News", 'description': "The CHIPS Act: Rebuilding America's technological infrastructure\xa0\xa0CBS News", 'url': 'https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMiXGh0dHBzOi8vd3d3LmNic25ld3MuY29tL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv0gFgaHR0cHM6Ly93d3cuY2JzbmV3cy5jb20vYW1wL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv?oc%3D5&gl=FR&hl=en-US&pc=n&src=1', 'urlToImage': None, 'publishedAt': '2023-03-05T14:11:07Z', 'content': 'We use cookies and data to<ul><li>Deliver and maintain Google services</li><li>Track outages and protect against spam, fraud, and abuse</li><li>Measure audience engagement and site statistics to unde… [+1131 chars]'}]


### Check the datatype for each key


```python
print(type(response_json['status']))
print(type(response_json['totalResults']))
print(type(response_json['articles']))
```

    <class 'str'>
    <class 'int'>
    <class 'list'>


### Make sure the list reads as a dictionary


```python
type(response_json['articles'][0])
```




    dict



### Convert the JSON key into a Pandas Dataframe


```python
df_articles = pd.DataFrame(response_json['articles'])
df_articles
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>author</th>
      <th>title</th>
      <th>description</th>
      <th>url</th>
      <th>urlToImage</th>
      <th>publishedAt</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'id': 'engadget', 'name': 'Engadget'}</td>
      <td>Igor Bonifacic</td>
      <td>Biden administration bars CHIPS Act funding recipients from expanding in China</td>
      <td>Chipmakers hoping to tap into the Biden administration’s $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they won’t expand production capacity in China. The requirement was among a handful of funding conditions t…</td>
      <td>https://www.engadget.com/biden-administration-bars-chips-act-funding-recipients-from-expanding-in-china-172637590.html</td>
      <td>https://s.yimg.com/uu/api/res/1.2/O0kRn2KWxWw2Gij4lWICow--~B/Zmk9ZmlsbDtoPTYzMDtweW9mZj0wO3c9MTIwMDthcHBpZD15dGFjaHlvbg--/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2023-02/f7fa4d60-bab0-11ed-afae-99a6c04fc149.cf.jpg</td>
      <td>2023-03-04T17:26:37Z</td>
      <td>Chipmakers hoping to tap into the Biden administrations $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they wont expand production capacity in China. T… [+1800 chars]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>S. Korea's trade minister to raise concerns over Chips Act in US ... - Reuters</td>
      <td>S. Korea's trade minister to raise concerns over Chips Act in US ...  Reuters</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMicGh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3RlY2hub2xvZ3kvcy1rb3JlYXMtdHJhZGUtbWluaXN0ZXItcmFpc2UtY29uY2VybnMtb3Zlci1jaGlwcy1hY3QtdXMtbWVldGluZ3MtMjAyMy0wMy0wOC_SAQA?oc%3D5&amp;gl=FR&amp;hl=en-US&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-08T03:55:00Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'id': 'bbc-news', 'name': 'BBC News'}</td>
      <td>https://www.facebook.com/bbcnews</td>
      <td>US-China chip war: Netherlands moves to restrict some exports</td>
      <td>The measures will affect Dutch firm ASML, which is a key part of the global microchip supply chain.</td>
      <td>https://www.bbc.co.uk/news/business-64897794</td>
      <td>https://ichef.bbci.co.uk/news/1024/branded_news/898D/production/_128931253_gettyimages-1354885833.png</td>
      <td>2023-03-09T03:46:28Z</td>
      <td>The Dutch government says before the summer it will put restrictions on the country's "most advanced" chip exports to protect its national security, following a similar move by the US.\r\nIt will inclu… [+2395 chars]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'id': None, 'name': 'Slashdot.org'}</td>
      <td>EditorDavid</td>
      <td>Worf's Final Act: a 'Star Trek' Legend Looks Back</td>
      <td>The final season of Star Trek: Picard features the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi's most beloved supporting characters something that's usually reserved only for Captains and Admirals: a glorious thir…</td>
      <td>https://entertainment.slashdot.org/story/23/03/05/0313230/worfs-final-act-a-star-trek-legend-looks-back</td>
      <td>https://a.fsdn.com/sd/topics/tv_64.png</td>
      <td>2023-03-05T09:04:00Z</td>
      <td>The final season of Star Trek: Picardfeatures the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi's most beloved supporting characters something that's usual… [+3499 chars]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'id': None, 'name': 'NPR'}</td>
      <td>Andrea Hsu</td>
      <td>Biden has big ideas for fixing child care. For now a small workaround will have to do</td>
      <td>The administration is turning to semiconductors in the hopes of expanding affordable child care.</td>
      <td>https://www.npr.org/2023/03/17/1162869162/child-care-chips-semiconductors-manufacturing-raimondo-subsidies</td>
      <td>https://media.npr.org/assets/img/2023/03/16/gettyimages-161145124_wide-1f062d47e8a8d91d840a9c29aba33d5c52586cde-s1400-c100.jpg</td>
      <td>2023-03-17T09:00:36Z</td>
      <td>President Biden's ambitious proposals to address the high cost and short supply of child care haven't garnered enough support in Congress, so now his administration has come up with a workaround.\r\nGe… [+6055 chars]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{'id': None, 'name': 'Theregister.com'}</td>
      <td>Laura Dobberstein</td>
      <td>US details CHIPS Act rules that give China and South Korea some comfort</td>
      <td>Allows a little expansion of output from existing Middle Kingdom facilities\nThe US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers…</td>
      <td>https://www.theregister.com/2023/03/22/chips_act_rules_floated/</td>
      <td>https://regmedia.co.uk/2023/03/22/shutterstock_us_korea_taiwan_china_motherboard.jpg</td>
      <td>2023-03-22T06:32:06Z</td>
      <td>The US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers adversarial.\r\nT… [+2712 chars]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>{'id': 'politico', 'name': 'Politico'}</td>
      <td>Brendan Bordelon and Caitlin Oprysko</td>
      <td>Everybody in Washington wants a byte of the CHIPS law</td>
      <td>Lobbyists are descending on Washington, all hungry for a piece of the CHIPS and Science Act — even if some bids have little chance of making the cut.</td>
      <td>https://www.politico.com/news/2023/03/17/chips-law-companies-washington-lobbying-00086687</td>
      <td>https://static.politico.com/a5/e9/ed7ea593463aac5b690046668670/https-delivery-gettyimages.com/downloads/586113576</td>
      <td>2023-03-17T08:30:00Z</td>
      <td>But the new law has also attracted an armada of lobbyists repping a wide range of industries. Some may be making long-shot bids lobbyists for Snap, for example, plan to ask Washington to subsidize pl… [+11802 chars]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>U.S. CHIPS Act Recipients Face Expansion Hurdles in China</td>
      <td>More regulations and requirements are being released for the CHIPS Act that semiconductor investors might want to know about.</td>
      <td>https://www.fool.com/investing/2023/03/24/us-chips-act-recipients-face-expansion-hurdles-in/</td>
      <td>https://g.foolcdn.com/editorial/images/725864/copy-of-jose-najarro-2023-03-24t153421346.png</td>
      <td>2023-03-24T20:01:00Z</td>
      <td>Suzanne Frey, an executive at Alphabet, is a member of The Motley Fool’s board of directors. John Mackey, former CEO of Whole Foods Market, an Amazon subsidiary, is a member of The Motley Fool’s boar… [+1048 chars]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>The CHIPS Act Is Open for Business -- Which Stocks Will Benefit Most?</td>
      <td>Applications for the CHIPS Act are now open. It comes with some rules that semiconductor companies must follow, from profit sharing to buyback and international expansion regulations.</td>
      <td>https://www.fool.com/investing/2023/03/08/the-chips-act-is-open-for-business-which-stocks-wi/</td>
      <td>https://g.foolcdn.com/editorial/images/723864/copy-of-jose-najarro-2023-03-08t090543667.png</td>
      <td>2023-03-08T14:56:40Z</td>
      <td>Check out this short video to learn what semiconductor investors Jose Najarro, Nicholas Rossolillo, and Billy Duberstein had to say about the CHIPS Act and how companies like Intel (INTC 0.37%) can b… [+268 chars]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>{'id': None, 'name': 'Biztoc.com'}</td>
      <td>cbsnews.com</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure</td>
      <td>You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They're also in your clocks, toys, thermostats, and every single thing in your kitchen. "Our demand …</td>
      <td>https://biztoc.com/x/582cdb2369a83030</td>
      <td>https://c.biztoc.com/p/582cdb2369a83030/og.webp</td>
      <td>2023-03-05T14:24:04Z</td>
      <td>You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They're also in your clocks, toys, therm… [+297 chars]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Nicholas Rossolillo)</td>
      <td>The CHIPS Act Is Accepting Applications -- Does That Make GlobalFoundries Stock a Buy?</td>
      <td>Mature chip manufacturing could get a big boost from government-led investment in the coming years.</td>
      <td>https://www.fool.com/investing/2023/03/23/the-chips-act-is-accepting-applications-does-that/</td>
      <td>https://g.foolcdn.com/editorial/images/725359/semiconductor-technician-with-wafer-in-manufacturing-plant.jpg</td>
      <td>2023-03-23T12:45:00Z</td>
      <td>The bear market of 2022 bludgeoned technology stocks, but it also brought an end to the chip shortage for many markets -- at exactly the wrong time for investors. Supply of high-end chips for smartph… [+5097 chars]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>{'id': None, 'name': 'Science Daily'}</td>
      <td>None</td>
      <td>Origami-inspired robots can sense, analyze and act in challenging environments</td>
      <td>A multidisciplinary team has created a new fabrication technique for fully foldable robots that can perform a variety of complex tasks without relying on semiconductors.</td>
      <td>https://www.sciencedaily.com/releases/2023/04/230403162632.htm</td>
      <td>https://www.sciencedaily.com/images/scidaily-icon.png</td>
      <td>2023-04-03T20:26:32Z</td>
      <td>Roboticists have been using a technique similar to the ancient art of paper folding to develop autonomous machines out of thin, flexible sheets. These lightweight robots are simpler and cheaper to ma… [+4224 chars]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>The CHIPS Act's International Hurdles -- What It Means for Semiconductor Investors</td>
      <td>The CHIPS Act is meant to improve semiconductor manufacturing and design in the U.S. Still, the requirements for these subsidies might be too much for certain international semiconductor companies.</td>
      <td>https://www.fool.com/investing/2023/03/13/the-chips-acts-international-hurdles-what-it-means/</td>
      <td>https://g.foolcdn.com/editorial/images/724405/copy-of-jose-najarro-2023-03-12t201709621.png</td>
      <td>2023-03-13T10:16:00Z</td>
      <td>While numerous U.S.-based chip design companies exist, most of the chip manufacturing is done by international players. Unfortunately, some of these global players believe requirements for the CHIPS … [+340 chars]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>{'id': None, 'name': 'Autoblog'}</td>
      <td>Reuters</td>
      <td>Tesla could begin producing electric cars in Mexico next year</td>
      <td>Filed under:\n Green,Plants/Manufacturing,Tesla,Electric\n Continue reading Tesla could begin producing electric cars in Mexico next year\nTesla could begin producing electric cars in Mexico next year originally appeared on Autoblog on Tue, 7 Mar 2023 08:43:00 E…</td>
      <td>https://www.autoblog.com/2023/03/07/tesla-mexico-production-2024/</td>
      <td>https://o.aolcdn.com/images/dims3/GLOB/crop/4368x2457+0+0/resize/800x450!/format/jpg/quality/85/https://s.yimg.com/os/creatr-uploaded-images/2023-02/ec46bca0-bc14-11ed-9c5d-895f539777f5</td>
      <td>2023-03-07T13:43:00Z</td>
      <td>MEXICO CITY Tesla could begin producing its first cars in Mexico next year, with the electric vehicle maker close to receiving its final permits allowing factory construction to begin in Nuevo Leon n… [+1628 chars]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>CHIPS Act: Biden increases government to increase control over everyone's business - Washington Examiner</td>
      <td>CHIPS Act: Biden increases government to increase control over everyone's business  Washington Examiner</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMie2h0dHBzOi8vd3d3Lndhc2hpbmd0b25leGFtaW5lci5jb20vb3Bpbmlvbi9jaGlwcy1hY3QtYmlkZW4taW5jcmVhc2VzLWdvdmVybm1lbnQtdG8taW5jcmVhc2UtY29udHJvbC1vdmVyLWV2ZXJ5b25lcy1idXNpbmVzc9IBAA?oc%3D5%26hl%3Den-CA%26gl%3DCA%26ceid%3DCA:en&amp;gl=FR&amp;hl=en-CA&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-07T23:38:00Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>{'id': 'al-jazeera-english', 'name': 'Al Jazeera English'}</td>
      <td>Amy Chew</td>
      <td>Malaysia’s Anwar faces balancing act on first China trip</td>
      <td>Malaysian leader expected to navigate between deepening economic ties and tackling differences in talks with Xi Jinping.</td>
      <td>https://www.aljazeera.com/economy/2023/3/30/malaysias-anwar-faces-balancing-act-on-first-china-trip</td>
      <td>https://www.aljazeera.com/wp-content/uploads/2023/03/AP23061118140422-1.jpg?resize=1920%2C1440</td>
      <td>2023-03-30T01:09:48Z</td>
      <td>Kuala Lumpur, Malaysia Malaysias Anwar Ibrahim is expected to navigate between deepening economic ties with his countrys biggest trading partner and tackling thorny issues such as the South China Sea… [+6866 chars]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>{'id': None, 'name': 'Restofworld.org'}</td>
      <td>Digg Editors</td>
      <td>US Senator Mark Warner On TikTok, China And The End Of Techno-Optimism</td>
      <td>With the RESTRICT Act, Sen. Warner wants to reset the way the US treats China's tech industry.</td>
      <td>https://restofworld.org/2023/mark-warner-interview-restrict-act/</td>
      <td>https://149346090.v2.pressablecdn.com/wp-content/uploads/2023/03/cropped-230329_GK_MarkWarner057-copy.jpg</td>
      <td>2023-03-29T19:18:13Z</td>
      <td>Officials in Washington have been sounding the security alarm about TikTok for years, but now the United States is closer than ever to blocking the social app from operating within its borders. TikTo… [+8017 chars]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>{'id': None, 'name': 'Biztoc.com'}</td>
      <td>fortune.com</td>
      <td>Chance to host semiconductor factories under CHIPS Act has Oregon reconsidering rules against urban sprawl</td>
      <td>Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Just over the horizon loomed a gigantic building of the semicon…</td>
      <td>https://biztoc.com/x/3511e057c4d9f5c2</td>
      <td>https://c.biztoc.com/p/3511e057c4d9f5c2/og.webp</td>
      <td>2023-03-26T14:36:04Z</td>
      <td>Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Jus… [+306 chars]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>{'id': None, 'name': 'The Atlantic'}</td>
      <td>Elliot Ackerman</td>
      <td>The Arsenal of Democracy Is Reopening for Business</td>
      <td>But after decades of decay, will it happen quickly enough to save Ukraine?</td>
      <td>https://www.theatlantic.com/ideas/archive/2023/03/american-defense-manufacturing-ukraine-aid-arkansas/673327/?utm_source=feed</td>
      <td>None</td>
      <td>2023-03-09T18:04:56Z</td>
      <td>Lockheed Martin builds its advanced mobile rocket launchers in a converted diaper factory, of all places. When I visited the plant in southern Arkansas at the end of February, I found it humming with… [+9948 chars]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure - CBS News</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure  CBS News</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMiXGh0dHBzOi8vd3d3LmNic25ld3MuY29tL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv0gFgaHR0cHM6Ly93d3cuY2JzbmV3cy5jb20vYW1wL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv?oc%3D5&amp;gl=FR&amp;hl=en-US&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-05T14:11:07Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
    </tr>
  </tbody>
</table>
</div>



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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>author</th>
      <th>title</th>
      <th>description</th>
      <th>url</th>
      <th>urlToImage</th>
      <th>publishedAt</th>
      <th>content</th>
      <th>scraped_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'id': 'engadget', 'name': 'Engadget'}</td>
      <td>Igor Bonifacic</td>
      <td>Biden administration bars CHIPS Act funding recipients from expanding in China</td>
      <td>Chipmakers hoping to tap into the Biden administration’s $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they won’t expand production capacity in China. The requirement was among a handful of funding conditions t…</td>
      <td>https://www.engadget.com/biden-administration-bars-chips-act-funding-recipients-from-expanding-in-china-172637590.html</td>
      <td>https://s.yimg.com/uu/api/res/1.2/O0kRn2KWxWw2Gij4lWICow--~B/Zmk9ZmlsbDtoPTYzMDtweW9mZj0wO3c9MTIwMDthcHBpZD15dGFjaHlvbg--/https://media-mbst-pub-ue1.s3.amazonaws.com/creatr-uploaded-images/2023-02/f7fa4d60-bab0-11ed-afae-99a6c04fc149.cf.jpg</td>
      <td>2023-03-04T17:26:37Z</td>
      <td>Chipmakers hoping to tap into the Biden administrations $39 billion semiconductor manufacturing subsidy program will need to sign agreements promising they wont expand production capacity in China. T… [+1800 chars]</td>
      <td>&lt;!doctype html&gt;&lt;html id="atomic" class="desktop bktengadget-def-bucket ua-undefined ua-undefined" lang="en-US"&gt;&lt;head&gt;&lt;script&gt;\n        window.performance.mark('PageStart');\n        document.documentElement.className += ' JsEnabled jsenabled';\n        &lt;/script&gt;&lt;title&gt;Biden administration bars CHIPS Act funding recipients from expanding in China | Engadget&lt;/title&gt;&lt;meta http-equiv="content-type...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>S. Korea's trade minister to raise concerns over Chips Act in US ... - Reuters</td>
      <td>S. Korea's trade minister to raise concerns over Chips Act in US ...  Reuters</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMicGh0dHBzOi8vd3d3LnJldXRlcnMuY29tL3RlY2hub2xvZ3kvcy1rb3JlYXMtdHJhZGUtbWluaXN0ZXItcmFpc2UtY29uY2VybnMtb3Zlci1jaGlwcy1hY3QtdXMtbWVldGluZ3MtMjAyMy0wMy0wOC_SAQA?oc%3D5&amp;gl=FR&amp;hl=en-US&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-08T03:55:00Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
      <td>&lt;!DOCTYPE html&gt;&lt;html lang="en"&gt;&lt;head&gt;&lt;title&gt;S. Korea&amp;#x27;s trade minister to raise concerns over Chips Act in US meetings | Reuters&lt;/title&gt;&lt;meta name="viewport" content="width=device-width, initial-scale=1"/&gt;&lt;meta name="apple-itunes-app" content="app-id=602660809" app-argument="https://www.reuters.com/technology/s-koreas-trade-minister-raise-concerns-over-chips-act-us-meetings-2023-03-08/?id=...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{'id': 'bbc-news', 'name': 'BBC News'}</td>
      <td>https://www.facebook.com/bbcnews</td>
      <td>US-China chip war: Netherlands moves to restrict some exports</td>
      <td>The measures will affect Dutch firm ASML, which is a key part of the global microchip supply chain.</td>
      <td>https://www.bbc.co.uk/news/business-64897794</td>
      <td>https://ichef.bbci.co.uk/news/1024/branded_news/898D/production/_128931253_gettyimages-1354885833.png</td>
      <td>2023-03-09T03:46:28Z</td>
      <td>The Dutch government says before the summer it will put restrictions on the country's "most advanced" chip exports to protect its national security, following a similar move by the US.\r\nIt will inclu… [+2395 chars]</td>
      <td>&lt;!DOCTYPE html&gt;&lt;html lang="en-GB" class="no-js"&gt;&lt;head&gt;&lt;meta charSet="utf-8" /&gt;&lt;meta name="viewport" content="width=device-width, initial-scale=1" /&gt;&lt;title data-rh="true"&gt;US-China chip war: Netherlands moves to restrict some tech exports - BBC News&lt;/title&gt;&lt;meta data-rh="true" name="description" content="The measures will affect Dutch firm ASML, which is a key part of the global microchip supply...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{'id': None, 'name': 'Slashdot.org'}</td>
      <td>EditorDavid</td>
      <td>Worf's Final Act: a 'Star Trek' Legend Looks Back</td>
      <td>The final season of Star Trek: Picard features the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi's most beloved supporting characters something that's usually reserved only for Captains and Admirals: a glorious thir…</td>
      <td>https://entertainment.slashdot.org/story/23/03/05/0313230/worfs-final-act-a-star-trek-legend-looks-back</td>
      <td>https://a.fsdn.com/sd/topics/tv_64.png</td>
      <td>2023-03-05T09:04:00Z</td>
      <td>The final season of Star Trek: Picardfeatures the return of the Klingon Worf, reports Polygon, calling it "the chance to give one of sci-fi's most beloved supporting characters something that's usual… [+3499 chars]</td>
      <td>&lt;!-- html-header type=current begin --&gt;\n\t\n\t&lt;!DOCTYPE html&gt;\n\t\n\t&lt;html lang="en"&gt;\n\t&lt;head&gt;\n\t&lt;!-- Render IE9 --&gt;\n\t&lt;meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1"&gt;\n\n\t\n\n\t&lt;script id="before-content" type="text/javascript"&gt;\n(function () {\n    if (typeof window.sdmedia !== 'object') {\n         window.sdmedia = {};\n    }\n    if (typeof window.sdmedia.site !== 'objec...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{'id': None, 'name': 'NPR'}</td>
      <td>Andrea Hsu</td>
      <td>Biden has big ideas for fixing child care. For now a small workaround will have to do</td>
      <td>The administration is turning to semiconductors in the hopes of expanding affordable child care.</td>
      <td>https://www.npr.org/2023/03/17/1162869162/child-care-chips-semiconductors-manufacturing-raimondo-subsidies</td>
      <td>https://media.npr.org/assets/img/2023/03/16/gettyimages-161145124_wide-1f062d47e8a8d91d840a9c29aba33d5c52586cde-s1400-c100.jpg</td>
      <td>2023-03-17T09:00:36Z</td>
      <td>President Biden's ambitious proposals to address the high cost and short supply of child care haven't garnered enough support in Congress, so now his administration has come up with a workaround.\r\nGe… [+6055 chars]</td>
      <td>&lt;!doctype html&gt;&lt;html class="no-js" lang="en"&gt;&lt;head&gt;&lt;!-- OneTrust Cookies Consent Notice start for npr.org --&gt;\n&lt;script type="text/javascript" src="https://cdn.cookielaw.org/consent/82089dfe-410c-4e1b-a7f9-698174b62a86/OtAutoBlock.js" &gt;&lt;/script&gt;\n&lt;script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js"  type="text/javascript" charset="UTF-8" data-domain-script="82089dfe-410c-4e1b-a7...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{'id': None, 'name': 'Theregister.com'}</td>
      <td>Laura Dobberstein</td>
      <td>US details CHIPS Act rules that give China and South Korea some comfort</td>
      <td>Allows a little expansion of output from existing Middle Kingdom facilities\nThe US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers…</td>
      <td>https://www.theregister.com/2023/03/22/chips_act_rules_floated/</td>
      <td>https://regmedia.co.uk/2023/03/22/shutterstock_us_korea_taiwan_china_motherboard.jpg</td>
      <td>2023-03-22T06:32:06Z</td>
      <td>The US Commerce Department proposed rules on Tuesday that would limit the amount of CHIPS Act recipients can invest to expand semiconductor manufacturing in countries the US considers adversarial.\r\nT… [+2712 chars]</td>
      <td>403 forbidden.  The Register apologises but your traffic appears to be tickling our robot sensors.  If this problem persists, webmaster@theregister.co.uk</td>
    </tr>
    <tr>
      <th>6</th>
      <td>{'id': 'politico', 'name': 'Politico'}</td>
      <td>Brendan Bordelon and Caitlin Oprysko</td>
      <td>Everybody in Washington wants a byte of the CHIPS law</td>
      <td>Lobbyists are descending on Washington, all hungry for a piece of the CHIPS and Science Act — even if some bids have little chance of making the cut.</td>
      <td>https://www.politico.com/news/2023/03/17/chips-law-companies-washington-lobbying-00086687</td>
      <td>https://static.politico.com/a5/e9/ed7ea593463aac5b690046668670/https-delivery-gettyimages.com/downloads/586113576</td>
      <td>2023-03-17T08:30:00Z</td>
      <td>But the new law has also attracted an armada of lobbyists repping a wide range of industries. Some may be making long-shot bids lobbyists for Snap, for example, plan to ask Washington to subsidize pl… [+11802 chars]</td>
      <td>\n&lt;!DOCTYPE html&gt;\n&lt;!--[if lt IE 7]&gt;&lt;html lang="en" class="no-js lt-ie9 lt-ie8 lt-ie7"&gt; &lt;![endif]--&gt;\n&lt;!--[if IE 7]&gt;&lt;html lang="en" class="no-js lt-ie9 lt-ie8"&gt; &lt;![endif]--&gt;\n&lt;!--[if IE 8]&gt;    &lt;html lang="en" class="no-js ie ie8 lte8 lt-ie9"&gt; &lt;![endif]--&gt;\n&lt;!--[if IE 9]&gt;    &lt;html lang="en" class="no-js ie ie9 lte9"&gt; &lt;![endif]--&gt;\n&lt;!--[if gt IE 9]&gt; &lt;html lang="en" class="no-js"&gt; &lt;![endif]--&gt;\n&lt;...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>U.S. CHIPS Act Recipients Face Expansion Hurdles in China</td>
      <td>More regulations and requirements are being released for the CHIPS Act that semiconductor investors might want to know about.</td>
      <td>https://www.fool.com/investing/2023/03/24/us-chips-act-recipients-face-expansion-hurdles-in/</td>
      <td>https://g.foolcdn.com/editorial/images/725864/copy-of-jose-najarro-2023-03-24t153421346.png</td>
      <td>2023-03-24T20:01:00Z</td>
      <td>Suzanne Frey, an executive at Alphabet, is a member of The Motley Fool’s board of directors. John Mackey, former CEO of Whole Foods Market, an Amazon subsidiary, is a member of The Motley Fool’s boar… [+1048 chars]</td>
      <td>\n&lt;!DOCTYPE html&gt;\n&lt;html lang="en" prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# article: http://ogp.me/ns/article#"&gt;\n&lt;head&gt;\n\n\n&lt;script type="text/javascript" src="https://cdn.cookielaw.org/consent/02abb198-81a8-49e5-a9b1-f69a5dd9c039/OtAutoBlock.js"&gt;&lt;/script&gt;\n&lt;script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js" type="text/javascript" charset="UTF-8" data-domain-sc...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>The CHIPS Act Is Open for Business -- Which Stocks Will Benefit Most?</td>
      <td>Applications for the CHIPS Act are now open. It comes with some rules that semiconductor companies must follow, from profit sharing to buyback and international expansion regulations.</td>
      <td>https://www.fool.com/investing/2023/03/08/the-chips-act-is-open-for-business-which-stocks-wi/</td>
      <td>https://g.foolcdn.com/editorial/images/723864/copy-of-jose-najarro-2023-03-08t090543667.png</td>
      <td>2023-03-08T14:56:40Z</td>
      <td>Check out this short video to learn what semiconductor investors Jose Najarro, Nicholas Rossolillo, and Billy Duberstein had to say about the CHIPS Act and how companies like Intel (INTC 0.37%) can b… [+268 chars]</td>
      <td>\n&lt;!DOCTYPE html&gt;\n&lt;html lang="en" prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# article: http://ogp.me/ns/article#"&gt;\n&lt;head&gt;\n\n\n&lt;script type="text/javascript" src="https://cdn.cookielaw.org/consent/02abb198-81a8-49e5-a9b1-f69a5dd9c039/OtAutoBlock.js"&gt;&lt;/script&gt;\n&lt;script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js" type="text/javascript" charset="UTF-8" data-domain-sc...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>{'id': None, 'name': 'Biztoc.com'}</td>
      <td>cbsnews.com</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure</td>
      <td>You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They're also in your clocks, toys, thermostats, and every single thing in your kitchen. "Our demand …</td>
      <td>https://biztoc.com/x/582cdb2369a83030</td>
      <td>https://c.biztoc.com/p/582cdb2369a83030/og.webp</td>
      <td>2023-03-05T14:24:04Z</td>
      <td>You already know that there are computer chips in your computer and your phone. But you may not realize just how many other things in your life rely on chips. They're also in your clocks, toys, therm… [+297 chars]</td>
      <td>&lt;!doctype html&gt;&lt;html lang="en" itemscope id="page_home" itemtype="http://schema.org/WebPage" prefix="og: http://ogp.me/ns#"&gt; &lt;head&gt;&lt;meta charset="utf-8"&gt;&lt;meta property="og:locale" content="en_US"&gt;&lt;meta name="viewport" content="width=device-width,initial-scale=1,minimum-scale=1"&gt;&lt;link rel="canonical" href="https://biztoc.com/"&gt;&lt;title&gt;BizToc&lt;/title&gt;&lt;script type="text/javascript"&gt;\n\n        cons...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Nicholas Rossolillo)</td>
      <td>The CHIPS Act Is Accepting Applications -- Does That Make GlobalFoundries Stock a Buy?</td>
      <td>Mature chip manufacturing could get a big boost from government-led investment in the coming years.</td>
      <td>https://www.fool.com/investing/2023/03/23/the-chips-act-is-accepting-applications-does-that/</td>
      <td>https://g.foolcdn.com/editorial/images/725359/semiconductor-technician-with-wafer-in-manufacturing-plant.jpg</td>
      <td>2023-03-23T12:45:00Z</td>
      <td>The bear market of 2022 bludgeoned technology stocks, but it also brought an end to the chip shortage for many markets -- at exactly the wrong time for investors. Supply of high-end chips for smartph… [+5097 chars]</td>
      <td>\n&lt;!DOCTYPE html&gt;\n&lt;html lang="en" prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# article: http://ogp.me/ns/article#"&gt;\n&lt;head&gt;\n\n\n&lt;script type="text/javascript" src="https://cdn.cookielaw.org/consent/02abb198-81a8-49e5-a9b1-f69a5dd9c039/OtAutoBlock.js"&gt;&lt;/script&gt;\n&lt;script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js" type="text/javascript" charset="UTF-8" data-domain-sc...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>{'id': None, 'name': 'Science Daily'}</td>
      <td>None</td>
      <td>Origami-inspired robots can sense, analyze and act in challenging environments</td>
      <td>A multidisciplinary team has created a new fabrication technique for fully foldable robots that can perform a variety of complex tasks without relying on semiconductors.</td>
      <td>https://www.sciencedaily.com/releases/2023/04/230403162632.htm</td>
      <td>https://www.sciencedaily.com/images/scidaily-icon.png</td>
      <td>2023-04-03T20:26:32Z</td>
      <td>Roboticists have been using a technique similar to the ancient art of paper folding to develop autonomous machines out of thin, flexible sheets. These lightweight robots are simpler and cheaper to ma… [+4224 chars]</td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>{'id': None, 'name': 'Motley Fool'}</td>
      <td>newsfeedback@fool.com (Jose Najarro, Nicholas Rossolillo, and Billy Duberstein)</td>
      <td>The CHIPS Act's International Hurdles -- What It Means for Semiconductor Investors</td>
      <td>The CHIPS Act is meant to improve semiconductor manufacturing and design in the U.S. Still, the requirements for these subsidies might be too much for certain international semiconductor companies.</td>
      <td>https://www.fool.com/investing/2023/03/13/the-chips-acts-international-hurdles-what-it-means/</td>
      <td>https://g.foolcdn.com/editorial/images/724405/copy-of-jose-najarro-2023-03-12t201709621.png</td>
      <td>2023-03-13T10:16:00Z</td>
      <td>While numerous U.S.-based chip design companies exist, most of the chip manufacturing is done by international players. Unfortunately, some of these global players believe requirements for the CHIPS … [+340 chars]</td>
      <td>\n&lt;!DOCTYPE html&gt;\n&lt;html lang="en" prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# article: http://ogp.me/ns/article#"&gt;\n&lt;head&gt;\n\n\n&lt;script type="text/javascript" src="https://cdn.cookielaw.org/consent/02abb198-81a8-49e5-a9b1-f69a5dd9c039/OtAutoBlock.js"&gt;&lt;/script&gt;\n&lt;script src="https://cdn.cookielaw.org/scripttemplates/otSDKStub.js" type="text/javascript" charset="UTF-8" data-domain-sc...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>{'id': None, 'name': 'Autoblog'}</td>
      <td>Reuters</td>
      <td>Tesla could begin producing electric cars in Mexico next year</td>
      <td>Filed under:\n Green,Plants/Manufacturing,Tesla,Electric\n Continue reading Tesla could begin producing electric cars in Mexico next year\nTesla could begin producing electric cars in Mexico next year originally appeared on Autoblog on Tue, 7 Mar 2023 08:43:00 E…</td>
      <td>https://www.autoblog.com/2023/03/07/tesla-mexico-production-2024/</td>
      <td>https://o.aolcdn.com/images/dims3/GLOB/crop/4368x2457+0+0/resize/800x450!/format/jpg/quality/85/https://s.yimg.com/os/creatr-uploaded-images/2023-02/ec46bca0-bc14-11ed-9c5d-895f539777f5</td>
      <td>2023-03-07T13:43:00Z</td>
      <td>MEXICO CITY Tesla could begin producing its first cars in Mexico next year, with the electric vehicle maker close to receiving its final permits allowing factory construction to begin in Nuevo Leon n… [+1628 chars]</td>
      <td>&lt;!DOCTYPE html&gt;\n\n&lt;html lang="en" class="flexbox"&gt;\n\t&lt;head&gt;\n\t\t&lt;meta charset="utf-8" /&gt;\n&lt;meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" /&gt;\n\n&lt;title&gt;Tesla could begin producing electric cars in Mexico next year - Autoblog&lt;/title&gt;\n&lt;meta name="description" content="Tesla Chief Executive Elon Musk announced that Tesla had selected Mexico for its next &amp;quot;gigafactory&amp;quot; wi...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>CHIPS Act: Biden increases government to increase control over everyone's business - Washington Examiner</td>
      <td>CHIPS Act: Biden increases government to increase control over everyone's business  Washington Examiner</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMie2h0dHBzOi8vd3d3Lndhc2hpbmd0b25leGFtaW5lci5jb20vb3Bpbmlvbi9jaGlwcy1hY3QtYmlkZW4taW5jcmVhc2VzLWdvdmVybm1lbnQtdG8taW5jcmVhc2UtY29udHJvbC1vdmVyLWV2ZXJ5b25lcy1idXNpbmVzc9IBAA?oc%3D5%26hl%3Den-CA%26gl%3DCA%26ceid%3DCA:en&amp;gl=FR&amp;hl=en-CA&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-07T23:38:00Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
      <td>&lt;!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd"&gt;\n&lt;HTML&gt;&lt;HEAD&gt;&lt;META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=iso-8859-1"&gt;\n&lt;TITLE&gt;ERROR: The request could not be satisfied&lt;/TITLE&gt;\n&lt;/HEAD&gt;&lt;BODY&gt;\n&lt;H1&gt;403 ERROR&lt;/H1&gt;\n&lt;H2&gt;The request could not be satisfied.&lt;/H2&gt;\n&lt;HR noshade size="1px"&gt;\nRequest blocked.\nWe can't connect to the ...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>{'id': 'al-jazeera-english', 'name': 'Al Jazeera English'}</td>
      <td>Amy Chew</td>
      <td>Malaysia’s Anwar faces balancing act on first China trip</td>
      <td>Malaysian leader expected to navigate between deepening economic ties and tackling differences in talks with Xi Jinping.</td>
      <td>https://www.aljazeera.com/economy/2023/3/30/malaysias-anwar-faces-balancing-act-on-first-china-trip</td>
      <td>https://www.aljazeera.com/wp-content/uploads/2023/03/AP23061118140422-1.jpg?resize=1920%2C1440</td>
      <td>2023-03-30T01:09:48Z</td>
      <td>Kuala Lumpur, Malaysia Malaysias Anwar Ibrahim is expected to navigate between deepening economic ties with his countrys biggest trading partner and tackling thorny issues such as the South China Sea… [+6866 chars]</td>
      <td>&lt;!doctype html&gt;&lt;html lang="en" dir="ltr" class="theme-aje"&gt;&lt;head&gt;&lt;meta charset="utf-8"/&gt;&lt;meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no"/&gt;&lt;meta http-equiv="Content-Type" content="text/html;charset=utf-8"&gt;&lt;link rel="shortcut icon" href="/favicon_aje.ico"&gt;&lt;title data-rh="true" data-reactroot=""&gt;Malaysia’s Anwar faces balancing act on first China trip | Internat...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>{'id': None, 'name': 'Restofworld.org'}</td>
      <td>Digg Editors</td>
      <td>US Senator Mark Warner On TikTok, China And The End Of Techno-Optimism</td>
      <td>With the RESTRICT Act, Sen. Warner wants to reset the way the US treats China's tech industry.</td>
      <td>https://restofworld.org/2023/mark-warner-interview-restrict-act/</td>
      <td>https://149346090.v2.pressablecdn.com/wp-content/uploads/2023/03/cropped-230329_GK_MarkWarner057-copy.jpg</td>
      <td>2023-03-29T19:18:13Z</td>
      <td>Officials in Washington have been sounding the security alarm about TikTok for years, but now the United States is closer than ever to blocking the social app from operating within its borders. TikTo… [+8017 chars]</td>
      <td>&lt;!doctype html&gt;\n&lt;html lang="en-US" class="palette-10 ltr" data-env="prod"&gt;\n&lt;head&gt;\n\t&lt;meta charset="UTF-8" /&gt;\n\t&lt;meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" /&gt;\n\t&lt;link rel="profile" href="https://gmpg.org/xfn/11" /&gt;\n\t&lt;link rel="shortcut icon" type="image/png" href="/wp-content/themes/orbis/static-assets/media/favicon-32.png" /&gt;\n\t&lt;title&gt;The RES...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>{'id': None, 'name': 'Biztoc.com'}</td>
      <td>fortune.com</td>
      <td>Chance to host semiconductor factories under CHIPS Act has Oregon reconsidering rules against urban sprawl</td>
      <td>Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Just over the horizon loomed a gigantic building of the semicon…</td>
      <td>https://biztoc.com/x/3511e057c4d9f5c2</td>
      <td>https://c.biztoc.com/p/3511e057c4d9f5c2/og.webp</td>
      <td>2023-03-26T14:36:04Z</td>
      <td>Aaron Nichols walked past rows of kale growing on his farm, his knee-high brown rubber boots speckled with some of the richest soil on earth, and gazed with concern toward fields in the distance. Jus… [+306 chars]</td>
      <td>&lt;!doctype html&gt;&lt;html lang="en" itemscope id="page_post" itemtype="http://schema.org/Article" prefix="og: http://ogp.me/ns# article: http://ogp.me/ns/article#"&gt; &lt;head&gt;&lt;meta charset="utf-8"&gt;&lt;meta property="og:locale" content="en_US"&gt;&lt;meta name="viewport" content="width=device-width,initial-scale=1,minimum-scale=1"&gt;&lt;link rel="canonical" href="https://biztoc.com/x/3511e057c4d9f5c2"&gt;&lt;title&gt;Chance t...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>{'id': None, 'name': 'The Atlantic'}</td>
      <td>Elliot Ackerman</td>
      <td>The Arsenal of Democracy Is Reopening for Business</td>
      <td>But after decades of decay, will it happen quickly enough to save Ukraine?</td>
      <td>https://www.theatlantic.com/ideas/archive/2023/03/american-defense-manufacturing-ukraine-aid-arkansas/673327/?utm_source=feed</td>
      <td>None</td>
      <td>2023-03-09T18:04:56Z</td>
      <td>Lockheed Martin builds its advanced mobile rocket launchers in a converted diaper factory, of all places. When I visited the plant in southern Arkansas at the end of February, I found it humming with… [+9948 chars]</td>
      <td>&lt;!DOCTYPE html&gt;&lt;html lang="en" dir="ltr"&gt;&lt;head&gt;&lt;meta charSet="utf-8"/&gt;&lt;meta name="viewport" content="width=device-width,initial-scale=1"/&gt;&lt;link rel="icon" href="https://cdn.theatlantic.com/_next/static/images/favicon-3888b0e329526a975703e3059a02b92d.ico"/&gt;&lt;link rel="apple-touch-icon" href="https://cdn.theatlantic.com/_next/static/images/apple-touch-icon-default-b504d70343a9438df64c32ce339c7ebc...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>{'id': 'google-news', 'name': 'Google News'}</td>
      <td>None</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure - CBS News</td>
      <td>The CHIPS Act: Rebuilding America's technological infrastructure  CBS News</td>
      <td>https://consent.google.com/ml?continue=https://news.google.com/rss/articles/CBMiXGh0dHBzOi8vd3d3LmNic25ld3MuY29tL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv0gFgaHR0cHM6Ly93d3cuY2JzbmV3cy5jb20vYW1wL25ld3MvdGhlLWNoaXBzLWFjdC1yZWJ1aWxkaW5nLWFtZXJpY2FzLXRlY2hub2xvZ2ljYWwtaW5mcmFzdHJ1Y3R1cmUv?oc%3D5&amp;gl=FR&amp;hl=en-US&amp;pc=n&amp;src=1</td>
      <td>None</td>
      <td>2023-03-05T14:11:07Z</td>
      <td>We use cookies and data to&lt;ul&gt;&lt;li&gt;Deliver and maintain Google services&lt;/li&gt;&lt;li&gt;Track outages and protect against spam, fraud, and abuse&lt;/li&gt;&lt;li&gt;Measure audience engagement and site statistics to unde… [+1131 chars]</td>
      <td>&lt;!DOCTYPE html&gt;\n&lt;html lang="en-US" class="theme--responsive\n  device--type-desktop\n  device--platform-web\n  device--size-\n  content--type-article\n  context--slug-the-chips-act-rebuilding-americas-technological-infrastructure\n  page--type-news-item\n     is--show \n      show--name-sunday-morning\n  \n  \n  edition--us\n\n  \n  \n   has__article-hero-video-player"&gt;&lt;head prefix="og: http:...</td>
    </tr>
  </tbody>
</table>
</div>



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




    ['files/15.txt',
     'files/14.txt',
     'files/16.txt',
     'files/17.txt',
     'files/13.txt',
     'files/12.txt',
     'files/10.txt',
     'files/11.txt',
     'files/9.txt',
     'files/8.txt',
     'files/5.txt',
     'files/4.txt',
     'files/6.txt',
     'files/7.txt',
     'files/3.txt',
     'files/2.txt',
     'files/1.txt',
     'files/20.txt',
     'files/19.txt',
     'files/18.txt']



### Let's run the NER on a single article first


```python
filepath = "files/3.txt"
text = open(filepath, encoding='utf-8').read()
document = nlp(text)
```

### Let's use displacy to visualize our results


```python
displacy.render(document, style="ent")
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
-
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 chip war: 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 moves to restrict some tech exports - 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 NewsBBC 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HomepageSkip
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 to contentAccessibility 
<mark class="entity" style="background: #bfeeb7; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    HelpYour accountHomeNewsSportReelWorklifeTravelFutureMore
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PRODUCT</span>
</mark>
 menuMore menuSearch BBCHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsClose 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    menuBBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 NewsMenuHomeWar in 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    UkraineClimateVideoWorldUS &amp; CanadaUKBusinessTechScienceMoreStoriesEntertainment &amp; ArtsHealthIn
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>

<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    PicturesReality CheckWorld News
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 TVNewsbeatLong ReadsBusinessMarket 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    DataNew EconomyNew Tech EconomyCompaniesTechnology
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 of BusinessEconomyCEO 
<mark class="entity" style="background: #bfeeb7; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    SecretsGlobal TradeCost
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PRODUCT</span>
</mark>
 of LivingUS-
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 chip war: 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 moves to restrict some tech exportsPublished9 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    MarchShareclose
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 panelShare pageCopy linkAbout sharingImage source, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Getty ImagesBy Annabelle  
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
LiangBusiness reporterThe 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 government is to put restrictions on the country's &quot;most advanced&quot; microchip technology exports to protect national security, following a similar move by the US.It will include products by chip equipment maker 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ASML
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, a key firm in the global microchip supply chain.In response, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 has launched a formal complaint against the move.It said it hoped the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 would not &quot;follow the abuse of export control measures by certain countries&quot;.
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 has frequently called the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 a &quot;tech hegemony&quot; in response to export controls imposed by 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Washington
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
.Semiconductors, which power everything from mobile phones to military hardware, are at the centre of a bitter dispute between the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
.A spokeswoman for 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Chinese Foreign Ministry
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Mao Ning
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, said the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 move aimed to deprive 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 of its right to develop.
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dexter Roberts
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, a senior fellow at the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Washington
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
-based 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Atlantic Council
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 think tank, told the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 that the decision by the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 was &quot;a real step forward, a real victory for the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and also very bad news for China&quot;.&quot;US-
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 relations are already in a pretty bad place. This clearly will make things even worse.&quot;The measures will affect &quot;very specific technologies in the semiconductor production cycle,&quot; the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 trade minister 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Liesje Schreinemacher
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 said.&quot;The 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 considers it necessary on national and international security grounds that this technology is brought under control as soon as possible,&quot; she said in a letter to lawmakers on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Wednesday
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
.
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Ms Schreinemacher
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 added that the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 government had considered &quot;the technological developments and geopolitical context,&quot; without naming 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 or ASML.Under the new rules, companies would have to apply for licences to export technology including &quot;the most advanced 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Deep Ultra Violet
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 (
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    DUV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
) immersion lithography and deposition&quot;.
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ASML
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 said in a statement that it expects the restrictions to apply to its &quot;most advanced immersion 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    DUV
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 systems&quot;.The company added that &quot;based on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    today
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
's announcement, our expectation of the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 government's licensing policy, and the current market situation, we do not expect these measures to have a material effect on our financial outlook.&quot;Lithography machines use lasers to print miniscule patterns on silicon as part of the manufacturing process of microchips.Since 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    2019
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 the 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Dutch
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 government has stopped 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    ASML
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 from selling its most advanced lithography machines to 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
.In 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    October
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Washington
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 announced that it would require licences for companies exporting chips to 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 using 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 tools or software, no matter where they are made in the world.The 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 has been pushing the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Netherlands
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Japan
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 to adopt similar restrictions.Meanwhile, 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    South Korea's
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 trade ministry raised concerns over the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 policy on semiconductors 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    earlier this week
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
.&quot;The 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    South Korean
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 government will make it clear that the conditions of 
<mark class="entity" style="background: #ff8197; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the Chips Act
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LAW</span>
</mark>
 could deepen business uncertainties, violate companies' management and technology rights as well as make 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the United States
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 less attractive as an investment option,&quot; the ministry said.
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    South Korea
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 is home to major microprocessor manufacturers including the world's biggest memory chip maker 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Samsung
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
.You may also be interested in:This video can not be playedTo play this video you need to enable 
<mark class="entity" style="background: #bfeeb7; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    JavaScript
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PRODUCT</span>
</mark>
 in your browser.Media caption, Watch: How the semiconductor shortage could be a problem for 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    youRelated TopicsCompaniesNetherlandsSemiconductorsChina-US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 relationsMore on this storyHow ASML became 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Europe
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
’s most valuable tech firm21 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    FebruaryMajor
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 microchip firm says 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 employee stole data16 FebruaryHow 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 sneaks out 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    America
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
's technology secrets16 JanuaryThe 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 is beating 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 in the battle for 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    chips13
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 JanuaryHow the 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    US
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
-
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    China
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 chip war is playing out16 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    December 2022Top
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 StoriesUS going to hell, says 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 after being 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    chargedPublished2 hours
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 agoTrump's historic court appearance in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    77 seconds
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
. 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    VideoTrump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's historic court appearance in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    77 secondsPublished14 hours
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 agoIs 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Taiwan
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 in danger of being loved to death?Published11 hours agoFeaturesThe 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    one
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 thing 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's day in court tells usDonald 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 has been charged. What happens next?How the world reacted to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's arrest‘You aimed at my eyes but my heart still 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    beats’Is Taiwan
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 in danger of being loved to death?Super 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Mario
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
: 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Jack Black
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 on rise of game adaptationsInside the life coaching cult that takes over livesWhat clues does new 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Russian
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 bomb footage 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    reveal?'I'm
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 not selling. Am I part of the housing problem?'Elsewhere on the BBCWhy method acting is so controversialThe return of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the X-planes?The NYC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 restaurant with 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    only one
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 tableMost Read1US going to hell, says 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 after being charged2Sturgeon's husband arrested in 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    SNP
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 finance 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    probe3Who
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 is 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Karen McDougal
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and is she linked to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's case?4What the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    34
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 felony charges against 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 reveal5Rupert Murdoch's engagement called off - reports6'Queen 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Camilla
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
' title debuts on coronation invite7How the world reacted to 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Trump
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
's arrest8Australian jailed for abducting girl from campsite9Liberals win control of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Wisconsin Supreme
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 Court10‘You aimed at my eyes but my heart still beats’BBC News ServicesOn your mobileOn smart speakersGet news alertsContact 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBC NewsHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsTerms
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    UseAbout
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 the BBCPrivacy PolicyCookiesAccessibility HelpParental GuidanceContact the BBCGet Personalised 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    NewslettersWhy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 you can trust the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBCAdvertise
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 with usAdChoices / Do Not Sell My Info© 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    2023
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>

<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
. The 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    BBC
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 is not responsible for the content of external sites. Read about our approach to external linking.</div></span>


### Let's see a list of the identified entities


```python
document.ents
```




    (US,
     China,
     Netherlands,
     BBC,
     HomepageSkip,
     HelpYour accountHomeNewsSportReelWorklifeTravelFutureMore,
     menuBBC,
     UkraineClimateVideoWorldUS & CanadaUKBusinessTechScienceMoreStoriesEntertainment & ArtsHealthIn,
     PicturesReality CheckWorld News,
     DataNew EconomyNew Tech EconomyCompaniesTechnology,
     SecretsGlobal TradeCost,
     China,
     Netherlands,
     MarchShareclose,
     Getty ImagesBy Annabelle  ,
     Dutch,
     ASML,
     China,
     Netherlands,
     China,
     US,
     Washington,
     US,
     China,
     the Chinese Foreign Ministry,
     Mao Ning,
     Dutch,
     China,
     Dexter Roberts,
     Washington,
     Atlantic Council,
     BBC,
     Netherlands,
     US,
     China,
     Dutch,
     Liesje Schreinemacher,
     Netherlands,
     Wednesday,
     Ms Schreinemacher,
     Dutch,
     China,
     Deep Ultra Violet,
     DUV,
     ASML,
     DUV,
     today,
     Dutch,
     2019,
     Dutch,
     ASML,
     China,
     October,
     Washington,
     China,
     US,
     US,
     Netherlands,
     Japan,
     South Korea's,
     US,
     earlier this week,
     South Korean,
     the Chips Act,
     the United States,
     South Korea,
     Samsung,
     JavaScript,
     youRelated TopicsCompaniesNetherlandsSemiconductorsChina-US,
     Europe,
     FebruaryMajor,
     China,
     China,
     America,
     US,
     China,
     chips13,
     US,
     China,
     December 2022Top,
     Trump,
     chargedPublished2 hours,
     77 seconds,
     VideoTrump,
     77 secondsPublished14 hours,
     Taiwan,
     one,
     Trump,
     Trump,
     Trump,
     beats’Is Taiwan,
     Mario,
     Jack Black,
     Russian,
     reveal?'I'm,
     the X-planes?The NYC,
     only one,
     Trump,
     SNP,
     probe3Who,
     Karen McDougal,
     Trump,
     34,
     Trump,
     Camilla,
     Trump,
     Wisconsin Supreme,
     BBC NewsHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsTerms,
     UseAbout,
     NewslettersWhy,
     BBCAdvertise,
     2023,
     BBC,
     BBC)



### Let's add the entity label next to each entity: 


```python
for named_entity in document.ents:
    print(named_entity, named_entity.label_)
```

    US GPE
    China GPE
    Netherlands GPE
    BBC ORG
    HomepageSkip ORG
    HelpYour accountHomeNewsSportReelWorklifeTravelFutureMore PRODUCT
    menuBBC ORG
    UkraineClimateVideoWorldUS & CanadaUKBusinessTechScienceMoreStoriesEntertainment & ArtsHealthIn ORG
    PicturesReality CheckWorld News ORG
    DataNew EconomyNew Tech EconomyCompaniesTechnology ORG
    SecretsGlobal TradeCost PRODUCT
    China GPE
    Netherlands GPE
    MarchShareclose PERSON
    Getty ImagesBy Annabelle   PERSON
    Dutch NORP
    ASML ORG
    China GPE
    Netherlands GPE
    China GPE
    US GPE
    Washington GPE
    US GPE
    China GPE
    the Chinese Foreign Ministry ORG
    Mao Ning PERSON
    Dutch NORP
    China GPE
    Dexter Roberts PERSON
    Washington GPE
    Atlantic Council ORG
    BBC ORG
    Netherlands GPE
    US GPE
    China GPE
    Dutch NORP
    Liesje Schreinemacher PERSON
    Netherlands GPE
    Wednesday DATE
    Ms Schreinemacher PERSON
    Dutch NORP
    China GPE
    Deep Ultra Violet ORG
    DUV ORG
    ASML ORG
    DUV ORG
    today DATE
    Dutch NORP
    2019 DATE
    Dutch NORP
    ASML ORG
    China GPE
    October DATE
    Washington GPE
    China GPE
    US GPE
    US GPE
    Netherlands GPE
    Japan GPE
    South Korea's GPE
    US GPE
    earlier this week DATE
    South Korean NORP
    the Chips Act LAW
    the United States GPE
    South Korea GPE
    Samsung ORG
    JavaScript PRODUCT
    youRelated TopicsCompaniesNetherlandsSemiconductorsChina-US ORG
    Europe LOC
    FebruaryMajor ORG
    China GPE
    China GPE
    America GPE
    US GPE
    China GPE
    chips13 CARDINAL
    US GPE
    China GPE
    December 2022Top DATE
    Trump PERSON
    chargedPublished2 hours TIME
    77 seconds TIME
    VideoTrump ORG
    77 secondsPublished14 hours TIME
    Taiwan GPE
    one CARDINAL
    Trump ORG
    Trump ORG
    Trump ORG
    beats’Is Taiwan ORG
    Mario PERSON
    Jack Black PERSON
    Russian NORP
    reveal?'I'm GPE
    the X-planes?The NYC ORG
    only one CARDINAL
    Trump PERSON
    SNP ORG
    probe3Who PERSON
    Karen McDougal PERSON
    Trump ORG
    34 CARDINAL
    Trump ORG
    Camilla ORG
    Trump ORG
    Wisconsin Supreme ORG
    BBC NewsHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsTerms ORG
    UseAbout ORG
    NewslettersWhy ORG
    BBCAdvertise ORG
    2023 DATE
    BBC ORG
    BBC ORG


### Let's filter for the results to see all entities labelled as "PERSON":


```python
for named_entity in document.ents:
    if named_entity.label_ == "PERSON":
        print(named_entity)
```

    MarchShareclose
    Getty ImagesBy Annabelle  
    Mao Ning
    Dexter Roberts
    Liesje Schreinemacher
    Ms Schreinemacher
    Trump
    Mario
    Jack Black
    Trump
    probe3Who
    Karen McDougal


### Let's filter for the results to see all entities labelled as "NORP":


```python
for named_entity in document.ents:
    if named_entity.label_ == "NORP":
        print(named_entity)
```

    Dutch
    Dutch
    Dutch
    Dutch
    Dutch
    Dutch
    South Korean
    Russian


### Let's filter for the results to see all entities labelled as "GPE":


```python
for named_entity in document.ents:
    if named_entity.label_ == "GPE":
        print(named_entity)
```

    US
    China
    Netherlands
    China
    Netherlands
    China
    Netherlands
    China
    US
    Washington
    US
    China
    China
    Washington
    Netherlands
    US
    China
    Netherlands
    China
    China
    Washington
    China
    US
    US
    Netherlands
    Japan
    South Korea's
    US
    the United States
    South Korea
    China
    China
    America
    US
    China
    US
    China
    Taiwan
    reveal?'I'm


### Let's filter for the results to see all entities labelled as "LOC":


```python
for named_entity in document.ents:
    if named_entity.label_ == "LOC":
        print(named_entity)
```

    Europe


### Let's filter for the results to see all entities labelled as "FAC":


```python
for named_entity in document.ents:
    if named_entity.label_ == "FAC":
        print(named_entity)
```

### Let's filter for the results to see all entities labelled as "ORG":


```python
for named_entity in document.ents:
    if named_entity.label_ == "ORG":
        print(named_entity)
```

    BBC
    HomepageSkip
    menuBBC
    UkraineClimateVideoWorldUS & CanadaUKBusinessTechScienceMoreStoriesEntertainment & ArtsHealthIn
    PicturesReality CheckWorld News
    DataNew EconomyNew Tech EconomyCompaniesTechnology
    ASML
    the Chinese Foreign Ministry
    Atlantic Council
    BBC
    Deep Ultra Violet
    DUV
    ASML
    DUV
    ASML
    Samsung
    youRelated TopicsCompaniesNetherlandsSemiconductorsChina-US
    FebruaryMajor
    VideoTrump
    Trump
    Trump
    Trump
    beats’Is Taiwan
    the X-planes?The NYC
    SNP
    Trump
    Trump
    Camilla
    Trump
    Wisconsin Supreme
    BBC NewsHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsTerms
    UseAbout
    NewslettersWhy
    BBCAdvertise
    BBC
    BBC


### Now, let's define a function that will run this process across our entire collection of texts:


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

    [{'File_name': 'files/15.txt', 'Entity_type': ['CARDINAL', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON'], 'Entity_identified': ['403', 'Request', 'CloudFront', 'CloudFront', 'CloudFront', 'Request', 'ksF1nKbP2kaMk6WLD8ZzAd8dibgqi9k0KFdn5eagZ3MDjdzDA9wZiw==']}, {'File_name': 'files/14.txt', 'Entity_type': ['ORG', 'GPE', 'DATE', 'ORG', 'ORG', 'WORK_OF_ART', 'PERSON', 'PERSON', 'PERSON', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'PERSON', 'PRODUCT', 'DATE', 'EVENT', 'ORG', 'GPE', 'DATE', 'PERSON', 'ORG', 'GPE', 'DATE', 'TIME', 'CARDINAL', 'ORG', 'ORDINAL', 'GPE', 'DATE', 'PERSON', 'ORG', 'DATE', 'DATE', 'DATE', 'PERSON', 'PERSON', 'DATE', 'DATE', 'ORDINAL', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE', 'GPE', 'NORP', 'MONEY', 'PERSON', 'GPE', 'ORG', 'QUANTITY', 'DATE', 'PERSON', 'ORG', 'ORG', 'NORP', 'PERSON', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'CARDINAL', 'ORG', 'MONEY', 'PERSON', 'DATE', 'PERSON', 'PERSON', 'DATE', 'ORG', 'GPE', 'CARDINAL', 'ORG', 'CARDINAL', 'PERSON', 'WORK_OF_ART', 'ORG', 'ORG', 'MONEY', 'MONEY', 'MONEY', 'MONEY', 'MONEY', 'MONEY', 'MONEY', 'WORK_OF_ART', 'ORG', 'WORK_OF_ART', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'DATE', 'EVENT', 'ORG', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'PRODUCT', 'CARDINAL', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'DATE', 'ORG', 'DATE', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'GPE', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'NORP', 'WORK_OF_ART', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'GPE', 'CARDINAL', 'ORG', 'GPE', 'CARDINAL', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'DATE', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'GPE', 'ORG', 'PERSON', 'PERSON', 'WORK_OF_ART', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'PRODUCT', 'NORP'], 'Entity_identified': ['Tesla', 'Mexico', 'next year', 'Luxury News', 'Hybrid News\nElectric News', 'Weird Car News', 'Junkyard Gems\n', 'Newsletters\nPhotos\nNews', 'Car Reviews', 'Most Reliable Cars\nBuying Guides\nVideos', 'Car Values\nCar Finder\nCompare Vehicles\nDealers', 'Car Insurance\nRepair Shops', 'Buyer', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Jeep', 'Lexus', 'Toyota', 'Volvo', 'Cars for Sale\n\n\nCars for Sale\n\n\nNew Cars for Sale\nUsed Cars for Sale\n\n\n\nDeals\n\n\nDeals\n\n\n', 'Door Openers', 'Best Power Generators\nBest Car Vacuums', 'Gas Prices Near You\n\n\n\n\nElectric Vehicles\n\n\n\n\n\n\n\nMake\n\n\nModel\n\nCancel\nResearch\n\n\n\nFacebook\n\n\n', 'Twitter', 'Cancel\nSearch', '2023', 'New York Auto Show', 'https://www.parsintl.com/publication/autoblog/\n\n\n\n\n\n\n\n\n\t\t\t\t\t\t\t\t\tReport\n\n\nReport', 'Mexico', 'next year', 'Elon Musk', 'Tesla', 'Mexico', 'Mar 7th 2023', '8:43AM', '0', 'Tesla', 'first', 'Mexico', 'next year', 'Nuevo Leon', 'U.S.-Mexico', 'Monday', 'this very month', 'March', 'Nuevo Leon', 'Samuel Garcia', 'next year', '2024', 'first', 'Tesla', 'Elon Musk', 'last week', 'Austin', 'Texas', 'Mexico', 'Mexican', '$5 billion', 'Garcia', 'Santa Catarina', 'Monterrey', 'several thousand acres', 'last week', 'Garcia', 'Tesla', 'Tesla', 'Mexican', 'Andres Manuel Lopez Obrador', 'Lopez Obrador', 'Tesla', 'Mexico', 'Tesla\n\n\n\n\t\t\t\t\t\t\t\t\t\t\t\t\tElectric\n\n\n\n\t\t\t\t\t\t\t\t\t\t\t\tMexico', 'License\nLicense\n\n\n\n\n\nView 0', 'License\nLicense', '2023', 'Mitsubishi Colt', '5', 'Volvo', 'less than $100', 'Junkyard Gem:', '1993', 'Chevrolet', 'Silverado Crew-Cab', '2024', 'Toyota', 'Tacoma', '10', 'Genesis', '7', 'Follow Us', 'Facebook Share', 'Flipboard Share', 'Feeds Share\n\n\n\n\n\n\n\n\n\n\n\n\n\t\t\t\t\tResearch', '$29,995 - $', '44,995', '37,090', '83,790', '$52,000 - $', '75,600', '$49,000 - $57,000', 'View More \n\n\n\n\n\nElectric Vehicles', 'Volkswagen', 'Preview Drive', 'VW', '2023', 'Mercedes-Benz EQE', 'First Drive Review: Easy', '2023', 'New York Auto Show Live Updates', 'Kia EV9', 'Autoblog Daily Roundup\nNews', 'Reviews', 'Photos', 'Videos', 'Honda', 'Civic', '2023', 'Ford', 'Chevrolet', '1500', 'Ford', 'Toyota', '4Runner', 'Popular Used Vehicles\n\n\n\n\t\t\t\t\t\t\t   2021', 'Jeep Grand Cherokee', 'Toyota', 'Honda', 'Accord\n\n\n\n\t\t\t\t\t\t\t   ', '2021', 'Chevrolet', '1500\n\n\n\n\t\t\t\t\t\t\t   ', '2017', 'Chevrolet', '2018', 'Jeep Grand Cherokee', 'Honda', 'Accord\n\n\n\n\t\t\t\t\t\t\t   2020', 'Honda', 'Civic', 'Volkswagen', 'Chevrolet', 'Silverado', '2500HD', 'Popular Electric Vehicles\n\n\n\n\t\t\t\t\t\t\t   2023', 'BMW', 'GMC HUMMER', 'GMC HUMMER', '2023', 'Rivian', 'Popular Truck Vehicles\n\n\n\n\t\t\t\t\t\t\t   2023', 'Toyota', 'Toyota', 'Chevrolet', '1500\n\n\n\n\t\t\t\t\t\t\t   ', 'Ford', 'Ford', 'Toyota', 'Chevrolet', 'Silverado', '2023', 'Chevrolet', 'Colorado', '1500', 'Popular Crossover Vehicles\n\n\n\n\t\t\t\t\t\t\t   2023', '2023', 'Toyota', 'Honda', 'CR-V\n\n\n\n\t\t\t\t\t\t\t   2023', 'Honda', 'Volkswagen Tiguan\n\n\n\n\t\t\t\t\t\t\t   2023', 'Toyota', 'Highlander', '2022', 'BMW M760', '2023', 'Porsche', 'Mercedes-Benz E-Class\n\n\n\n\t\t\t\t\t\t\t   2023', 'Mercedes-Benz G-Class\n\n\n\n\t\t\t\t\t\t\t   2023', 'Mercedes-Benz C-Class\n\n\n\n\t\t\t\t\t\t\t   2023', 'Mercedes-Benz GLE 350\n\n\n\n\t\t\t\t\t\t\t   ', 'Mercedes-Benz S-Class\n\n\n\n\t\t\t\t\t\t\t   2023', 'Acura', 'Popular Hybrid Vehicles\n\n\n\n\t\t\t\t\t\t\t   2023', 'Ford', 'Ford', 'Ford', 'Toyota', 'Honda', 'Accord Hybrid', 'Toyota', 'Toyota', 'Ford', 'Honda', 'CR-V Hybrid', 'Toyota', 'Crown', 'Honda', 'Chevrolet', 'Dodge', 'Cadillac', 'Hyundai', 'Jeep', 'Lexus\n\n\n\n\t\t\t\t\t\t\t   Subaru', 'Sitemap', 'Us', 'Us', 'iTunes\n\n\nArchives\n\n\n\n\nAdvertising', 'Dashboard', 'Trademarks', 'Facebook Share', 'RSS Share', '2023 Yahoo Inc.', 'JavaScript', 'Autoblog', 'JavaScript', 'Cancel\nChange Name', 'Reddit']}, {'File_name': 'files/16.txt', 'Entity_type': ['GPE', 'PERSON', 'ORDINAL', 'GPE', 'ORG', 'ORG', 'FAC', 'WORK_OF_ART', 'ORG', 'PERSON', 'ORDINAL', 'GPE', 'NORP', 'GPE', 'PERSON', 'NORP', 'PERSON', 'FAC', 'GPE', 'GPE', 'DATE', 'PERSON', 'CARDINAL', 'DATE', 'GPE', 'GPE', 'GPE', 'PERSON', 'LOC', 'GPE', 'PERSON', 'NORP', 'PERSON', 'DATE', 'DATE', 'NORP', 'PERSON', 'PERSON', 'PERSON', 'NORP', 'PERSON', 'DATE', 'PERSON', 'GPE', 'DATE', 'ORG', 'PERSON', 'ORG', 'GPE', 'LOC', 'PERSON', 'GPE', 'ORDINAL', 'DATE', 'GPE', 'GPE', 'CARDINAL', 'GPE', 'GPE', 'DATE', 'MONEY', 'DATE', 'GPE', 'GPE', 'CARDINAL', 'MONEY', 'DATE', 'NORP', 'GPE', 'ORDINAL', 'PERCENT', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'PERSON', 'ORG', 'ORG', 'GPE', 'NORP', 'GPE', 'PERSON', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORG', 'GPE', 'GPE', 'GPE', 'PERSON', 'PERSON', 'ORG', 'GPE', 'PERSON', 'GPE', 'DATE', 'ORDINAL', 'GPE', 'GPE', 'DATE', 'ORDINAL', 'NORP', 'PERSON', 'NORP', 'PERSON', 'DATE', 'PERSON', 'ORG', 'LOC', 'GPE', 'GPE', 'GPE', 'GPE', 'NORP', 'DATE', 'ORG', 'GPE', 'ORDINAL', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'MONEY', 'NORP', 'MONEY', 'DATE', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'GPE', 'GPE', 'DATE', 'MONEY', 'MONEY', 'GPE', 'GPE', 'PERSON', 'GPE', 'NORP', 'PERSON', 'GPE', 'ORG', 'PERSON', 'ORG', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'LOC', 'GPE', 'NORP', 'GPE', 'PERSON', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'PERSON', 'GPE', 'PERSON', 'GPE', 'PERSON', 'ORG', 'ORG', 'LOC', 'PERSON', 'GPE', 'ORDINAL', 'ORG', 'PERSON', 'PERSON', 'LOC', 'GPE', 'PERCENT', 'NORP', 'GPE', 'ORG', 'GPE', 'LOC', 'GPE', 'ORG', 'DATE', 'CARDINAL', 'NORP', 'QUANTITY', 'QUANTITY', 'GPE', 'NORP', 'GPE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'LOC', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'NORP', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'LANGUAGE', 'CARDINAL', 'ORG'], 'Entity_identified': ['Malaysia', 'Anwar', 'first', 'China', 'International Trade', 'Al Jazeera', 'Skip linksSkip', 'Contentplay Live Show', 'TradeMalaysia', 'Anwar', 'first', 'China', 'Malaysian', 'China', 'Xi Jinping', 'Malaysian', 'Anwar Ibrahim', 'Rizal Park', 'Manila', 'Philippines', 'Thursday, March 2, 2023', 'Lisa Marie David/Pool Photo', '30', '202330', '2023Kuala Lumpur', 'Malaysia', 'Malaysia', 'Anwar Ibrahim', 'the South China Sea', 'China', 'Anwar', 'Chinese', 'Xi Jinping', 'Friday', 'four-day', 'Chinese', 'Li Qiang', 'Anwar', 'Xi', 'Malaysian', 'Zambry Abdul Kadir', 'Wednesday', 'Anwar', 'China', 'Wednesday', 'National People’s Congress', 'Zhao Leji', 'the China Communications Construction Company', 'Malaysia', 'East Coast Rail Link', 'Anwar', 'Malaysia', '10th', 'November', 'United States-China', 'Malaysia', 'two', 'China', 'Malaysia', '14 consecutive years', '203.6bn', '2022', 'Kuala Lumpur', 'US', 'two', '72.9bn', 'last year', 'Chinese', 'Malaysia', 'sixth', '6.3 percent', 'US', 'China', 'Malaysia', 'China', 'US', 'US', 'China', 'Hoo Chiew Ping', 'the National University of Malaysia', 'Al Jazeera', 'US', 'Malaysian', 'China', 'Ngeow Chow Bing', 'Universiti Malaya’s Institute of China Studies', 'Al Jazeera', 'Shahriman Lockman', 'Kuala Lumpur', 'Institute of Strategic and International Studies', 'ISIS', 'Malaysia', 'US', 'China', 'Anwar', 'Lockman', 'Al Jazeera', 'China', 'Anwar', 'Beijing', 'this year', '10th', 'Malaysia', 'China', 'next year', '50th', 'Malaysian', 'Anwar Ibrahim', 'Chinese', 'Xi Jinping', 'Friday', 'Alexei Maishev/Kremlin', 'Reuters', 'Southeast Asia', 'US', 'China', 'Malaysia', 'US', 'Chinese', '2019', 'Nomura', 'Malaysia', 'fourth', 'US', 'Vietnam', 'Taiwan', 'Chile', 'China', 'Malaysia', '9.7 billion', 'Malaysian', '2.2bn', '2022', '23.5 percent', '7.9 billion', '1.8bn', '2021', 'US', 'Malaysia', 'last year', '43.9 billion', '9.9bn', 'Singapore', 'Japan', 'Anwar', 'Malaysia', 'Chinese', 'Yeah Kim Leng', 'Malaysia', 'Sunway University', 'Anwar', 'Al Jazeera', 'Malaysia', 'China', 'China', 'Malaysia', 'Malaysia', 'China', 'The South China Sea', 'China', 'Southeast Asian', 'Malaysia', 'Ritchie B. Tongo', 'AP', 'Anwar', 'China', 'Uighurs', 'Muslims', 'Anwar', 'China', 'Anwar', 'China', 'Lockman', 'the Institute of Strategic and International Studies', 'Uighurs', 'the South China Sea', 'Anwar', 'Malaysia', 'first', 'Ngeow of Universiti Malaya', 'Anwar', 'Uighur', 'South China Sea', 'Beijing', 'more than 90 percent', 'Southeast Asian', 'Malaysia', 'Chinese Coast Guard', 'Malaysia', 'the South China Sea', 'Malaysia', 'Petronas', '2021', '16', 'Chinese', '60 nautical miles', '112 km', 'Kuala Lumpur', 'Chinese', 'Beijing', 'China', 'Kuala Lumpur', 'Hoo', 'the University of Malaysia', 'Anwar', 'South China Sea', 'One', 'the Coast Guard', 'PLAAF', 'People’s Liberation Army Air Force', 'Malaysian', 'airspace', 'Hoo', 'Al Jazeeraaj-logoaj-logoaj-logoAboutShow', 'UsCode of EthicsTerms', 'usAppsChannel', 'TipOur', 'Jazeera ArabicAl Jazeera', 'Jazeera Investigative UnitAl Jazeera MubasherAl Jazeera DocumentaryAl', 'BalkansAJ+Our NetworkShow', 'Jazeera Centre', 'StudiesAl Jazeera', 'InstituteLearn ArabicAl Jazeera Centre for Public Liberties & Human', 'RightsAl Jazeera', 'ForumAl Jazeera', 'Al Jazeera', 'English', '2023', 'Al Jazeera']}, {'File_name': 'files/17.txt', 'Entity_type': ['LAW', 'PERSON', 'PRODUCT', 'GPE', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'PERSON', 'PERSON', 'ORG', 'GPE', 'ORG', 'LAW', 'ORG', 'GPE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'GPE', 'ORG', 'DATE', 'GPE', 'ORG', 'PERSON', 'ORG', 'DATE', 'ORG', 'NORP', 'ORG', 'ORG', 'DATE', 'PERSON', 'NORP', 'GPE', 'LAW', 'GPE', 'ORG', 'ORG', 'ORDINAL', 'CARDINAL', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'ORG', 'GPE', 'GPE', 'ORG', 'NORP', 'DATE', 'GPE', 'DATE', 'GPE', 'ORG', 'PERSON', 'GPE', 'ORG', 'GPE', 'ORG', 'GPE', 'GPE', 'DATE', 'LAW', 'GPE', 'GPE', 'CARDINAL', 'ORG', 'ORG', 'ORDINAL', 'LAW', 'ORG', 'ORG', 'DATE', 'GPE', 'GPE', 'GPE', 'DATE', 'PERSON', 'ORG', 'ORG', 'NORP', 'GPE', 'DATE', 'ORG', 'NORP', 'NORP', 'ORG', 'ORG', 'GPE', 'PERSON', 'GPE', 'ORG', 'GPE', 'DATE', 'GPE', 'GPE', 'GPE', 'NORP', 'ORG', 'NORP', 'ORG', 'ORG', 'NORP', 'NORP', 'NORP', 'NORP', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'PERSON', 'LAW', 'GPE', 'NORP', 'PERSON', 'GPE', 'PERSON', 'GPE', 'ORG', 'ORG', 'ORG', 'NORP', 'LAW', 'ORG', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'NORP', 'GPE', 'NORP', 'ORG', 'DATE', 'NORP', 'PRODUCT', 'NORP', 'CARDINAL', 'ORG', 'ORG', 'DATE', 'NORP', 'GPE', 'ORG', 'GPE', 'GPE', 'FAC', 'CARDINAL', 'GPE', 'PERSON', 'PERSON', 'GPE', 'GPE', 'GPE', 'LOC', 'LOC', 'LOC', 'PERSON', 'GPE', 'PERSON', 'GPE', 'ORG', 'ORG', 'GPE', 'ORG', 'ORG', 'GPE', 'NORP', 'PERSON', 'PERSON', 'GPE', 'GPE', 'GPE', 'NORP', 'GPE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'LAW'], 'Entity_identified': ['The RESTRICT Act', 'Mark Warner', 'TikTok - Rest of World', 'U.S.', 'TikTok', 'Latest Stories\nAccess & Connectivity\nCreators & Communities', 'The Platform Economy\n \n\n\n\n\n\nLearn', 'Team news', 'Email', 'Dark Mode', 'TikTok Domination\nMeet', 'U.S.', 'TikTok', 'the RESTRICT Act', 'Warner', 'U.S.', 'China', 'Greg Kahn', 'Greg Kahn', 'Russell Brandom', '29 March 2023', 'Washington', 'TikTok', 'years', 'the United States', 'TikTok', 'Shou Zi Chew’s', 'the House Energy and Commerce Committee', 'earlier this month', 'TikTok', 'Chinese', 'TikTok', 'Senate', 'earlier this month', 'Mark Warner', 'Democrat', 'Virginia', 'the RESTRICT Act', 'U.S.', 'RESTRICT', 'TikTok', 'first', 'six', 'China', 'Russia', 'Iran', 'North Korea', 'Cuba', 'Venezuela', 'RESTRICT', 'U.S.', 'U.S.', 'Apple', 'Chinese', 'decades', 'U.S.', 'the days', 'Washington', 'Rest of World', 'Warner', 'China', 'TikTok', 'U.S.', 'Rest of World', 'Brazil', 'Nigeria', 'five or 10 years', 'The RESTRICT Act', 'U.S.', 'U.S.', 'One', 'TikTok', 'quantum', 'first', 'The RESTRICT Act', 'TikTok', 'TikTok', 'the past 20 years', 'U.S.', 'China', 'U.S.', 'the beginning of the first decade', 'Xi', 'CCP', 'Chinese Communist Party', 'Orwellian', 'China', '2017', 'CCP', 'Chinese', 'American', 'CCP', 'Uyghurs', 'Hong Kong', 'Beyond TikTok', 'U.S.', 'TikTok', 'America', 'Three years ago', 'India', 'Canada', 'U.K.', 'Danish', 'TikTok', 'Chinese', 'TikTok', 'TikTok', 'American', 'Indian', 'French', 'Brazilian', 'Russia', 'China', 'Iran', 'North Korea', 'Cuba', 'Mark Warner', 'the RESTRICT Act', 'U.S.', 'Chinese', 'Xi', 'Russia', 'Xi', 'Taiwan', 'House', 'YouTube', 'TikTok', 'American', 'Section 230', 'TikTok', 'the end of the day', 'Meta', 'the end of the day', 'Meta', 'the Communist Party of China', 'Meta', 'the end of the day', 'Meta', 'the Communist Party of China', 'Chinese', 'China', 'American', 'Google', 'years', 'Chinese', 'Twitter', 'Chinese', 'One', 'Huawei', 'CHIPS', 'the 21st century', 'Chinese', 'China', 'quantum', 'Russia', 'China', 'Belt and Road Initiative', 'three', 'China', 'Covid', 'Covid', 'China', 'Russia', 'China', 'Africa', 'South America', 'Asia', 'Biden', 'Singapore', 'Russell Brandom', 'U.S.', 'Rest of World', 'Creators & Communities', 'Nigeria', 'Digital', 'TikTok', 'India', 'Indians', 'Kusha Kapila', 'Allahbadia', 'Instagram', 'U.S.', 'China', 'American', 'China', 'Taiwan', 'Johanna M. Costigan', 'Aidan Powers-Riggs', 'Newsletters\nLabs\nContact us\n \n\t\t\t', 'World 2020–2023']}, {'File_name': 'files/13.txt', 'Entity_type': ['LAW', 'ORG', 'PERSON', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORG', 'ORDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'WORK_OF_ART', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORDINAL', 'PERSON', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'LAW', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'ORG', 'DATE', 'LAW', 'GPE', 'LAW', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'DATE', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'PERSON', 'MONEY', 'DATE', 'LOC', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'CARDINAL', 'PERCENT', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'WORK_OF_ART', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG'], 'Entity_identified': ["The CHIPS Act's", 'The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease', 'Javascript', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Start Saving Now', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\n\n\nPlanning for Retirement\n\n\n\nHow Much Do I Need', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'Bank & Loans', 'First', '101', 'Bank', '101', '101', 'Us', '1993', 'Tom', 'David Gardner', 'millions', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\n\n\nOther Services', 'The Ascent\n\n\n\n\n\n\n\n\n\n', 'Latest Stock Picks\n\n\n\n\n\n\nBars\n\n\n\nTimes\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\n\nOur Services', 'Investing 101', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks\n\n\n\nIndustries', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Getting Started', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\nPlanning for Retirement\n\n\n\n', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies\n\n\n\nRetired:', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'First', 'Guides', '101', 'Bank', '101', '101', 'Us', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\nOther Services', '1993', 'Tom', 'David Gardner', 'millions', 'Log In\n\n\n\n\nHelp', 'Latest Stock Picks', 'The Motley Fool Foundation', 'our first year', "The CHIPS Act's", 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'Mar 13, 2023', 'The Motley Fool’s Premium Investing Services', 'today', 'The CHIPS Act', 'U.S.', 'the CHIPS Act', 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'March 9, 2023', 'March 10, 2023', 'The Motley Fool', 'Jose Najarro', 'The Motley Fool', 'The Motley Fool', 'Related Articles', 'Has Intuit Stock Become', 'Meta Platforms Stock: Time to Sell', 'CrowdStrike', 'AMD', 'Corsair', '$7 Trillion', '10 Years', 'Buy Now', '416%', '200,000', '$1 Million', '2033', 'AI Could Have a', '$7 Trillion', '10 Years', '1', '697%', '2', 'The Motley Fool', "The Motley Fool's", 'View Premium Services', 'Linked', 'LinkedIn', 'YouTube', 'YouTube', 'Instagram\n\nInstagram\n\n\n\n\nTiktok', 'Xignite', 'The Motley Fool', 'Us', 'Careers\nResearch\nNewsroom', 'The Ascent\nAll Services', 'UK', 'Australia', 'Fool Canada\n\n\n\nFree Tools\n\nCAPS Stock Ratings\nDiscussion Boards\nCalculators', 'Lakehouse Capital', 'Trademark and Patent Information \nTerms and Conditions']}, {'File_name': 'files/12.txt', 'Entity_type': [], 'Entity_identified': []}, {'File_name': 'files/10.txt', 'Entity_type': ['PERSON', 'ORG', 'WORK_OF_ART', 'PERSON', 'PERSON', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'GPE', 'ORG', 'ORG', 'PERSON', 'ORDINAL', 'ORG', 'FAC', 'PERSON', 'DATE', 'DATE', 'ORG', 'DATE', 'NORP', 'FAC', 'ORG', 'GPE', 'PERSON', 'GPE', 'MONEY', 'ORG', 'ORG', 'ORG', 'NORP', 'ORG', 'ORG', 'GPE', 'DATE', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'GPE', 'ORG', 'CARDINAL', 'ORG', 'GPE', 'ORG', 'ORDINAL', 'DATE', 'ORG', 'ORG', 'PERSON', 'FAC', 'MONEY', 'PERSON', 'ORG', 'CARDINAL', 'ORG', 'FAC', 'DATE', 'PERSON', 'GPE', 'GPE', 'GPE', 'EVENT', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORDINAL', 'GPE', 'NORP', 'ORG', 'MONEY', 'ORG', 'ORG', 'FAC', 'ORG', 'FAC', 'GPE', 'ORG', 'MONEY', 'GPE', 'DATE', 'MONEY', 'LOC', 'ORG', 'CARDINAL', 'ORG', 'GPE', 'ORG', 'CARDINAL', 'DATE', 'ORG', 'PERSON', 'ORG', 'PERCENT', 'DATE', 'ORG', 'PERSON', 'PERCENT', 'CARDINAL', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'DATE', 'DATE', 'GPE', 'ORG', 'MONEY', 'WORK_OF_ART', 'MONEY', 'PERSON', 'FAC', 'ORG', 'GPE', 'NORP', 'MONEY', 'MONEY', 'GPE', 'DATE', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'ORG', 'ORG', 'GPE', 'GPE', 'PERSON', 'NORP', 'PERSON', 'ORG', 'NORP', 'PERSON', 'NORP', 'NORP', 'ORG', 'GPE', 'GPE', 'PERSON', 'PERSON', 'ORG', 'PERCENT', 'GPE', 'ORG', 'ORG', 'MONEY', 'PERSON', 'NORP', 'FAC', 'NORP', 'DATE', 'GPE', 'PERSON', 'GPE', 'GPE', 'GPE', 'FAC', 'PERSON', 'ORG', 'ORG', 'ORG', 'GPE', 'ORG', 'ORG', 'ORDINAL', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'PERCENT', 'ORG', 'PERSON', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERCENT', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'PERSON', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'PERCENT', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'GPE', 'DATE', 'NORP', 'PERSON', 'PERSON', 'GPE', 'PERSON', 'ORG', 'NORP', 'ORG', 'LAW', 'DATE', 'NORP', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'ORG', 'ORG', 'PRODUCT', 'PRODUCT', 'DATE', 'PRODUCT', 'LOC', 'ORG', 'ORG', 'PERSON', 'GPE', 'PERSON', 'PERSON', 'PRODUCT', 'ORG', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'CARDINAL', 'PRODUCT', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'MONEY', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'DATE', 'CARDINAL', 'ORG', 'PRODUCT', 'GPE', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'DATE', 'DATE', 'ORG', 'MONEY', 'ORG', 'WORK_OF_ART', 'GPE', 'ORG', 'LOC', 'ORG', 'DATE', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'PERSON', 'GPE', 'ORG', 'GPE', 'GPE', 'DATE', 'WORK_OF_ART', 'GPE', 'WORK_OF_ART', 'NORP', 'ORG', 'NORP', 'NORP', 'NORP', 'GPE', 'ORG', 'GPE', 'NORP', 'NORP', 'WORK_OF_ART', 'PERSON', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'WORK_OF_ART', 'ORG', 'ORG', 'LAW', 'PERSON', 'DATE', 'PERSON', 'NORP', 'GPE', 'PERSON', 'GPE', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'ORG', 'ORDINAL', 'ORG', 'DATE', 'GPE', 'ORG', 'MONEY', 'GPE', 'ORG', 'PERSON', 'ORG', 'CARDINAL', 'ORG', 'PERSON', 'PERSON', 'WORK_OF_ART', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'PERSON', 'DATE', 'ORG', 'CARDINAL', 'ORG', 'MONEY', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'NORP', 'PERSON', 'GPE', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'WORK_OF_ART', 'PERSON', 'GPE', 'ORG', 'ORG', 'DATE', 'ORG', 'DATE', 'GPE', 'MONEY', 'PERSON', 'ORG', 'CARDINAL', 'ORG', 'FAC', 'CARDINAL', 'PERCENT', 'ORG', 'ORG', 'DATE', 'ORG', 'PERSON', 'ORG', 'TIME', 'GPE', 'TIME', 'PERSON', 'ORG', 'ORG', 'ORG', 'GPE', 'FAC', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'GPE', 'PERSON', 'GPE', 'DATE', 'TIME', 'PERSON', 'DATE', 'PERSON', 'PERSON', 'TIME', 'PERSON', 'GPE', 'PERSON', 'ORDINAL', 'CARDINAL', 'GPE', 'GPE', 'TIME', 'PERSON', 'ORG', 'NORP', 'NORP', 'ORG', 'ORG', 'GPE', 'GPE', 'PERCENT', 'MONEY', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'DATE', 'GPE', 'GPE', 'ORG', 'GPE', 'PERCENT', 'DATE', 'ORG', 'ORG', 'GPE', 'PERCENT', 'ORG', 'CARDINAL', 'MONEY', 'DATE', 'PERSON', 'ORG', 'WORK_OF_ART', 'PRODUCT', 'PERSON', 'PERSON', 'ORG', 'ORG', 'MONEY', 'PERSON', 'ORG', 'PERSON', 'WORK_OF_ART', 'NORP', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'LAW', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'DATE', 'ORG', 'PERSON', 'ORG', 'ORG', 'MONEY', 'MONEY', 'GPE', 'CARDINAL', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERCENT', 'PERSON', 'GPE', 'PERSON', 'GPE', 'PERSON', 'GPE', 'GPE', 'GPE', 'ORG', 'GPE', 'ORG', 'ORG', 'NORP', 'NORP', 'ORG', 'ORG', 'NORP', 'ORG', 'LOC', 'WORK_OF_ART', 'PRODUCT', 'CARDINAL', 'GPE', 'GPE', 'WORK_OF_ART', 'ORG', 'CARDINAL', 'PERSON', 'MONEY', 'MONEY', 'GPE', 'MONEY', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'GPE', 'PERSON', 'GPE', 'MONEY', 'WORK_OF_ART', 'PERSON', 'MONEY', 'ORG', 'MONEY', 'NORP', 'ORG', 'ORG', 'GPE', 'GPE', 'PRODUCT', 'ORG', 'PERCENT', 'NORP', 'PERSON', 'NORP', 'ORG', 'GPE', 'ORG', 'NORP', 'GPE', 'DATE', 'WORK_OF_ART', 'ORG', 'DATE', 'MONEY', 'DATE', 'DATE', 'DATE', 'ORG', 'PERCENT', 'PERSON', 'DATE', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'NORP', 'TIME', 'PERSON', 'ORG', 'MONEY', 'CARDINAL', 'NORP', 'PERSON', 'ORG', 'CARDINAL', 'MONEY', 'PERSON', 'LOC', 'PERSON', 'MONEY', 'ORG', 'PERSON', 'ORG', 'PERSON', 'PRODUCT', 'ORG', 'DATE', 'CARDINAL', 'ORG', 'DATE', 'LOC', 'PERSON', 'PERSON', 'GPE', 'GPE', 'MONEY', 'ORG', 'ORG', 'LOC', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'MONEY', 'GPE', 'GPE', 'ORG', 'PERSON', 'PERSON', 'ORG', 'PERSON', 'ORG', 'PERSON', 'ORG', 'GPE', 'PRODUCT', 'PERSON', 'NORP', 'FAC', 'MONEY', 'CARDINAL', 'GPE', 'NORP', 'PERSON', 'GPE', 'GPE', 'PERCENT', 'ORG', 'GPE', 'GPE', 'CARDINAL', 'ORG', 'PERSON', 'ORG', 'ORG', 'DATE', 'NORP', 'MONEY', 'ORG', 'LOC', 'PERSON', 'ORG', 'ORG', 'DATE', 'ORG', 'DATE', 'ORG', 'PERSON', 'DATE', 'GPE', 'DATE', 'PERSON', 'ORG', 'NORP', 'ORG', 'ORG', 'GPE', 'DATE', 'NORP', 'ORG', 'LOC', 'CARDINAL', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'PERSON', 'ORG', 'WORK_OF_ART', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'MONEY', 'GPE', 'EVENT', 'PERSON', 'PERSON', 'PERSON', 'TIME', 'MONEY', 'TIME', 'DATE', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'CARDINAL', 'ORG', 'PERSON', 'WORK_OF_ART', 'ORG', 'DATE', 'ORG', 'WORK_OF_ART', 'GPE', 'CARDINAL', 'CARDINAL', 'WORK_OF_ART', 'ORG', 'DATE', 'ORG', 'GPE', 'PERSON', 'PERSON', 'GPE', 'NORP', 'LOC', 'GPE', 'GPE', 'GPE', 'GPE', 'NORP', 'ORDINAL', 'WORK_OF_ART', 'ORG', 'PERSON', 'ORG', 'DATE', 'DATE', 'WORK_OF_ART', 'PERSON', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'MONEY', 'CARDINAL', 'GPE', 'GPE', 'CARDINAL', 'GPE', 'DATE', 'PERSON', 'MONEY', 'ORDINAL', 'GPE', 'MONEY', 'ORG', 'ORG', 'CARDINAL', 'NORP', 'WORK_OF_ART', 'ORG', 'PERSON', 'ORG', 'ORG', 'CARDINAL', 'GPE', 'ORG', 'GPE', 'CARDINAL', 'WORK_OF_ART', 'GPE', 'NORP', 'DATE', 'GPE', 'ORG', 'PERSON', 'PERSON', 'LAW', 'DATE', 'WORK_OF_ART', 'ORG', 'ORG', 'GPE', 'PERSON', 'ORG', 'ORG', 'GPE', 'GPE', 'EVENT', 'PERSON', 'PERSON', 'EVENT', 'ORG', 'ORG', 'WORK_OF_ART', 'GPE', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG', 'NORP', 'GPE', 'WORK_OF_ART', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'PERSON', 'LOC', 'NORP', 'ORG', 'ORG', 'MONEY', 'CARDINAL', 'ORG', 'PERSON', 'GPE', 'ORG', 'MONEY', 'PERSON', 'PERSON', 'GPE', 'ORG', 'WORK_OF_ART', 'DATE', 'ORG', 'ORG', 'GPE', 'WORK_OF_ART', 'GPE', 'ORG', 'DATE', 'ORG', 'GPE', 'DATE', 'ORG', 'PERSON', 'LAW', 'ORG', 'ORG', 'GPE', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'ORG', 'GPE', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'PRODUCT', 'ORG', 'GPE', 'PERSON', 'ORG', 'ORG', 'ORG', 'PERSON', 'CARDINAL', 'GPE', 'CARDINAL', 'MONEY', 'GPE', 'ORG', 'GPE', 'ORG', 'CARDINAL', 'GPE', 'LOC', 'WORK_OF_ART', 'PERSON', 'PERSON', 'ORG', 'LOC', 'MONEY', 'DATE', 'ORG', 'ORG', 'ORG', 'NORP', 'PERSON', 'ORG', 'ORDINAL', 'GPE', 'PERSON', 'ORG', 'DATE', 'ORG', 'ORG', 'PERSON', 'ORG', 'CARDINAL', 'DATE', 'WORK_OF_ART', 'ORG', 'CARDINAL', 'MONEY', 'CARDINAL', 'ORDINAL', 'MONEY', 'PERSON', 'ORG', 'ORG', 'QUANTITY', 'ORG', 'WORK_OF_ART', 'ORG', 'PERSON', 'WORK_OF_ART', 'ORG', 'DATE', 'WORK_OF_ART', 'WORK_OF_ART', 'GPE', 'ORDINAL', 'DATE', 'DATE', 'PERSON', 'ORG', 'PERSON', 'GPE', 'WORK_OF_ART', 'LOC', 'DATE', 'QUANTITY', 'CARDINAL', 'ORG', 'ORG', 'GPE', 'GPE', 'GPE', 'NORP', 'NORP', 'ORG', 'WORK_OF_ART', 'GPE', 'ORG', 'ORG', 'GPE', 'MONEY', 'ORG', 'GPE', 'GPE', 'ORG', 'DATE', 'ORG', 'ORG', 'ORDINAL', 'GPE', 'ORDINAL', 'ORG', 'GPE', 'GPE', 'ORG', 'PERSON', 'ORG', 'ORG', 'MONEY', 'GPE', 'ORG', 'DATE', 'ORG', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'WORK_OF_ART', 'ORG', 'ORG', 'GPE', 'NORP', 'GPE', 'PERCENT', 'CARDINAL', 'DATE', 'GPE', 'ORG', 'CARDINAL', 'MONEY', 'ORG', 'GPE', 'NORP', 'WORK_OF_ART', 'DATE', 'NORP', 'NORP', 'CARDINAL', 'ORG', 'NORP', 'ORG', 'GPE', 'CARDINAL', 'NORP', 'GPE', 'GPE', 'CARDINAL', 'ORG', 'ORG', 'NORP', 'GPE', 'ORG', 'GPE', 'NORP', 'PERSON', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'DATE', 'NORP', 'GPE', 'NORP', 'ORG', 'GPE', 'CARDINAL', 'NORP', 'ORG', 'GPE', 'GPE', 'PERSON', 'GPE', 'NORP', 'GPE', 'NORP', 'PERSON', 'LOC', 'ORG', 'TIME', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'GPE', 'GPE', 'PERSON', 'PERSON', 'LAW', 'GPE', 'PERCENT', 'ORG', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'GPE', 'GPE', 'GPE', 'ORG', 'CARDINAL', 'CARDINAL', 'PERSON', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG', 'GPE', 'ORG', 'ORDINAL', 'ORG', 'GPE', 'ORG', 'GPE', 'GPE', 'NORP', 'ORG', 'CARDINAL', 'NORP', 'ORG', 'NORP', 'ORG', 'PERSON', 'ORG', 'GPE', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'GPE', 'ORG', 'GPE', 'GPE', 'ORG', 'DATE', 'ORG', 'GPE', 'PERSON', 'GPE', 'NORP', 'NORP', 'ORG', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'MONEY', 'ORG', 'ORG', 'TIME', 'TIME', 'CARDINAL', 'DATE', 'ORG', 'ORG', 'MONEY', 'DATE', 'DATE', 'ORG', 'PERSON', 'WORK_OF_ART'], 'Entity_identified': ['Login / Sign Up  Crypto News', 'Media News', 'Hot  Live  Light Mode  Imagery  Search  Customize News Grid', 'App', 'Telegram Discord', '4,124 0.12%', 'S&P', '0.58%', '33,402 0.59%', '0.52%', 'Russel', '1,770 1.81%', 'VIX', '19.69 6.15%', '10Y 3.34 0.00%', '40,523 0.72%', 'FTSE', '7,663 0.37%', 'EuroStoxx', '0.13%', '15,562 0.27%', '20,275 0.66%', '27,813 1.68%', '0.24%', '0.07%', '28,550 1.36%', 'Fear\xa0&\xa0Greed', '50/100', 'Intel', 'Micron Fall After', 'China', 'Japanese Semiconductor Restrictions', 'Credit Suisse', 'Slip', 'first', 'van ++ 6 Ways to Grow the Seeds of Greatness Within You Into Your Full Potential', '++ How Disney', "Ron DeSantis ++ https://s.nikkei.com/3ZHQqDK ++ Bitcoin's", "11 Months ++ Egypt's", '2023', '++ U.S. Job Openings Dropped', 'February ++', '+ German', "++ JPMorgan's", 'Dimon', 'US', 'Brandon Johnson', 'Chicago', '$175 million', '++ C3.ai', 'FBI', 'Genesis', '+ German', 'Apple ++ Ofcom', 'Google', 'UK', 'October 2022', 'SNP', 'Tom Steyer', 'US', 'Pimco', '++ Live Coverage ++ Women in Finance See Double', 'the Wage Disparity', 'UK', 'EU Sustainability', 'US', '++ Bank', '3', 'Buy After OPEC Production Cuts +', 'Wisconsin', 'Supreme Court', 'first', '15 years ++', 'Cleveland Fed', 'Reiterates Central Bank’s Resolve', 'Serie A Clubs', '++ Gold Holds', '2,000', 'Elon Musk', 'Forbes ++ Workers', '33', 'Cub Foods', '++ SFPD Investigates Main Street Homicide', '23-039', 'Strengthens Defenses', 'Occupied Ukraine', 'Zelensky Visits', 'Poland', 'the Great Financial Crisis', '++ Mexican Government', 'Buy Power Plants', 'Spain', 'Iberdrola', 'first', 'Scotland', 'Israeli', 'Al-Aqsa Mosque', '$10.8 Billion', 'Advisor Team', 'Morgan Stanley ++', '++ CZ', 'FUD', 'Schiphol Airport', 'UK', 'TikTok', '16', 'Australia', 'July ++ Cities', 'nearly $200M', 'Silicon Valley ++ ‘Criminal', 'Trump', '34', 'Felony', 'Florida', 'GM', '5,000', 'a month', 'Google-Owner', 'Alphabet', 'Sustain Latest Rally      Stocks', '67%', 'years', 'the Bank of America', 'Mester', '5%', 'Half', '10', '2022', 'the United States', 'Bernard Arnault', 'Elon Musk', 'Forbes      Workers', 'AEP', 'this month', '7-Month', 'Mexico', 'Iberdrola', '6', 'More      Reddit     Frank Founder Faces Criminal Charges Over', '$175 Million', 'JPMorgan      Need', 'Robinhood', 'Iberdrola', 'US', 'Mexican', 'FOUR', '81', 'US', 'April 05, 2023    ', '80.87%', 'CDLX Cardlytics Inc        ', '14.96%', 'IONQ IonQ Inc        ', '12.08%', '11.04%', 'EHC Encompass Health Corp        ', '10.45%', 'AVAV AeroVironment Inc.', '9.30%', '32.41%', 'Guardforce AI Co Ltd        ', '26.32%', '15.52%', 'Micromobility.com', '15.02%', '14.38%', 'BigBear.ai Holdings Inc        ', '14.03%', 'DYN Dyne Therapeutics Inc', '04/11GOLDEN HEAVEN GROUP', 'GDHG', 'manhattan', 'florida', 'alvinbragg ukraine', 'republicans', 'jpmorgan', 'elonmusk google treasury', 'democrats', 'jamiedimon apple', 'russian', 'democrat', 'AP News     ', 'China', 'US', 'Biden', 'Ford Mustang Mach-E', 'Hyundai', '5      ', 'Germany', 'Lufthansa', 'LSG Group      Cities', 'nearly $200M', 'Wall St', 'Dutch', 'Schiphol Airport', 'Democrats', '2024', 'Russia', 'Strengthens Defenses', 'Occupied Ukraine', 'Zelensky Visits', 'Poland', 'Lucrative Exit      Everywhere Ventures Bucks Trend', 'Lou Paskalis Joins', 'Encourage Brands', 'Advertise on News    More      Financial Times', 'EU', 'China', 'The White House', 'Credit Suisse', 'first', 'Scotland      UBS', 'Credit Suisse', 'SNP', 'More      Reddit Trending       ', '83.90%', 'IFRX', 'InflaRx N.V.        ', '27.93%', 'BFRG Bullfrog AI Holdings Inc        ', '8.88%', '6.38%', '4.17%', 'STRC Sarcos Technology', '2.69%', 'PET Wag Group Co        ', '1.47%', '0.88%', '0.75%', 'CANO Cano Health Inc - Class A        ', '0.15%', 'RUN Sunrun Inc        ', '0.54%', 'OXY Occidental Petroleum Corp.', '1.49%', 'CTXR Citius Pharmaceuticals Inc        ', '1.74%', 'TD Toronto Dominion Bank', '1.83%', 'NVDA NVIDIA Corp        ', '2.14%', 'Snap Inc - Class A        ', '4.61%', 'ALBT Avalon GloboCare Corp        ', '5.18%', 'WLDN Willdan Group Inc        ', '7.26%', 'CEI Camber Energy Inc        Bloomberg     Biden Trade', 'US Working With Allies to Counter China      Apple’s Complex', 'Secretive Gamble', 'Move Beyond', 'Tom Steyer', 'Australia', 'July', 'Italian', 'Berlusconi Hospitalized', 'Ansa Reports', 'U.S.', 'Tai', 'NYC', 'German', 'Apple      Exclusive-EU', 'Chips Act', 'April 18', 'Canadian', 'one year', 'More      Marketwatch     Nicola Sturgeon’s', '83,200', 'last year', 'Social Security', 'Treasury', 'Kelley Blue Book: Gone', 'V-10', '2023', 'R8      NerdWallet', 'Midwest', 'OPEC', 'Production Cut Complicates The Inflation Fight', 'Suncor Erupt', 'Germany', "Mark Zuckerberg's", 'Meta Looks To Revolutionize Ads Using Generative', 'Twitter Blue', 'Apple Shortcut Checks      Tesla Puts Power In Your Hands', 'Spring 2023', 'Apple', '17', 'NASDAQ', 'Productivity Software Stocks', '2U', 'Q4 Earnings', 'Zuora', 'NYSE', 'ZUO', 'NYSE', 'NYSE', 'United Parcel Service Stock', 'These Two Covered Call Ideas      Stock Index Futures Move Lower', 'SEC', 'FED', 'SEC', '2022', 'Inclusion Initiatives and Progress &plus;&plus;&plus', 'SEC', '$175 million', 'Student Loan Assistance Company &plus;&plus;&plus', 'SEC', 'Merrill Lynch for Failing to Disclose Foreign Exchange Fees', 'Clients &plus;&plus;&plus', 'SEC', 'Chatham Asset Management', 'Anthony Melchiorre', 'Improper Fixed Income Securities Trading &plus;&plus;&plus', 'SEC', 'Highlight Free Investor Education Resources During Financial Capability', 'this week', '#', 'ArtificialIntelligence', 'MarketWatch Oil ticks lower', 'U.S.', 'CNBC', 'NATO', 'Ukraine      The Economist With', '“Life of Pi', 'Bloomberg Toronto', 'a second month', 'the traditional spring selling season', '@MIT', '#Concrete #', 'Apple', 'China      The Wall Street Journal Raine Group', 'San Francisco', 'boutique bank Code Advisors', 'the Silicon Valley      ', 'Bloomberg Economics Latest', 'quarterly', 'Apple', 'iPhones', 'one', 'Bloomberg TV Ukrainian', 'Volodymyr Zelenskiy', 'Poland', 'NATO', 'Argentina', 'California', 'years', 'More      ZeroHedge     The Countries Bailed Out By China', 'Paris', 'The Mad Emperor Of Ice Cream      The Rising Prevalence Of Autism      Saudi Arabia Makes Its', 'Eurasian', 'New Airfare Savings Program', 'J&J', 'European', 'German', 'Wisconsin', 'Supreme Court', 'U.S.', 'European', 'Asian', 'More      Morning Brew', 'Trump', '34', 'NWSL', 'landmark investment &plus;&plus;&plus', 'Downtowns', 'Trump', 'copycats &plus;&plus;&plus', 'Gen Z', 'Gen Z', 'Clubhouse', 'Virgin Orbit Files For', 'Chapter 11 &plus;&plus;&plus', 'Jamie Dimon Dishes', 'Annual', 'Barbie Biden Bragg', 'Democrats', 'New York', 'Hillary Dems', 'Georgia', 'NATO', 'Ukraine      Lack', 'Switzerland', 'Credit Suisse', 'Swiss', 'EV', 'UBS', 'first', 'Credit Suisse', '9-year-old', 'L.A.', 'Johnson & Johnson', 'nearly $9 billion', 'U.S.', 'JPMorgan Chase', 'Jamie Dimon', 'Volkswagen', '143,000', 'SEC', 'Frank', 'Charlie Javice', 'More      NYT Business     For Lower-Income Students', 'the Gas-Engine Camaro Opens', 'Airplanes      Help!', 'a Carrier Rule That Does', 'Whose E*Trade Transformed Stock Trading', '79', 'Whose Code Transformed Stock Trading, Dies', '79', 'Johnson & Johnson Reaches', '$8.9 Billion', 'Alphabet Inc - Class', 'PLTR Palantir Technologies Inc', 'CDLX Cardlytics Inc', 'DLO DLocal Limited', 'Philip Morris', 'NVDA NVIDIA Corp   PHVS', 'Pharvaris NV', 'Motorola Solutions Inc', 'Johnson & Johnson', 'HP Helmerich & Payne', 'JBL Jabil Inc', 'KTB Kontoor Brands Inc   ', 'Assured Guaranty Ltd', 'Trump', 'Melania      ', 'Democrat', 'Brandon Johnson', 'Chicago', 'NBC News', 'Janet Protasiewicz', 'Wisconsin Supreme Court', 'NBC News', 'Trump', 'Trump', '34', 'More      NBC', 'Bob Lee', 'San Francisco', 'Trump', 'GOP', '44-year', 'Trump', '18-year-old', 'California', '13', 'Charles', 'Catan board', '70', 'Trump', 'Mar-a-Lago', '34', '67%', "Wendy's Stock Gets Key Rating Upgrade", "Kerrisdale Capital's", '16-Month', 'The U.S. Economy With The Economic Optimism Index    More      Google Trends    REAL-TIME     Hail', 'Thunderstorm', 'National Weather Service', 'afternoon', 'Chicago', '24 HOURS', 'Donald Trump', 'En el', 'sur de la', 'Trump', 'cerraba', 'la noche', 'un', 'Nunca', 'Estados Unidos', 'Wisconsin Supreme Court', 'Janet Protasiewicz', 'the Supreme Court', 'Chicago', 'Brandon Johnson', 'Chicago', 'Tuesday', 'night', 'Paul Vallas', 'Tuesday', 'LeBron James', 'Anthony Davis', 'minutes', 'Barbie', "Greta Gerwig's", 'Barbie', 'first', 'two', 'Liverpool', 'LONDON', '90 seconds', "N'Golo Kante", 'ABC News', 'German', 'Swiss', 'Credit Suisse', 'UBS', 'San Francisco', "New Zealand's", '5.25%', '$175 million', 'Mexico', 'GM', 'Ford', 'EV', 'Tesla', 'fiscal 2024', '2025', 'China', 'Malaysia', 'Asian Monetary Fund', 'US', '65%', '2026', 'Credit Suisse', 'UBS', 'Japan', '82%', 'GM', '5,000', '$1 billion', 'first quarter', 'Musk', "Forbes World's", 'More      CNN     Kid Rock', 'Bud Light', 'Tucker Carlson', 'Sean Hannity', 'Fox', 'Johnson & Johnson', '$8.9 billion', 'Ron DeSantis', 'NBA', 'Miami Heat', 'More      Youtube         UBS Chair: Credit Suisse Integration to Take Up to 4 Years         ', 'Israeli', 'Al-Aqsa Mosque', '2024', 'Trump', 'Arnault', 'Fortune Soars Past $200Billion', '10', 'Pricey Foods Are Even More Expensive Because of Climate Change         ', 'Gavin Maloof Talks', 'New Sports Investment', 'Las Vegas', 'NBA', 'Sacramento Kings         How To Buy A', '24 2023', 'Stanford Conference on Organizations and Environmental Sustainability         Energy', 'Williams', 'UFC', 'WWE', '#wwe', '#ufc', 'US', '30', '6.4%', '6.45%', 'US MBA Mortgage Applications Actual', 'Forecast', '2.9%', 'Brexit', 'UK', 'Brxit', 'UK', 'Brexit', 'UK', 'Turkey', 'Sweden', 'NATO', 'Brazil', 'The Stock Slips      Live Coverage      Dogecoin Is Falling but Elon Musk Means', 'the Real Star      Live Coverage    ', 'Chinese', 'Democrats', 'Trump', 'IMF', 'Swiss', 'Credit Suisse      Why', 'Asia', 'More      MarketBeat     At 3x Earnings, Avis Budget Is Worth Taking for a Spin      Toyota Is The Reliable Value Car Play      High-Growth', 'Lyft Be a Gamechanger', '3', 'China', 'US', 'More      The Hill     New', 'Trump', '34', 'Charlie Javice', '175', '16', 'UK', '$100 million', 'three years', 'IRS', 'BBC     ', 'Treasury', 'CBI', 'Amazon', 'UK', "Rupert Murdoch's", 'NI', '2', 'More      ', 'Bernard Arnault', '200bn', 'Franco Manca', '93', 'Japanese', 'Amazon', 'Microsoft', 'UK', 'UK', 'More      YouGov', 'Trump', '56%', 'Americans', "Donald Trump's", 'Americans', 'COVID-19 fears &plus;&plus;&plus', 'US', 'Americans &plus;&plus;&plus', 'Americans', 'U.S.', 'today', 'Used Car Models That Have Dropped', 'Johnson & Johnson Talc Settlement', 'Investor Day', '$2 billion', "6-month '", 'months', 'March', 'Twitter Blue', 'An estimated 4%', 'Mario Gabelli', '2nd-half', 'Google', 'AI', 'Bard', 'Almost half', 'Americans', 'their work hours', 'Musk', 'Shiba Inu', '$4 per', 'gallon', 'Republican', 'Biden', 'Kraft Heinz', '3', 'more than $3 million', 'Bob Iger', 'DeSantis', 'Johnson', '$9 billion', 'No Venture Funding', 'Detail Breakdown', 'Social Media Ended Tomorrow', 'Variance      Donald', 'AR-15', 'AI & What it Means for The Creative Industry      ', 'February', '5', 'Worst-Performing Mega-Cap Stocks', 'March 2023', 'Permian Basin With Big Acquisition', 'Exit      Nano', 'Stratasys', 'Brazil', 'SoCal', 'as little as $1.58', 'Labor Department', 'Virgin Orbit', 'Bay Area', 'Social Security', 'one', 'WWE', 'UFC', '$21.4-billion', 'California', 'L.A', 'US Working With Allies to Counter China      Analysis      Advice      ', 'Biden', 'Ford Mustang Mach-E', 'Hyundai', 'Wendy Suzuki', 'AI', 'Musk', 'Trump', 'Illinois', 'Twitter', 'Elon Musk’s', 'RxBar', 'Lincoln Park', '6.5', 'Fewer than half', 'US', 'Asian', 'Wall St', 'San Francisco', 'New Zealand’s', '5.25%', 'Starbucks', 'South Carolina', 'Mexico', '74', 'JPMorgan', 'Jamie Dimon', 'SVB', 'IRS', 'February', 'Brits', '$15.9M', 'Team USA', 'the Bay Area      Lauder', 'Pierre Karl Péladeau', 'Videotron', 'Freedom Mobile      Canada', 'second half of 2023', 'Shaw', 'one day', 'Bank of Canada', 'Paul Beaudry', 'July', 'Calgary', 'March', 'Coinsquare', 'CoinSmart', 'Canadian', 'Champagne', 'FCA', 'US', 'September 2024', 'European', 'Borsa Italiana', 'ICE Clear Europe      ‘', 'only two', 'Credit Suisse', 'AGM      ', 'UBS', 'Broadridge DLT', 'Gordon', 'Winterflood Securities', 'More      Observer     Cris Valenzuela’s', 'AI Video', 'Google', 'Google', 'Sundar Pichai Clarifies', 'Bob Iger', 'Ariel Emanuel', '$20 Billion', 'WWE Juggernaut', 'Price War Strategy    ', 'Barry Ritholtz', 'Transcript', 'Ken Kencel &plus;&plus;&plus', '10 Tuesday AM', '$33 Trillion', '10 Monday AM', '1958', 'Porsche', 'Whistles', "Donald Trump's", 'Trump', 'today', 'Wisconsin', 'Chicago', 'Denver      ', 'Trump', '34', 'Trump', 'Biden', 'More      Reason     Today', 'Supreme Court', 'April 5, 1982      Dump', 'Trump', '\'      Devin Nunes\' Sues for Libel Over "Investigators Examined Trump Media for Possible Money Laundering" Article      Trump\'s', 'New York', 'one', '34', 'More      Nikkei Asia     Meta', 'AI', 'this year', 'CTO', 'Taiwan', 'Tsai', 'McCarthy', 'Japan', 'Australian', 'Asia', 'Pakistan', 'Taiwan', 'Japan', 'China', 'Japanese', 'first', 'More      Abnormal Returns   Research', 'obsession &plus;&plus;&plus', 'Adviser', 'spreadsheet knowledge &plus;&plus;&plus', 'Sunday', 'this week', 'Abnormal Returns      Forbes     What’s', 'Mixed Q1 Delivery Report', 'RS Group', 'FTSE', 'Lower As Revenues Growth Slows Sharply      JPM’s', 'Regulation Plea Means Heads The Bank Wins', 'Credit Rating', 'Advanced To Frozen Four', 'Bill Gates', 'Elon Musk-', 'A.I      J&J', '8.9bn', '60,000', 'San Francisco', 'U.S.', '2', 'America', '40 years ago', 'Bernard Arnault’s', '$200 billion', 'first', 'Florida', '$17 Billion', 'State', 'Ways to Minimize Isolation', '6', 'Dems', 'Trump', 'Wisconsin Supreme Court', "Alvin Bragg's", 'Trump', 'Trump', '34', 'Manhattan', 'Trump', 'California', 'one', 'More      Vox     ', 'Florida', 'Republicans', '6-week', 'Chicago', 'Trump', 'Donald Trump', 'Donald Trump', 'The CHIPS Act', '2033', 'More      Bloomberg Quicktake         Warner Bros. Nears Deal', 'Harry Potter TV Series', 'HBO', 'US', 'Mecca Flashes Warning', 'Health Systems Are Failing Their Patients         Finland Joins', 'NATO', 'Blow', 'Russia', 'Ukraine War         ', 'Alvin Bragg', "Donald Trump's", 'the AI Revolution Through the Power of Legal Liability      Could ChatGPT', 'Pose a Threat', "Google's Dominance in Search", 'The Label’s Actually Part of the Problem', 'America', 'China', 'the United States', 'GOP', 'NATO', 'Wisconsin Supreme Court', 'Israeli', 'Jerusalem', 'More      Newsweek     Mysterious Russian', 'Tatarsky', 'Aaron Gunches', 'Katie Hobbs', 'Samuel L. Jackson', 'Trump', 'Brie Larson', 'NYC', 'Republican', 'House', 'Johnson & Johnson', '$8.9 billion', 'Hundreds', "McDonald's", 'Laid', 'U.K.', 'TikTok', 'nearly $16 million', "Richard Branson's", 'Virgin Orbit', 'Australia', 'TikTok', 'More      Globe And Mail     Opinion:', '36', 'UBS', 'Credit Suisse', 'U.S.', 'food      Opinion: A', 'Canada', 'LNG', 'Months', 'Canada Jetlines', 'Calgary', '2018-2021', 'Volkswagen', "Richard Branson's", 'Chapter 11', 'UFC', 'WWE', 'US', 'Shaw', 'Aritzia', 'More      Techmeme     Sources', 'Apple', 'China', 'AI', 'Nvidia', 'Twitter', 'NPR', 'Twitter', 'NPR', 'Germany', 'Bundeskartellamt', 'Apple', 'Silicon Valley Bank', 'Brex', 'Twitter Blue', 'One', 'Albany', 'Five', '$3.6 million', 'Palo Alto', 'Disney', 'Florida', 'Catan board', '70', 'Oakland', 'Bay Area', 'More      Techcrunch     ', 'Bob Lee', 'Cash App', 'CTO', 'Africa      Kia', '122B', '2030', 'AI', 'Morgan Stanley', 'BP Ventures', 'Indian', 'Magenta      ', 'Apple', 'first', 'India', 'Twitter', "Removing Your Verification Today & it's", 'April', 'U-Turn On Remote Work    More      PYMNTS     Supplemental Incomes Make Household Finances More Manageable', 'API-Enable All Customer', 'Journeys      ', 'AI Tools Center of New Regulation-Innovation Tug of War      Built Technologies Creating Solutions for Commercial Property Developers      Belgium’s Real-Time Payments Volume', '1B', '2026', 'More      Medium     How Net Worth', 'Financial Freedom Connect', '6', '15', '7', 'First', '$258 Billion Dollars', 'Meme Money', 'Too      Trump Lawyers Warn Against', "Media '", '4.3.23      Jared Kushner Got His      Regulators Really Have Given Up Trying To Teach Wells Fargo A Lesson    More      Mises Institute     To Fight the State', 'the State      The Fed’s', 'Capital Goes Negative      Why the Regime Needs the Dollar to Be the Global Reserve Currency      What Our Energy Future Be', 'Bail Out the World', 'Khan Academy Joins', 'OpenAI      How to visit Italy      ', 'AI', 'Tuesday', 'More      ', 'Yellowstone      How Petteri Orpo', 'Finland', 'third', '2024', '2023', 'Trump', 'Stormy Daniels', 'Brandon Johnson', 'Chicago', 'More      PBS Newshour     News Wrap:', 'South', 'more storms days', '32      ', '34', 'Trump      Finland', 'NATO', 'Russia', 'Ukraine', 'Tennessee', 'Republicans', 'Democratic', 'Capitol      Trump', 'More      Naked Capitalism     Links 4/5/2023      The Shambolic Criminal Case Against Donald', 'Saudi Arabia', 'Shanghai Cooperation Organization      Continuing', 'Undies', 'Hong Kong’s', 'as much as US$122.5 million &plus;&plus;&plus', 'MTR Lab', 'Cyberport', 'Hong Kong', 'alfred24 &plus;&plus;&plus', '2008', 'lurk &plus;&plus;&plus', 'Apple', 'first', 'India', 'second', 'Mumbai &plus;&plus;&plus', 'China', 'China', 'Arts Economics', 'Art Basel', 'UBS', 'TikTok', '€14.5 million', 'UK', 'TikTok', 'next year', 'NASA', 'Artemis', 'More      Asia Financial     EU Leaders Back', 'China', 'Ukraine      ', 'US Chip Firms, Others      China Calls', 'WTO', 'Review Chip Export Curbs Led', 'US', 'Arab', 'Qatar', '8%', 'Q4', '2022', 'Saudi Arabia', 'Saudi Electricity Co.', 'two', '2bn', 'Sanabil Investments', 'Iraq', 'Kurdish', 'More      Wikipedia Current Events     ', 'April 5', 'Israeli', 'Palestinian', '2023', 'Al-Aqsa', 'Israeli', 'Al-Aqsa Mosque', 'Jerusalem', 'seven', 'Palestinians', 'the West Bank', 'Gaza', 'nine', 'Arab News', 'Volodymyr Zelenskyy', 'Russian', 'Ukraine', 'Ukraine Volodymyr Zelenskyy', 'Poland', 'Polish', 'Andrzej Duda', 'Zelenskyy', 'the Order of the White Eagle', 'Poland', 'Reuters', 'European Pravda', 'April 4', 'Syrian', 'Israel', 'Syrian', 'The Israeli Air Force', 'Damascus', 'two', 'Syrian', 'The Syrian Observatory for Human Rights', 'Iran', 'al-Kiswah', 'Rif Dimashq Governorate', 'France', 'Islamic', 'U.S.', 'Islamic', "Khalid 'Aydd Ahmad al-Jabouri", 'Europe', 'Easter', 'overnight', 'Syria', 'U.S.', 'Reuters', 'Islamic State', 'Taliban', 'Six', 'Taliban', 'Balkh Province', 'Afghanistan', 'Al Arabiya', 'Virgin Orbit', 'Chapter 11', 'California', '85%', 'Reuters', '2023', 'One', '50', 'nineteen', 'Voorschoten', 'South Holland', 'Netherlands', 'BBC News', 'Seven', 'eleven', 'Nathu La', 'Sikkim', 'India', 'AP', 'NATO Finland', 'NATO', 'Finland', 'NATO', '31st', 'NATO', 'Russia', 'BBC News', 'China', 'India', 'Sino-Indian', 'The Ministry of Civil Affairs of China', '11', 'Indian', 'Arunachal Pradesh', 'Chinese', 'Times of India', 'Law', 'Donald Trump Indictment', 'U.S.', 'Donald Trump', 'Stormy Daniels', 'Manhattan', 'CBS News', 'The Independent) Manhattan District', 'Alvin Bragg', 'Trump', 'Bronx Daily', 'United Kingdom', 'the High Court of Justice', 'Cuba', 'Cuba', 'Central Bank', 'the 1980s', 'Reuters', 'Chicago', 'Brandon Johnson', 'Chicago', 'Peruvian', 'Peruvian', 'The Congress of Peru', 'Dina Boluarte', 'Reuters', "The United Kingdom's", 'ICO', 'TikTok', '12.7', 'ICO', 'BBC News', '11 hours', '16 hours', '#', 'Yesterday', '@costplusdrugs', 'Invokana', 'around $244', 'Yesterday', 'today', '@mcuban @costplusdrug', 'Telegram Discord', "Status      The Web's Most Comprehensive Business News Site"]}, {'File_name': 'files/11.txt', 'Entity_type': ['LAW', 'ORG', 'PERSON', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORG', 'ORDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'WORK_OF_ART', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORDINAL', 'PERSON', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'LAW', 'PERSON', 'DATE', 'ORG', 'ORG', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'PERSON', 'PERCENT', 'MONEY', 'DATE', 'TIME', 'PERSON', 'ORG', 'DATE', 'DATE', 'DATE', 'DATE', 'GPE', 'LAW', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'DATE', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'LAW', 'MONEY', 'DATE', 'GPE', 'ORG', 'PRODUCT', 'ORG', 'DATE', 'PRODUCT', 'DATE', 'DATE', 'DATE', 'ORG', 'ORG', 'DATE', 'PERCENT', 'DATE', 'PERCENT', 'PERCENT', 'DATE', 'PRODUCT', 'CARDINAL', 'DATE', 'CARDINAL', 'DATE', 'LAW', 'ORG', 'ORG', 'ORG', 'ORG', 'LAW', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'MONEY', 'ORG', 'ORG', 'ORG', 'MONEY', 'PERCENT', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'DATE', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'CARDINAL', 'PERCENT', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'WORK_OF_ART', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG'], 'Entity_identified': ['The CHIPS Act Is Accepting Applications', 'The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease', 'Javascript', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\n\n\nPlanning for Retirement\n\n\n\nHow Much Do I Need', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'Bank & Loans', 'First', '101', 'Bank', '101', '101', 'Us', '1993', 'Tom', 'David Gardner', 'millions', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\n\n\nOther Services', 'The Ascent\n\n\n\n\n\n\n\n\n\n', 'Latest Stock Picks\n\n\n\n\n\n\nBars\n\n\n\nTimes\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\n\nOur Services', 'Investing 101', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks\n\n\n\nIndustries', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Getting Started', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\nPlanning for Retirement\n\n\n\n', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies\n\n\n\nRetired:', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'First', 'Guides', '101', 'Bank', '101', '101', 'Us', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\nOther Services', '1993', 'Tom', 'David Gardner', 'millions', 'Log In\n\n\n\n\nHelp', 'Latest Stock Picks', 'The Motley Fool Foundation', 'our first year', 'The CHIPS Act Is Accepting Applications', 'Nicholas Rossolillo', 'Mar 23, 2023', 'IPO', 'GlobalFoundries', 'this year', 'All In” Buy Alert', '39B', 'Today', 'Arrow-Thin-Down', '(-1.94%', '69.34', 'April 4, 2023', '10:00 a.m.', 'ET', 'The Motley Fool’s Premium Investing Services', 'today', 'the coming years', '2022', 'two years', 'U.S.', 'Science Act', 'the coming years', 'GlobalFoundries', 'GlobalFoundries', 'GlobalFoundries', 'AMD', 'a decade ago', 'AMD', 'GlobalFoundries', 'late 2021', '2018', 'GlobalFoundries', 'Taiwan Semiconductor Manufacturing', 'GlobalFoundries', 'General Motors', 'NYSE', 'GM', 'the CHIPS Act', '$52 billion', 'the beginning of March 2023', 'U.S.', 'GlobalFoundries', 'GlobalFoundries', 'GlobalFoundries', 'years', 'GlobalFoundries', 'the first half of 2023', 'the year ahead', '2023', 'GlobalFoundries', 'GlobalFoundries', 'Q1 2023', '5.7%', 'year over year', 'about 14%', '11.6%', 'Q1 2022', 'GlobalFoundries', 'nearly 26', 'trailing-12-month', '21', 'one-year', 'CHIPS Act', 'GlobalFoundries', 'Texas Instruments', 'NASDAQ', 'TXN', 'the CHIPS Act', 'Applied Materials', 'Nicholas Rossolillo', 'Advanced Micro Devices', 'Applied Materials', 'The Motley Fool', 'Advanced Micro Devices', 'Applied Materials', 'Taiwan Semiconductor Manufacturing', 'Texas Instruments', 'General Motors', 'January 2025', '25', 'General Motors', 'The Motley Fool', 'Stocks Mentioned', '69.34', '-1.94%', 'GlobalFoundries', 'General Motors Team Up', 'This Semiconductor Manufacturing Company', 'GlobalFoundries', 'Micron Technology', 'Applied Materials Soared', 'Today', 'Today', '416%', '200,000', '$1 Million', '2033', 'AI Could Have a', '$7 Trillion', '10 Years', '1', '697%', '2', 'The Motley Fool', "The Motley Fool's", 'View Premium Services', 'Linked', 'LinkedIn', 'YouTube', 'YouTube', 'Instagram\n\nInstagram\n\n\n\n\nTiktok', 'Xignite', 'The Motley Fool', 'Us', 'Careers\nResearch\nNewsroom', 'The Ascent\nAll Services', 'UK', 'Australia', 'Fool Canada\n\n\n\nFree Tools\n\nCAPS Stock Ratings\nDiscussion Boards\nCalculators', 'Lakehouse Capital', 'Trademark and Patent Information \nTerms and Conditions']}, {'File_name': 'files/9.txt', 'Entity_type': ['LAW', 'ORG', 'PERSON', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORG', 'ORDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'WORK_OF_ART', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORDINAL', 'PERSON', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'LAW', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'ORG', 'DATE', 'ORG', 'PERSON', 'MONEY', 'DATE', 'PERSON', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'TIME', 'PERSON', 'WORK_OF_ART', 'PERSON', 'PERSON', 'PERSON', 'LAW', 'ORG', 'PERCENT', 'DATE', 'DATE', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'ORG', 'DATE', 'MONEY', 'ORG', 'DATE', 'MONEY', 'ORG', 'DATE', 'MONEY', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'MONEY', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'DATE', 'ORG', 'PERCENT', 'DATE', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'CARDINAL', 'PERCENT', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'WORK_OF_ART', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG'], 'Entity_identified': ['The CHIPS Act Is Open for Business -- Which Stocks Will', 'The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease', 'Javascript', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', '401k Basics', 'HSA Basics\n\n\n\n\n\nPlanning for Retirement\n\n\n\n', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'Bank & Loans', 'First', '101', 'Bank', '101', '101', 'Us', '1993', 'Tom', 'David Gardner', 'millions', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\n\n\nOther Services', 'The Ascent\n\n\n\n\n\n\n\n\n\n', 'Latest Stock Picks\n\n\n\n\n\n\nBars\n\n\n\nTimes\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\n\nOur Services', 'Investing 101', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks\n\n\n\nIndustries', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Getting Started', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\nPlanning for Retirement\n\n\n\n', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies\n\n\n\nRetired:', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'First', 'Guides', '101', 'Bank', '101', '101', 'Us', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\nOther Services', '1993', 'Tom', 'David Gardner', 'millions', 'Log In\n\n\n\n\nHelp', 'Latest Stock Picks', 'The Motley Fool Foundation', 'our first year', 'The CHIPS Act Is Open for Business -- Which Stocks Will', 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'Mar 8, 2023', 'The Motley Fool’s Premium Investing Services', 'today', 'Intel', 'Cap', '136B', 'Today', 'Arrow-Thin-Down', '0.64%', '0.21', '33.10', 'April 4, 2023', '10:00 a.m.', 'ET', 'Applications for the CHIPS Act', 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'the CHIPS Act', 'Intel', '0.64%', 'March 2, 2023', 'March 3, 2023', 'Billy Duberstein', 'Jose Najarro', 'Nicholas Rossolillo', 'The Motley Fool', 'Intel', 'January 2023', '57.50', 'Intel', 'January 2025', '45', 'Intel', 'January 2025', '45', 'Intel', 'The Motley Fool', 'Jose Najarro', 'The Motley Fool', 'The Motley Fool', 'Stocks Mentioned', 'Intel', 'INTC', '33.10', '0.64%', '0.21', 'Why Intel Outperformed', 'the Semiconductor Sector Today', 'Intel', 'Best Semiconductor Stock', 'AMD', 'Intel', 'Micron\n\n\n\nWhy Intel Stock Thumped', 'Thursday', 'Why Intel Stock Soared', '30%', 'Just a Month', '416%', '200,000', '$1 Million', '2033', 'AI Could Have a', '$7 Trillion', '10 Years', '1', '697%', '2', 'The Motley Fool', "The Motley Fool's", 'View Premium Services', 'Linked', 'LinkedIn', 'YouTube', 'YouTube', 'Instagram\n\nInstagram\n\n\n\n\nTiktok', 'Xignite', 'The Motley Fool', 'Us', 'Careers\nResearch\nNewsroom', 'The Ascent\nAll Services', 'UK', 'Australia', 'Fool Canada\n\n\n\nFree Tools\n\nCAPS Stock Ratings\nDiscussion Boards\nCalculators', 'Lakehouse Capital', 'Trademark and Patent Information \nTerms and Conditions']}, {'File_name': 'files/8.txt', 'Entity_type': ['GPE', 'ORG', 'PERSON', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'WORK_OF_ART', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORG', 'ORDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'ORG', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'PERCENT', 'PERCENT', 'PERSON', 'PERCENT', 'PERCENT', 'WORK_OF_ART', 'WORK_OF_ART', 'PRODUCT', 'ORG', 'LAW', 'ORG', 'DATE', 'ORG', 'ORG', 'PERCENT', 'ORDINAL', 'PERSON', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'GPE', 'GPE', 'PERSON', 'ORG', 'WORK_OF_ART', 'GPE', 'PERSON', 'ORG', 'ORG', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'WORK_OF_ART', 'PERSON', 'MONEY', 'DATE', 'PERSON', 'PERCENT', 'MONEY', 'DATE', 'TIME', 'PERSON', 'LAW', 'LAW', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'DATE', 'PERSON', 'GPE', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'PERSON', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORG', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'GPE', 'ORG', 'ORG', 'GPE', 'GPE', 'ORG', 'GPE', 'ORG', 'ORG', 'ORG', 'GPE', 'LOC', 'GPE', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'MONEY', 'PERCENT', 'MONEY', 'PERSON', 'MONEY', 'MONEY', 'WORK_OF_ART', 'ORG', 'ORG', 'MONEY', 'DATE', 'MONEY', 'CARDINAL', 'MONEY', 'ORG', 'DATE', 'WORK_OF_ART', 'ORG', 'ORG', 'PERCENT', 'MONEY', 'MONEY', 'DATE', 'WORK_OF_ART', 'MONEY', 'DATE', 'CARDINAL', 'PERCENT', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'WORK_OF_ART', 'GPE', 'GPE', 'ORG', 'ORG', 'ORG'], 'Entity_identified': ['China', 'The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease', 'Javascript', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\n\n\nPlanning for Retirement\n\n\n\nHow Much Do I Need', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'Bank & Loans', 'First', '101', 'Bank', '101', '101', 'Us', '1993', 'Tom', 'David Gardner', 'millions', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\n\n\nOther Services', 'The Ascent\n\n\n\n\n\n\n\n\n\n', 'Latest Stock Picks\n\n\n\n\n\n\nBars\n\n\n\nTimes\n\n\n\n\n\n\n\n\n\n\n\nSearch\n\n\n\n\n\n\nOur Services', 'Investing 101', 'Stocks', 'Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes', 'Dow Jones', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Value Stocks', 'Dividend Stocks', 'Large Cap Stocks', 'Blue Chip Stocks', 'Buy Stocks\n\n\n\nIndustries', 'Consumer Goods\n\n\n\n\nTechnology\n\n\n\n\nEnergy', 'Healthcare', '2023', 'Premium Services\n\n\nStock Advisor', '416%', '120%', 'Rule Breakers', '213%', '102%', 'Getting Started', 'Types of Retirement Accounts', '401k Basics', 'HSA Basics\n\n\n\nPlanning for Retirement\n\n\n\n', 'the Full Retirement Age', 'Investing for Retirement\n\n\n\n\nRetirement Strategies\n\n\n\nRetired:', '2023', 'Withdrawal Strategies\n\n\n\n\nHealthcare', 'Credit Cards', '0%', 'First', 'Guides', '101', 'Bank', '101', '101', 'Us', 'Us', 'Reviews', 'The Motley Fool Foundation', 'Newsroom', 'Us', 'Twitter', 'YouTube', 'CAPS - Stock Picking Community\n\n\n\nOther Services', '1993', 'Tom', 'David Gardner', 'millions', 'Log In\n\n\n\n\nHelp', 'Latest Stock Picks', 'The Motley Fool Foundation', 'our first year', 'China', 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'Mar 24, 2023', 'The Motley Fool’s Premium Investing Services', 'today', 'NYSE', 'TSM', 'Taiwan Semiconductor Manufacturing', 'Cap', '454B', 'Today', 'Arrow-Thin-Down', '-0.72%', '92.17', 'April 4, 2023', '10:00 a.m.', 'ET', 'the CHIPS Act', 'CHIPS Act', 'China', 'Jose Najarro', 'Nicholas Rossolillo', 'Billy Duberstein', 'March 23, 2023', 'March 24, 2023', 'Suzanne Frey', 'Alphabet', 'The Motley Fool’s', 'John Mackey', 'Whole Foods Market', 'Amazon', 'The Motley Fool’s', 'Randi Zuckerberg', 'Meta Platforms', 'Mark Zuckerberg', "The Motley Fool's", 'Billy Duberstein', 'Alphabet', 'Amazon.com', 'Meta Platforms', 'Microsoft', 'Taiwan Semiconductor Manufacturing', 'Jose Najarro', 'Alphabet', 'Meta Platforms', 'Microsoft', 'Nvidia', 'Shopify', 'Snap', 'Taiwan Semiconductor Manufacturing', 'Nicholas Rossolillo', 'Alphabet', 'Amazon.com', 'Meta Platforms', 'Nvidia', 'Shopify', 'The Motley Fool', 'Alphabet', 'Amazon.com', 'Meta Platforms', 'Microsoft', 'Nvidia', 'Pinterest', 'Shopify', 'Taiwan Semiconductor Manufacturing', 'The Motley Fool', 'Jose Najarro', 'The Motley Fool', 'The Motley Fool', 'Stocks Mentioned', 'Taiwan Semiconductor Manufacturing\n', 'TSM', '92.17', '-0.72%', '0.67', 'Snap', '11.00', '0.24', 'Taiwan Semiconductor Manufacturing', 'IBM', 'Will TSMC Be a', 'Trillion-Dollar', '2030', '3,000', '3', '2,000', 'TSMC', '2014', 'This Is How Much You Would Have Today\n\n\n\nWhy Semiconductor Stocks Taiwan Semiconductor Manufacturing', 'Micron', 'Aehr Test Systems Rallied Today', '416%', '200,000', '$1 Million', '2033', 'AI Could Have a', '$7 Trillion', '10 Years', '1', '697%', '2', 'The Motley Fool', "The Motley Fool's", 'View Premium Services', 'Linked', 'LinkedIn', 'YouTube', 'YouTube', 'Instagram\n\nInstagram\n\n\n\n\nTiktok', 'Xignite', 'The Motley Fool', 'Us', 'Careers\nResearch\nNewsroom', 'The Ascent\nAll Services', 'UK', 'Australia', 'Fool Canada\n\n\n\nFree Tools\n\nCAPS Stock Ratings\nDiscussion Boards\nCalculators', 'Lakehouse Capital', 'Trademark and Patent Information \nTerms and Conditions']}, {'File_name': 'files/5.txt', 'Entity_type': ['ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'WORK_OF_ART', 'ORDINAL', 'PERSON', 'PRODUCT', 'PERSON', 'DATE', 'PERSON', 'PERSON', 'CARDINAL', 'ORG', 'ORG', 'GPE', 'NORP', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'ORG', 'ORG', 'GPE', 'PERSON', 'NORP', 'PERSON', 'MONEY', 'WORK_OF_ART', 'ORG', 'PERSON', 'PERSON', 'LAW', 'DATE', 'GPE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'LAW', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'DATE', 'MONEY', 'GPE', 'DATE', 'ORG', 'MONEY', 'ORG', 'PERSON', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'CARDINAL', 'PERSON', 'ORG', 'ORG', 'CARDINAL', 'GPE', 'ORG', 'ORG', 'PERSON', 'PERSON', 'ORG', 'CARDINAL', 'PERSON', 'CARDINAL', 'PERSON', 'ORG', 'PERSON', 'DATE', 'ORG', 'ORG', 'GPE', 'DATE', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'GPE', 'ORG', 'DATE', 'ORDINAL', 'ORG', 'PRODUCT', 'PERSON', 'PERSON', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG'], 'Entity_identified': ['NPR', 'Keyboard', 'Navigation Menu', 'NPR Shop', 'Navigation Menu', 'Home\n\n\n\nNews\n', 'Expand', 'News\n\n\nNational\nWorld\nPolitics\nBusiness\nHealth\nScience\nClimate', 'Pop Culture\nFood\nArt & Design \nPerforming Arts\nLife Kit\n\n\n\n\nMusic\nExpand', 'Music\n\n\n\n        Tiny Desk', 'Expand', 'Podcasts & Shows\n\n\nDaily\n\n\n\n\n                                    Morning Edition\n                                \n\n\n\n\n                                    Weekend Edition', 'Saturday\n                                \n\n\n\n\n                                    ', 'Weekend Edition', 'First', 'Biden', 'Twitter', 'Flipboard\nEmail', 'March 17', 'Andrea Hsu', 'Biden', '3:54', 'Toggle', 'Download\n\n\nEmbed', 'height="290', 'Transcript', 'Biden', 'Congress', 'Maskot', 'Biden', 'Congress', 'Getty Images/Maskot', 'America', 'Biden', 'Americans', 'Biden', 'a record $600 billion', 'Build Back Better', 'Senate', 'Gina Raimondo', 'Biden', 'the CHIPS Act', 'July 25, 2022', 'Washington', 'D.C.', 'Anna Moneymaker/Getty Images', 'Anna Moneymaker/Getty Images', 'Gina Raimondo', 'Biden', 'the CHIPS Act', 'July 25, 2022', 'Washington', 'D.C.', 'Anna Moneymaker/Getty Images', 'Congress', 'last summer', '$39 billion', 'U.S.', 'late last month', 'the Commerce Department', 'more than $150 million', 'Intel', 'Keyvan Esfarjani', 'CBS News', 'months', 'Intel', 'the U.S. Chamber of Commerce', 'Commerce', 'Gina Raimondo', 'Gina Raimondo', '@SecRaimondo', 'February 27', '2023', 'Stephen Kramer', 'Bright Horizons', 'Bright Horizons', '600', 'U.S.', 'Toyota', 'Tyson Foods', 'Kramer', 'Julie Kashen', 'The Century Foundation', '90,000', 'Kashen', 'millions', 'Kashen', 'The American Government Once', 'Kashen', '1980', 'Corning', 'Corning', 'New York', 'annual', 'Kashen', 'the Commerce Department', 'Annie Dade', "Berkeley's", 'Center for the Study of Child Care Employment', 'Biden', 'Daycare Is Costly In The U.S.', 'Biden', 'Dade', 'U.S.', 'The Commerce Department', 'March 31', 'first', 'Commerce Department', 'Twitter', 'Flipboard\nEmail', 'Newsletters\nFacebook', 'Twitter\n', 'Instagram\nPress\nContact & Help', 'NPR', 'NPR', 'NPR Careers', 'NPR Shop', 'NPR Events', 'NPR Extra', 'MessageBecome', 'NPR']}, {'File_name': 'files/4.txt', 'Entity_type': ['WORK_OF_ART', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'DATE', 'TIME', 'NORP', 'NORP', 'DATE', 'WORK_OF_ART', 'CARDINAL', 'DATE', 'DATE', 'WORK_OF_ART', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORDINAL', 'DATE', 'LAW', 'PERSON', 'ORDINAL', 'ORG', 'DATE', 'TIME', 'PERSON', 'CARDINAL', 'ORDINAL', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'ORG', 'ORG', 'DATE', 'WORK_OF_ART', 'PERSON', 'DATE', 'WORK_OF_ART', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'CARDINAL', 'PERSON', 'DATE', 'CARDINAL', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'GPE', 'PERSON', 'ORG', 'ORG', 'MONEY', 'GPE', 'LANGUAGE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'WORK_OF_ART', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'DATE', 'DATE', 'MONEY', 'DATE', 'PERSON', 'PERSON', 'GPE', 'ORG', 'DATE', 'DATE', 'DATE', 'MONEY', 'ORG', 'ORDINAL', 'PERSON', 'DATE', 'ORDINAL', 'PERSON', 'CARDINAL', 'DATE', 'PERSON', 'PRODUCT', 'WORK_OF_ART', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'CARDINAL', 'DATE', 'PERSON', 'ORG', 'ORDINAL', 'ORG', 'ORG', 'DATE', 'PERSON', 'DATE', 'ORDINAL', 'ORG', 'DATE', 'ORG', 'ORG', 'ORDINAL', 'ORG', 'DATE', 'WORK_OF_ART', 'GPE', 'CARDINAL', 'PERSON', 'DATE', 'ORG', 'DATE', 'DATE', 'DATE', 'MONEY', 'ORDINAL', 'CARDINAL', 'CARDINAL', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'PERSON', 'DATE', 'ORDINAL', 'CARDINAL', 'WORK_OF_ART', 'GPE', 'ORG', 'CARDINAL', 'WORK_OF_ART', 'GPE', 'DATE', 'DATE', 'DATE', 'MONEY', 'CARDINAL', 'WORK_OF_ART', 'WORK_OF_ART', 'PERSON', 'CARDINAL', 'DATE', 'PRODUCT', 'PERSON', 'ORG', 'DATE', 'DATE', 'CARDINAL', 'MONEY', 'ORG', 'PERSON', 'PERSON', 'DATE', 'DATE', 'MONEY', 'ORG', 'PERSON', 'PERSON', 'GPE', 'PERSON', 'GPE', 'DATE', 'WORK_OF_ART', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'GPE', 'ORG', 'DATE', 'DATE', 'MONEY', 'ORG', 'ORDINAL', 'DATE', 'ORG', 'ORG', 'PERSON', 'DATE', 'ORG', 'DATE', 'PERSON', 'ORG', 'DATE', 'ORG', 'PERSON', 'ORG', 'WORK_OF_ART', 'ORDINAL', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'DATE', 'PERSON', 'DATE', 'PERSON', 'GPE', 'GPE', 'LOC', 'LOC', 'DATE', 'DATE', 'ORG', 'DATE', 'PERSON', 'DATE', 'ORG', 'ORDINAL', 'PERSON', 'PERSON', 'DATE', 'ORG', 'DATE', 'ORG', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'DATE', 'LOC', 'ORG', 'DATE', 'LOC', 'CARDINAL', 'PERSON', 'DATE', 'WORK_OF_ART', 'LOC', 'PRODUCT', 'PRODUCT', 'DATE', 'WORK_OF_ART', 'CARDINAL', 'DATE', 'PERSON', 'ORG', 'DATE', 'DATE', 'CARDINAL', 'ORG', 'DATE', 'EVENT', 'ORG', 'DATE', 'ORG', 'DATE', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'DATE', 'DATE', 'MONEY', 'WORK_OF_ART', 'WORK_OF_ART', 'ORG', 'LOC', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'DATE', 'DATE', 'DATE', 'MONEY', 'ORG', 'PERSON', 'WORK_OF_ART', 'PERSON', 'ORG', 'DATE', 'DATE', 'ORG', 'ORG', 'DATE', 'DATE', 'DATE', 'CARDINAL', 'MONEY', 'PERSON', 'LANGUAGE', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'CARDINAL', 'TIME', 'CARDINAL', 'PERSON', 'DATE', 'PERSON', 'LANGUAGE', 'GPE', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'DATE', 'ORG', 'GPE', 'DATE', 'PERSON', 'ORG', 'ORG', 'LOC', 'ORG', 'DATE', 'DATE', 'DATE', 'MONEY', 'DATE', 'PERSON', 'DATE', 'PERSON', 'GPE', 'DATE', 'CARDINAL', 'DATE', 'ORG', 'PERSON', 'PERSON', 'NORP', 'DATE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'CARDINAL', 'DATE', 'ORG', 'DATE', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'PERSON', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'DATE', 'GPE', 'CARDINAL', 'PERSON', 'DATE', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'ORG', 'DATE', 'ORG', 'ORG', 'DATE', 'PERSON', 'WORK_OF_ART', 'ORG', 'DATE', 'PERSON', 'PERSON', 'DATE', 'PERSON', 'GPE', 'CARDINAL', 'WORK_OF_ART', 'WORK_OF_ART', 'GPE', 'DATE', 'WORK_OF_ART', 'GPE', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'CARDINAL', 'PRODUCT', 'CARDINAL', 'CARDINAL', 'DATE', 'CARDINAL', 'CARDINAL', 'DATE', 'DATE', 'DATE', 'ORG', 'DATE', 'PERSON', 'CARDINAL', 'PERSON', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'DATE', 'MONEY', 'ORG', 'ORG', 'MONEY', 'CARDINAL', 'LANGUAGE', 'PERSON', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'GPE', 'PERSON', 'CARDINAL', 'ORG', 'CARDINAL', 'CARDINAL', 'ORG', 'CARDINAL', 'ORG', 'WORK_OF_ART', 'CARDINAL', 'ORG'], 'Entity_identified': ["Worf's Final Act", 'Apparel\nNewsletter', 'RSS', 'Youtube', 'Mastodon\nNewsletter\n\n\n\n\n\n\n\n\n\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\tFollow Slashdot', 'RSS', 'GitHub', 'GitHub', 'GitHub', 'SourceForge', 'nearly 30 million', 'monthly', 'less than a minute', 'Slashdot', 'Slashdot', '170474827', "Worf's Final Act", '70', 'Sunday March 05', 'The final season', 'Star Trek: Picard', 'the Klingon Worf', 'Polygon', "sci-fi's", 'Captains and Admirals', 'third', '1987', 'Star Trek', 'Gene Roddenberry', 'first', 'Star Trek TV', '1987', 'last-minute', 'Worf', 'one', 'first', 'Worf', 'Roddenberry', 'Worf', 'Worf', "Star Trek's", 'Roddenberry', 'Worf', 'Klingon', 'Worf', 'the years', 'Next Gen', 'Star Trek', 'The third season', 'Star Trek: Picard', 'Worf', 'the past 20 years', 'Star Trek', 'Quentin Tarantino', 'Worf', 'Klingon', 'Kill Bill', 'One', 'Pai Mei', 'the past 20 years', 'One', 'Starfleet', 'Klingon', 'Brent Spiner', 'Patrick Stewart', 'Chernobyl', 'Stray Dogs', 'Tolkien Fans React', 'Amazon', "$465M Series '", 'France', 'English Gaming Tech', 'Push To Preserve Language Purity', 'Amazon', 'Congress', 'Marvel Movies No Longer Guaranteed Blockbusters', 'Washington Post Urges Funding Office', "Worf's Final Act", '5', '4', '3', '2', '1', '0', 'Comments Filter', 'Cap Worf', 'YetAnotherDrew', '664604', 'Sunday March 05', '2023', '63343967', '196126', 'Worf', 'Cap Worf', 'Interesting', 'Kokuyo', '549451', 'Sunday March 05,', '2023', '63344099', 'Journal', 'first', 'Parent Share', '1175323', 'first', 'Jim', '1', '1175323', 'Worf', 'Discovery', 'Lower Decks', 'Star Trek', 'Herculean', 'Clancy', 'Hubris', 'At least Discovery', '153816', 'Worf', 'MMO', 'second', 'Polygon', 'technoviking1', '6415930', 'Megane', '129182', 'first', 'STD', '20 years ago', 'ESPN', 'SeaFox', 'first', 'STD', '20 years ago', 'ESPN.Ironically Star Trek: Picard', 'Paramount+', '1', 'Kunedog', '1033226', 'Andor', '196126', 'Sunday March 05', '2023', '63344061', 'first', '2.The', 'more than half', 'season 1', 'Borg', 'Season 3', 'Wrath of Khan', 'Paramount', 'Trek', 'Strange New Worlds', 'Parent Share', 'LondoMollari', '172563', 'Paramount', 'SNW', 'TOS', 'Kyle', '196126', 'first', 'One', 'Star Trek', 'Starfleet', 'Starfleet', 'One', "Captain's", 'Interesting', '1175323', 'Sunday March 05', '2023', '63344145', 'One', "Captain's", 'Star Trek', 'Worf', 'one', 'the present day', 'Discovery', 'Parent Share', 'Opportunist', '166417', 'Sunday March 05', '2023', '63344005', 'TNG', 'Get Wesley', 'Wil Wheaton', '2434720', 'Sunday March 05, 2023', '63344027', 'TNG', 'Get Wesley', 'Wil Wheaton', 'probably).Funny', 'Parent Share', 'Interesting', '196126', 'Star Trek', 'Star Trek', 'TNG', 'Trek', 'Data', 'Spock', 'Picard', 'Interesting', 'Opportunist', '166417', 'Sunday March 05, 2023 @11:06AM', '63344347', 'TNG', 'first', 'years', 'TOS', 'TNG', 'SciFi', 'the late 80s/early 90s', 'TNG', 'the late 90s', 'SciFi', 'TNG', '87', 'Airwolf', 'Knight Rider', 'TNG', 'Kirk', 'first', 'Spock', 'Kirk', 'Checkov', 'Crosby', 'TOS', 'Space 1999', 'Kirk', 'the late 80s', 'Kirk', 'the Soviet Union', 'Russia', 'Lower Decks', 'Lower Decks', 'the week', 'today', 'Piccard', 'every other week', 'Parent Share', '196126', 'Troi', 'first', 'Geordi', 'Scottie', 'the first season', 'Opportunist', '166417', 'Piccard', 'One', 'navy', '24/7', 'more than one', '196126', 'Lower Decks', 'Opportunist', '166417', 'Lower Decks', '1', 'Cyberax', '705495', 'Lower Decks', 'Lower Decks', 'StarTrek', 'StartTrek', 'about 25 years old', 'Lower Decks', '1', '1287354', 'Get Wesley', 'Opportunist', '166417', '9395567', 'two', 'Butchering People\n ', '690967', 'a Holy War', 'Opportunist', '166417', 'Fox News', '690967', 'Fox News', 'Wil Wheaton', 'Wil Wheaton', 'Wil Wheaton', 'LondoMollari', '172563', 'Sunday March 05, 2023', '63344049', 'Star Trek', 'The Expanse', 'Enterprise', 'Earth', 'TOS', 'TAS', 'TNG', 'VOY', 'ENT', 'SNW', 'Kirk', 'YMMV', '196126', 'Sunday March 05', '2023', '63344059', 'TNG', 'Kelvin', 'The Abrams', 'Parent Share', 'LondoMollari', '172563', '1175323', 'FTFY', 'LondoMollari', '172563', '643147', 'Sunday March 05', '2023', '63344343', 'Picard S1', 'S2', 'S3.However', 'Picard', 'Federation', 'Federation', 'Picard', 'Romulan', 'Seven', 'the last seconds', '2.0.Great', 'Parent Share', '1175323', 'Picard S1', 'S2', 'S3.However', 'Picard', 'Federation', '21st Century', 'IMO', 'Star Trek', '1960', 'Federation', 'Bourdain', '683477', 'Anonymous Coward', 'TNG', 'TNG', 'Earth', 'Federation', '1290626', 'Sunday March 05', '2023', '63344055', '1287354', 'Season 2', 'season 3', 'Anonymous Coward', 'Methadras', '1912048', '1', '4828467', 'Gates', 'Roddenberry', 'Vassili', 'Soviet', '1678196', 'Russia', 'Turkey', 'India', 'Tunisia', 'Peru', 'Hungry etc', 'One', '909048', 'Gates', 'mid 20th century', 'TNG', 'TNG', 'SuperKendall', '25149', 'S3', '909048', 'Agree', 'TNG', 'TNG', 'TNG', '20', 'WaffleMonster', '969671', 'Humanity', '2', 'Hairy Gorilla', '9839972', 'Discovery', 'Booooorrrinnnggg', 'Star Trek', 'WaffleMonster', '969671', 'STNG', 'The Worf Effect\n (Score:1', '7004192', 'Worf', 'Star Trek', 'VAXcat', '674775', 'Worf', 'Worf', '863552', 'Worf', 'SF', 'one', 'Bible', 'Trekonomics', 'Saadia', '643147', 'Trekonomics', 'Saadia', 'Star Trek', 'Economics', 'bill_mcgonigle', '4333', 'Ferengi', 'Ferengi', 'one', 'Voyager', 'only much one', 'one', '643147', 'one', '10', 'years', '6155920', '50 years old', 'Good Tea - Nice House\n ', '871664', 'Worf', 'one', "Patrick Stewart's", '3', 'JavaScript', 'Classic Discussion System', 'Related Links\nTop', 'week', '302 commentsTolkien', 'Fans React', 'Amazon', "$465M Series '", '291', 'English', 'Push To Preserve Language Purity', '288', '286', '285', 'Washington Post Urges Funding Office', '172', 'Chernobyl', 'Stray Dogs', '28', 'Slashdot Top Deals', '70', '70', 'Universe', 'zero', 'FAQ', 'Story Archive\nHall of Fame\nAdvertising\nTerms\nPrivacy Statement', '2023', 'SlashdotMedia']}, {'File_name': 'files/6.txt', 'Entity_type': ['CARDINAL', 'ORG', 'ORG'], 'Entity_identified': ['403', 'Register', 'webmaster@theregister.co.uk']}, {'File_name': 'files/7.txt', 'Entity_type': ['GPE', 'PERSON', 'ORG', 'ORG', 'WORK_OF_ART', 'CARDINAL', 'ORG', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'ORG', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'LOC', 'GPE', 'GPE', 'PRODUCT', 'PERSON', 'PERSON', 'GPE', 'GPE', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'TIME', 'ORG', 'GPE', 'MONEY', 'PERSON', 'GPE', 'PRODUCT', 'DATE', 'LAW', 'PERSON', 'DATE', 'GPE', 'DATE', 'LOC', 'ORG', 'ORG', 'PERSON', 'PERSON', 'GPE', 'GPE', 'MONEY', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'ORG', 'GPE', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'ORG', 'PERSON', 'ORG', 'MONEY', 'MONEY', 'ORG', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'DATE', 'PERSON', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'GPE', 'DATE', 'PERSON', 'PERSON', 'ORG', 'ORG', 'MONEY', 'MONEY', 'CARDINAL', 'CARDINAL', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'CARDINAL', 'ORG', 'PERSON', 'PERSON', 'ORG', 'CARDINAL', 'MONEY', 'CARDINAL', 'ORG', 'PERSON', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'GPE', 'DATE', 'PERSON', 'GPE', 'ORG', 'PERSON', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'DATE', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG', 'ORG', 'FAC', 'GPE', 'ORG', 'DATE', 'GPE', 'ORG', 'ORG', 'GPE', 'DATE', 'ORG', 'MONEY', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'LAW', 'MONEY', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'DATE', 'NORP', 'ORG', 'ORG', 'ORG', 'GPE', 'NORP', 'DATE', 'GPE', 'PERSON', 'ORG', 'PRODUCT', 'GPE', 'GPE', 'GPE', 'LAW', 'MONEY', 'CARDINAL', 'ORG', 'DATE', 'ORG', 'PERSON', 'DATE', 'ORG', 'PERSON', 'PERSON', 'ORG', 'ORG', 'PERSON', 'ORG', 'MONEY', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'GPE', 'GPE', 'ORG', 'DATE'], 'Entity_identified': ['Washington', 'Politico Logo', 'Congress', 'White House', 'Congress Minutes', 'Fifty', 'Playbook', 'Nightly', 'John Harris', 'Alex Burns', 'Jonathan Martin', 'Michael Schaffer', 'Jack Shafer', 'Matt Wuerker', 'Cartoon Carousel', 'Energy & Environment\nFinance & Tax\nHealth Care\nImmigration\nLabor\nSpace\nSustainability\nTechnology\nTrade\nTransportation', 'California', 'Canada', 'Florida', 'New Jersey', 'New York', 'Europe', 'Brussels', 'United Kingdom\n\n\n\n\n\n\nFollow', 'Twitter', 'Log In', 'Log Out\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n', 'Washington', 'Washington', 'Snap', 'Carl Court', 'Brendan Bordelon', 'Caitlin Oprysko\n', '04:30 AM EDT', 'Link Copied', 'Washington', 'an unprecedented $52 billion', 'Snap', 'FedEx', 'Coinbase', 'the last three months of 2022', 'Science Act', 'Joe Biden', 'last August', 'U.S.', 'decades', 'Asia', 'Intel', 'Samsung', 'Uncle Sam', 'Snap', 'Washington', 'Washington', 'just a few dollars', 'Scott Lincicome', 'Cato Institute', 'Lincicome', 'Josh Teitelbaum', 'Hauer & Feld', 'U.S.', 'Teitelbaum', 'Snap', 'Snap', 'Peter Boogard', 'the Commerce Department', 'Teitelbaum', 'Snap', 'Teitelbaum', 'the tens of billions', 'roughly $200 billion', 'the National Science Foundation', 'Congress', 'Lincicome', 'the XR Association', 'Science', 'The XR Association', 'the final months of 2022', 'Miranda Lutz', 'Capitol Hill', 'NSF', 'the National Institute of Standards and Technology', 'XR R&D', 'Lutz', 'Cisco Systems', 'Washington', 'the end of last year', 'Justin Sullivan/Getty Images', 'Snap', 'the XR Association', 'Science', 'tens of billions of dollars', 'hundreds of billions of dollars', 'two', 'five', 'U.S.', 'Meta', 'Microsoft', 'Google', 'Amazon', 'Apple', 'the last three months of 2022', 'five', 'Meta', 'Andy Stone', 'Kate Frischmann', 'Microsoft', 'two', '$1.5 billion', 'one', 'Microsoft', 'José Castañeda', 'Science', 'last summer', 'Google', 'Amazon', 'Amazon Web Services', 'the final months of 2022', 'Amazon', 'Apple', 'Apple', 'Cisco Systems', 'Washington', 'the end of last year', 'Allen Tsai', 'Cisco', 'Grumman', 'Steve Helber', 'Science', 'Northrop Grumman', 'General Dynamics', 'Carrier', 'Trane', 'CONSOL Energy', 'the American Coatings Association', 'one', 'TST Inc.', 'the Commerce Department', 'late last year', 'two', 'the AFL-CIO', 'the Communications Workers of America', 'the Commerce Department', 'last November', 'the International Association of Sheet Metal', 'Air, Rail and Transportation Workers', 'late last month', 'the Commerce Department', 'the Los Angeles County Metropolitan Transportation Authority', 'the Greater Pittsburgh Chamber of Commerce', 'the Port of Portland', 'the Republic of Korea', 'the Commerce Department', 'late last year', 'U.S.', 'Dish Network', 'Baxter Healthcare', 'Illumina', 'the early months', 'Congress', '$2 billion', 'Ford', 'General Motors', 'Toyota', 'Nissan', 'Hyundai', 'Honda', 'the last quarter of 2022', 'Science Act', 'the hundreds of billions of dollars', 'the University of California to', 'MIT', 'Ohio State University', 'the University of Central Florida', 'Harvard', 'the State University of New York', 'Coinbase', 'the end of last year', 'Coinbase', 'the White House Office of Science and Technology Policy', 'NSF', 'AIPAC', 'Washington', 'pro-Israel', 'late last year', 'FedEx', 'Audible', 'Amazon', 'Q4', 'Newark', 'Newark', 'New Jersey', 'Science Act’s', '$2.5 billion', 'one', 'the Flexible Packaging Association', 'the end of 2022', 'FPA', 'Alison Keane', 'late last month', 'Commerce', 'Gina Raimondo', 'Snap', 'Teitelbaum', 'Teitelbaum', 'Lincicome', 'the Commerce Department', 'more than $150 million', 'Snap', 'Lincicome', 'Lincicome', 'Lobbying', 'Google', 'Apple', 'Northrop', 'Grumman', 'Link Copied', 'Us', 'Us', 'Terms of Service', '2023']}, {'File_name': 'files/3.txt', 'Entity_type': ['GPE', 'GPE', 'GPE', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'GPE', 'GPE', 'PERSON', 'PERSON', 'NORP', 'ORG', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'ORG', 'PERSON', 'NORP', 'GPE', 'PERSON', 'GPE', 'ORG', 'ORG', 'GPE', 'GPE', 'GPE', 'NORP', 'PERSON', 'GPE', 'DATE', 'PERSON', 'NORP', 'GPE', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'NORP', 'DATE', 'NORP', 'ORG', 'GPE', 'DATE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'DATE', 'NORP', 'LAW', 'GPE', 'GPE', 'ORG', 'PRODUCT', 'ORG', 'LOC', 'ORG', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'CARDINAL', 'GPE', 'GPE', 'DATE', 'PERSON', 'TIME', 'TIME', 'ORG', 'TIME', 'GPE', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'PERSON', 'PERSON', 'NORP', 'GPE', 'ORG', 'CARDINAL', 'PERSON', 'ORG', 'PERSON', 'PERSON', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'DATE', 'ORG', 'ORG'], 'Entity_identified': ['US', 'China', 'Netherlands', 'BBC', 'HomepageSkip', 'HelpYour accountHomeNewsSportReelWorklifeTravelFutureMore', 'menuBBC', 'UkraineClimateVideoWorldUS & CanadaUKBusinessTechScienceMoreStoriesEntertainment & ArtsHealthIn', 'PicturesReality CheckWorld News', 'DataNew EconomyNew Tech EconomyCompaniesTechnology', 'SecretsGlobal TradeCost', 'China', 'Netherlands', 'MarchShareclose', 'Getty ImagesBy Annabelle  ', 'Dutch', 'ASML', 'China', 'Netherlands', 'China', 'US', 'Washington', 'US', 'China', 'the Chinese Foreign Ministry', 'Mao Ning', 'Dutch', 'China', 'Dexter Roberts', 'Washington', 'Atlantic Council', 'BBC', 'Netherlands', 'US', 'China', 'Dutch', 'Liesje Schreinemacher', 'Netherlands', 'Wednesday', 'Ms Schreinemacher', 'Dutch', 'China', 'Deep Ultra Violet', 'DUV', 'ASML', 'DUV', 'today', 'Dutch', '2019', 'Dutch', 'ASML', 'China', 'October', 'Washington', 'China', 'US', 'US', 'Netherlands', 'Japan', "South Korea's", 'US', 'earlier this week', 'South Korean', 'the Chips Act', 'the United States', 'South Korea', 'Samsung', 'JavaScript', 'youRelated TopicsCompaniesNetherlandsSemiconductorsChina-US', 'Europe', 'FebruaryMajor', 'China', 'China', 'America', 'US', 'China', 'chips13', 'US', 'China', 'December 2022Top', 'Trump', 'chargedPublished2 hours', '77 seconds', 'VideoTrump', '77 secondsPublished14 hours', 'Taiwan', 'one', 'Trump', 'Trump', 'Trump', 'beats’Is Taiwan', 'Mario', 'Jack Black', 'Russian', "reveal?'I'm", 'the X-planes?The NYC', 'only one', 'Trump', 'SNP', 'probe3Who', 'Karen McDougal', 'Trump', '34', 'Trump', 'Camilla', 'Trump', 'Wisconsin Supreme', 'BBC NewsHomeNewsSportReelWorklifeTravelFutureCultureMusicTVWeatherSoundsTerms', 'UseAbout', 'NewslettersWhy', 'BBCAdvertise', '2023', 'BBC', 'BBC']}, {'File_name': 'files/2.txt', 'Entity_type': ['PERSON', 'LAW', 'GPE', 'ORG', 'TIME', 'GPE', 'LAW', 'GPE', 'PRODUCT', 'DATE', 'ORG', 'DATE', 'ORG', 'GPE', 'GPE', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'ORG', 'LAW', 'PERSON', 'DATE', 'MONEY', 'LAW', 'LAW', 'PERSON', 'GPE', 'GPE', 'NORP', 'ORG', 'ORG', 'NORP', 'NORP', 'LAW', 'GPE', 'GPE', 'LAW', 'GPE', 'GPE', 'DATE', 'ORG', 'GPE', 'PERSON', 'PERSON', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'GPE', 'CARDINAL', 'CARDINAL', 'ORDINAL', 'ORG', 'LAW', 'DATE', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'NORP', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'DATE', 'ORG', 'PRODUCT', 'ORG', 'ORG', 'PERSON', 'PRODUCT', 'PERSON', 'ORG', 'ORG', 'GPE', 'ORG', 'ORG', 'ORG', 'FAC', 'ORG', 'ORG', 'TIME', 'ORG'], 'Entity_identified': ["S. Korea's", 'Chips Act', 'US', 'Reuters', '20233:55 AM', 'Korea', 'Chips Act', 'US', 'YangSemiconductor', 'February 17, 2023', 'REUTERS', 'March 8', 'Reuters', "South Korea's", 'Washington', 'this week', 'U.S.', "South Korea's", 'Ahn Duk', 'the U.S. Commerce Department', 'White House', 'the Chips Act', 'Biden', 'last month', '$52.7 billion', 'the Chips Act', 'The Chips Act', 'Biden', 'U.S.', 'China', 'South Korean', 'Samsung Electronics Co Ltd', 'SK Hynix Inc', 'South Korean', 'South Korean', 'the Chips Act', 'the United States', "South Korea's", 'the Chips Act', 'U.S.', 'China', '10 years', 'Samsung Electronics', 'China', 'Heekyong Yang', 'Jamie FreedOur Standards', 'The Thomson Reuters Trust Principles', 'AI', 'Nvidia', 'A100', 'Japan', '4', '2023TechnologycategoryAustralian', 'first', 'EU', 'Chips Act', 'April 18', 'AM UTCSite IndexBrowseWorldBusinessLegalMarketsBreakingviewsTechnologyInvestigations', 'tabLifestyleAbout ReutersAbout Reuters', 'tabCareers', 'tabReuters News Agency', 'Attribution Guidelines', 'tabReuters', 'tabReuters Fact Check', 'tabReuters Diversity Report', 'InformedDownload the App', 'tabNewsletters', 'tabInformation', 'Thomson Reuters', 'billions', 'every day', 'Reuters', 'ProductsWestlaw', 'tabBuild', 'Checkpoint', 'Refinitiv ProductsRefinitiv Workspace', 'Access', 'Refinitiv Data Catalogue', 'Refinitiv World-Check', 'tabScreen', 'Us', 'tabAdvertising Guidelines', 'tabCookies', 'tabPrivacy', 'tabDigital Accessibility', 'tabCorrections', 'Feedback', '15 minutes', 'Reuters']}, {'File_name': 'files/1.txt', 'Entity_type': ['PERSON', 'PRODUCT', 'GPE', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'PERSON', 'DATE', 'ORG', 'PERSON', 'MONEY', 'GPE', 'ORG', 'DATE', 'LAW', 'DATE', 'ORG', 'MONEY', 'DATE', 'MONEY', 'GPE', 'ORG', 'DATE', 'DATE', 'ORG', 'PERSON', 'ORG', 'ORG', 'GPE', 'GPE', 'ORG', 'LAW', 'CARDINAL', 'ORG', 'GPE', 'MONEY', 'ORG', 'ORG', 'MONEY', 'ORG', 'LAW', 'ORG', 'ORG', 'ORG', 'ORG', 'PRODUCT', 'ORG', 'CARDINAL', 'DATE', 'WORK_OF_ART', 'DATE', 'GPE', 'ORG'], 'Entity_identified': ['Biden', 'CHIPS Act', 'China', 'TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest', 'TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest', 'GuidesGamingGearEntertainmentTomorrowDealsNewsVideoPodcastsLoginSponsored LinksBiden', 'CHIPS Act', 'ChinaChipmakers', 'Florence Lo', '4, 2023', 'PMChipmakers', 'Biden', '$39 billion', 'China', 'the US Commerce Department', 'this week', 'the CHIPS Act', 'late June', 'Congress', '$280 billion', 'last July', '$52 billion', 'US', 'Recipients', 'a period of', '10 years', 'Commerce', 'Gina Raimondo', 'the Financial Times', 'Raimondo', 'China', 'US', 'Raimondo', 'CHIPS Act', 'one', 'Ford', 'China', 'no CHIPS dollars', 'Raimondo', 'The Commerce Department', 'more than $150 million', 'Raimondo', 'the CHIPS Act', 'the Engadget Deals NewsletterGreat', 'Engadget', "Engadget's Terms and Privacy Policy", 'CommentsBiden', 'CHIPS Act', 'ChinaChips', 'two', 'weekly', 'dealsThe Morning After - A', 'daily', 'Us', 'UsReprints']}, {'File_name': 'files/20.txt', 'Entity_type': ['LAW', 'ORG', 'PERSON', 'GPE', 'ORG', 'PERSON', 'ORG', 'ORG', 'GPE', 'ORG', 'ORG', 'ORG', 'CARDINAL', 'ORG', 'ORDINAL', 'PERSON', 'ORG', 'PERSON', 'GPE', 'ORG', 'NORP', 'ORG', 'ORG', 'CARDINAL', 'PERSON', 'PERSON', 'GPE', 'CARDINAL', 'ORG', 'ORG', 'WORK_OF_ART', 'PERSON', 'ORG', 'ORG', 'GPE', 'ORG', 'PERSON', 'ORG', 'DATE', 'LAW', 'PERSON', 'DATE', 'ORG', 'LAW', 'LAW', 'GPE', 'PERSON', 'ORG', 'ORG', 'WORK_OF_ART', 'ORG', 'ORG', 'LOC', 'GPE', 'PERCENT', 'DATE', 'CARDINAL', 'ORG', 'ORG', 'PERSON', 'PERSON', 'PERSON', 'CARDINAL', 'DATE', 'DATE', 'PERSON', 'PERSON', 'GPE', 'GPE', 'GPE', 'GPE', 'DATE', 'MONEY', 'DATE', 'GPE', 'PERCENT', 'DATE', 'NORP', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'NORP', 'PERSON', 'ORG', 'NORP', 'LOC', 'LAW', 'PERSON', 'PERSON', 'DATE', 'GPE', 'PERSON', 'ORG', 'CARDINAL', 'CARDINAL', 'LAW', 'GPE', 'MONEY', 'MONEY', 'MONEY', 'ORG', 'PERSON', 'EVENT', 'NORP', 'CARDINAL', 'ORG', 'GPE', 'QUANTITY', 'MONEY', 'PERSON', 'ORG', 'PERSON', 'LAW', 'CARDINAL', 'LOC', 'CARDINAL', 'ORG', 'ORG', 'CARDINAL', 'CARDINAL', 'MONEY', 'PERSON', 'LAW', 'CARDINAL', 'ORG', 'PERSON', 'PERSON', 'MONEY', 'QUANTITY', 'GPE', 'NORP', 'ORG', 'CARDINAL', 'QUANTITY', 'GPE', 'LAW', 'CARDINAL', 'CARDINAL', 'GPE', 'CARDINAL', 'GPE', 'GPE', 'ORG', 'MONEY', 'CARDINAL', 'NORP', 'PERSON', 'GPE', 'PERSON', 'PERSON', 'GPE', 'PRODUCT', 'ORG', 'ORG', 'ORG', 'PERSON', 'ORG', 'GPE', 'ORG', 'ORG', 'PERSON', 'PERSON', 'ORG', 'GPE', 'PERSON', 'PERSON', 'ORG', 'PERSON', 'WORK_OF_ART', 'ORG', 'ORDINAL', 'DATE', 'CARDINAL', 'ORG', 'WORK_OF_ART', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'GPE', 'ORG', 'GPE', 'ORG', 'ORG', 'PERSON', 'PERSON', 'ORDINAL', 'WORK_OF_ART'], 'Entity_identified': ["The CHIPS Act: Rebuilding America's", 'CBS News', 'Donald Trump Indictment', 'Finland', 'NATO', 'Biden', 'CBS News Live', 'Newsletters\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nNews', 'US', 'HealthWatch', 'MoneyWatch', 'CBS Village\n\n\n\n\nTechnology\n\n\n\n\nScience\n\n\n\n\nCrime\n\n\n\n\nSports\n\n\n\n\nEssentials', '34', 'Trump', 'first', 'Janet Protasiewicz', 'Wisconsin Supreme Court', 'Brandon Johnson', 'Chicago', 'Taliban', 'Afghan', 'U.N.', 'Trump', '34', "Donald Trump's", 'Clashes', 'Jerusalem', 'hundreds', 'Trump', 'Stormy Daniels', 'Shows\n\n\n\n\nLive\n\n\n\n\nLocal\n\n\n\n\nMore\n\n\n\n\nLatest\n\n\n\n\nVideo\n\n\n\n\nPhotos\n\n\n\n\nPodcasts\n\n\n\n\nIn Depth\n\n\n\n\nLocal\n\n\n\n\nGlobal Thought Leaders\n\n\n\n\nInnovators & Disruptors\n\n\n\n\n\n', 'Newsletters', 'RSS', 'CBS Store', 'Paramount+', 'Join Our Talent Community', 'Davos 2023', 'CBS News', 'Sunday', "The CHIPS Act: Rebuilding America's", 'David Pogue', 'March 5, 2023 /', 'CBS News', 'The CHIPS Act: Made in America', 'The CHIPS Act', 'America', 'Chris Miller', "Tuft University's", 'Fletcher School', "Chip War: The Fight for the World's Most Critical Technology", 'CBS', 'Simon & Schuster', 'East Asia', 'Taiwan', '90 percent', 'the last 30 years', 'one', 'the Taiwan Semiconductor Manufacturing Company', 'TSMC', 'Pogue', 'Miller', 'Miller', 'hundreds', 'weeks', 'months', 'Miller', 'Miller', 'China', 'Taiwan', 'China', 'Taiwan', 'many years', 'the trillions of dollars', 'the 1990s', "the United States'", '37% to 12%', 'Today', 'American', 'Apple', 'AMD', 'nVidia', 'TSMC', 'TSMC', 'Intel', 'American', 'Al Thompson', 'Intel', 'East Asian', 'East Asia', 'The CHIPS Act', 'Trump', 'Biden', 'last August', 'America', 'Biden', 'Thompson', 'two', 'two', '"The CHIPS Act', 'America', '$13 billion', '$39 billion', '$24 billion', 'Intel', 'Pat Gelsinger', 'World War II."If', 'American', 'two', 'Intel', 'Arizona', '650,000 square feet', 'a little over $20 billion', 'Keyvan Esfarjani', 'Intel', 'Pogue', 'the CHIPS Act', 'One', 'Earth', 'thousand', 'Intel', 'CBS News', 'billions', 'one', 'hundreds of billions', 'angstrom', 'the CHIPS Act', 'One', 'Intel', 'Keyvan Esfarjani', 'Chris Miller', '$52 billion', 'the CHIPS Act', 'Taiwan', 'American', 'Intel', 'eight', '2,000 acres', 'Ohio', 'the CHIPS Act', '14', '22', 'America', 'two', 'Arizona', 'Taiwan', 'TSMC', '$160 billion', '28,000', 'American', 'Al Thompson', 'U.S.', 'Chris Miller', 'Scribner', 'Hardcover', 'eBook', 'Audio', 'Amazon', 'Barnes & Noble', 'IndieboundChristopher Miller', 'Tufts UniversityTaiwan Semiconductor Manufacturing Company', 'Hsinchu', 'TaiwanIntelSemiconductor Degrees Program', 'Purdue University\xa0 \xa0 \xa0Story', 'Mark Hudspeth', 'Lauren Barnello', 'Trending News', 'U.S.', 'John Fetterman', 'Michael Cohen', 'Trump', 'Neil Diamond', 'A Beautiful Noise', 'COVID', 'First', 'March 5, 2023 / 9:11 AM', '2023', 'CBS Interactive Inc.', 'All Rights Reserved', 'CBS NEWS', '2023', 'CBS Interactive Inc.', 'CBS News Live', 'Paramount+', 'CBS News Store', 'Us', 'View CBS News In', 'CBS News App', 'Chrome', 'Safari', 'first', 'Turn On\n\n\n']}, {'File_name': 'files/19.txt', 'Entity_type': ['ORG', 'ORG', 'LOC', 'ORG', 'CARDINAL', 'ORG', 'ORG', 'DATE', 'PERSON', 'ORG', 'PERSON', 'GPE', 'DATE', 'GPE', 'PERSON', 'GPE', 'GPE', 'DATE', 'GPE', 'PERSON', 'GPE', 'NORP', 'PERSON', 'ORG', 'TIME', 'GPE', 'GPE', 'DATE', 'DATE', 'ORG', 'ORG', 'ORG', 'EVENT', 'TIME', 'GPE', 'CARDINAL', 'PERSON', 'CARDINAL', 'PRODUCT', 'CARDINAL', 'ORG', 'ORG', 'GPE', 'ORG', 'GPE', 'GPE', 'NORP', 'ORDINAL', 'DATE', 'ORG', 'ORG', 'GPE', 'CARDINAL', 'DATE', 'ORG', 'GPE', 'CARDINAL', 'PERCENT', 'MONEY', 'ORG', 'DATE', 'GPE', 'MONEY', 'PERSON', 'PERSON', 'PERSON', 'DATE', 'ORG', 'CARDINAL', 'DATE', 'DATE', 'PRODUCT', 'CARDINAL', 'ORG', 'PRODUCT', 'NORP', 'ORG', 'DATE', 'GPE', 'GPE', 'DATE', 'PERSON', 'ORG', 'GPE', 'GPE', 'CARDINAL', 'CARDINAL', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'CARDINAL', 'CARDINAL', 'ORG', 'DATE', 'CARDINAL', 'DATE', 'ORG', 'GPE', 'QUANTITY', 'GPE', 'PERSON', 'GPE', 'LOC', 'QUANTITY', 'ORG', 'EVENT', 'DATE', 'ORG', 'GPE', 'EVENT', 'DATE', 'TIME', 'ORG', 'GPE', 'ORG', 'DATE', 'GPE', 'CARDINAL', 'FAC', 'ORG', 'GPE', 'MONEY', 'ORG', 'GPE', 'NORP', 'NORP', 'GPE', 'GPE', 'DATE', 'NORP', 'FAC', 'PERSON', 'WORK_OF_ART', 'ORG', 'ORG', 'DATE', 'DATE', 'GPE', 'GPE', 'ORG', 'GPE', 'EVENT', 'PERCENT', 'GPE', 'MONEY', 'DATE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'GPE', 'LAW', 'DATE', 'MONEY', 'GPE', 'DATE', 'PERCENT', 'GPE', 'NORP', 'GPE', 'DATE', 'DATE', 'GPE', 'CARDINAL', 'CARDINAL', 'ORG', 'GPE', 'PRODUCT', 'ORG', 'GPE', 'ORG', 'CARDINAL', 'LOC', 'ORG', 'ORG', 'PRODUCT', 'GPE', 'ORG', 'CARDINAL', 'PERSON', 'QUANTITY', 'CARDINAL', 'ORG', 'PERSON', 'GPE', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'DATE', 'PERSON', 'ORG', 'DATE', 'CARDINAL', 'DATE', 'CARDINAL', 'GPE', 'GPE', 'GPE', 'CARDINAL', 'CARDINAL', 'CARDINAL', 'DATE', 'PERSON', 'PERSON', 'DATE', 'CARDINAL', 'ORG'], 'Entity_identified': ['The American Factories Building an Arsenal for Ukraine and Democracy - The AtlanticSkip', 'NavigationThe', 'The Atlantic ArchivePlay', 'Trump', '34', 'IdeasThe Arsenal of Democracy Is Reopening', 'BusinessBut', 'decades', 'Elliot AckermanKevin Lamarque / ReutersMarch', '2023ShareSaved', 'Lockheed Martin', 'Arkansas', 'the end of February', 'America', 'Biden', 'Ukraine', 'the United States', 'decades', 'America', 'Elliot Ackerman', 'Ukraine', 'American', 'Becky Withrow', 'Lockheed', '90 minutes', 'Little Rock', 'East Camden', 'opening-day', '2017', 'Ford', 'Willow Run', 'B-24 Liberator', 'the Second World War', 'every hour', 'Ukraine', 'Dozens', 'Withrow', 'two', 'M270', 'M270', 'Lockheed', 'Red River Army Depot', 'Texas', 'Lockheed', 'Camden', 'Ukraine', 'American', 'first', '2013', 'Lockheed', 'HIMARS', 'the United Arab Emirates', '12', '2017', 'NATO', 'Ukraine', 'at least 20', '10 M270s', '$67.1 billion', 'Congress', 'last year', 'Ukraine', '$631 million', 'Lockheed Martin', 'Withrow', 'Dennis Truelove', '40-year', 'Lockheed', 'M270', 'decades', 'today', 'M270', 'More than one', 'Lockheed', 'M270', 'Russian', 'Lockheed', 'decades', 'Truelove', 'Arkansas', '2022', 'Winner', 'Cheetos', 'Arkansas', 'Camden', '48', '48', 'each year', '96', 'the third quarter of 2025', 'two and a half years', 'One', '1,300', 'Lockheed', 'four days', '200', 'the next five years', 'Lockheed', 'Camden', '2,427 acres', 'Camden', 'Truelove', 'U.S.', 'Highland Industrial Park', '18,500 acres', 'the Shumaker Ammunition Depot', 'the Second World War', 'March 2023', 'Navy', 'East Camden', 'the Second World War', 'today', '30 minutes', 'Boots & Liquor', 'East Camden', 'Lockheed', 'the past four years', 'Camden', 'more than 1,000', 'Highland Industrial Park', 'General Dynamics, Raytheon', 'Aerojet Rocketdyne', '$67.1 billion', 'Congress', 'Ukraine', 'Ukrainian', 'American', 'Ukraine', 'America', 'Nine months', 'Japanese', 'Pearl Harbor', 'Franklin Roosevelt', 'An Act to Promote the Defense of the United States', 'the Lend-Lease Act', 'Congress', '1935', '1937', 'the United States', 'America', 'Lend-Lease', 'America', 'the Second World War', '17 percent', 'U.S.', '$719 billion', 'today', 'Britain', 'France', 'the Soviet Union', 'China', 'U.S.', 'U.S.', 'Science Act', 'last August', '$280 billion', 'the United States', 'Today', 'more than 90 percent', 'Taiwan', 'Chinese', 'Ukraine', 'decades', 'an average day', 'Ukraine', 'two', 'approximately 30,000', 'NATO', 'Russia', 'M270', 'GMLRS', 'Camden', 'GMLRS', 'hundreds', 'the Highland Industrial Park', 'Lockheed', 'GMLRS', 'M270', 'Camden', 'GMLRS', 'half a dozen', 'Jervis Webb', '200-pound', 'six', 'Lockheed', 'Eliot A. Cohen', 'Ukraine', 'a dozen', 'one', '52', 'the end of the day', 'Jay Price', 'Lockheed', 'last year', '7,500', 'This year', '10,000', 'Ukraine', 'China', 'Taiwan', 'Fifty-two', '365', '18,980', 'each year', 'Price', 'Price', 'daily', '52', 'Lockheed']}, {'File_name': 'files/18.txt', 'Entity_type': ['LAW', 'GPE', 'PERSON', 'PERSON', 'ORG', 'WORK_OF_ART', 'PERSON', 'PERSON', 'LAW', 'GPE', 'PERSON', 'ORG', 'DATE', 'GPE', 'DATE', 'PERSON', 'WORK_OF_ART'], 'Entity_identified': ['CHIPS Act', 'Oregon', 'Login / Sign Up', 'Crypto News', 'Media News', 'Hot  Live  Light Mode  Imagery  Search  Customize News Grid', 'App', 'Telegram Discord', 'CHIPS Act', 'Oregon', 'Aaron Nichols', 'Intel', 'exactly 50 years', 'Oregon', '2023-03-26', 'Telegram Discord', "Status      The Web's Most Comprehensive Business News Site"]}]


### Let's visualize our results in a Pandas Dataframe sorted by the file name


```python
df_NER = pd.DataFrame(all_entities)
df_NER = df_NER.sort_values(by='File_name', ascending=True)
df_NER 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File_name</th>
      <th>Entity_type</th>
      <th>Entity_identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>files/1.txt</td>
      <td>[PERSON, PRODUCT, GPE, ORG, ORG, ORG, PRODUCT, ORG, PERSON, DATE, ORG, PERSON, MONEY, GPE, ORG, DATE, LAW, DATE, ORG, MONEY, DATE, MONEY, GPE, ORG, DATE, DATE, ORG, PERSON, ORG, ORG, GPE, GPE, ORG, LAW, CARDINAL, ORG, GPE, MONEY, ORG, ORG, MONEY, ORG, LAW, ORG, ORG, ORG, ORG, PRODUCT, ORG, CARDINAL, DATE, WORK_OF_ART, DATE, GPE, ORG]</td>
      <td>[Biden, CHIPS Act, China, TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest, TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest, GuidesGamingGearEntertainmentTomorrowDealsNewsVideoPodcastsLoginSponsored LinksBiden, CHIPS Act, ChinaChipmakers, Florence Lo, 4, 2023, PMChipmakers, Biden, $39 billion, Chin...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>files/10.txt</td>
      <td>[PERSON, ORG, WORK_OF_ART, PERSON, PERSON, PERCENT, ORG, PERCENT, PERCENT, PERCENT, ORG, PERCENT, ORG, PERCENT, PERCENT, PERCENT, ORG, PERCENT, ORG, PERCENT, PERCENT, PERCENT, PERCENT, PERCENT, PERCENT, PERCENT, ORG, CARDINAL, ORG, ORG, GPE, ORG, ORG, PERSON, ORDINAL, ORG, FAC, PERSON, DATE, DATE, ORG, DATE, NORP, FAC, ORG, GPE, PERSON, GPE, MONEY, ORG, ORG, ORG, NORP, ORG, ORG, GPE, DATE, ORG...</td>
      <td>[Login / Sign Up  Crypto News, Media News, Hot  Live  Light Mode  Imagery  Search  Customize News Grid, App, Telegram Discord, 4,124 0.12%, S&amp;P, 0.58%, 33,402 0.59%, 0.52%, Russel, 1,770 1.81%, VIX, 19.69 6.15%, 10Y 3.34 0.00%, 40,523 0.72%, FTSE, 7,663 0.37%, EuroStoxx, 0.13%, 15,562 0.27%, 20,275 0.66%, 27,813 1.68%, 0.24%, 0.07%, 28,550 1.36%, Fear &amp; Greed, 50/100, Intel, Micron Fall After,...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>files/11.txt</td>
      <td>[LAW, ORG, PERSON, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, GPE, ORG, ORG, ORG, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, ORG, ORG, PERSON, ORG, ORG, ORG, ORG, DATE, WORK_OF_ART, PRODUCT, ORG, LAW, ORG, DATE, ORG, ORG, PERCENT, ORG, ORDINAL, CARDINAL, ORG, CARDINAL, CARDINAL, GPE, DATE, PERSON, PERSON, CARDINAL, GPE, PERSON, ORG, WORK_OF_ART, GPE, PERSON, ORG, ORG, ORG, ORG, FAC...</td>
      <td>[The CHIPS Act Is Accepting Applications, The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease, Javascript, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Stocks, Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes, Dow Jones, Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>files/12.txt</td>
      <td>[]</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>files/13.txt</td>
      <td>[LAW, ORG, PERSON, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, GPE, ORG, ORG, ORG, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, ORG, ORG, PERSON, ORG, ORG, ORG, ORG, DATE, ORG, WORK_OF_ART, PRODUCT, ORG, LAW, ORG, DATE, ORG, ORG, PERCENT, ORG, ORDINAL, CARDINAL, ORG, CARDINAL, CARDINAL, GPE, DATE, PERSON, PERSON, CARDINAL, GPE, PERSON, ORG, WORK_OF_ART, GPE, PERSON, ORG, ORG, ORG, ORG...</td>
      <td>[The CHIPS Act's, The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease, Javascript, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Stocks, Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes, Dow Jones, Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Value Stocks, D...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>files/14.txt</td>
      <td>[ORG, GPE, DATE, ORG, ORG, WORK_OF_ART, PERSON, PERSON, PERSON, WORK_OF_ART, PRODUCT, ORG, PERSON, ORG, ORG, ORG, ORG, ORG, ORG, ORG, ORG, ORG, ORG, WORK_OF_ART, PERSON, PRODUCT, WORK_OF_ART, PERSON, PRODUCT, DATE, EVENT, ORG, GPE, DATE, PERSON, ORG, GPE, DATE, TIME, CARDINAL, ORG, ORDINAL, GPE, DATE, PERSON, ORG, DATE, DATE, DATE, PERSON, PERSON, DATE, DATE, ORDINAL, ORG, PERSON, DATE, GPE, G...</td>
      <td>[Tesla, Mexico, next year, Luxury News, Hybrid News\nElectric News, Weird Car News, Junkyard Gems\n, Newsletters\nPhotos\nNews, Car Reviews, Most Reliable Cars\nBuying Guides\nVideos, Car Values\nCar Finder\nCompare Vehicles\nDealers, Car Insurance\nRepair Shops, Buyer, Chevrolet, Chrysler, Dodge, Ford, GMC, Honda, Jeep, Lexus, Toyota, Volvo, Cars for Sale\n\n\nCars for Sale\n\n\nNew Cars for ...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>files/15.txt</td>
      <td>[CARDINAL, PERSON, ORG, ORG, ORG, ORG, PERSON]</td>
      <td>[403, Request, CloudFront, CloudFront, CloudFront, Request, ksF1nKbP2kaMk6WLD8ZzAd8dibgqi9k0KFdn5eagZ3MDjdzDA9wZiw==]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>files/16.txt</td>
      <td>[GPE, PERSON, ORDINAL, GPE, ORG, ORG, FAC, WORK_OF_ART, ORG, PERSON, ORDINAL, GPE, NORP, GPE, PERSON, NORP, PERSON, FAC, GPE, GPE, DATE, PERSON, CARDINAL, DATE, GPE, GPE, GPE, PERSON, LOC, GPE, PERSON, NORP, PERSON, DATE, DATE, NORP, PERSON, PERSON, PERSON, NORP, PERSON, DATE, PERSON, GPE, DATE, ORG, PERSON, ORG, GPE, LOC, PERSON, GPE, ORDINAL, DATE, GPE, GPE, CARDINAL, GPE, GPE, DATE, MONEY, ...</td>
      <td>[Malaysia, Anwar, first, China, International Trade, Al Jazeera, Skip linksSkip, Contentplay Live Show, TradeMalaysia, Anwar, first, China, Malaysian, China, Xi Jinping, Malaysian, Anwar Ibrahim, Rizal Park, Manila, Philippines, Thursday, March 2, 2023, Lisa Marie David/Pool Photo, 30, 202330, 2023Kuala Lumpur, Malaysia, Malaysia, Anwar Ibrahim, the South China Sea, China, Anwar, Chinese, Xi J...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>files/17.txt</td>
      <td>[LAW, PERSON, PRODUCT, GPE, ORG, ORG, WORK_OF_ART, ORG, PERSON, PERSON, ORG, GPE, ORG, LAW, ORG, GPE, GPE, PERSON, PERSON, PERSON, DATE, GPE, ORG, DATE, GPE, ORG, PERSON, ORG, DATE, ORG, NORP, ORG, ORG, DATE, PERSON, NORP, GPE, LAW, GPE, ORG, ORG, ORDINAL, CARDINAL, GPE, GPE, GPE, GPE, GPE, GPE, ORG, GPE, GPE, ORG, NORP, DATE, GPE, DATE, GPE, ORG, PERSON, GPE, ORG, GPE, ORG, GPE, GPE, DATE, LA...</td>
      <td>[The RESTRICT Act, Mark Warner, TikTok - Rest of World, U.S., TikTok, Latest Stories\nAccess &amp; Connectivity\nCreators &amp; Communities, The Platform Economy\n \n\n\n\n\n\nLearn, Team news, Email, Dark Mode, TikTok Domination\nMeet, U.S., TikTok, the RESTRICT Act, Warner, U.S., China, Greg Kahn, Greg Kahn, Russell Brandom, 29 March 2023, Washington, TikTok, years, the United States, TikTok, Shou Z...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>files/18.txt</td>
      <td>[LAW, GPE, PERSON, PERSON, ORG, WORK_OF_ART, PERSON, PERSON, LAW, GPE, PERSON, ORG, DATE, GPE, DATE, PERSON, WORK_OF_ART]</td>
      <td>[CHIPS Act, Oregon, Login / Sign Up, Crypto News, Media News, Hot  Live  Light Mode  Imagery  Search  Customize News Grid, App, Telegram Discord, CHIPS Act, Oregon, Aaron Nichols, Intel, exactly 50 years, Oregon, 2023-03-26, Telegram Discord, Status      The Web's Most Comprehensive Business News Site]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>files/19.txt</td>
      <td>[ORG, ORG, LOC, ORG, CARDINAL, ORG, ORG, DATE, PERSON, ORG, PERSON, GPE, DATE, GPE, PERSON, GPE, GPE, DATE, GPE, PERSON, GPE, NORP, PERSON, ORG, TIME, GPE, GPE, DATE, DATE, ORG, ORG, ORG, EVENT, TIME, GPE, CARDINAL, PERSON, CARDINAL, PRODUCT, CARDINAL, ORG, ORG, GPE, ORG, GPE, GPE, NORP, ORDINAL, DATE, ORG, ORG, GPE, CARDINAL, DATE, ORG, GPE, CARDINAL, PERCENT, MONEY, ORG, DATE, GPE, MONEY, PE...</td>
      <td>[The American Factories Building an Arsenal for Ukraine and Democracy - The AtlanticSkip, NavigationThe, The Atlantic ArchivePlay, Trump, 34, IdeasThe Arsenal of Democracy Is Reopening, BusinessBut, decades, Elliot AckermanKevin Lamarque / ReutersMarch, 2023ShareSaved, Lockheed Martin, Arkansas, the end of February, America, Biden, Ukraine, the United States, decades, America, Elliot Ackerman,...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>files/2.txt</td>
      <td>[PERSON, LAW, GPE, ORG, TIME, GPE, LAW, GPE, PRODUCT, DATE, ORG, DATE, ORG, GPE, GPE, DATE, GPE, GPE, PERSON, ORG, ORG, LAW, PERSON, DATE, MONEY, LAW, LAW, PERSON, GPE, GPE, NORP, ORG, ORG, NORP, NORP, LAW, GPE, GPE, LAW, GPE, GPE, DATE, ORG, GPE, PERSON, PERSON, ORG, ORG, ORG, PRODUCT, GPE, CARDINAL, CARDINAL, ORDINAL, ORG, LAW, DATE, ORG, ORG, ORG, ORG, ORG, NORP, ORG, ORG, WORK_OF_ART, ORG,...</td>
      <td>[S. Korea's, Chips Act, US, Reuters, 20233:55 AM, Korea, Chips Act, US, YangSemiconductor, February 17, 2023, REUTERS, March 8, Reuters, South Korea's, Washington, this week, U.S., South Korea's, Ahn Duk, the U.S. Commerce Department, White House, the Chips Act, Biden, last month, $52.7 billion, the Chips Act, The Chips Act, Biden, U.S., China, South Korean, Samsung Electronics Co Ltd, SK Hyni...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>files/20.txt</td>
      <td>[LAW, ORG, PERSON, GPE, ORG, PERSON, ORG, ORG, GPE, ORG, ORG, ORG, CARDINAL, ORG, ORDINAL, PERSON, ORG, PERSON, GPE, ORG, NORP, ORG, ORG, CARDINAL, PERSON, PERSON, GPE, CARDINAL, ORG, ORG, WORK_OF_ART, PERSON, ORG, ORG, GPE, ORG, PERSON, ORG, DATE, LAW, PERSON, DATE, ORG, LAW, LAW, GPE, PERSON, ORG, ORG, WORK_OF_ART, ORG, ORG, LOC, GPE, PERCENT, DATE, CARDINAL, ORG, ORG, PERSON, PERSON, PERSON...</td>
      <td>[The CHIPS Act: Rebuilding America's, CBS News, Donald Trump Indictment, Finland, NATO, Biden, CBS News Live, Newsletters\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nNews, US, HealthWatch, MoneyWatch, CBS Village\n\n\n\n\nTechnology\n\n\n\n\nScience\n\n\n\n\nCrime\n\n\n\n\nSports\n\n\n\n\nEssentials, 34, Trump, first, Janet Protasiewicz, Wisconsin Supreme Court, Brandon Johnson, Chicago, Taliban, Afg...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>files/3.txt</td>
      <td>[GPE, GPE, GPE, ORG, ORG, PRODUCT, ORG, ORG, ORG, ORG, PRODUCT, GPE, GPE, PERSON, PERSON, NORP, ORG, GPE, GPE, GPE, GPE, GPE, GPE, GPE, ORG, PERSON, NORP, GPE, PERSON, GPE, ORG, ORG, GPE, GPE, GPE, NORP, PERSON, GPE, DATE, PERSON, NORP, GPE, ORG, ORG, ORG, ORG, DATE, NORP, DATE, NORP, ORG, GPE, DATE, GPE, GPE, GPE, GPE, GPE, GPE, GPE, GPE, DATE, NORP, LAW, GPE, GPE, ORG, PRODUCT, ORG, LOC, ORG...</td>
      <td>[US, China, Netherlands, BBC, HomepageSkip, HelpYour accountHomeNewsSportReelWorklifeTravelFutureMore, menuBBC, UkraineClimateVideoWorldUS &amp; CanadaUKBusinessTechScienceMoreStoriesEntertainment &amp; ArtsHealthIn, PicturesReality CheckWorld News, DataNew EconomyNew Tech EconomyCompaniesTechnology, SecretsGlobal TradeCost, China, Netherlands, MarchShareclose, Getty ImagesBy Annabelle  , Dutch, ASML,...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>files/4.txt</td>
      <td>[WORK_OF_ART, ORG, ORG, PERSON, ORG, ORG, ORG, ORG, ORG, ORG, CARDINAL, DATE, TIME, NORP, NORP, DATE, WORK_OF_ART, CARDINAL, DATE, DATE, WORK_OF_ART, PERSON, ORG, ORG, ORG, ORDINAL, DATE, LAW, PERSON, ORDINAL, ORG, DATE, TIME, PERSON, CARDINAL, ORDINAL, PERSON, PERSON, PERSON, PERSON, ORG, PERSON, PERSON, PERSON, PERSON, DATE, ORG, ORG, DATE, WORK_OF_ART, PERSON, DATE, WORK_OF_ART, PERSON, PER...</td>
      <td>[Worf's Final Act, Apparel\nNewsletter, RSS, Youtube, Mastodon\nNewsletter\n\n\n\n\n\n\n\n\n\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\tFollow Slashdot, RSS, GitHub, GitHub, GitHub, SourceForge, nearly 30 million, monthly, less than a minute, Slashdot, Slashdot, 170474827, Worf's Final Act, 70, Sunday March 05, The final season, Star Trek: Picard, the Klingon Worf, Polygon, sci-fi's, Captains and Ad...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>files/5.txt</td>
      <td>[ORG, ORG, ORG, ORG, ORG, ORG, ORG, ORG, WORK_OF_ART, WORK_OF_ART, ORG, ORG, DATE, WORK_OF_ART, ORDINAL, PERSON, PRODUCT, PERSON, DATE, PERSON, PERSON, CARDINAL, ORG, ORG, GPE, NORP, PERSON, ORG, PERSON, PERSON, ORG, ORG, GPE, PERSON, NORP, PERSON, MONEY, WORK_OF_ART, ORG, PERSON, PERSON, LAW, DATE, GPE, GPE, PERSON, PERSON, PERSON, PERSON, LAW, DATE, GPE, GPE, PERSON, ORG, DATE, MONEY, GPE, D...</td>
      <td>[NPR, Keyboard, Navigation Menu, NPR Shop, Navigation Menu, Home\n\n\n\nNews\n, Expand, News\n\n\nNational\nWorld\nPolitics\nBusiness\nHealth\nScience\nClimate, Pop Culture\nFood\nArt &amp; Design \nPerforming Arts\nLife Kit\n\n\n\n\nMusic\nExpand, Music\n\n\n\n        Tiny Desk, Expand, Podcasts &amp; Shows\n\n\nDaily\n\n\n\n\n                                    Morning Edition\n                     ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>files/6.txt</td>
      <td>[CARDINAL, ORG, ORG]</td>
      <td>[403, Register, webmaster@theregister.co.uk]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>files/7.txt</td>
      <td>[GPE, PERSON, ORG, ORG, WORK_OF_ART, CARDINAL, ORG, GPE, PERSON, PERSON, PERSON, PERSON, PERSON, PERSON, ORG, ORG, GPE, GPE, GPE, GPE, GPE, LOC, GPE, GPE, PRODUCT, PERSON, PERSON, GPE, GPE, PERSON, PERSON, PERSON, PERSON, TIME, ORG, GPE, MONEY, PERSON, GPE, PRODUCT, DATE, LAW, PERSON, DATE, GPE, DATE, LOC, ORG, ORG, PERSON, PERSON, GPE, GPE, MONEY, PERSON, ORG, PERSON, PERSON, ORG, GPE, ORG, P...</td>
      <td>[Washington, Politico Logo, Congress, White House, Congress Minutes, Fifty, Playbook, Nightly, John Harris, Alex Burns, Jonathan Martin, Michael Schaffer, Jack Shafer, Matt Wuerker, Cartoon Carousel, Energy &amp; Environment\nFinance &amp; Tax\nHealth Care\nImmigration\nLabor\nSpace\nSustainability\nTechnology\nTrade\nTransportation, California, Canada, Florida, New Jersey, New York, Europe, Brussels,...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>files/8.txt</td>
      <td>[GPE, ORG, PERSON, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, GPE, ORG, ORG, ORG, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, ORG, ORG, PERSON, ORG, ORG, ORG, ORG, DATE, WORK_OF_ART, PRODUCT, ORG, LAW, WORK_OF_ART, DATE, ORG, ORG, PERCENT, ORG, ORDINAL, CARDINAL, ORG, CARDINAL, CARDINAL, GPE, DATE, PERSON, PERSON, CARDINAL, GPE, PERSON, ORG, WORK_OF_ART, GPE, PERSON, ORG, ORG, ORG, ...</td>
      <td>[China, The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease, Javascript, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Stocks, Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes, Dow Jones, Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Value Stocks, Dividend St...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>files/9.txt</td>
      <td>[LAW, ORG, PERSON, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, GPE, ORG, ORG, ORG, ORG, PERCENT, PERCENT, PERSON, PERCENT, PERCENT, ORG, ORG, PERSON, ORG, ORG, ORG, ORG, DATE, PRODUCT, ORG, LAW, ORG, DATE, ORG, ORG, PERCENT, ORG, ORDINAL, CARDINAL, ORG, CARDINAL, CARDINAL, GPE, DATE, PERSON, PERSON, CARDINAL, GPE, PERSON, ORG, WORK_OF_ART, GPE, PERSON, ORG, ORG, ORG, ORG, FAC, GPE, ORG, O...</td>
      <td>[The CHIPS Act Is Open for Business -- Which Stocks Will, The Motley Fool\n\n  \n\n\n\n\n\n\n\n\n\n\n\nPlease, Javascript, Premium Services\n\n\nStock Advisor, 416%, 120%, Rule Breakers, 213%, 102%, Stocks, Types of Stocks\n\n\n\n\nStock Market Sectors\n\n\n\n\nStock Market Indexes, Dow Jones, Nasdaq Composite\n\n\n\n\n\n\n\n\n\nStock Market, Premium Services\n\n\nStock Advisor, 416%, 120%, Ru...</td>
    </tr>
  </tbody>
</table>
</div>



### Let's explode our Dataframe so we have just one entity value per row pegged to the file name


```python
df_NER = df_NER.set_index(['File_name'])
df_NER = df_NER.apply(pd.Series.explode).reset_index()
df_NER[:25]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File_name</th>
      <th>Entity_type</th>
      <th>Entity_identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>files/1.txt</td>
      <td>PERSON</td>
      <td>Biden</td>
    </tr>
    <tr>
      <th>1</th>
      <td>files/1.txt</td>
      <td>PRODUCT</td>
      <td>CHIPS Act</td>
    </tr>
    <tr>
      <th>2</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>3</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest</td>
    </tr>
    <tr>
      <th>4</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>TechHands-OnView all ReviewsBuying GuidesBest Wireless EarbudsBest Robot VacuumsBest LaptopsBest</td>
    </tr>
    <tr>
      <th>5</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>GuidesGamingGearEntertainmentTomorrowDealsNewsVideoPodcastsLoginSponsored LinksBiden</td>
    </tr>
    <tr>
      <th>6</th>
      <td>files/1.txt</td>
      <td>PRODUCT</td>
      <td>CHIPS Act</td>
    </tr>
    <tr>
      <th>7</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>ChinaChipmakers</td>
    </tr>
    <tr>
      <th>8</th>
      <td>files/1.txt</td>
      <td>PERSON</td>
      <td>Florence Lo</td>
    </tr>
    <tr>
      <th>9</th>
      <td>files/1.txt</td>
      <td>DATE</td>
      <td>4, 2023</td>
    </tr>
    <tr>
      <th>10</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>PMChipmakers</td>
    </tr>
    <tr>
      <th>11</th>
      <td>files/1.txt</td>
      <td>PERSON</td>
      <td>Biden</td>
    </tr>
    <tr>
      <th>12</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$39 billion</td>
    </tr>
    <tr>
      <th>13</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>14</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>the US Commerce Department</td>
    </tr>
    <tr>
      <th>15</th>
      <td>files/1.txt</td>
      <td>DATE</td>
      <td>this week</td>
    </tr>
    <tr>
      <th>16</th>
      <td>files/1.txt</td>
      <td>LAW</td>
      <td>the CHIPS Act</td>
    </tr>
    <tr>
      <th>17</th>
      <td>files/1.txt</td>
      <td>DATE</td>
      <td>late June</td>
    </tr>
    <tr>
      <th>18</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>Congress</td>
    </tr>
    <tr>
      <th>19</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$280 billion</td>
    </tr>
    <tr>
      <th>20</th>
      <td>files/1.txt</td>
      <td>DATE</td>
      <td>last July</td>
    </tr>
    <tr>
      <th>21</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$52 billion</td>
    </tr>
    <tr>
      <th>22</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>23</th>
      <td>files/1.txt</td>
      <td>ORG</td>
      <td>Recipients</td>
    </tr>
    <tr>
      <th>24</th>
      <td>files/1.txt</td>
      <td>DATE</td>
      <td>a period of</td>
    </tr>
  </tbody>
</table>
</div>



### Let's filter our results by GPE


```python
df_NER[df_NER['Entity_type'] == 'GPE'][:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File_name</th>
      <th>Entity_type</th>
      <th>Entity_identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>13</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>22</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>30</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>31</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>36</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>53</th>
      <td>files/1.txt</td>
      <td>GPE</td>
      <td>Us</td>
    </tr>
    <tr>
      <th>85</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>China</td>
    </tr>
    <tr>
      <th>100</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>102</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>Chicago</td>
    </tr>
    <tr>
      <th>110</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>114</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>118</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>UK</td>
    </tr>
    <tr>
      <th>120</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>124</th>
      <td>files/10.txt</td>
      <td>GPE</td>
      <td>Wisconsin</td>
    </tr>
  </tbody>
</table>
</div>



### Let's filter our results by LAW


```python
df_NER[df_NER['Entity_type'] == 'LAW'][:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File_name</th>
      <th>Entity_type</th>
      <th>Entity_identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>files/1.txt</td>
      <td>LAW</td>
      <td>the CHIPS Act</td>
    </tr>
    <tr>
      <th>33</th>
      <td>files/1.txt</td>
      <td>LAW</td>
      <td>CHIPS Act</td>
    </tr>
    <tr>
      <th>42</th>
      <td>files/1.txt</td>
      <td>LAW</td>
      <td>the CHIPS Act</td>
    </tr>
    <tr>
      <th>317</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>Chips Act</td>
    </tr>
    <tr>
      <th>426</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>Chapter 11 &amp;plus;&amp;plus;&amp;plus</td>
    </tr>
    <tr>
      <th>597</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>Pricey Foods Are Even More Expensive Because of Climate Change</td>
    </tr>
    <tr>
      <th>903</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>The CHIPS Act</td>
    </tr>
    <tr>
      <th>965</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>Chapter 11</td>
    </tr>
    <tr>
      <th>1168</th>
      <td>files/10.txt</td>
      <td>LAW</td>
      <td>Chapter 11</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>The CHIPS Act Is Accepting Applications</td>
    </tr>
    <tr>
      <th>1277</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>the Full Retirement Age</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>the Full Retirement Age</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>The CHIPS Act Is Accepting Applications</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>Science Act</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>files/11.txt</td>
      <td>LAW</td>
      <td>the CHIPS Act</td>
    </tr>
  </tbody>
</table>
</div>



### Let's filter our results by Money


```python
df_NER[df_NER['Entity_type'] == 'MONEY'][:15]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>File_name</th>
      <th>Entity_type</th>
      <th>Entity_identified</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$39 billion</td>
    </tr>
    <tr>
      <th>19</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$280 billion</td>
    </tr>
    <tr>
      <th>21</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>$52 billion</td>
    </tr>
    <tr>
      <th>37</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>no CHIPS dollars</td>
    </tr>
    <tr>
      <th>40</th>
      <td>files/1.txt</td>
      <td>MONEY</td>
      <td>more than $150 million</td>
    </tr>
    <tr>
      <th>103</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>$175 million</td>
    </tr>
    <tr>
      <th>132</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>2,000</td>
    </tr>
    <tr>
      <th>152</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>$10.8 Billion</td>
    </tr>
    <tr>
      <th>160</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>16</td>
    </tr>
    <tr>
      <th>163</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>nearly $200M</td>
    </tr>
    <tr>
      <th>192</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>6</td>
    </tr>
    <tr>
      <th>194</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>$175 Million</td>
    </tr>
    <tr>
      <th>200</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>FOUR</td>
    </tr>
    <tr>
      <th>201</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>81</td>
    </tr>
    <tr>
      <th>246</th>
      <td>files/10.txt</td>
      <td>MONEY</td>
      <td>nearly $200M</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
