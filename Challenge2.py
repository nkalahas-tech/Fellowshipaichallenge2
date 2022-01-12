{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6726bdca",
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "4da7ab59",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import random\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "id": "74d8556e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /Users/nkalahas/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package punkt to /Users/nkalahas/nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package averaged_perceptron_tagger to\n",
      "[nltk_data]     /Users/nkalahas/nltk_data...\n",
      "[nltk_data]   Package averaged_perceptron_tagger is already up-to-\n",
      "[nltk_data]       date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     /Users/nkalahas/nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Importing all the necessary packages needed for Explaratory Data Analysis\n",
    "import re, string\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "nltk.download('stopwords')\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import SnowballStemmer\n",
    "from nltk.corpus import wordnet\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "nltk.download('punkt')\n",
    "nltk.download('averaged_perceptron_tagger')\n",
    "nltk.download('wordnet')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "id": "04c8ff76",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class Index</th>\n",
       "      <th>Question</th>\n",
       "      <th>Questions Content</th>\n",
       "      <th>Solution</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5</td>\n",
       "      <td>why doesn't an optical mouse work on a glass t...</td>\n",
       "      <td>or even on some surfaces?</td>\n",
       "      <td>Optical mice use an LED and a camera to rapidl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>6</td>\n",
       "      <td>What is the best off-road motorcycle trail ?</td>\n",
       "      <td>long-distance trail throughout CA</td>\n",
       "      <td>i hear that the mojave road is amazing!&lt;br /&gt;\\...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>What is Trans Fat? How to reduce that?</td>\n",
       "      <td>I heard that tras fat is bad for the body.  Wh...</td>\n",
       "      <td>Trans fats occur in manufactured foods during ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7</td>\n",
       "      <td>How many planes Fedex has?</td>\n",
       "      <td>I heard that it is the largest airline in the ...</td>\n",
       "      <td>according to the www.fedex.com web site:\\nAir ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>In the san francisco bay area, does it make se...</td>\n",
       "      <td>the prices of rent and the price of buying doe...</td>\n",
       "      <td>renting vs buying depends on your goals. &lt;br /...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Class Index                                           Question  \\\n",
       "0            5  why doesn't an optical mouse work on a glass t...   \n",
       "1            6       What is the best off-road motorcycle trail ?   \n",
       "2            3             What is Trans Fat? How to reduce that?   \n",
       "3            7                         How many planes Fedex has?   \n",
       "4            7  In the san francisco bay area, does it make se...   \n",
       "\n",
       "                                   Questions Content  \\\n",
       "0                          or even on some surfaces?   \n",
       "1                  long-distance trail throughout CA   \n",
       "2  I heard that tras fat is bad for the body.  Wh...   \n",
       "3  I heard that it is the largest airline in the ...   \n",
       "4  the prices of rent and the price of buying doe...   \n",
       "\n",
       "                                            Solution  \n",
       "0  Optical mice use an LED and a camera to rapidl...  \n",
       "1  i hear that the mojave road is amazing!<br />\\...  \n",
       "2  Trans fats occur in manufactured foods during ...  \n",
       "3  according to the www.fedex.com web site:\\nAir ...  \n",
       "4  renting vs buying depends on your goals. <br /...  "
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initilization of the train df\n",
    "df_train = pd.read_csv('Downloads/yahoo_answers_csv/train.csv', names=['Class Index','Question', 'Questions Content', 'Solution'])\n",
    "df_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "0a485e10",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class Index</th>\n",
       "      <th>Question</th>\n",
       "      <th>Questions Content</th>\n",
       "      <th>Answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>9</td>\n",
       "      <td>What makes friendship click?</td>\n",
       "      <td>How does the spark keep going?</td>\n",
       "      <td>good communication is what does it.  Can you m...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Why does Zebras have stripes?</td>\n",
       "      <td>What is the purpose or those stripes? Who do t...</td>\n",
       "      <td>this provides camouflage - predator vision is ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4</td>\n",
       "      <td>What did the itsy bitsy sipder climb up?</td>\n",
       "      <td>NaN</td>\n",
       "      <td>waterspout</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>What is the difference between a Bachelors and...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>One difference between a Bachelors and a Maste...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>Why do women get PMS?</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Premenstrual syndrome (PMS) is a group of symp...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Class Index                                           Question  \\\n",
       "0            9                       What makes friendship click?   \n",
       "1            2                      Why does Zebras have stripes?   \n",
       "2            4           What did the itsy bitsy sipder climb up?   \n",
       "3            4  What is the difference between a Bachelors and...   \n",
       "4            3                              Why do women get PMS?   \n",
       "\n",
       "                                   Questions Content  \\\n",
       "0                     How does the spark keep going?   \n",
       "1  What is the purpose or those stripes? Who do t...   \n",
       "2                                                NaN   \n",
       "3                                                NaN   \n",
       "4                                                NaN   \n",
       "\n",
       "                                              Answer  \n",
       "0  good communication is what does it.  Can you m...  \n",
       "1  this provides camouflage - predator vision is ...  \n",
       "2                                         waterspout  \n",
       "3  One difference between a Bachelors and a Maste...  \n",
       "4  Premenstrual syndrome (PMS) is a group of symp...  "
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Initilization of the test df\n",
    "df_test = pd.read_csv('Downloads/yahoo_answers_csv/test.csv', names=['Class Index','Question', 'Questions Content', 'Answer']) \n",
    "df_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "402bc5a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def random_insertion(sentence, K):\n",
    "  \n",
    "# initializing add list \n",
    "    add_list = [\"category\", \"pattern\", \"fulfilled\"]\n",
    "  \n",
    "# initializing K \n",
    "  \n",
    "    for idx in range(K):\n",
    "      \n",
    "    # choosing index to enter element\n",
    "        index = random.randint(0, len(sentence))\n",
    "      \n",
    "    # reforming list and getting random element to add\n",
    "        sentence = sentence[:index] + [random.choice(add_list)] + sentence[index:]\n",
    "    return sentence\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 150,
   "id": "1d3b5520",
   "metadata": {},
   "outputs": [],
   "source": [
    "def random_swap(sentence, n):\n",
    "    for i in range(n):\n",
    "        i1, i2 = random.sample(range(len(sentence)), 2)\n",
    "        sentence[i1], sentence[i2] = sentence[i2], sentence[i1]\n",
    "    return sentence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "89c6ebaa",
   "metadata": {},
   "outputs": [],
   "source": [
    "def delete_random_elems(sentence, n):\n",
    "    to_delete = set(random.sample(range(len(sentence)), n))\n",
    "    return [x for i,x in enumerate(sentence) if not i in to_delete]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "7b6e5925",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['good', 'communication', 'is', 'what', 'does', 'it', '.', 'Can', 'you', 'move', 'beyond', 'small', 'talk', 'and', 'say', 'what', \"'s\", 'really', 'on', 'your', 'mind', '.', 'If', 'you', 'start', 'doing', 'this', ',', 'my', 'expereince', 'is', 'that', 'potentially', 'good', 'friends', 'will', 'respond', 'or', 'shun', 'you', '.', 'Then', 'you', 'know', 'who', 'the', 'really', 'good', 'friends', 'are', '.']\n",
      "BEFORE: \n",
      "['good', 'communication', '.', 'Can', 'move', 'beyond', 'small', 'talk', 'say', \"'s\", 'really', 'mind', '.', 'If', 'start', ',', 'expereince', 'potentially', 'good', 'friends', 'respond', 'shun', '.', 'Then', 'know', 'really', 'good', 'friends', '.']\n",
      "AFTER INSERTION: \n",
      "['good', 'communication', '.', 'Can', 'move', 'beyond', 'small', 'talk', 'fulfilled', 'say', \"'s\", 'really', 'mind', '.', 'If', 'category', 'start', ',', 'expereince', 'pattern', 'potentially', 'good', 'friends', 'respond', 'shun', '.', 'Then', 'know', 'really', 'good', 'friends', '.']\n",
      "AFTER SWAP: \n",
      "['good', 'communication', 'good', 'Can', 'move', 'beyond', 'small', 'talk', 'say', \"'s\", 'really', 'mind', '.', 'If', 'start', ',', 'friends', 'potentially', 'good', 'friends', 'respond', 'shun', '.', 'Then', 'know', 'really', '.', 'expereince', '.']\n",
      "AFTER DELETION: \n",
      "['good', 'communication', 'good', 'Can', 'move', 'beyond', 'small', 'talk', 'say', \"'s\", 'really', 'mind', '.', 'If', 'start', 'friends', 'good', 'friends', 'shun', '.', 'Then', 'know', 'really', '.', 'expereince', '.']\n",
      "['this', 'provides', 'camouflage', '-', 'predator', 'vision', 'is', 'such', 'that', 'it', 'is', 'usually', 'difficult', 'for', 'them', 'to', 'see', 'complex', 'patterns']\n",
      "BEFORE: \n",
      "['provides', 'camouflage', '-', 'predator', 'vision', 'usually', 'difficult', 'see', 'complex', 'patterns']\n",
      "AFTER INSERTION: \n",
      "['category', 'provides', 'camouflage', 'fulfilled', '-', 'predator', 'vision', 'pattern', 'usually', 'difficult', 'see', 'complex', 'patterns']\n",
      "AFTER SWAP: \n",
      "['provides', 'camouflage', '-', 'vision', 'difficult', 'usually', 'predator', 'see', 'complex', 'patterns']\n",
      "AFTER DELETION: \n",
      "['provides', 'camouflage', 'vision', 'difficult', 'usually', 'predator', 'patterns']\n",
      "['waterspout']\n",
      "BEFORE: \n",
      "['waterspout']\n",
      "AFTER INSERTION: \n",
      "['category', 'pattern', 'fulfilled', 'waterspout']\n",
      "AFTER SWAP: \n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Sample larger than population or is negative",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/qt/dfpmp9sx3j95lgb20n7cn7500000gn/T/ipykernel_32535/1089490969.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     18\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandom_insertion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfiltered_sentence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"AFTER SWAP: \"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 20\u001b[0;31m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrandom_swap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfiltered_sentence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     21\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"AFTER DELETION: \"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     22\u001b[0m     \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdelete_random_elems\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfiltered_sentence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/var/folders/qt/dfpmp9sx3j95lgb20n7cn7500000gn/T/ipykernel_32535/1002059464.py\u001b[0m in \u001b[0;36mrandom_swap\u001b[0;34m(sentence, n)\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mrandom_swap\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msentence\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mi\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mn\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m         \u001b[0mi1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi2\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msample\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msentence\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m         \u001b[0msentence\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msentence\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi2\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msentence\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi2\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msentence\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi1\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0msentence\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/opt/anaconda3/lib/python3.9/random.py\u001b[0m in \u001b[0;36msample\u001b[0;34m(self, population, k, counts)\u001b[0m\n\u001b[1;32m    447\u001b[0m         \u001b[0mrandbelow\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_randbelow\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    448\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;36m0\u001b[0m \u001b[0;34m<=\u001b[0m \u001b[0mk\u001b[0m \u001b[0;34m<=\u001b[0m \u001b[0mn\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 449\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Sample larger than population or is negative\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    450\u001b[0m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    451\u001b[0m         \u001b[0msetsize\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m21\u001b[0m        \u001b[0;31m# size of a small set minus size of an empty list\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: Sample larger than population or is negative"
     ]
    }
   ],
   "source": [
    "for i in df_test['Answer'].head():\n",
    " \n",
    "    stop_words = set(stopwords.words('english'))\n",
    " \n",
    "    word_tokens = word_tokenize(i)\n",
    " \n",
    "    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]\n",
    " \n",
    "    filtered_sentence = []\n",
    " \n",
    "    for w in word_tokens:\n",
    "        if w not in stop_words:\n",
    "            filtered_sentence.append(w)\n",
    "    print(word_tokens)\n",
    "    print(\"BEFORE: \")\n",
    "    print(filtered_sentence)\n",
    "    print(\"AFTER INSERTION: \")\n",
    "    print(random_insertion(filtered_sentence, 3))\n",
    "    print(\"AFTER SWAP: \")\n",
    "    print(random_swap(filtered_sentence, 2))\n",
    "    print(\"AFTER DELETION: \")\n",
    "    print(delete_random_elems(filtered_sentence, 3))\n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "bc20c018",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Optical', 'mice', 'use', 'an', 'LED', 'and', 'a', 'camera', 'to', 'rapidly', 'capture', 'images', 'of', 'the', 'surface', 'beneath', 'the', 'mouse', '.', 'The', 'infomation', 'from', 'the', 'camera', 'is', 'analyzed', 'by', 'a', 'DSP', '(', 'Digital', 'Signal', 'Processor', ')', 'and', 'used', 'to', 'detect', 'imperfections', 'in', 'the', 'underlying', 'surface', 'and', 'determine', 'motion', '.', 'Some', 'materials', ',', 'such', 'as', 'glass', ',', 'mirrors', 'or', 'other', 'very', 'shiny', ',', 'uniform', 'surfaces', 'interfere', 'with', 'the', 'ability', 'of', 'the', 'DSP', 'to', 'accurately', 'analyze', 'the', 'surface', 'beneath', 'the', 'mouse', '.', '\\\\nSince', 'glass', 'is', 'transparent', 'and', 'very', 'uniform', ',', 'the', 'mouse', 'is', 'unable', 'to', 'pick', 'up', 'enough', 'imperfections', 'in', 'the', 'underlying', 'surface', 'to', 'determine', 'motion', '.', 'Mirrored', 'surfaces', 'are', 'also', 'a', 'problem', ',', 'since', 'they', 'constantly', 'reflect', 'back', 'the', 'same', 'image', ',', 'causing', 'the', 'DSP', 'not', 'to', 'recognize', 'motion', 'properly', '.', 'When', 'the', 'system', 'is', 'unable', 'to', 'see', 'surface', 'changes', 'associated', 'with', 'movement', ',', 'the', 'mouse', 'will', 'not', 'work', 'properly', '.']\n",
      "BEFORE: \n",
      "['Optical', 'mice', 'use', 'LED', 'camera', 'rapidly', 'capture', 'images', 'surface', 'beneath', 'mouse', '.', 'The', 'infomation', 'camera', 'analyzed', 'DSP', '(', 'Digital', 'Signal', 'Processor', ')', 'used', 'detect', 'imperfections', 'underlying', 'surface', 'determine', 'motion', '.', 'Some', 'materials', ',', 'glass', ',', 'mirrors', 'shiny', ',', 'uniform', 'surfaces', 'interfere', 'ability', 'DSP', 'accurately', 'analyze', 'surface', 'beneath', 'mouse', '.', '\\\\nSince', 'glass', 'transparent', 'uniform', ',', 'mouse', 'unable', 'pick', 'enough', 'imperfections', 'underlying', 'surface', 'determine', 'motion', '.', 'Mirrored', 'surfaces', 'also', 'problem', ',', 'since', 'constantly', 'reflect', 'back', 'image', ',', 'causing', 'DSP', 'recognize', 'motion', 'properly', '.', 'When', 'system', 'unable', 'see', 'surface', 'changes', 'associated', 'movement', ',', 'mouse', 'work', 'properly', '.']\n",
      "AFTER INSERTION: \n",
      "['Optical', 'mice', 'use', 'LED', 'camera', 'rapidly', 'capture', 'images', 'surface', 'beneath', 'mouse', '.', 'The', 'category', 'infomation', 'camera', 'analyzed', 'DSP', '(', 'Digital', 'Signal', 'Processor', ')', 'used', 'detect', 'imperfections', 'underlying', 'surface', 'determine', 'motion', '.', 'Some', 'materials', ',', 'glass', ',', 'mirrors', 'shiny', ',', 'uniform', 'surfaces', 'interfere', 'ability', 'DSP', 'accurately', 'analyze', 'surface', 'beneath', 'mouse', '.', '\\\\nSince', 'glass', 'transparent', 'uniform', ',', 'mouse', 'unable', 'pick', 'enough', 'imperfections', 'underlying', 'surface', 'determine', 'motion', 'pattern', '.', 'Mirrored', 'surfaces', 'also', 'problem', ',', 'since', 'constantly', 'reflect', 'back', 'image', ',', 'causing', 'DSP', 'recognize', 'motion', 'properly', '.', 'When', 'system', 'unable', 'see', 'surface', 'changes', 'associated', 'movement', 'category', ',', 'mouse', 'work', 'properly', '.']\n",
      "AFTER SWAP: \n",
      "['Optical', 'mice', 'use', 'LED', 'camera', 'rapidly', 'capture', 'images', 'surface', 'beneath', 'mouse', '.', 'The', 'infomation', 'camera', 'analyzed', 'DSP', '(', 'Digital', 'Signal', 'Processor', ')', 'used', 'detect', 'imperfections', 'underlying', 'surface', 'determine', 'motion', '.', 'Some', 'materials', ',', 'glass', ',', 'mirrors', 'shiny', ',', 'uniform', 'surfaces', 'interfere', 'ability', 'DSP', 'accurately', 'analyze', 'surface', 'beneath', 'mouse', '.', '\\\\nSince', 'glass', 'transparent', 'uniform', ',', 'mouse', 'unable', 'pick', 'surface', 'imperfections', 'underlying', 'enough', 'determine', 'motion', '.', 'Mirrored', ',', 'also', 'problem', ',', 'since', 'constantly', 'reflect', 'back', 'image', 'surfaces', 'causing', 'DSP', 'recognize', 'motion', 'properly', '.', 'When', 'system', 'unable', 'see', 'surface', 'changes', 'associated', 'movement', ',', 'mouse', 'work', 'properly', '.']\n",
      "AFTER DELETION: \n",
      "['Optical', 'mice', 'use', 'LED', 'camera', 'capture', 'images', 'surface', 'beneath', 'mouse', '.', 'The', 'infomation', 'camera', 'analyzed', 'DSP', '(', 'Digital', 'Signal', 'Processor', ')', 'used', 'detect', 'imperfections', 'underlying', 'surface', 'determine', 'motion', '.', 'Some', 'materials', ',', ',', 'mirrors', 'shiny', ',', 'uniform', 'interfere', 'ability', 'DSP', 'accurately', 'analyze', 'surface', 'beneath', 'mouse', '.', '\\\\nSince', 'glass', 'transparent', 'uniform', ',', 'mouse', 'unable', 'pick', 'surface', 'imperfections', 'underlying', 'enough', 'determine', 'motion', '.', 'Mirrored', ',', 'also', 'problem', ',', 'since', 'constantly', 'reflect', 'back', 'image', 'surfaces', 'causing', 'DSP', 'recognize', 'motion', 'properly', '.', 'When', 'system', 'unable', 'see', 'surface', 'changes', 'associated', 'movement', ',', 'mouse', 'work', 'properly', '.']\n",
      "['i', 'hear', 'that', 'the', 'mojave', 'road', 'is', 'amazing', '!', '<', 'br', '/', '>', '\\\\nsearch', 'for', 'it', 'online', '.']\n",
      "BEFORE: \n",
      "['hear', 'mojave', 'road', 'amazing', '!', '<', 'br', '/', '>', '\\\\nsearch', 'online', '.']\n",
      "AFTER INSERTION: \n",
      "['hear', 'mojave', 'road', 'amazing', 'fulfilled', '!', 'pattern', '<', 'pattern', 'br', '/', '>', '\\\\nsearch', 'online', '.']\n",
      "AFTER SWAP: \n",
      "['hear', 'mojave', 'amazing', 'road', '!', '<', 'br', '/', '\\\\nsearch', '>', 'online', '.']\n",
      "AFTER DELETION: \n",
      "['hear', 'road', '!', '<', 'br', '\\\\nsearch', '>', 'online', '.']\n",
      "['Trans', 'fats', 'occur', 'in', 'manufactured', 'foods', 'during', 'the', 'process', 'of', 'partial', 'hydrogenation', ',', 'when', 'hydrogen', 'gas', 'is', 'bubbled', 'through', 'vegetable', 'oil', 'to', 'increase', 'shelf', 'life', 'and', 'stabilize', 'the', 'original', 'polyunsatured', 'oil', '.', 'The', 'resulting', 'fat', 'is', 'similar', 'to', 'saturated', 'fat', ',', 'which', 'raises', '``', 'bad', \"''\", 'LDL', 'cholesterol', 'and', 'can', 'lead', 'to', 'clogged', 'arteries', 'and', 'heart', 'disease', '.', '\\\\nUntil', 'very', 'recently', ',', 'food', 'labels', 'were', 'not', 'required', 'to', 'list', 'trans', 'fats', ',', 'and', 'this', 'health', 'risk', 'remained', 'hidden', 'to', 'consumers', '.', 'In', 'early', 'July', ',', 'FDA', 'regulations', 'changed', ',', 'and', 'food', 'labels', 'will', 'soon', 'begin', 'identifying', 'trans', 'fat', 'content', 'in', 'processed', 'foods', '.']\n",
      "BEFORE: \n",
      "['Trans', 'fats', 'occur', 'manufactured', 'foods', 'process', 'partial', 'hydrogenation', ',', 'hydrogen', 'gas', 'bubbled', 'vegetable', 'oil', 'increase', 'shelf', 'life', 'stabilize', 'original', 'polyunsatured', 'oil', '.', 'The', 'resulting', 'fat', 'similar', 'saturated', 'fat', ',', 'raises', '``', 'bad', \"''\", 'LDL', 'cholesterol', 'lead', 'clogged', 'arteries', 'heart', 'disease', '.', '\\\\nUntil', 'recently', ',', 'food', 'labels', 'required', 'list', 'trans', 'fats', ',', 'health', 'risk', 'remained', 'hidden', 'consumers', '.', 'In', 'early', 'July', ',', 'FDA', 'regulations', 'changed', ',', 'food', 'labels', 'soon', 'begin', 'identifying', 'trans', 'fat', 'content', 'processed', 'foods', '.']\n",
      "AFTER INSERTION: \n",
      "['Trans', 'fats', 'occur', 'manufactured', 'foods', 'process', 'partial', 'hydrogenation', ',', 'hydrogen', 'gas', 'bubbled', 'vegetable', 'oil', 'increase', 'shelf', 'life', 'stabilize', 'original', 'polyunsatured', 'oil', '.', 'The', 'pattern', 'resulting', 'fat', 'similar', 'saturated', 'fat', ',', 'raises', '``', 'bad', \"''\", 'LDL', 'category', 'cholesterol', 'lead', 'clogged', 'arteries', 'heart', 'disease', '.', '\\\\nUntil', 'recently', ',', 'food', 'labels', 'required', 'list', 'trans', 'fats', ',', 'health', 'risk', 'remained', 'hidden', 'consumers', '.', 'In', 'pattern', 'early', 'July', ',', 'FDA', 'regulations', 'changed', ',', 'food', 'labels', 'soon', 'begin', 'identifying', 'trans', 'fat', 'content', 'processed', 'foods', '.']\n",
      "AFTER SWAP: \n",
      "['Trans', 'fats', 'occur', 'manufactured', 'foods', 'process', 'partial', 'hydrogenation', ',', 'hydrogen', 'gas', 'consumers', 'vegetable', 'oil', 'increase', 'shelf', 'life', 'stabilize', 'original', 'polyunsatured', 'oil', '.', 'The', 'resulting', 'fat', 'similar', 'saturated', 'fat', ',', 'raises', '``', 'bad', \"''\", 'LDL', 'cholesterol', 'lead', 'clogged', 'arteries', 'disease', 'heart', '.', '\\\\nUntil', 'recently', ',', 'food', 'labels', 'required', 'list', 'trans', 'fats', ',', 'health', 'risk', 'remained', 'hidden', 'bubbled', '.', 'In', 'early', 'July', ',', 'FDA', 'regulations', 'changed', ',', 'food', 'labels', 'soon', 'begin', 'identifying', 'trans', 'fat', 'content', 'processed', 'foods', '.']\n",
      "AFTER DELETION: \n",
      "['Trans', 'fats', 'occur', 'manufactured', 'foods', 'process', 'partial', 'hydrogenation', ',', 'hydrogen', 'gas', 'consumers', 'vegetable', 'oil', 'increase', 'shelf', 'life', 'stabilize', 'polyunsatured', 'oil', '.', 'The', 'resulting', 'fat', 'similar', 'saturated', 'fat', ',', 'raises', '``', 'bad', \"''\", 'LDL', 'cholesterol', 'lead', 'arteries', 'disease', 'heart', '.', '\\\\nUntil', ',', 'food', 'labels', 'required', 'list', 'trans', 'fats', ',', 'health', 'risk', 'remained', 'hidden', 'bubbled', '.', 'In', 'early', 'July', ',', 'FDA', 'regulations', 'changed', ',', 'food', 'labels', 'soon', 'begin', 'identifying', 'trans', 'fat', 'content', 'processed', 'foods', '.']\n",
      "['according', 'to', 'the', 'www.fedex.com', 'web', 'site', ':', '\\\\nAir', 'Fleet', '<', 'br', '/', '>', '\\\\n', '<', 'br', '/', '>', '\\\\n670', 'aircraft', ',', 'including', ':', '<', 'br', '/', '>', '\\\\n47', 'Airbus', 'A300-600s', '17', 'Boeing', 'DC10-30s', '<', 'br', '/', '>', '\\\\n62', 'Airbus', 'A310-200/300s', '36', 'Boeing', 'MD10-10s', '<', 'br', '/', '>', '\\\\n2', 'ATR', '72s', '5', 'Boeing', 'MD10-30s', '<', 'br', '/', '>', '\\\\n29', 'ATR', '42s', '57', 'Boeing', 'MD11s', '<', 'br', '/', '>', '\\\\n18', 'Boeing', '727-100s', '10', 'Cessna', '208As', '<', 'br', '/', '>', '\\\\n94', 'Boeing', '727-200s', '246', 'Cessna', '208Bs', '<', 'br', '/', '>', '\\\\n30', 'Boeing', 'DC10-10s', '17', 'Fokker', 'F-27s']\n",
      "BEFORE: \n",
      "['according', 'www.fedex.com', 'web', 'site', ':', '\\\\nAir', 'Fleet', '<', 'br', '/', '>', '\\\\n', '<', 'br', '/', '>', '\\\\n670', 'aircraft', ',', 'including', ':', '<', 'br', '/', '>', '\\\\n47', 'Airbus', 'A300-600s', '17', 'Boeing', 'DC10-30s', '<', 'br', '/', '>', '\\\\n62', 'Airbus', 'A310-200/300s', '36', 'Boeing', 'MD10-10s', '<', 'br', '/', '>', '\\\\n2', 'ATR', '72s', '5', 'Boeing', 'MD10-30s', '<', 'br', '/', '>', '\\\\n29', 'ATR', '42s', '57', 'Boeing', 'MD11s', '<', 'br', '/', '>', '\\\\n18', 'Boeing', '727-100s', '10', 'Cessna', '208As', '<', 'br', '/', '>', '\\\\n94', 'Boeing', '727-200s', '246', 'Cessna', '208Bs', '<', 'br', '/', '>', '\\\\n30', 'Boeing', 'DC10-10s', '17', 'Fokker', 'F-27s']\n",
      "AFTER INSERTION: \n",
      "['according', 'www.fedex.com', 'web', 'site', ':', '\\\\nAir', 'Fleet', '<', 'br', '/', '>', '\\\\n', '<', 'br', '/', 'fulfilled', '>', '\\\\n670', 'aircraft', ',', 'including', ':', '<', 'br', '/', '>', '\\\\n47', 'Airbus', 'A300-600s', '17', 'Boeing', 'DC10-30s', '<', 'br', '/', '>', '\\\\n62', 'Airbus', 'A310-200/300s', '36', 'Boeing', 'MD10-10s', '<', 'br', '/', 'fulfilled', '>', '\\\\n2', 'ATR', '72s', '5', 'Boeing', 'MD10-30s', '<', 'br', '/', '>', '\\\\n29', 'ATR', '42s', '57', 'Boeing', 'MD11s', '<', 'br', 'fulfilled', '/', '>', '\\\\n18', 'Boeing', '727-100s', '10', 'Cessna', '208As', '<', 'br', '/', '>', '\\\\n94', 'Boeing', '727-200s', '246', 'Cessna', '208Bs', '<', 'br', '/', '>', '\\\\n30', 'Boeing', 'DC10-10s', '17', 'Fokker', 'F-27s']\n",
      "AFTER SWAP: \n",
      "['according', 'www.fedex.com', 'web', 'site', ':', '\\\\nAir', 'Fleet', '<', 'br', '/', '>', '\\\\n', '<', 'br', '/', '>', 'A300-600s', 'aircraft', ',', 'including', ':', '<', 'br', '/', '>', '\\\\n47', 'Airbus', '\\\\n670', '17', 'Boeing', 'DC10-30s', '<', 'br', '/', '>', '\\\\n62', 'Airbus', 'A310-200/300s', '36', 'Boeing', 'MD10-10s', '<', 'br', '/', '>', '\\\\n2', 'ATR', '72s', '5', 'Boeing', 'Cessna', '<', 'br', '/', '>', '\\\\n29', 'ATR', '42s', '57', 'Boeing', 'MD11s', '<', 'br', '/', '>', '\\\\n18', 'Boeing', '727-100s', '10', 'MD10-30s', '208As', '<', 'br', '/', '>', '\\\\n94', 'Boeing', '727-200s', '246', 'Cessna', '208Bs', '<', 'br', '/', '>', '\\\\n30', 'Boeing', 'DC10-10s', '17', 'Fokker', 'F-27s']\n",
      "AFTER DELETION: \n",
      "['according', 'www.fedex.com', 'web', 'site', ':', '\\\\nAir', 'Fleet', '<', 'br', '/', '>', '\\\\n', '<', 'br', '/', '>', 'A300-600s', 'aircraft', ',', 'including', ':', '<', 'br', '/', '>', '\\\\n47', 'Airbus', '\\\\n670', '17', 'Boeing', 'DC10-30s', '<', 'br', '/', '>', '\\\\n62', 'Airbus', 'A310-200/300s', '36', 'Boeing', 'MD10-10s', '<', 'br', '/', '>', 'ATR', '72s', '5', 'Boeing', 'Cessna', '<', 'br', '/', '>', '\\\\n29', 'ATR', '42s', '57', 'Boeing', 'MD11s', '<', 'br', '/', '>', '\\\\n18', 'Boeing', '727-100s', '10', 'MD10-30s', '208As', '<', 'br', '/', '>', '\\\\n94', '727-200s', '246', 'Cessna', '208Bs', '<', 'br', '/', '>', '\\\\n30', 'Boeing', 'DC10-10s', '17', 'F-27s']\n",
      "['renting', 'vs', 'buying', 'depends', 'on', 'your', 'goals', '.', '<', 'br', '/', '>', '\\\\ngenerally', 'thinking', 'is', 'that', 'buying', 'is', 'better', 'b/c', 'the', 'payments', 'that', 'would', 'go', 'into', 'the', 'rent', 'start', 'building', 'equity', 'in', 'your', 'home', '.', 'the', 'govt', 'also', 'incentivizes', 'you', 'to', 'buy', 'by', 'making', 'your', 'property', 'tax', 'payments', 'and', 'mortgage', 'interest', 'payments', 'tax', 'deductible.\\\\nhaving', 'said', 'that', 'current', 'housing', 'status', 'in', 'the', 'bay', 'area', 'is', 'such', 'that', 'housing', 'cost', 'to', 'purchase', 'is', 'relatively', 'high', 'and', 'rental', 'prices', '(', 'compared', 'to', 'ownership', 'cost', ')', 'are', 'relatively', 'low', '(', 'relative', 'to', 'the', 'rest', 'of', 'the', 'country', ')', '.', 'it', 'makes', 'lese', 'sense', 'to', 'buy', 'vs.', 'other', 'places.\\\\nbottom', 'line', 'you', 'should', 'base', 'your', 'decision', 'on', 'whether', 'you', 'think', 'the', 'market', 'will', 'keep', 'going', 'up', 'or', 'not', '.', 'the', 'other', 'numbers', 'tend', 'to', 'even', 'out', ',', 'the', 'main', 'gain', 'or', 'loss', 'in', 'buying', 'comes', 'from', 'appreciation/depreciation', '.']\n",
      "BEFORE: \n",
      "['renting', 'vs', 'buying', 'depends', 'goals', '.', '<', 'br', '/', '>', '\\\\ngenerally', 'thinking', 'buying', 'better', 'b/c', 'payments', 'would', 'go', 'rent', 'start', 'building', 'equity', 'home', '.', 'govt', 'also', 'incentivizes', 'buy', 'making', 'property', 'tax', 'payments', 'mortgage', 'interest', 'payments', 'tax', 'deductible.\\\\nhaving', 'said', 'current', 'housing', 'status', 'bay', 'area', 'housing', 'cost', 'purchase', 'relatively', 'high', 'rental', 'prices', '(', 'compared', 'ownership', 'cost', ')', 'relatively', 'low', '(', 'relative', 'rest', 'country', ')', '.', 'makes', 'lese', 'sense', 'buy', 'vs.', 'places.\\\\nbottom', 'line', 'base', 'decision', 'whether', 'think', 'market', 'keep', 'going', '.', 'numbers', 'tend', 'even', ',', 'main', 'gain', 'loss', 'buying', 'comes', 'appreciation/depreciation', '.']\n",
      "AFTER INSERTION: \n",
      "['renting', 'vs', 'buying', 'depends', 'goals', '.', '<', 'br', '/', '>', '\\\\ngenerally', 'thinking', 'buying', 'better', 'b/c', 'payments', 'would', 'go', 'rent', 'start', 'building', 'equity', 'pattern', 'home', '.', 'govt', 'also', 'incentivizes', 'buy', 'making', 'property', 'tax', 'payments', 'mortgage', 'interest', 'payments', 'category', 'tax', 'deductible.\\\\nhaving', 'said', 'current', 'housing', 'status', 'bay', 'area', 'housing', 'cost', 'purchase', 'relatively', 'high', 'rental', 'prices', '(', 'compared', 'ownership', 'cost', ')', 'relatively', 'low', '(', 'relative', 'rest', 'country', ')', '.', 'makes', 'lese', 'sense', 'buy', 'vs.', 'places.\\\\nbottom', 'line', 'base', 'decision', 'whether', 'think', 'pattern', 'market', 'keep', 'going', '.', 'numbers', 'tend', 'even', ',', 'main', 'gain', 'loss', 'buying', 'comes', 'appreciation/depreciation', '.']\n",
      "AFTER SWAP: \n",
      "['renting', 'vs', 'buying', 'depends', 'go', '.', 'status', 'br', '/', '>', '\\\\ngenerally', 'thinking', 'buying', 'better', 'b/c', 'payments', 'would', 'goals', 'rent', 'start', 'building', 'equity', 'home', '.', 'govt', 'also', 'incentivizes', 'buy', 'making', 'property', 'tax', 'payments', 'mortgage', 'interest', 'payments', 'tax', 'deductible.\\\\nhaving', 'said', 'current', 'housing', '<', 'bay', 'area', 'housing', 'cost', 'purchase', 'relatively', 'high', 'rental', 'prices', '(', 'compared', 'ownership', 'cost', ')', 'relatively', 'low', '(', 'relative', 'rest', 'country', ')', '.', 'makes', 'lese', 'sense', 'buy', 'vs.', 'places.\\\\nbottom', 'line', 'base', 'decision', 'whether', 'think', 'market', 'keep', 'going', '.', 'numbers', 'tend', 'even', ',', 'main', 'gain', 'loss', 'buying', 'comes', 'appreciation/depreciation', '.']\n",
      "AFTER DELETION: \n",
      "['renting', 'vs', 'buying', 'depends', 'go', '.', 'status', 'br', '/', '>', '\\\\ngenerally', 'thinking', 'buying', 'better', 'b/c', 'payments', 'would', 'goals', 'rent', 'start', 'building', 'equity', 'home', '.', 'govt', 'also', 'incentivizes', 'buy', 'making', 'property', 'tax', 'payments', 'mortgage', 'interest', 'payments', 'tax', 'deductible.\\\\nhaving', 'said', 'current', 'housing', '<', 'bay', 'area', 'housing', 'cost', 'purchase', 'relatively', 'high', 'rental', '(', 'compared', 'ownership', 'cost', ')', 'relatively', 'low', '(', 'relative', 'rest', 'country', ')', '.', 'makes', 'lese', 'sense', 'buy', 'vs.', 'places.\\\\nbottom', 'decision', 'whether', 'think', 'market', 'keep', 'going', '.', 'numbers', 'tend', 'even', ',', 'main', 'gain', 'loss', 'buying', 'comes', 'appreciation/depreciation', '.']\n"
     ]
    }
   ],
   "source": [
    "#Display the sentences/words after filtering out the stopwords\n",
    "# Then inserting, swapping, and deleting randomly n times\n",
    "\n",
    "for i in df_train['Solution'].head():\n",
    " \n",
    "    stop_words = set(stopwords.words('english'))\n",
    " \n",
    "    word_tokens = word_tokenize(i)\n",
    " \n",
    "    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]\n",
    " \n",
    "    filtered_sentence = []\n",
    " \n",
    "    for w in word_tokens:\n",
    "        if w not in stop_words:\n",
    "            filtered_sentence.append(w)\n",
    "    print(word_tokens)\n",
    "    print(\"BEFORE: \")\n",
    "    print(filtered_sentence)\n",
    "    print(\"AFTER INSERTION: \")\n",
    "    print(random_insertion(filtered_sentence, 3))\n",
    "    print(\"AFTER SWAP: \")\n",
    "    print(random_swap(filtered_sentence, 2))\n",
    "    print(\"AFTER DELETION: \")\n",
    "    print(delete_random_elems(filtered_sentence, 3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "8c9a4120",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Class Index              0\n",
       "Question                 0\n",
       "Questions Content    27106\n",
       "Answer                1034\n",
       "dtype: int64"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking the amount of nulls per column\n",
    "df_test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "e09f5fd1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Class Index               0\n",
       "Question                  0\n",
       "Questions Content    631675\n",
       "Solution              24579\n",
       "dtype: int64"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking the amount of nulls per column\n",
    "df_train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "b6c2a70a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To start off, I filtered out a column, which has no relation to the questions and answers in each dataframe. \n",
    "# There are exactly 631675 nulls for Question#2 and 24759 nulls in the train dataframe , while there are a lot less null spaces in the test dataframe with 27106 spaces for Question #2 and 1034 spaces for Answer.\n",
    "# After taking out the stopwords from the dataframes, I was able to filter the sentence and then randomly insert k amount of sentences, then randomly swap 2 values k times for each of the split sentences. Finally, I implemented a randomly delete function, which randomly removes each word in a sentence with probability p. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "f8a0083c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1.400000e+06</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.500000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.872282e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.500000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>8.000000e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.000000e+01</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Class Index\n",
       "count  1.400000e+06\n",
       "mean   5.500000e+00\n",
       "std    2.872282e+00\n",
       "min    1.000000e+00\n",
       "25%    3.000000e+00\n",
       "50%    5.500000e+00\n",
       "75%    8.000000e+00\n",
       "max    1.000000e+01"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train.describe()\n",
    "#Based on the Class Index column in the train dataframe, the index mean is 5.5 (Computers and Internet) and the max is Politics and Government. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "id": "0f78c3d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Class Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>60000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.872305</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>8.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Class Index\n",
       "count  60000.000000\n",
       "mean       5.500000\n",
       "std        2.872305\n",
       "min        1.000000\n",
       "25%        3.000000\n",
       "50%        5.500000\n",
       "75%        8.000000\n",
       "max       10.000000"
      ]
     },
     "execution_count": 159,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.describe()\n",
    "#Based on the Class Index column in the test dataframe, the index mean is 5.5 (Computers and Internet) and the max is Politics and Government. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17e9509d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 'What should I do?' is the most common question for the first column as the frequency is always so high. The answer 'no' is also very high as the frequency is 1244 and 51, for each dataframe respectively.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "id": "0cdf899d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Class Index'>"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# the boxplot for the test dataframe regarding the index\n",
    "sns.boxplot(x=df_test['Class Index'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "id": "69615867",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='Class Index'>"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# the boxplot for the test dataframe regarding the index\n",
    "sns.boxplot(x=df_train['Class Index'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "id": "2a92f261",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1400000, 4)"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "f5681b9a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 4)"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1038786",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
