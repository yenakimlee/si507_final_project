import pandas as pd
import numpy as np
import requests
import json
import bs4 as bs
import re
from datetime import datetime
from datetime import date
import snscrape.modules.twitter as sntwitter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from spacy.tokens import Doc
import sqlite3 as sql
import matplotlib.pyplot as plt
import networkx as nx
import plotly.express as px
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

import warnings
warnings.filterwarnings("ignore")

# Import mass_shootings data

mass_shootings = pd.read_csv("./Mass Shootings Dataset.csv", encoding = "latin1")
mass_shootings = mass_shootings.drop(["S#", "Latitude", "Longitude"], axis = 1)
mass_shootings_date = mass_shootings[["Date", "Fatalities", "Injured", "Total victims"]]
mass_shootings_date["Date"] = mass_shootings_date["Date"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%Y'))
mass_shootings_date = mass_shootings_date.groupby([mass_shootings_date["Date"]]).sum().reset_index()

# Importing Data from CDC Wonder

b_parameters = {
    "B_1": "D76.V1-level1",
    "B_2": "D76.V1-level2",
    "B_3": "D76.V8",
    "B_4": "D76.V2-level3",
    "B_5": "*None*"
}

m_parameters = {
    "M_1": "D76.M1",   # Deaths, must be included
    "M_2": "D76.M2",   # Population, must be included
    "M_3": "D76.M3",   # Crude rate, must be included
    #"M_31": "D76.M31",        # Standard error (crude rate)
    #"M_32": "D76.M32"         # 95% confidence interval (crude rate)
    "M_41": "D76.M41", # Standard error (age-adjusted rate)
    "M_42": "D76.M42"  # 95% confidence interval (age-adjusted rate)
}

f_parameters = {
    "F_D76.V1": ["*All*"], # year/month
    "F_D76.V10": ["*All*"], # Census Regions - dont change
    #"F_D76.V2": ["X85-Y09"], # ICD-10 Codes
    "F_D76.V2": ["X93", "X94", "X95",
                 "W32", "W33", "W34",
                 "Y22", "Y23", "Y24",
                 "U01.4"],
    "F_D76.V27": ["*All*"], # HHS Regions - dont change
    "F_D76.V9": ["*All*"] # State County - dont change
}

i_parameters = {
    "I_D76.V1": "*All* (All Dates)",  # year/month
    "I_D76.V10": "*All* (The United States)", # Census Regions - dont change
    "I_D76.V2": "V01-Y89 (External causes of morbidity and mortality)", # ICD-10 Codes
    "I_D76.V27": "*All* (The United States)", # HHS Regions - dont change
    "I_D76.V9": "*All* (The United States)" # State County - dont change
}

v_parameters = {
    "V_D76.V1": "",         # Year/Month
    "V_D76.V10": "*All*",   # Census Regions
    "V_D76.V11": "*All*",   # 2006 Urbanization
    "V_D76.V12": "*All*",   # ICD-10 130 Cause List (Infants)
    "V_D76.V17": "*All*",   # Hispanic Origin
    "V_D76.V19": "*All*",   # 2013 Urbanization
    "V_D76.V2": "",         # ICD-10 Codes
    "V_D76.V20": "*All*",   # Autopsy
    "V_D76.V21": "*All*",   # Place of Death
    "V_D76.V22": "*All*",   # Injury Intent
    "V_D76.V23": "*All*",   # Injury Mechanism and All Other Leading Causes
    "V_D76.V24": "*All*",   # Weekday
    "V_D76.V25": "*All*",   # Drug/Alcohol Induced Causes
    "V_D76.V27": "",        # HHS Regions
    "V_D76.V4": "*All*",    # ICD-10 113 Cause List
    "V_D76.V5": ["15-24", "25-34", "35-44"], # Ten-Year Age Groups
    "V_D76.V51": "*All*",   # Five-Year Age Groups
    "V_D76.V52": "*All*",   # Single-Year Ages
    "V_D76.V6": "00",       # Infant Age Groups
    "V_D76.V7": "*All*",    # Gender
    "V_D76.V8": "*All*",    # Race
    "V_D76.V9": ""          # State/County
}

o_parameters = {
    "O_V10_fmode": "freg",    # Use regular finder and ignore v parameter value
    "O_V1_fmode": "freg",     # Use regular finder and ignore v parameter value
    "O_V27_fmode": "freg",    # Use regular finder and ignore v parameter value
    "O_V2_fmode": "freg",     # Use regular finder and ignore v parameter value
    "O_V9_fmode": "freg",     # Use regular finder and ignore v parameter value
    "O_aar": "aar_std",       # age-adjusted rates
    "O_aar_pop": "0000",      # population selection for age-adjusted rates
    "O_age": "D76.V5",        # select age-group (e.g. ten-year, five-year, single-year, infant groups)
    "O_javascript": "on",     # Set to on by default
    "O_location": "D76.V10",   # select location variable to use (e.g. state/county, census, hhs regions)
    "O_precision": "1",       # decimal places
    "O_rate_per": "100000",   # rates calculated per X persons
    "O_show_totals": "false",  # Show totals for 
    "O_timeout": "300",
    "O_title": "Assault by Age Group",    # title for data run
    "O_ucd": "D76.V2",        # select underlying cause of death category
    "O_urban": "D76.V19"      # select urbanization category
}

vm_parameters = {
    "VM_D76.M6_D76.V10": "",        # Location
    "VM_D76.M6_D76.V17": "*All*",   # Hispanic-Origin
    "VM_D76.M6_D76.V1_S": "*All*",  # Year
    "VM_D76.M6_D76.V7": "*All*",    # Gender
    "VM_D76.M6_D76.V8": "*All*"     # Race
}

misc_parameters = {
    "action-Send": "Send",
    "finder-stage-D76.V1": "codeset",
    "finder-stage-D76.V1": "codeset",
    "finder-stage-D76.V2": "codeset",
    "finder-stage-D76.V27": "codeset",
    "finder-stage-D76.V9": "codeset",
    "stage": "request"
}

def createParameterList(parameterList):
    """Helper function to create a parameter list from a dictionary object"""

    parameterString = ""

    for key in parameterList:
        parameterString += "<parameter>\n"
        parameterString += "<name>" + key + "</name>\n"

        if isinstance(parameterList[key], list):
            for value in parameterList[key]:
                parameterString += "<value>" + value + "</value>\n"
        else:
            parameterString += "<value>" + parameterList[key] + "</value>\n"
        parameterString += "</parameter>\n"
    return parameterString


xml_request = "<request-parameters>\n"
xml_request += createParameterList(b_parameters)
xml_request += createParameterList(m_parameters)
xml_request += createParameterList(f_parameters)
xml_request += createParameterList(i_parameters)
xml_request += createParameterList(o_parameters)
xml_request += createParameterList(vm_parameters)
xml_request += createParameterList(v_parameters)
xml_request += createParameterList(misc_parameters)
xml_request += "</request-parameters>"


def cache_to_file(file_name):
    def decorator(func):
        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        def new_func(param):
            if param not in cache:
                cache[(param)] = func(param)
                json.dump(cache, open(file_name, 'w'))
                print('Fetching WONDER Data')
                return cache[param]
            print('Using Cache to Retrieve WONDER data')
            return cache[param]
        return new_func
    return decorator

@cache_to_file('cache_wonder.dat')
def get_wonder_data(site_url):
    response = requests.post(site_url, data={"request_xml": xml_request, "accept_datause_restrictions": "true"})
    if response.status_code == 200:
        data = response.text
    else:
        print("something went wrong")
    return data

def xml2df(xml_data):
    """ This function grabs the root of the XML document and iterates over
        the 'r' (row) and 'c' (column) tags of the data-table
        Rows with a 'v' attribute contain a numerical value
        Rows with a 'l attribute contain a text label and may contain an
        additional 'r' (rowspan) tag which identifies how many rows the value
        should be added. If present, that label will be added to the following
        rows of the data table.
        Function returns a two-dimensional array or data frame that may be
        used by the pandas library."""

    root = bs.BeautifulSoup(xml_data,"lxml")
    all_records = []
    row_number = 0
    rows = root.find_all("r")

    for row in rows:
        if row_number >= len(all_records):
            all_records.append([])
        for cell in row.find_all("c"):
            if 'v' in cell.attrs:
                try:
                    all_records[row_number].append(float(cell.attrs["v"].replace(',','')))
                except ValueError:
                    all_records[row_number].append(cell.attrs["v"])
            else:
                if 'r' not in cell.attrs:
                    all_records[row_number].append(cell.attrs["l"])
                else:
                    for row_index in range(int(cell.attrs["r"])):
                        if (row_number + row_index) >= len(all_records):
                            all_records.append([])
                            all_records[row_number + row_index].append(cell.attrs["l"])
                        else:
                            all_records[row_number + row_index].append(cell.attrs["l"])
        row_number += 1
    return all_records

data_frame = xml2df(get_wonder_data("https://wonder.cdc.gov/controller/datarequest/D76"))

wonder_df = pd.DataFrame(data=data_frame, columns=["Year", "Month", "Race", "Death-Type", "Deaths", "Population", "Crude Rate", "Age-adjusted Rate", "Age-adjusted Rate Standard Error"])

wonder_df = wonder_df[["Year", "Month", "Race", "Death-Type", "Deaths"]]

conditions = [
    (wonder_df['Death-Type'] == "Assault by rifle, shotgun and larger firearm discharge"),
    (wonder_df['Death-Type'] == "Assault by other and unspecified firearm discharge"),
    (wonder_df['Death-Type'] == "Handgun discharge"),
    (wonder_df['Death-Type'] == "Rifle, shotgun and larger firearm discharge"),
    (wonder_df['Death-Type'] == "Discharge from other and unspecified firearms"),
    (wonder_df['Death-Type'] == "Assault by handgun discharge"),
    (wonder_df['Death-Type'] == "Other and unspecified firearm discharge, undetermined intent"),
    (wonder_df['Death-Type'] == "Handgun discharge, undetermined intent"),
    (wonder_df['Death-Type'] == "Rifle, shotgun and larger firearm discharge, undetermined intent"),
    (wonder_df['Death-Type'] == "Terrorism involving firearms"),
]

values = ["W32", "X95", "X93", "W33", "W34", "X93", "Y24", "Y22", "Y23", "U01.4"]

wonder_df['ICD-10'] = np.select(conditions, values)
wonder_df["Month"] = wonder_df["Month"].apply(lambda x: re.sub("[.,]", "", x))
wonder_df["Month"] = wonder_df["Month"].apply(lambda x: datetime.strptime(x.strip(), '%b %Y'))
wonder_date = wonder_df[["Year", "Deaths"]]
wonder_date = wonder_date.groupby([wonder_date["Year"]]).sum().reset_index()

# Importing Tweets from Twitter

# Setting variables to be used below
maxTweets = 100

# Creating list to append tweet data to
tweets_list = []

date_range = pd.date_range(start = '2006-01-01', end = date.today(), freq = 'W') #Twitter was created in 2006
date_range_str = [str(date)[:-9] for date in date_range]

search_phrases = ["gun violence",
                  "gun regulations",
                  "progun",
                  "gun control",
                  "guns",
                  "firearms",
                  "weapons",
                  "rifle",
                  "pistol",
                  "bullets",
                  "2A",
                  "secondamendment",
                  "2ndamendment"
                  "shooting",
                  "gun"
                 ]


def cache_to_file(file_name):
    def decorator(func):

        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        def new_func(parameter):
            for param in parameter:
                if param not in cache:
                    cache[(param)] = func(param)
                    json.dump(cache, open(file_name, 'w'), default=str)
                    print('Fetching Twitter Data')
                    return cache[param]
                print('Using Cache to Retrieve Twitter Data')
                return cache[param]

        return new_func

    return decorator


@cache_to_file('cache_twitter.dat')
def get_twitter_data(date_range):
    i = 0

    while i < len(date_range):
        for phrase in search_phrases:
            if i == len(date_range) - 1:
                for x,tweet in enumerate(sntwitter.TwitterSearchScraper(f"{phrase} since:{date_range[i]} until:{date.today()}").get_items()):
                    if x>maxTweets:
                        break

                    tweets_list.append([tweet.date, tweet.content])
            else:
                for x,tweet in enumerate(sntwitter.TwitterSearchScraper(f"{phrase} since:{date_range[i]} until:{date_range[i+1]}").get_items()):
                    if x>maxTweets:
                        break

                    tweets_list.append([tweet.date, tweet.content])

        i += 1
    return tweets_list


tweets_list_df = pd.DataFrame(get_twitter_data(date_range_str), columns =['Date',
                                                                          'Content'])

sent_analyzer = SentimentIntensityAnalyzer()
def sentiment_scores(docx):
    return sent_analyzer.polarity_scores(docx.text)

nlp = spacy.load('en_core_web_sm')
Doc.set_extension("sentimenter",getter=sentiment_scores, force = True)


tweets_list_df['rating'] = tweets_list_df['Content'].apply(lambda x: nlp(x)._.sentimenter['compound'])
tweets_list_df["Date"] = tweets_list_df["Date"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S+00:00').strftime('%Y'))
tweets_list_df = tweets_list_df.groupby(by=["Date"]).agg({'Content': 'count', 
                         'rating':'mean'}).reset_index()


# ## Create a database
database = "gun_violence.db"
connection = sql.connect(database)

tweets_list_df.to_sql("tweet_list_df", connection, if_exists = "replace")
wonder_date.to_sql("wonder_date", connection, if_exists = "replace")
mass_shootings_date.to_sql("mass_shootings_date", connection, if_exists = "replace")
mass_shootings.to_sql("mass_shootings", connection, if_exists = "replace")

cur = connection.cursor()


# Modify table structure

cur.execute('''
DROP TABLE IF EXISTS mass_shootings_df''')

cur.execute('''
CREATE TABLE mass_shootings_df
(
  row_count INT,
  title VARCHAR,
  location VARCHAR,
  date DATE,
  summary VARCHAR,
  fatalities INT,
  injured INT,
  total_victims INT,
  mental_health_issues VARCHAR,
  race VARCHAR,
  gender VARCHAR
)
''')

cur.execute('''INSERT INTO mass_shootings_df
SELECT * FROM mass_shootings''')

cur.execute('''
DROP TABLE IF EXISTS mass_shootings;
''')


# Create tables for graph

cur.execute('''
DROP TABLE IF EXISTS tweet_list''')

cur.execute('''
CREATE TABLE tweet_list
(
  "row_count" INTEGER,
  "tweet_date" DATE NOT NULL,
  "tweet_count" INTEGER,
  "mean_sentiment" FLOAT,
  CONSTRAINT tweet_pk PRIMARY KEY (tweet_date)
)
''')

cur.execute('''INSERT INTO tweet_list
SELECT * FROM tweet_list_df''')

cur.execute('''
DROP TABLE IF EXISTS tweet_list_df;
''')

cur.execute('''
DROP TABLE IF EXISTS wonder_data''')

cur.execute('''
CREATE TABLE wonder_data
(
  "row_count" INTEGER,
  "wonder_date" DATE NOT NULL,
  "death_count" INTEGER,
  FOREIGN KEY ( wonder_date ) REFERENCES [tweet_list] ( tweet_date ) ON UPDATE  NO ACTION  ON DELETE  CASCADE
)
''')

cur.execute('''
            INSERT INTO wonder_data
            SELECT * FROM wonder_date''')

cur.execute('''
DROP TABLE IF EXISTS wonder_date;
''')

cur.execute('''
DROP TABLE IF EXISTS shootings_data;''')

cur.execute('''
CREATE TABLE shootings_data
(
  "row_count" INTEGER,
  "shooting_date" DATE NOT NULL,
  "fatalities" INTEGER,
  "injured" INTEGER,
  "total_victims" INTEGER,
  CONSTRAINT gun_injuries_pk PRIMARY KEY (shooting_date) 
  FOREIGN KEY ( shooting_date ) REFERENCES [tweet_list] ( tweet_date ) ON UPDATE  NO ACTION  ON DELETE  CASCADE
)
''')

cur.execute('''
            INSERT INTO shootings_data
            SELECT * FROM mass_shootings_date''')

cur.execute('''
DROP TABLE IF EXISTS mass_shootings_date;
''')

combined_data = pd.read_sql_query('''
                                 SELECT 
                                     shootings_data.shooting_date,
                                     shootings_data.fatalities AS shooting_fatalities,
                                     wonder_data.death_count AS wonder_death_count,
                                     tweet_list.tweet_count,
                                     tweet_list.mean_sentiment
                                 FROM wonder_data 
                                     LEFT JOIN shootings_data ON wonder_date = shooting_date
                                     LEFT JOIN tweet_list ON wonder_date = tweet_date''', connection)


# Create Graph - Directed Cyclic Graph

combined_data.sort_values("shooting_date")
combined_data = combined_data.fillna('')

#fill in missing dates
i = 19
year = 2018
while i < 22:
    combined_data.shooting_date[i] = int(year)
    i += 1
    year += 1

combined_data.shooting_date = combined_data.shooting_date.astype(int)
combined_data_dict = combined_data.set_index("shooting_date").agg(list, axis=1).to_dict()

G = nx.Graph()
for dict_key in combined_data_dict:
    G.add_nodes_from([(dict_key, 
                       {"fatalities": combined_data_dict[dict_key][0],
                        "death_count": combined_data_dict[dict_key][1],
                        "tweet_count": combined_data_dict[dict_key][2],
                        "sentiment": combined_data_dict[dict_key][3]}),])

#add edges
for i in range(len(combined_data_dict.keys())):
    if i != len(combined_data_dict.keys()) - 1:
        G.add_edge(combined_data.shooting_date[i], combined_data.shooting_date[i+1])

#export to json
from networkx.readwrite import json_graph

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open('./graph_structure.json', 'w') as f:
    json.dump(json_graph.node_link_data(G), f, cls=NpEncoder)


# User Interaction

valid_years =["1999", "2000", "2001", "2002", "2003",
                "2004", "2005", "2006", "2007", "2008",
                "2009", "2010", "2011", "2012", "2013",
                "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

year1 = input("Please enter the starting year between 1999-2020. ")
if year1 == "exit":
    print("User requested to exit the program")
    sys.exit()

year2 = input("Please enter the ending year between 1999-2020. ")
if year2 == "exit":
    print("User requested to exit the program")
    sys.exit()

if (year1 not in valid_years) | (year2 not in valid_years) | (int(year1) > int(year2)):
    print("Invalid date range")
    sys.exit()

year1 = datetime.strptime("1/1/" + year1, '%d/%M/%Y').strftime('%m/%d/%Y')
year2 = datetime.strptime("12/31/" + year2, '%d/%M/%Y').strftime('%m/%d/%Y')

year1 = year1[-4:]
year2 = year2[-4:]

try:
    if year1 == year2:
        year1 = int(year1)
        if int(year1) <= 2017:
            print(f'''In the year {year1}, there were {G.nodes[year1]["fatalities"]} fatalities due to mass shootings.
    There were a total of {G.nodes[year1]["death_count"]} individuals dead due to firearm related injuries.
    During that time there were {G.nodes[year1]["tweet_count"]} tweets on guns and gun regulations with a mean sentiment score of {str(round(G.nodes[year1]["sentiment"], 2))}
    ''')
        else:
            print(f'''In the year {year1}, there were a total of {G.nodes[year1]["death_count"]} individuals dead due to firearm related injuries.
    During that time there were {G.nodes[year1]["tweet_count"]} tweets on guns and gun regulations with a mean sentiment score of {str(round(G.nodes[year1]["sentiment"], 2))}
    ''')
    else:
        for path in nx.all_simple_paths(G, source= int(year1), target= int(year2)):
            for node in path:
                if node <= 2017:
                    print(f"In the year {node}, there were {G.nodes[node]['fatalities']} fatalities due to mass shootings")
                print(f"There were a total of {G.nodes[node]['death_count']} individuals dead due to firearm related injuries")
                if node >= 2011:
                    print(f"During that time there were {G.nodes[node]['tweet_count']} tweets on guns and gun regulations with a mean sentiment score of {str(round(G.nodes[node]['sentiment'], 2))}\n")
                else:
                    print("\n")

except:
    raise ValueError("Please input valid numbers")

year2 = str(int(year2) + 1)

while True:
    shootings_output = input("Would you like to see written descriptions of the mass shootings that occurred in your selected timeframe? Y/N ")
    if shootings_output in ["Y", "y", "N", "n"]:
        break
if shootings_output in ["Y", "y", "Yes", "YES", "yes"]:
    shootings_data = pd.read_sql_query('''
                                SELECT
                                    *
                                FROM mass_shootings_df''', connection)

    shootings_data["date"] = shootings_data["date"].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
    shootings_data["year"] = shootings_data["date"].apply(lambda x: x.strftime('%Y'))

    if year1 == int(year2)-1:
        queried_shootings_data = shootings_data[(shootings_data["year"] == str(year1))].reset_index()
    else:
        queried_shootings_data = shootings_data[(shootings_data["year"] >= year1) & (shootings_data["year"] < year2)].reset_index()

    if not queried_shootings_data.empty:
        print(f'''These are the mass shootings that happened between {year1} and {str(int(year2) -1)}: \n''')
        for index, row in queried_shootings_data.iterrows():
            print(f'''{index}: {row['title']} in {row['location']} on {row['date'].strftime('%m/%d/%Y')}.
    Summary: {row['summary']}
    There were {row['fatalities']} fatalities and {row['injured']} people injured for a total of {row['total_victims']} total_victims.
    The race of the shooter is {row['race']} and their gender is {row['gender']}.\n\n''')

    else:
        if year1 == int(year2)-1:
            print(f"There were no mass shootings that happened in {year1}")
        else:
            print(f"There were no mass shootings that happened between {year1} and {str(int(year2) -1)}")

queried_wonder_data = wonder_df[(wonder_df["Month"] > year1) & (wonder_df["Month"] <= year2)].reset_index()

queried_wonder_data_race = queried_wonder_data[["Race", "Deaths"]].groupby("Race").agg(sum).reset_index()

while True:
    race_output = input("Would you like to see information on the race of victims of firearm related injuries? Y/N ")
    if shootings_output in ["Y", "y", "N", "n"]:
        break

if race_output in ["Y", "y"]:
    race = px.bar(x = queried_wonder_data_race.Race, y = queried_wonder_data_race.Deaths)
    print(f"The race of the individuals who were victims of firearm related injuries is as follows:")
    race.show()

while True:
    death_type_output = input("Would you like to see information on the type of deaths of victims of firearm related deaths? Y/N ")
    if death_type_output in ["Y", "y", "N", "n"]:
        break

if death_type_output in ["Y", "y"]:
    queried_wonder_data_type = queried_wonder_data[["Death-Type", "Deaths"]].groupby("Death-Type").agg(sum).reset_index()
    death_type = px.bar(x = queried_wonder_data_type["Death-Type"], y = queried_wonder_data_type.Deaths)
    print(f"The type of death of individuals who were victims of firearm related injuries is as follows:")
    death_type.show()

while True:
    num_deaths_output = input("Would you like to see information on the number of firearm related injuries? Y/N ")
    if num_deaths_output in ["Y", "y", "N", "n"]:
        break

if num_deaths_output in ["Y", "y"]:
    print(f"The # of deaths of individuals who were victims of firearm related injuries is as follows:")
    queried_wonder_data_count = queried_wonder_data[["Month", "Deaths"]].groupby("Month").agg(sum).reset_index()
    death_count = px.line(x = queried_wonder_data_count["Month"], y = queried_wonder_data_count.Deaths)
    death_count.show()

if int(year1) >= 2011:
    while True:
        tweet_output = input("Would you like to see information regarding tweets? Y/N ")
        if tweet_output in ["Y", "y", "N", "n"]:
            break

    queried_twitter_data =  tweets_list_df[(tweets_list_df["Date"] >= year1) & (tweets_list_df["Date"] < year2)].reset_index()

    if tweet_output in ["Y", "y"]:
        trace1 = go.Line(
        x= queried_twitter_data["Date"],
        y= queried_twitter_data["Content"],
        name="Count of Tweets",
        )
        trace2 = go.Line(
            x=queried_twitter_data["Date"],
            y=queried_twitter_data["rating"],
            name='Sentiment',
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(trace1)
        fig.add_trace(trace2,secondary_y=True)
        fig['layout'].update(height = 600, width = 800, title = "Tweets",xaxis=dict(
            tickangle=-90
            ))

        iplot(fig)

if int(year1) >= 2011:

    while True:
        comparison_output = input('''What dataset would you like to compare the sentiment of tweets with?
                                    1. Mass shootings dataset
                                    2. Firearm related injuries
                                    Please input the number 1 or 2''')
        if comparison_output in ["1", "2"]:
            break


    if int(comparison_output) == 1:

        mass_shootings_date = mass_shootings_date[(mass_shootings_date["Date"] >= year1) & (mass_shootings_date["Date"] < year2)].reset_index()
        mass_shootings_date_query = mass_shootings_date.groupby("Date").agg(sum).reset_index()

        trace1 = go.Line(
            x= queried_twitter_data["Date"],
            y= queried_twitter_data["rating"],
            name="Sentiment",
        )

        while True:
            injured_death = input('''Would you like to compare it with:
                                1. Total Victims
                                2. Injured
                                3. Fatalities
                                Please press 1, 2, or 3''')
            if injured_death in ["1", "2", "3"]:
                break

        type_injury= ["Total victims", "Injured", "Fatalities"]
        x = type_injury[int(injured_death) - 1]

        trace2 = go.Line(
            x = mass_shootings_date_query["Date"],
            y = mass_shootings_date_query[x],
            name='Firearm Related Injuries',
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(trace1)
        fig.add_trace(trace2,secondary_y=True)
        fig['layout'].update(height = 600, width = 800, title = "Tweets",xaxis=dict(
            tickangle=-90
            ))

        iplot(fig)

    elif int(comparison_output) == 2:
        queried_wonder_data_count = queried_wonder_data[["Month", "Deaths"]].groupby("Month").agg(sum).reset_index()
        queried_wonder_data_count["Month"] = queried_wonder_data_count["Month"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d 00:00:00').strftime('%Y'))
        queried_wonder_data_count = queried_wonder_data_count[["Month", "Deaths"]].groupby("Month").agg(sum).reset_index()

        trace1 = go.Line(
            x= queried_twitter_data["Date"],
            y= queried_twitter_data["rating"],
            name="Sentiment",
        )
        trace2 = go.Line(
            x = queried_wonder_data_count["Month"],
            y = queried_wonder_data_count.Deaths,
            name='Firearm Related Injuries',
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(trace1)
        fig.add_trace(trace2,secondary_y=True)
        fig['layout'].update(height = 600, width = 800, title = "Tweets",xaxis=dict(
            tickangle=-90
            ))

        iplot(fig)