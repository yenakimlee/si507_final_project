# si507_final_project

## Introduction

The purpose of this project is to be able to visualize the changing number of shootings and firearm related injuries. My hypothesis is that in months where there have been more 
mass shootings, or more firearm-related injuries and deaths, sentiments will be more negative, and the # of tweets regarding guns will increase. 
I would also like to see the demographic behind the victims of firearm related injuries as well as their type of deaths. 

## Data Sources
- CDC Wonder API Data: https://wonder.cdc.gov/controller/datarequest/D76
  - The ICD-10 codes used to extract the data are: 
    - Firearm Injury: Assault (Gun Homicide, attempted or completed)
      - X93: Assault by handgun discharge
      - X94: Assault by rifle, shotgun, and larger firearm discharge
      - X95: Assault by other and unspecified firearm discharge
    - Firearm Injury: Unintentional (Unintentional shooting, fatal or non-fatal)
      - W32: Handgun discharge
      - W33: Rifle, shotgun, and larger firearm discharge
      - W34: Discharge from other and unspecified firearms
    - Firearm Injury: Undetermined Intent (Unknown cause, fatal or non-fatal)
      - Y22: Handgun discharge and undetermined intent
      - Y23: Rifle shotgun & larger firearm discharge undetermined intent
      - Y24: Other & unspecified firearm discharge undetermined intent
    - Firearm Injury: Terrorism (Gun Terrorism, fatal or non-fatal)
      - U01.4: Terrorism involving firearms (homicide, completed or attempted)
- US Mass Shootings: https://www.kaggle.com/zusmani/us-mass-shootings-last-50-years
  - **Please download this dataset before running the program**
  - This dataset includes information on US mass shootings as well as information on the total number of victims, 
  race and gender of the assailant as well as location.
- Twitter 
  - To gain a better understanding of changing sentiments regarding mass shootings and firearm related injuries, the search phrases we used are: 
    - gun violence
    - gun regulations
    - progun
    - gun control
    - guns
    - firearms
    - weapons
    - rifle
    - pistol
    - bullets
    - 2A
    - secondamendment
    - 2ndamendment
    - shooting
    - gun
  - We scraped for a max of 100 tweets per search phrase, for every week between the 1 Jan 2006 and today

## Packages required to be installed

The following packages are required for the program to run: 
- pandas
- numpy
- requests
- json
- bs4
- re
- datetime
- seaborn
- snscrape
- snscrape.modules.twitter
- nltk
- spacy
- sqlite3
- matplotlib.pyplot
- networkx
- plotly.express
- sys
- plotly.graph_objects
- plotly.subplots
- plotly.offline

## Running the program
You will be given the choice to select two dates between the ranges of 1999-2020.
You will then be given the following choices:
- View written description of mass shootings
- View information on race of firearm victims
- View information on the type of deaths of victims of firearm related deaths
- View information on the number of firearm related injuries
- View information regarding tweets
- Select the dataset you want to compare the sentiment of tweets with
  - If you select the mass shootings dataset, you will then be able to select whether you want to see the number of total victims, number of total individuals injured, or the number of fatalities
Giving an erroneous year will lead the program to automatically stop. Please follow the instructions given.
