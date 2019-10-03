
# coding: utf-8

# # Problem 2

# We have provided files that provide yearly game appearance statistics for every
# player to have played in Major League Baseball between the years 1871 and 2014.
# 
# The files can be found on:
# 
# https://s3.amazonaws.com/dd-interview-data/data_scientist/baseball/appearances/YYYY/YYYY-0,000
# 
# 
# The header for these files is as follows:
# 
# 
# Year,Team,League,Player ID code,Player Name,Total games played,Games started,Games in which player batted,Games in which player appeared on defense,Games as pitcher,Games as catcher,Games as firstbaseman,Games as secondbaseman,Games as thirdbaseman,Games as shortstop,Games as leftfielder,Games as centerfielder,Games as right fielder,Games as outfielder,Games as designated hitter,Games as pinch hitter,Games as pinch runner
# 
# 
# 
# Write a program that downloads these files and produces a list of triples of
#  teams for which at least 50 players have played for all three teams.
# 
# For instance, Alex Rodriguez has played for the Mariners, Rangers, and Yankees,
# and thus he would count once for the Mariners/Rangers/Yankees triple.
# 

# ### Software prerequesites

# This notebook requires the following dependencies:
# 
# * `numpy`
# * `pandas`
# * `jupyter`
# * `tqdm`
# 
# Please install them using `conda` or using `pip` before running the following code

# In[12]:


print("Please consider using jupyter notebook to read the script\nMore details in \"README_BEFORE_RUN.txt\" ")

#%matplotlib inline
import numpy as np
import pandas as pd

# ### Load data

# In[13]:


import requests
from tqdm import tqdm

#Define the headers
HEADERS = ("Year,team,League,player_id,Player Name,Total games played,"
"Games started,Games in which player batted," 
"Games in which player appeared on defense,Games as pitcher,"
"Games as catcher,Games as firstbaseman,Games as secondbaseman,"
"Games as thirdbaseman,Games as shortstop,Games as leftfielder,"
"Games as centerfielder,Games as right fielder,Games as outfielder,"
"Games as designated hitter,Games as pinch hitter,Games as pinch runner")
HEADERS = HEADERS.split(",")
	
def read_data(year):
	'''read data from the given year, set them in a list'''
	#import data for the year given in parameters from the corresponding url
	
	year = str(year)
	url = 'https://s3.amazonaws.com/dd-interview-data/data_scientist/baseball/appearances/%s/%s-0,000' % (year,year)
	r = requests.get(url, allow_redirects=True).content
	
	#we get "bytes" data : it is necessary to decode them
	r = r.decode('utf-8')	
	r = r.split("\n")
	split_coma = lambda s:s.split(",")
	r = list(map(split_coma,r))
	
	# each file ends with a \n, then we have to remove the last entry that is empty
	return r[:-1] if r[-1]==[""] else r

def get_dataframe(year_start,year_end):
	entries = []
	print("Downloading the data ... please wait")
	for year in tqdm(range(year_start, year_end + 1)):
		entries += read_data(year)
	return pd.DataFrame(entries, columns=HEADERS)


# In[14]:


data = get_dataframe(1871,2014)


# In[15]:


data.head()


# In[16]:


teams = data["team"].unique()
n_teams = len(teams)
n_teams

latex_text = "We find that n_teams = 151. Thus, there is $binom{151}{3} = 562475$ possibilities. Of course, trying every player on every combination works, but let imagine we have 100 iterations/sec, as we have about 18000 players, it will take about frac{562475 times 18000}{100} = 10^9 seconds$ !"

print("\n(You can see the code with Latex by using jupyter or in the corresponding html file)\n\n",latex_text)
 
print("Obviously, we have to find another way to find the solution.")
print("\n\n")

# ### Find the players of each team

# In[17]:
print("We want to find the players associated to each team")

players_in_team = []
considered_teams = []
print("processing...")
for team in tqdm(data["team"].unique()):
	players = data[data["team"]==team]["player_id"].unique()
	
	# we can filter, because considering a team with less than 50 players is pointless
	filter_players = True
    
	# if you want to experience the speed boost granted by the filter, I invite you to change the value 
	# of filter_players to False (no filter)
	if len(players)>= (50 if filter_players else 0):
		considered_teams.append(team)
		players_in_team.append(players)


# In[18]:


len(considered_teams)


# ### Intersection between the different teams

# In[19]:


def intersection(l1, l2):  
	'''intersection between two lists'''
	tempset = set(l2) 
	l3 = [value for value in l1 if value in tempset] 
	return l3 

def intersection_3(container,i,j,k):
	'''Intersection between the i-th, j-th, and k-th list of container'''
	return intersection(container[i],intersection(container[j],container[k]))

latex_text = "We consider 68 teams so we have to process the triple intersection $binom{68}{3} approx 50000 times$. The triple intersection is really fast. The intersection is, in average, in $O(min(len(l_1),len(l_2)))$. Thus, we assume that the triple intersection has a linear complexity in the length of the lists given in argument. \n(In fact, we can have in the worst case: $ O(len(l_1) * len(l_2))$ according to https://wiki.python.org/moin/TimeComplexity#set. However, that \"worst case\" assumes data that is inappropriate for use in the hash table used by dict and set : so, in our case, we don't have to worry much about this.)"
print("\n(You can see the code with Latex by using jupyter or in the corresponding html file)\n\n",latex_text)



print("As a conclusion, processing the intersection between so much combinations is possible in a reasonable time.")

# In[20]:


solution = []

c = 0

print("Processing intersection between different possible triples")
n_possible_teams = len(considered_teams)
for i_team in tqdm(range(n_possible_teams-2)):
	for j_team in range(i_team+1,n_possible_teams-1):
		for k_team in range(j_team+1,n_possible_teams):
			c+=1
			if len(intersection_3(players_in_team,i_team,j_team,k_team))>50:
				solution.append((i_team,j_team,k_team))


# In[21]:


solution = [(considered_teams[i],considered_teams[j],considered_teams[k]) for i,j,k in solution]


# ## Conclusion

# ### Solution

# In[22]:


print("Finally,",len(solution),"triples of teams have at least 50 players who have played for all three teams.")
print("The solution is the following :")
for triples in solution:
	print(triples)


# ### Complexity study


latex_text = """Downloading the data will not count in the final complexity. Its complexity in time and space is linear in N, where N = dataset_size (assuming we do not encounter any network issues)\n
The intersection between 3 lists of players has a time complexity of $O($size_of_players_lists$)$ and have a constant space complexity.\n
The part that may take some time is the one in which we have to intersect every existing triples. \n
By removing every team with less than 50 players, we had a great boost in speed. We had 151 teams, and after the filter we have only 68 remaining. Then, instead of considering $binom{151}{3}=562475$ possible combinations, we consider only $binom{68}{3}=50116$ of them. It is about 10 times less, so it is a great boost. In practice we only have about 5 times less computing time, because all the removed list are very small (<50 players), then the intersect is even faster. 
\n
The space complexity is $O(n_{solution})$ and is very low in our case.

As a conclusion, we have a time complexity in the worst case (that is "every list contains every player") in $$C_{time} = O(n_{players} * binom{n_{team}}{3}) = O(n_{players} * frac{n_{team}!}{(n_{team}-3)!})$$
$$C_{time} = O(n_{players} * (n_{team})^3)$$
and a space complexity in the worst case ("every list contains every player") : 
$$C_{space} = O(n_{players} * n_{team} + N_{dataset})$$
\n\n
Please note that the space complexity can be reduced if we extract the players and the teams for each year at a time, and if we remove the entire dataset from the memory when we have extracted all the informations. In our case, there is no need to do such a thing ; however it is a good way to save some memory if the dataset proves to be too big."""

print("\nYou can see the code with Latex by using jupyter or in the corresponding html file\n\n",latex_text)

input("\n\n\nPress enter to exit")
