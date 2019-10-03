
# coding: utf-8

# # Problem 1

# You have been provided two files, scenario1.txt and scenario2.txt 
# which each contain simulated application logs in the following format:
# 
# Column 1: # seconds since time t=0
# 
# Column 2: # requests processed since last log
# 
# Column 3: Mean response time for requests processed since last log
# 
# 
# The logging frequency is 1 per second, but there are no logs for seconds in
# which there were no requests. The data span two simulated weeks and were 
# generated under an idealized/simplified model in which there is a single
# application server which processes requests sequentially using a single thread.
# If a request arrives while the server is busy, it waits in a queue until
# the server is free to process it. There is no limit to the size of the queue.
# 
# For each scenario, please answer the following questions.
# Note that we define "week 2" to begin at second 626400 (6 am on the 8th day).
# 
# 
# 1) How much has the mean response time (specifically, the mean of the response
# times for each individual request) changed from week 1 to week 2?
# 
# 
# 2) Create a plot illustrating the probability distribution of the amount of
# server time it takes to process a request (excluding the time the request
# spends waiting in the queue). How would you describe the distribution?
# 
# 
# 3) Propose a potential cause for the change in response times. 

# ## Software prerequesites

# This notebook requires the following dependencies:
# 
# * `numpy`
# * `pandas`
# * `matplotlib`
# * `jupyter`
# 
# Please install them using `conda` or using `pip` before running the following code

# In[3]:


#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# In[4]:


WEEK2 = 626400 


# In[5]:


#Time management
def convert_sec(day,hour = 6,minute = 0):
    '''convert to sec. Hour default is 6, since each day begin at 6am'''
    return (((day*24 + hour)*60) + minute)*60

print("Please consider using jupyter notebook to read the script\nMore details in \"README_BEFORE_RUN.txt\" ")

print("\n\nScenario 1\n\n")

# ## Load Data : scenario 1

# In[6]:


try:
    data = pd.read_csv('scenario1.txt', sep="\t", header=None)
except:
    input("You need to have the two scenarioX.txt files in the same directory. Press any key to exit ...")
    exit()
data.columns = ["time_sec","requests","response_time"]

week1 = data[data['time_sec']<WEEK2]
week2 = data[data['time_sec']>=WEEK2]


# ## Quick visualization on data

# In[7]:


print("data on approx",np.ceil(len(data)/3600/24),"days")
data.info()


# In[8]:


data.head()


# ### Filters on data

# #### Uniform type filter
# To complete the algorithm, each column has to be of the same type.

# In[9]:


AUTHORIZED_TYPES = [['int64', 'int'],
                   ['int64', 'int'],
                   ['float64', 'float']]


def columns_have_proper_type(data):
    '''Returns True if all columns types are the one specified'''
    types = data.dtypes
    for col in range(3):
        if types[col] not in AUTHORIZED_TYPES[col]:
            return False
    return True


# In[10]:


print("Data columns have proper type :",columns_have_proper_type(data))


print("\n\nVisualization\n\n")

# #### Global visualization of response time

# In[11]:


def highlight_overflow(threshold = np.max(week1['response_time'])):
    """
    Plots the data and highlights every point above the given threshold
    """
    fig = plt.figure(figsize = (15,10))
    plt.plot(week1['time_sec'],week1['response_time'],label="Week 1")
    plt.plot(week2['time_sec'],week2['response_time'],label="Week 2", color = "green")
    plt.title("response time of the server (sec) as a function of time (sec)")
    
    above_threshold = data[data['response_time']>threshold]
    plt.plot(above_threshold['time_sec'],
             above_threshold['response_time'], "r+", label ="overflow",)
    # Separate the days
    for day in range(1,14):
        plt.axvline(x=(day*24+6)*3600, color = "black")
    # Separate the days
    plt.axvline(x=(day*24+6)*3600, color = "black",label = "day separation")
    for day in range(2,14):
        plt.axvline(x=(day*24+6)*3600, color = "black")


    plt.axhline(threshold, color = "red")

    print("data on",len(above_threshold),"points is above given threshold:",threshold)
    
    plt.legend()
    plt.show()


# In[12]:


highlight_overflow(np.max(week1['response_time'])-1)


# #### Visualization of response time per request

# In[13]:


def get_time_per_request(data):
    """returns an altered dataframe where response_time is divided by the number of requests"""
    return pd.concat([data["time_sec"],pd.DataFrame(data['response_time']/data["requests"],columns=["response_time"])], axis = 1)

week1_time_per_request = get_time_per_request(week1)
week2_time_per_request = get_time_per_request(week2)


# In[14]:


fig = plt.figure(figsize = (15,10))
plt.plot(week1_time_per_request['time_sec'],week1_time_per_request['response_time'],label="Week 1")
plt.plot(week2_time_per_request['time_sec'],week2_time_per_request['response_time'],label="Week 2", color = "green")
plt.title("Average time per request as a function of time")

threshold = np.max(week1['response_time']-1)
above_threshold = data[data['response_time']>threshold]
plt.plot(above_threshold['time_sec'],
         np.zeros(len(above_threshold)), "r+", label ="overflow on response time\nprojected on axis time")

# Separate the days
plt.axvline(x=(1*24+6)*3600, color = "black",label = "day separation")
for day in range(2,14):
    plt.axvline(x=(day*24+6)*3600, color = "black")

plt.legend()
plt.show()


# <div class="alert-info" style="padding:15px"> 1) How much has the mean response time (specifically, the mean of the response
# times for each individual request) changed from week 1 to week 2?</div>

# In[15]:


##### Mean of response time per request 
print("Mean of response time per request - week 1 :",np.mean(week1_time_per_request["response_time"]))
print("Mean of response time per request - week 2 :",np.mean(week2_time_per_request["response_time"]))

quantile = week2_time_per_request["response_time"].quantile(.99)      
print("Mean of response time per request | quantile at 99% removed - week 2 :",
      np.mean(week2_time_per_request[week2_time_per_request['response_time']<quantile]["response_time"]))

#### Mean of number of requests 

print("\n")
print("Mean of number of requests - week 1 :",np.mean(week1["requests"]))
print("Mean of number of requests - week 2 :",np.mean(week2["requests"]))

quantile = week2["requests"].quantile(.99)
print("Mean of number of requests | quantile at 99% removed - week 2 :",
      np.mean(week2[week2['requests']<quantile]["requests"]))

quantile = week2["requests"].quantile(.95)
print("Mean of response time per request | quantile at 95% removed - week 2 :",
      np.mean(week2[week2['requests']<quantile]["requests"]))

##### Mean of total response time


print("\n")
print("Mean of total response time - week 1 :",np.mean(week1["response_time"]))
print("Mean of total response time - week 2 :",np.mean(week2["response_time"]))

quantile = week2["response_time"].quantile(.99)
print("Mean of response time per request | quantile at 99% removed - week 2 :",
      np.mean(week2[week2['response_time']<quantile]["response_time"]))

quantile = week2["response_time"].quantile(.95)
print("Mean of response time per request | quantile at 95% removed - week 2 :",
      np.mean(week2[week2['response_time']<quantile]["response_time"]))

print("""
We can see that the week 2 has some extreme values that will make the whole mean reach twice the value of the mean of week1.

The average is very sensitive to extreme values, which is why we have a very high mean in the 2nd week. Without these extreme values, the two means are of the same order of magnitude. So we can solve this by removing the quantile at 95% or at 99%. 
""")
# <div>

# ## Response time distribution

# <div class="alert-info" style="padding:15px">
# 2) Create a plot illustrating the probability distribution of the amount of
# server time it takes to process a request (excluding the time the request
# spends waiting in the queue). How would you describe the distribution?</div>

# In[16]:


def response_time_histogram(data,ax,title = "Data",bins = [i*0.01 for i in range(100)]):
    resp = data["response_time"]
    ax.hist(resp,bins=bins,density=True)
    ax.set_title(title)


# In[17]:


def plot_lognormal_hist(mean,var,size,ax, bins = [i*0.01 for i in range(100)]):
    fact = var/(mean**2)
    mean_ln = np.log(mean/(np.sqrt(1 + fact)))
    var_ln = np.log(1+fact)

    lognormal_sample = np.random.lognormal(mean = mean_ln, sigma = np.sqrt(var_ln),size = size)
    ax.hist(lognormal_sample,bins=bins,density=True)
    ax.set_title("Lognormal distribution")
    return np.mean(lognormal_sample), np.var(lognormal_sample)
    
def plot_power_hist(alpha,size,ax, bins = [i*0.01 for i in range(100)]): 
    power_sample = np.random.power(alpha,size = size)
    ax.set_title("Power-law distribution")
    ax.hist(power_sample,bins=bins,density=True)

def plot_gamma_hist(mean,var,size,ax, bins = [i*0.01 for i in range(100)]):
    shape = mean**2/var
    scale = var/mean
    
    gamma_sample = np.random.gamma(shape,scale,size = size)
    ax.hist(gamma_sample,bins=bins,density=True)
    ax.set_title("Gamma distribution")
    return np.mean(gamma_sample),np.var(gamma_sample)


# In[18]:


def law_comparison(data,data_label = "Data", bins = [i*0.01 for i in range(100)]):
    print("#############\n",data_label,"\n#############")
    
    axs = plt.subplots(2,2,figsize = (15,15),gridspec_kw = {'width_ratios':[4, 4]})
    response_time_histogram(data,axs[1][0,0],data_label,bins)
    
    resp = data["response_time"]
    mean_rt = np.mean(resp)
    var_rt = np.var(resp)
    print("mean :",mean_rt,"\nvar :",var_rt)
    
    generated_samples = 1000000
    m_ln, v_ln = plot_lognormal_hist(mean_rt,var_rt,generated_samples,axs[1][0,1],bins)
    plot_power_hist(1/2,generated_samples,axs[1][1,0],bins)
    m_g, v_g = plot_gamma_hist(mean_rt,var_rt,generated_samples,axs[1][1,1],bins)
    
    plt.show()
    
    print("#############\nEnd",data_label,"\n#############\n\n")


# In[19]:


bins = [i*0.001 for i in range(100)]
law_comparison(week1_time_per_request,"Data Week 1 | response time / request", bins = bins)
law_comparison(week2_time_per_request,"Data Week 2 | response time / request", bins = bins)


# In[20]:


print('There is some extreme data : we use a quantile to remove them and to be able to find a matching distribution.\n')
quantile = week2_time_per_request["response_time"].quantile(0.99)

law_comparison(week2_time_per_request[week2_time_per_request['response_time']<quantile],
               "Data per request Week 2 without quantile", bins = [i*0.001 for i in range(int(quantile//0.001))])

print("""
Two distributions may describe the distribution of response time : the lognormal and the gamma distribution.
We will also display a law power distribution which also comes close to distribution, but has too much weight on zero to compete with the two first distributions.\n\n
Finally, according to the graphs, the log-normal distribution appears to be the best to describe the response time distribution compared to the gamma distribution, especially because gamma distribution has too much weight close to zero.

However, the log-normal distribution is not close enough to the 2nd week data, because this distribution may model poorly the extreme values that we have highlighted previously. We would have to add some weight far from the mean, and then we will be able to model these extreme values.

As a conclusion, the distribution is a lognormal distribution with some weight far from the mean. 
""")
# Two distributions may describe the distribution of response time : the `lognormal` and the `gamma` distribution.<br>
# We will also display a `law power` distribution which also comes close to distribution, but has too much weight on zero to compete with the two first distributions.
# 
# <div class="alert-success" style="padding:15px;margin:10px">
#     Finally, according to the graphs, the <b>log-normal distribution</b> appears to be the best to describe the response time distribution compared to the gamma distribution, <i>especially because gamma distribution has too much weight close to zero.</i> 
#     
#    <br><br> However, the log-normal distribution is not close enough to the 2nd week data, because <b>this distribution may model poorly the extreme values</b> that we have highlighted previously. <b>We would have to add some weight</b> far from the mean, and then we will be able to model these extreme values.
#    
#    <br><br>As a conclusion, the distribution is a <b>lognormal distribution with some weight far from the mean.</b>
#     </div>

# <div class="alert-info" style="padding:15px">
# 3) Propose a potential cause for the change in response times. Give both a 
# qualitative answer as if you were explaining it to a client and a quantitative 
# answer as if you were explaining it to a statistician. Create 1 or 2 plots to 
# support and illustrate your argument</div>

# ## Simulation of the queue

# In[21]:


def queue_simulation(data):
    """
    Simulation of the queue 
    
    Returns :
    the history of the queue as a function of time in the form of a Dataframe.
    """
    print("Queue simulation, please wait ...(approx 45 sec )")
    last_t = data["time_sec"].iloc[0]
    #end = data["time_sec"].iloc[-1]

    queue = 0
    queue_history = []

    for t in tqdm(data.iterrows()):
        sec, response_time = t[1]["time_sec"],t[1]["response_time"]
        
        queue += response_time
        queue += last_t-sec
        queue = max(0,queue)
        
        queue_history.append((sec,queue))
        last_t = sec
        
    return pd.DataFrame(queue_history,columns=["time_sec","queue"])
    


# In[22]:


def plot_overflow(data,threshold):
    overflow = data[data["response_time"]>threshold]
    plt.plot(overflow["time_sec"],np.zeros(len(overflow)),"r+")


# In[23]:


df_queue = queue_simulation(data)


# In[24]:


plt.figure(figsize=(15,5))
plt.plot(df_queue["time_sec"],df_queue["queue"])
plt.title("Queue simulation")
plot_overflow(data,threshold=np.max(week1["response_time"]-1))
plt.show()

print("""
      We can imagine that the extremely high response time is due to an overloaded queue. Indeed, if the queue is overloaded, since there is only one thread, the incoming requests have to wait until the previous ones are treated. It is proved by the graph above : we have many red crosses (overflow in response time) when the queue is overloaded (blue peaks).
The blue peaks are huge, and match with the overflow, in other words they match with the extreme response time we experience in week 2. That could be a sign of a malfunction, an episodic slowdowns of the server.
The malfunction is often followed by a queue overload, according to the graph "queue simulation". That makes sense: after a slowdown, the requests are not treated. This also explain the massive peaks we have in the graph, as the requests are not treated for a while.

To the client we would say that there are two-phenomenons, and the one causes the second. The server may have a malfunction, and may encounter slowdowns that would explain the extreme reponse time we experience. Because of that, the one-thread server remains overloaded by all the pending requests, and the queue accumulates lateness.

To the statistician we would explain the same thing, but we will go a little further. It was not normal that distribution of the time requests had too much weight this far from the mean. We should have realized that something was not going well with the server.

      """)
# <div class="alert-success" style="padding:15px;margin:10px">
#     We can imagine that the extremely high response time is due to an overloaded queue. Indeed, if the queue is overloaded, since there is only one thread, the incoming requests have to wait until the previous ones are treated. It is proved by the graph above : we have many red crosses (overflow in response time) when the queue is overloaded (blue peaks). <br>
#     The blue peaks are huge, and match with the overflow, in other words they match with the extreme response time we experience in week 2. <b>That could be a sign of a malfunction, an episodic slowdowns of the server.</b><br>
#     The malfunction is often followed by a queue overload, according to the graph "queue simulation". That makes sense: after a slowdown, the requests are not treated. This also explain the massive peaks we have in the graph, as the requests are not treated for a while.
# <br><br>
#     <u>To the client</u> we would say that there are two-phenomenons, and the one causes the second. The server may have a malfunction, <b>and may encounter slowdowns</b> that would explain the extreme reponse time we experience. Because of that, the one-thread server remains overloaded by all the pending requests, and <b>the queue accumulates lateness.</b><br><br>
#     <u>To the statistician</u> we would explain the same thing, but we will go a little further. It was not normal that distribution of the time requests had too much weight this far from the mean. We should have realized that something was not going well with the server.
#     <br><br>
#     
# 
# </div>
#     
input("To continue to scenario 2, press any key")
print("###################################")
      
print("Scenario 2\n\n")

# ## Load Data : scenario 2

# In[25]:


#First, erase the previous data to free some memory
try:
    del data, week1, week2, week1_time_per_request, week2_time_per_request, df_queue
except:
    print("previous data already deleted")

data2 = pd.read_csv('scenario2.txt', sep="\t", header=None)
data2.columns = ["time_sec","requests","response_time"]

week1_2 = data2[data2['time_sec']<WEEK2]
week2_2 = data2[data2['time_sec']>=WEEK2]


# ## Quick visualization on data

# In[26]:


print("Data columns have proper type :",columns_have_proper_type(data2))


# In[27]:


week1_time_per_request2 = get_time_per_request(week1_2)
week2_time_per_request2 = get_time_per_request(week2_2)


# In[28]:


fig = plt.figure(figsize = (15,10))
plt.plot(week1_time_per_request2['time_sec'],week1_time_per_request2['response_time'],label="Week 1")
plt.plot(week2_time_per_request2['time_sec'],week2_time_per_request2['response_time'],label="Week 2", color = "green")
plt.title("Average time per request as a function of time")

threshold = np.max(week1_2['response_time']-1)
above_threshold = data2[data2['response_time']>threshold]
plt.plot(above_threshold['time_sec'],
         np.zeros(len(above_threshold)), "r+", label ="overflow on response time\nprojected on axis time")

# Separate the days
plt.axvline(x=(1*24+6)*3600, color = "black",label = "day separation")
for day in range(2,14):
    plt.axvline(x=(day*24+6)*3600, color = "black")

plt.legend()
plt.show()


# In[29]:


##### Mean of response time per request 
print("Mean of response time per request - week 1 :",np.mean(week1_time_per_request2["response_time"]))
print("Mean of response time per request - week 2 :",np.mean(week2_time_per_request2["response_time"]))

quantile = week2_time_per_request2["response_time"].quantile(.99)      
print("Mean of response time per request | quantile at 99% removed - week 2 :",
      np.mean(week2_time_per_request2[week2_time_per_request2['response_time']<quantile]["response_time"]))

#### Mean of number of requests 

print("\n")
print("Mean of number of requests - week 1 :",np.mean(week1_2["requests"]))
print("Mean of number of requests - week 2 :",np.mean(week2_2["requests"]))

quantile = week2_2["requests"].quantile(.99)
print("Mean of number of requests | quantile at 99% removed - week 2 :",
      np.mean(week2_2[week2_2['requests']<quantile]["requests"]))

quantile = week2_2["requests"].quantile(.95)
print("Mean of response time per request | quantile at 95% removed - week 2 :",
      np.mean(week2_2[week2_2['requests']<quantile]["requests"]))

##### Mean of total response time


print("\n")
print("Mean of total response time - week 1 :",np.mean(week1_2["response_time"]))
print("Mean of total response time - week 2 :",np.mean(week2_2["response_time"]))

quantile = week2_2["response_time"].quantile(.99)
print("Mean of response time per request | quantile at 99% removed - week 2 :",
      np.mean(week2_2[week2_2['response_time']<quantile]["response_time"]))

quantile = week2_2["response_time"].quantile(.95)
print("Mean of response time per request | quantile at 95% removed - week 2 :",
      np.mean(week2_2[week2_2['response_time']<quantile]["response_time"]))


print("""
The mean of the number of request does not change between the two weeks. 
However, both total response time and response time per request are higher in the 2nd week.
""")
# <div class=alert-success style="padding:15px">
# The mean of the number of request does not change between the two weeks. <br>However, both total response time and response time per request are higher in the 2nd week.
# <div>

# ## Response time distribution

# In[30]:


law_comparison(week1_time_per_request2,"Data Week 1",[i*0.001 for i in range(100)])
law_comparison(week2_time_per_request2,"Data Week 2",[i*0.001 for i in range(100)])


# In[31]:


print('There is some extreme data : we use a quantile to remove them and to be able to find a matching distribution')
quantile = week2_time_per_request2["response_time"].quantile(0.99)
print(quantile)
law_comparison(week2_time_per_request2[week2_time_per_request2['response_time']<quantile],
               "Data per request Week 2 without quantile", bins = [i*0.001 for i in range(int(quantile//0.001))])


print("""
      Finally, according to the graphs, the log-normal distribution appears to be the best to describe the response time distribution compared to the gamma distribution, especially because gamma distribution has too much weight close to zero.

Unlike the first scenario, a simple lognormal distribution is enough to describe the distribution. We simply notice that the distribution without quantile is closer to the lognormal, because there may be some long-processing requests for example, and because the lognormal distribution does not describe such phenomenon very well.

As a conclusion, the distribution is close to a lognormal distribution.
      """)
# <div class="alert-success" style="padding:15px;margin:10px">
#     Finally, according to the graphs, the <b>log-normal distribution</b> appears to be the best to describe the response time distribution compared to the gamma distribution, <i>especially because gamma distribution has too much weight close to zero.</i> 
#     
#    <br><br> Unlike the first scenario, a simple lognormal distribution is enough to describe the distribution. We simply notice that the distribution without quantile is closer to the lognormal, because there may be some long-processing requests for example, and because the lognormal distribution does not describe such phenomenon very well.
#    
#    <br><br>As a conclusion, the distribution is close to a <b>lognormal distribution</b>
#     </div>

# ## Simulation of the queue

# In[32]:


df_queue2 = queue_simulation(data2)


# In[33]:


plt.figure(figsize=(15,5))
plt.plot(df_queue2["time_sec"],df_queue2["queue"])
plt.title("Queue simulation")
plot_overflow(data2,threshold=np.max(week1_2["response_time"]-1))
plt.show()

print("""
The simulation the queue is very useful to understand what happens in the 2nd scenario. We have a very strong correlation between the queue overflow (the blue peaks) and the server time response above overflow (red crosses).

To the client we would just say that the one-thread server is sometimes overloaded, and the queue accumulates lateness.

To the statistician we would explain the same thing. Moreover, we would support this explanation by saying the following : the mean of the number of requests is the same between week 1 and 2, however the mean that changes is the one on the times (total response time and time/request). If during the 2nd week we have periods in which there are many requests, and periods with few requests, the mean of numbers of requests remains the same, however during "busy" periods we assume the queue will be overloaded. That would explain that the mean of the requests does not change, but the temporal means do.

Another theory would be to say the requests become more complex, but that is not sure, and we have a clear evidence that a high server time response is strongly correlated to queue overflow in the graph showing the queue simulation. The two phenomenons might occur at the same time, but we may need more data to investigate about this. 
""")
# <div class=alert-success style="padding:15px">
# The simulation the queue is very useful to understand what happens in the 2nd scenario. We have a very strong correlation between the queue overflow (the blue peaks) and the server time response above overflow (red crosses). <br><br>
#     <u>To the client</u> we would just say that the one-thread server is sometimes overloaded, and <b>the queue accumulates lateness.</b><br><br>
#     <u>To the statistician</u> we would explain the same thing. Moreover, we would support this explanation by saying the following : the mean of the number of requests is the same between week 1 and 2, however the mean that changes is the one on the times (total response time and time/request). If during the 2nd week we have periods in which there are many requests, and periods with few requests, the mean of numbers of requests remains the same, however during "busy" periods we assume the queue will be overloaded. That would explain that the mean of the requests does not change, but the temporal means do.
#     <br><br>
#     Another theory would be to say the requests become more complex, but that is not sure, and we have a clear evidence that a high server time response is strongly correlated to queue overflow in the graph showing the queue simulation. The two phenomenons might occur at the same time, but we may need more data to investigate about this.
# </div>
      
input("Press any key to exit")
