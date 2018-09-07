# Homework 2
## EE746 Neuromorphic Engineering
### Autumn 2018

Varshith Sreeramdass
150050084

Vikash Kumar Meena
150110050

Note:  
To run any of files `p1.py`, `p2.py`, `p3.py`, `p4.py`, `p5.py`:  
```
$ python3 <file>
``` 

#### Q1

The seed here is set to 3. Change the seed to observe effect across random initializations.  

a.  

The time instants when the stimuli occur are:  
0.0218 0.2221 0.2429 0.2473 0.2542 0.404  

b.  

The response for second setting is in `Q1.b.png`.

![Response](Q1.b.png ){width=75%}

Number of spikes are:  
11  


#### Q2

The seed here is set to 0. Change the seed to observe effect across random initializations.  

a.  

The response for first setting is in `Q2.a.png`.

![Response](Q2.a.png ){width=75%}

Number of spikes are:  
0  

b.  

The response for second setting is in `Q2.b.png`.

![Response](Q2.b.png ){width=75%}

Number of spikes are:  
11  

#### Q3

The seed here is set to 2. Change the seed to observe effect across random initializations.  
a.  

The weights for the tasks are contained in the variable `wsn` at the end of the code block.  

The response after causing spikes is in `Q3.a.png`.

![Response](Q3.a.png ){width=75%}

Number of iterations are:  
4  

b.  

The scatter plot for $\delta w_{k}, \delta t_{k}$ across synapses and iterations is available in `Q3.b.png`  

![Scatter Plot](Q3.b.png ){width=75%}


#### Q4

The seed here is set to 2. Change the seed to observe effect across random initializations.  

a.  

The weights for the tasks are contained in the variable `wsn` at the end of the code block.  

The response after killing spikes is in `Q4.a.png`.

![Response](Q4.a.png ){width=75%}

Average number of iterations are:  
2.5  

The above result are after running for the seeds:  
2, 0, 100, 20, 25, 30  
2.5 is average of 3, 2, 3, 3, 2, 2  


b.  

The scatter plot for $\delta w_{k}, \delta t_{k}$ across synapses and iterations is available in `Q4.b.png`  

![Scatter Plot](Q4.b.png ){width=75%}


#### Q5

The weights for any of the tasks are contained in the variable `wsn` at the end of the code block for that task.  
The seed here is set. Change the seed to observe effect across random initializations.  

a.

The response for S1 is in `Q5.a.1.png`.  
The response for S2 is in `Q5.a.2.png`.  

S1 Response
![S1 Response](Q5.a.1.png ){width=75%}

S2 Response
![S2 Response](Q5.a.2.png ){width=75%}

b.

The response for S2 with changed weights after removing the spikes is in the file `Q5.b.2.png`  

S2 Response after train
![S2 Response after train](Q5.b.2.png ){width=75%}

c.  

The response for S1 and S2 after causing spikes for S1 and removing spikes for S2 are as follows  

S1 Response after train
![S1 Response after train](Q5.c.1.png ){width=75%}

S2 Response after train
![S2 Response after train](Q5.c.2.png ){width=75%}

d.  

The response for S1 and S2 after removing spikes for S1 and causing spikes for S2 are as follows  

S1 Response after train
![S1 Response after train](Q5.d.1.png ){width=75%}

S2 Response after train
![S2 Response after train](Q5.d.2.png ){width=75%}

