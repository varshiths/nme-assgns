# Homework 3
## EE746 Neuromorphic Engineering
### Autumn 2018

Varshith Sreeramdass
150050084

Note:  
To run any of files `p1.py`, `p2.py`, `p3.py`, `p4.py`, `p5.py`:  
```
$ python3 <file>
``` 
The seed here is set to 3. Change the seed to observe effect across random initializations.  

#### Q1

a.  

The connectivity is stored in a matrix sized NxN in the variable named `weights` in the file `p1.py`.  

b.  

Case 1:  
Neuron a:  
![Response](P1.1.0.png ){width=75%}  

Neuron b:  
![Response](P1.1.1.png ){width=75%}  

Neuron c:  
![Response](P1.1.2.png ){width=75%}  

Neuron d:  
![Response](P1.1.3.png ){width=75%}  

Neuron e:  
![Response](P1.1.4.png ){width=75%}  

Case 2:  
Neuron a:  
![Response](P1.2.0.png ){width=75%}  

Neuron b:  
![Response](P1.2.1.png ){width=75%}  

Neuron c:  
![Response](P1.2.2.png ){width=75%}  

Neuron d:  
![Response](P1.2.3.png ){width=75%}  

Neuron e:  
![Response](P1.2.4.png ){width=75%}  


#### Q2

a.  

Raster Plot:  
![Response](P2.0.png ){width=75%}  

b.  

Spikes by excitatory neurons:  
![Response](P2.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P2.2.png ){width=75%}  

c.  

The network displays an oscillation with the excitatory and inhibitory neurons oscillating aternatively.  
This seems to arise out of the fact that the system is attempting to enter into an equilibrium.

It is when the excitatory neurons spike, that the entire system tends to spike. This is seen as a vertical bar in the raster plot.  
When the inhibitory neurons spike, the entire system tends to depress. This is seen as the empty space between two bars in the raster plot, correlated with the peaks in the $R_{i}(t)$ plot.  

#### Q3

a.  

Network with 200 Neurons:  
For $w_{e} = -w_{i} = 3000$:  

Raster Plot:  
![Response](P3_3000_1.0.png ){width=75%}  

Spikes by excitatory neurons:  
![Response](P3_3000_1.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P3_3000_1.2.png ){width=75%}  

b.  

As the weight changes, the total number of spikes does change, but this does not make the network behave as in the larger network, as both the excitatory and inhibitory spikes increase/decrease by the same amount.  

For $w_{e} = -w_{i} = 1000$:  
Raster Plot:  
![Response](P3_1000_1.0.png ){width=75%}  

Spikes by excitatory neurons:  
![Response](P3_1000_1.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P3_1000_1.2.png ){width=75%}  

For $w_{e} = -w_{i} = 4000$ the number of spikes are even more and the oscillatory trend does not occur.  

c.  

The net inhibition in the network must be increased in order to observe oscillatory behaviour. This is evident from the raster plots.  

d.  

Since the overall inhibition in the network must increase, the $w_{i}$ is increased in magnitude.  

For $w_{e} = - \gamma w_{i} = 1500$ where $\gamma = 0.6$:  
Raster Plot:  
![Response](P3_1500_0_6.0.png ){width=75%}  

Spikes by excitatory neurons:  
![Response](P3_1500_0_6.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P3_1500_0_6.2.png ){width=75%}  

#### Q4

a.  

Implementation in the file `p4.py`.  

b.  

The raster plot, and spike rates of excitatory and inhibitory neurons in a learning setup for configuration in 3(a) are in the files `P4.0.png`, `P4.1.png` and `P4.2.png` respectively.  

Raster Plot:  
![Response](P4.0.png ){width=75%}  

Spikes by excitatory neurons:  
![Response](P4.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P4.2.png ){width=75%}  

Variation in the average excitatory synaptic strength:  
![Response](P4.3.png ){width=75%}  

#### Q5

a.  

The additional rules considered are:  
1. Upstream excitatory synapses exhibit anti-STDP  
2. Downstream inhibitory synapses exhibit STDP  

Raster Plot:  
![Response](P5_1000.0.png ){width=75%}  
Spikes by excitatory neurons:  
![Response](P5_1000.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P5_1000.2.png ){width=75%}  

Variation in the average excitatory synaptic strength:  
![Response](P5_1000.3.png ){width=75%}  

b.  

Raster Plot:  
![Response](P5_2000.0.png ){width=75%}  

c.  

Spikes by excitatory neurons:  
![Response](P5_2000.1.png ){width=75%}  
Spikes by inhibitory neurons:  
![Response](P5_2000.2.png ){width=75%}  

Variation in the average excitatory synaptic strength:  
![Response](P5_2000.3.png ){width=75%}  
