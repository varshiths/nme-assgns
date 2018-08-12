# Homework 1
## EE746 Neuromorphic Engineering
### Autumn 2018

Note:  
To run any of files `p1.py`, `p2.py`, `p3.py`, `p4.py`:  
```
$ python3 <file>
``` 

#### Q1  

a.  

The expression for steady state value of the membrane potential on constant current $I_{o}$ is:  
$V_{ss}=\frac{I_{o}}{g_{L}} + E_{L}$  

The minimum value of steady state current to issue a spike is:  
$I_{c}=g_{L}(V_{T}-E_{L})$  

For the given values,  
$I_{c}=2.7nA$

b.  

Refer to the file `p1.py` for code.  
The function that computes the potential is `runge_kutta_2_and_reset`.  

c.  

The code stores the output in the matrix `V`. The data is output into a file `Q1.output.dat` in csv format.
The plots are contained in the files `Q1.n2.png`, `Q1.n4.png`, `Q1.n6.png`, `Q1.n8.png` produced by the program.

![Neuron 2](Q1.n2.png "N2"){width=50%}

![Neuron 4](Q1.n4.png "N4"){width=50%}

![Neuron 6](Q1.n6.png "N6"){width=50%}

![Neuron 8](Q1.n8.png "N8"){width=50%}


d.  

The code in `p1.py` also produces `Q1.avg.png`. This file has the variation of average time period with current.  

![Average Time Period](Q1.avg.png "Avg Time Period"){width=50%}

#### Q2

a.  

The expression for the steady state values for U, V are:  
$V_{ss}=E_{t}-\frac{b}{k_{z}}$  
$U_{ss}=b(E_{t}-E_{r})+\frac{b^{2}}{k_{z}}$  

Their values for different types of neurons are:  

type 		RS			IB			CH
--			--			--			--
U(SV)		-3.42e-11	1.70e-10	2.06e-11
V(mV)		-42.85		-40.83		-39.33
--			--			--			--

b.  

The difference equations of the simultaneous equations are:  
$hey$

c.  

The code is contained in the file `p2.py`. The iterative solution is implemented in the function `runge_kutta_4_sim_and_reset`. The code produces three plots into the files `Q2.0.png`, `Q2.3.png`, `Q2.6.png`  

Neuron RS
![Neuron RS](Q2.0.png "RS"){width=75%}

Neuron IB
![Neuron IB](Q2.3.png "IB"){width=75%}

Neuron CH
![Neuron CH](Q2.6.png "CH"){width=75%}


#### Q3

a.

The difference equations of the simultaneous equations are:  
$hey$

b.  

The values of U, V for different types of neurons are:  

type 		RS			IB			CH
--			--			--			--
U(SV)		-1.80e-10	1.08e-10	-6e-11
V(mV)		-160		-85			-88
--			--			--			--
*** check these values ***

c.  

The code is contained in the file `p3.py`. The iterative solution is implemented in the function `euler_sim_and_reset`. The code produces three plots into the files `Q3.0.png`, `Q3.3.png`, `Q3.6.png`  

Neuron RS
![Neuron RS](Q3.0.png "RS"){width=75%}

Neuron IB
![Neuron IB](Q3.3.png "IB"){width=75%}

Neuron CH
![Neuron CH](Q3.6.png "CH"){width=75%}
