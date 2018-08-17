# Homework 1
## EE746 Neuromorphic Engineering
### Autumn 2018

Varshith Sreeramdass
150050084

Vikash Kumar Meena
150110050

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

The code stores the output in the matrix `V`.
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

$U_{t}-U_{t-1}=\Delta t * (k_{1}+2k_{2}+2k_{3}+k_{4})/6$  
$V_{t}-V_{t-1}=\Delta t * (l_{1}+2l_{2}+2l_{3}+l_{4})/6$  

where; 

$k1 = f(U_{t-1}, V_{t-1})$  
$l1 = g(U_{t-1}, V_{t-1}, I_{t-1})$  

$k2 = f(U_{t-1} + \Delta t * k1/2, V_{t-1} + \Delta t * l1/2)$  
$l2 = g(U_{t-1} + \Delta t * k1/2, V_{t-1} + \Delta t * l1/2, I_{t-0.5})$  

$k3 = f(U_{t-1} + \Delta t * k2/2, V_{t-1} + \Delta t * l2/2)$  
$l3 = g(U_{t-1} + \Delta t * k2/2, V_{t-1} + \Delta t * l2/2, I_{t-0.5})$  

$k4 = f(U_{t-1} + \Delta t * k3, V_{t-1} + \Delta t * l3)$  
$l4 = g(U_{t-1} + \Delta t * k3, V_{t-1} + \Delta t * l3, I_{t})$  

and;

$f(u, v) = a ( b (v - E_{t}) - u )$  
$g(u, v, i) = k_{z} ( v - E_{r} ) (v - E_{t}) - u + i$  

Note: Increment of one in the above timescale implies an increment of $\Delta t$ in actual time.

c.  

The code is contained in the file `p2.py`. The iterative solution is implemented in the function `runge_kutta_4_sim_and_reset`. The code produces plots into the files `Q2.*.png`.  

![src](Q2.0.png){width=75%}

![src](Q2.1.png){width=75%}

![src](Q2.2.png){width=75%}

![src](Q2.3.png){width=75%}

![src](Q2.4.png){width=75%}

![src](Q2.5.png){width=75%}

![src](Q2.6.png){width=75%}

![src](Q2.7.png){width=75%}

![src](Q2.8.png){width=75%}

#### Q3

a.

The difference equations of the simultaneous equations are:  
<!-- 
k, l = f_and_g(U[mn, i-1], V[mn, i-1], i-1, model)
U[mn, i] = U[mn, i-1] + delta*k
V[mn, i] = V[mn, i-1] + delta*l
 -->

$U_{t}-U_{t-1}=\Delta t * k$  
$V_{t}-V_{t-1}=\Delta t * l$

where;

$k1 = f(U_{t-1}, V_{t-1})$  
$l1 = g(U_{t-1}, V_{t-1}, I_{t-1})$  

and;

$f(u, v) = a (v - E_{t}) - u$  
$g(u, v, i) = -g_{L}(v-E_{L}) + g_{L} \Delta_{T}exp(\frac{v-V_{T}}{\Delta_{T}}) - u + i$  

Note: Increment of one in the above timescale implies an increment of $\Delta t$ in actual time.

b.  

The values of U, V for different types of neurons are:  

type 		RS			IB			CH
--			--			--			--
U(SV)		5.09e-11	4.79e-11	2.38e-11
V(mV)		-44.5		-46.0		-46.1
--			--			--			--

c.  

The code is contained in the file `p3.py`. The iterative solution is implemented in the function `euler_sim_and_reset`. The code produces plots into the files `Q3.*.png`.  

![src](Q3.0.png){width=75%}

![src](Q3.1.png){width=75%}

![src](Q3.2.png){width=75%}

![src](Q3.3.png){width=75%}

![src](Q3.4.png){width=75%}

![src](Q3.5.png){width=75%}

![src](Q3.6.png){width=75%}

![src](Q3.7.png){width=75%}

![src](Q3.8.png){width=75%}


#### Q4

a.

The membrane potential is plot in the file `Q4.1.png` while the ion currents are plot in `Q4.2.png`.  

Membrane Potential
![Membrane Potential](Q4.1.png ){width=75%}

Ion Currents
![Ion Currents](Q4.2.png ){width=75%}

The code in conatined in the file `p4.py` and the function that implements the iteration is `euler_sim`.  

b.

The power dissipated per unit area is plot in the file `Q4.3.png`.

Power Dissipated
![Power Dissipated](Q4.3.png ){width=75%}

c.  

The dissipated by the ion currents and the energy spent to charge/discharge the membrane are as follows.

channel			Energy(J)
--				--		
Na				3.61278438e-10
K				1.06092303e-10
leak			3.90637506e-11
Membr.			3.8193537e-14
--				--		
