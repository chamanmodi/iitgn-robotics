#!/usr/bin/env python
# coding: utf-8

# ## **ME 639: Introduction to Robotics | Assignment - 5**
# ### Chaman Modi | 20310015

# ### Task 1 Control tutorial reviewed

# In[1]:


# Importing libraries and creating basic rotation functions
import numpy as np
import scipy as sci
import sympy as sp
sp.init_printing()
import matplotlib.pyplot as plt
import math


# ### Task 2 Stanford dynamics

# In[2]:


# Creating the joint variables and arm parameters (symbolic)
t = sp.Symbol('t')
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
d = sp.Function('d')(t)

l1, l2, m1, m2, m3 = sp.symbols("l1, l2, m1, m2, m3")
I1 = 0
I2 = m2*(l2**2)/12

# Jacobian of  the points at the center  of links i.e. c1,c2,c3
Jvc1 = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]])

Jvc2 = np.array([[-0.5*l2*sp.sin(q1)*sp.cos(q2), -0.5*l2*sp.cos(q1)*sp.sin(q2), 0],
                  [0.5*l2*sp.cos(q1)*sp.cos(q2), -0.5*l2*sp.sin(q1)*sp.sin(q2), 0],
                  [0, 0.5*l2*sp.cos(q2), 0]])

Jvc3 = np.array([[-(0.5*d+l2)*sp.sin(q1)*sp.cos(q2), -(0.5*d+l2)*sp.cos(q1)*sp.sin(q2), 0.5*sp.cos(q1)*sp.cos(q2)],
                  [(0.5*d+l2)*sp.cos(q1)*sp.cos(q2), -(d+l2)*sp.sin(q1)*sp.sin(q2), 0.5*sp.sin(q1)*sp.cos(q2)],
                  [0, (l2+0.5*d)*sp.cos(q2), 0.5*sp.sin(q2)]])


# Creating Dq matrix
Dq = m1*np.transpose(Jvc1)@Jvc1  +  m2*np.transpose(Jvc2)@Jvc2  +  m3*np.transpose(Jvc3)@Jvc3 + np.array([[I1+I2, 0, 0],[0, I2, 0],[0, 0, 0]])
print('D(q) = ')
sp.simplify(Dq)


# In[3]:


# Vector containing joint variables
q = sp.Array([q1,q2,d])

# Joint velocities (q_dot)
q_dot = sp.diff(q,t)

# Joint accelaration (q_dotdot)
q_dotdot = sp.diff(q_dot,t)

# Potential energy expression
g = 9.81
V = m1*g*l1/2 + m2*g*l1 + m3*g*l1

# Intializing list for christoffel symbols 
c = [[[0]*3]*3]*3

# Calculating christoffel symbols
for k in range(0,3):
    for i in range(0,3):
        for j in range(0,3):
            c[i][j][k] = 0.5*(sp.diff(Dq[k][j],q[i]) + sp.diff(Dq[k][i],q[j]) - sp.diff(Dq[i][j],q[k]))

phi = sp.zeros(3,1)
tau = sp.zeros(3,1)

# Finally creating the torque vector
for k in range(3):
    phi[k] = sp.diff(V, q[k])
    d_temp = 0
    ct = 0
    for j in range(3):
        d_temp = d_temp + Dq[k][j] * q_dotdot[j] 
        for i in range(3):
            ct = ct + c[i][j][k] * q_dot[i] * q_dotdot[j]
    tau[k] = d_temp + ct + phi[k]

print('tau = ')
sp.simplify(tau)


# ### Task 3 SCARA dynamics

# In[4]:


# Creating the joint variables and arm parameters (symbolic)
t = sp.Symbol('t')
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
d = sp.Function('d')(t)

l1, l2, m1, m2, m3 = sp.symbols("l1, l2, m1, m2, m3")
I1 = m1*(l1**2)/12
I2 = m2*(l2**2)/12

# Jacobian of  the points at the center  of links i.e. c1,c2,c3
Jvc1 = np.array([[-0.5*l1*sp.sin(q1), 0, 0],
                 [0.5*l1*sp.cos(q1), 0, 0],
                 [0, 0, 0]])

Jvc2 = np.array([[-l1*sp.sin(q1)-0.5*l2*sp.sin(q1+q2), -0.5*l2*sp.sin(q1+q2), 0],
                 [l1*sp.cos(q1)+0.5*l2*sp.cos(q1+q2), 0.5*l2*sp.cos(q1+q2), 0],
                 [0, 0, 0]])

Jvc3 = np.array([[-l1*sp.sin(q1)-l2*sp.sin(q1+q2), l2*sp.sin(q1+q2), 0],
                 [l1*sp.cos(q1)+l2*sp.cos(q1+q2), l2*sp.cos(q1+q2), 0],
                 [0, 0, -0.5]])

# Creating Dq matrix
Dq = m1*np.transpose(Jvc1)@Jvc1  +  m2*np.transpose(Jvc2)@Jvc2  +  m3*np.transpose(Jvc3)@Jvc3 + np.array([[I1+I2, I2, 0],[I2, I2, 0],[0, 0, 0]])
print('D(q) = ')
sp.simplify(Dq)


# In[5]:


# Vector containing joint variables
q = np.array([q1,q2,d])

# Joint velocities (q_dot)
q_dot = sp.diff(q,t)

# Joint accelaration (q_dotdot)
q_dotdot = sp.diff(q_dot,t)

# Potential energy expression
V = m3*9.81*(-d/2)

# Intializing list for christoffel symbols 
c = [[[0]*3]*3]*3

# Calculating christoffel symbols
for k in range(0,3):
    for i in range(0,3):
        for j in range(0,3):
            c[i][j][k] = 0.5*(sp.diff(Dq[k][j],q[i]) + sp.diff(Dq[k][i],q[j]) - sp.diff(Dq[i][j],q[k]))

phi = sp.zeros(3,1)
tau = sp.zeros(3,1)

# Finally creating the torque vector
for k in range(3):
    phi[k] = sp.diff(V, q[k])
    d_temp = 0
    ct = 0
    for j in range(3):
        d_temp = d_temp + Dq[k][j] * q_dotdot[j] 
        for i in range(3):
            ct = ct + c[i][j][k] * q_dot[i] * q_dotdot[j]
    tau[k] = d_temp + ct + phi[k]

print('tau = ')
sp.simplify(tau)


# ### Task 4 PUMA dynamics

# In[6]:


# Creating the joint variables and arm parameters (symbolic)
t = sp.Symbol('t')
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)
q3 = sp.Function('q3')(t)

l1, l2, l3, m1, m2, m3 = sp.symbols("l1, l2, l3, m1, m2, m3")
I1 = 0
I2 = m2*(l2**2)/12
I3 = m3*(l3**2)/12

# Jacobian of  the points at the center  of links i.e. c1,c2,c3
Jvc1 = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]])

Jvc2 = np.array([[-0.5*l2*sp.sin(q1)*sp.cos(q2), -0.5*l2*sp.cos(q1)*sp.sin(q2), 0],
                 [0.5*l2*sp.cos(q1)*sp.cos(q2), -0.5*l2*sp.sin(q1)*sp.sin(q2), 0],
                 [0, 0.5*l2*sp.cos(q2), 0]])

Jvc3 = np.array([[-(0.5*l3*sp.cos(q3)+l2)*sp.sin(q1)*sp.cos(q2), -(0.5*l3*sp.cos(q3)+l2)*sp.cos(q1)*sp.sin(q2), 0.5*l3*sp.cos(q1)*sp.cos(q2)*sp.sin(q3)],
                 [(0.5*l3*sp.cos(q3)+l2)*sp.cos(q1)*sp.cos(q2), -(0.5*l3*sp.cos(q3)+l2)*sp.sin(q1)*sp.sin(q2), 0.5*l3*sp.sin(q1)*sp.cos(q2)*sp.sin(q3)],
                 [0, l2*sp.cos(q2), 0.5*l3*sp.cos(q2)]])

# Creating Dq matrix
Dq = m1*np.transpose(Jvc1)@Jvc1  +  m2*np.transpose(Jvc2)@Jvc2  +  m3*np.transpose(Jvc3)@Jvc3 + np.array([[I1+I2+I3, 0, 0],[0, I2+I3, I3],[0, I3, I3]])
print('D(q) = ')
sp.simplify(Dq)


# In[7]:


# Vector containing joint variables
q = np.array([q1,q2,q3])

# Joint velocities (q_dot)
q_dot = sp.diff(q,t)

# Joint accelaration (q_dotdot)
q_dotdot = sp.diff(q_dot,t)

# Potential energy expression
g = 9.81
V = m1*g*l1/2 + m2*g*(l1+0.5*l2*sp.sin(q2)) + m3*g*(l1+l2*sp.sin(q2)+0.5*l3*sp.sin(q3))

# Intializing list for christoffel symbols 
c = [[[0]*3]*3]*3

# Calculating christoffel symbols
for k in range(0,3):
    for i in range(0,3):
        for j in range(0,3):
            c[i][j][k] = 0.5*(sp.diff(Dq[k][j],q[i]) + sp.diff(Dq[k][i],q[j]) - sp.diff(Dq[i][j],q[k]))

phi = sp.zeros(3,1)
tau = sp.zeros(3,1)

# Finally creating the torque vector
for k in range(3):
    phi[k] = sp.diff(V, q[k])
    d_temp = 0
    ct = 0
    for j in range(3):
        d_temp = d_temp + Dq[k][j] * q_dotdot[j] 
        for i in range(3):
            ct = ct + c[i][j][k] * q_dot[i] * q_dotdot[j]
    tau[k] = d_temp + ct + phi[k]

print('tau = ')
sp.simplify(tau)

