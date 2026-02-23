## II. SYSTEM MODEL

We consider an integrated **LEO satellite–terrestrial mobile edge computing (MEC)** system designed to support **Metaverse Play-to-Earn (P2E)** services. The system consists of one LEO satellite equipped with an edge server, one terrestrial base station (BS) with MEC capability, and a set $\mathcal U = \{u_1, u_2, \dots, u_N\}$ of $N$ Metaverse ground users (MGUs). In addition to serving MGUs, the LEO satellite simultaneously performs a **remote sensing image compression task** and transmits compressed images to a ground gateway.

---

### A. Offloading and Channel Model

The LEO satellite and the BS are equipped with $\nu^{\text{sat}}$ and $\nu^{\text{bs}}$ orthogonal OFDMA channels, respectively. Let
- [x] Channel sets definition
$
\mathcal C = \{c_1, \dots, c_{\nu^{\text{sat}}}\}, \qquad \hat{\mathcal C} = \{\hat c_1, \dots, \hat c_{\nu^{\text{bs}}}\}
$
denote the channel sets of the LEO satellite and the BS. The bandwidth of each satellite channel $c_q$ is $B^{\text{sat}}$, while that of each BS channel $\hat c_j$ is $B^{\text{bs}}$.

Each MGU $u_k$ chooses one offloading decision
- [x] Offloading decision set
$
a_k \in \{(1,q),(2,j)\},
$
where $(1,q)$ indicates offloading to the LEO satellite via channel $c_q$, and $(2,j)$ indicates offloading to the BS via channel $\hat c_j$. The joint offloading strategy is denoted by $\mathbf A = \{a_1, \dots, a_N\}$.

The set of MGUs associated with channel $c_q$ at the LEO satellite is
- [x] Satellite channel association set
$
\mathcal U^{\text{sat}}_q = \{u_k \in \mathcal U \mid a_k = (1,q)\},
$
and the set of MGUs associated with channel $\hat c_j$ at the BS is
- [x] BS channel association set
$
\mathcal U^{\text{bs}}_j = \{u_k \in \mathcal U \mid a_k = (2,j)\}.
$

---

### B. LEO Satellite Communication Geometry

Let $\zeta_k$ denote the elevation angle between MGU $u_k$ and the LEO satellite. Denote by $R_e$ the Earth radius and by $H_e$ the satellite orbit altitude. The distance between $u_k$ and the LEO satellite is
- [x] Satellite distance
$
d^{\text{sat}}_k = \sqrt{R_e^2 + (R_e+H_e)^2 - 2R_e(R_e+H_e)\cos\xi_k},
$
where
- [x] Angle $\xi_k$
$
\xi_k = \arccos
\left(
\frac{R_e}{R_e+H_e}\cos\zeta_k
\right) - \zeta_k.
$

The maximum arc length during which the satellite can communicate with $u_k$ is
- [x] Maximum arc length
$
G_k = 2(R_e+H_e)\xi_k.
$
Let $\omega = \sqrt{\mu/(R_e+H_e)}$ denote the orbital velocity, where $\mu$ is the Earth’s gravitational constant. The maximum available communication time is
- [x] Maximum communication time
$
T_k^{\max} = \frac{G_k}{\omega}.
$

---

### C. Metaverse Task Model

Each MGU generates a Metaverse computation task consisting of uplink and downlink data. The uplink data size $D_k^{\text{up}}$ is fixed, representing tracking and posture information. The downlink data size depends on the video resolution $r_k$:
- [x] Resolution bounds
$
r_{\min} \le r_k \le r_{\max}.
$
Each Metaverse frame consists of two images (one per eye), with 24 bits per pixel. With compression ratio $\chi_k$, the downlink data size is
- [x] Downlink data size
$
D_k^{\text{down}} = \frac{24 \times 2 \times r_k}{\chi_k}.
$

---

### D. Offloading to the LEO Satellite

#### 1) Communication Rates

The channel gain between MGU $u_k$ and the LEO satellite on channel $c_q$ is
- [x] Satellite channel gain
$
q^{\text{sat}}_{k,q} = L_{k,q} G^{\text{sat}}_q G^{\text{user}}_{k,q} |\mu_{k,q}|^2,
$
where
- [x] Satellite path loss
$
L_{k,q} = \left(\frac{c}{4\pi d^{\text{sat}}_k \Psi_q}\right)^2.
$

The uplink and downlink transmission rates are
- [x] Uplink rate to satellite
$
R^{\text{up,sat}}_{k,q} = \frac{B^{\text{sat}}}{|\mathcal U^{\text{sat}}_q|}
\log_2\left(1 + \frac{q^{\text{sat}}_{k,q} p_k}{B^{\text{sat}}\sigma^2/|\mathcal U^{\text{sat}}_q|}\right),
$
- [x] Downlink rate from satellite
$
R^{\text{down,sat}}_{k,q} = \frac{B^{\text{sat}}}{|\mathcal U^{\text{sat}}_q|}
\log_2\left(1 + \frac{q^{\text{sat}}_{k,q} p^{\text{sat}}}{B^{\text{sat}}\sigma^2/|\mathcal U^{\text{sat}}_q|}\right).
$

#### 2) Computation Model

Let $f^{\text{sat}}_q$ be the computation capacity allocated to channel $c_q$. Each associated MGU obtains
- [ ] Satellite compute allocation per user
$
v^{\text{sat}}_{k,q} = \frac{f^{\text{sat}}_q}{|\mathcal U^{\text{sat}}_q|}.
$

The computation latency and energy consumption are
- [x] Satellite computation latency
$
T^{\text{comp,sat}}_{k,q} = \frac{\tau^{\text{up}} D_k^{\text{up}} + \tau^{\text{down}} D_k^{\text{down}}}{C^{\text{sat}} v^{\text{sat}}_{k,q}},
$
- [x] Satellite computation energy
$
E^{\text{comp,sat}}_{k,q} = \kappa^{\text{sat}} (v^{\text{sat}}_{k,q})^2
(\tau^{\text{up}} D_k^{\text{up}} + \tau^{\text{down}} D_k^{\text{down}}) C^{\text{sat}}.
$

#### 3) Total Latency and Energy

The total latency experienced by (u_k) is
- [ ] Total satellite latency
$
T^{\text{sat}}_{k,q} =
\frac{D_k^{\text{up}}}{R^{\text{up,sat}}_{k,q}} +
\frac{D_k^{\text{down}}}{R^{\text{down,sat}}_{k,q}} +
T^{\text{comp,sat}}_{k,q} +
\frac{d^{\text{sat}}_k}{c}.
$

The total energy consumption is
- [ ] Total satellite energy
$
E^{\text{sat}}_{k,q} =
p_k \frac{D_k^{\text{up}}}{R^{\text{up,sat}}_{k,q}} +
E^{\text{comp,sat}}_{k,q} +
p^{\text{sat}} \frac{D_k^{\text{down}}}{R^{\text{down,sat}}_{k,q}}.
$

---

### E. Offloading to the BS

The uplink and downlink rates on BS channel $\hat c_j$ are
- [x] Uplink rate to BS
$
R^{\text{up,bs}}_{k,j} = \frac{B^{\text{bs}}}{|\mathcal U^{\text{bs}}_j|}
\log_2\left(1 + \frac{q^{\text{bs}}_k p_k}{B^{\text{bs}}\sigma^2/|\mathcal U^{\text{bs}}_j|}\right),
$
- [x] Downlink rate from BS
$
R^{\text{down,bs}}_{k,j} = \frac{B^{\text{bs}}}{|\mathcal U^{\text{bs}}_j|}
\log_2\left(1 + \frac{q^{\text{bs}}_k p^{\text{bs}}}{B^{\text{bs}}\sigma^2/|\mathcal U^{\text{bs}}_j|}\right).
$

Each MGU obtains computation resource
- [ ] BS compute allocation per user
$
\hat v^{\text{bs}}_{k,j} = \frac{f^{\text{bs}}_j}{|\mathcal U^{\text{bs}}_j|}.
$

The total latency and energy are
- [ ] Total BS latency
$
T^{\text{bs}}_{k,j} =
\frac{D_k^{\text{up}}}{R^{\text{up,bs}}_{k,j}} +
\frac{D_k^{\text{down}}}{R^{\text{down,bs}}_{k,j}} +
\frac{\tau^{\text{up}} D_k^{\text{up}} + \tau^{\text{down}} D_k^{\text{down}}}{C^{\text{bs}} \hat v^{\text{bs}}_{k,j}},
$
- [ ] Total BS energy
$
E^{\text{bs}}_{k,j} =
p_k \frac{D_k^{\text{up}}}{R^{\text{up,bs}}_{k,j}} +
p^{\text{bs}} \frac{D_k^{\text{down}}}{R^{\text{down,bs}}_{k,j}}.
$

Computation offloading cost for each $u_k$ is defined as
- [ ] Offloading cost definition
$
    Q_k(a_k, a_{-k}) = \bigg\{
    \begin{array}{cc}
         E^\text{sat}_{k, q}, & a_k = (1, q), \\
         E^\text{bs}_{k, j}, & a_k = (2, j),
    \end{array}
$
where $a_{-k} = \mathcal{A} \setminus \{a_k\}$ denotes the offloading decisions of all MGUs except $u_k$.


---

### F. Satellite Image Compression

The LEO satellite captures (M) images with sizes $B_z$. Let $\theta$ denote the compression ratio. The required CPU cycles per bit are
- [x] Compression CPU cycles per bit
$
F(\theta,\varsigma) = e^{\varsigma\theta} - e^{\varsigma}.
$

With computation resource 
- [ ] $f^{\text{cpr}}$ , the total compression latency and energy are
- [x] Compression latency
$
T^{\text{cpr}} = \sum_{z=1}^M \frac{B_z F(\theta,\varsigma)}{f^{\text{cpr}}},
$
- [x] Compression energy
$
E^{\text{cpr}} = \sum_{z=1}^M \kappa^{\text{sat}} (f^{\text{cpr}})^2 B_z F(\theta,\varsigma).
$

The compressed images are transmitted to the gateway at rate $\tilde R$, yielding transmission latency $\tilde T$ and energy $\tilde E$.

### G. Transmit data rate from satellite to gateway

Denote $l_z = B_z / \theta$ as the size of each compressed image. The channal gain between the LEO satellite and the ground gateway is

- [x] Gateway channel gain
$    
\tilde{q} = \frac{c}{4\pi \tilde{d} \tilde{\Psi}} \tilde{G}^\text{sat} \tilde{G}^\text{gate}|\tilde{\mu}|^2,
$

where $\tilde{\Psi}$ is the center frequency of the channel $\tilde{c}$, $\tilde{G}^\text{sat}$ and $\tilde{G}^\text{gate}$ are the antenna power gains of the LEO satellite and the gateway for channel $\tilde{c}$ respectively, and $|\tilde{\mu}|^2$ is the small-scale channel power gain following the shadowed-Rician model. The distance $\tilde{d}$ between the gateway and the LEO satellite. 


The downlink transmission rate between the LEO satellite and the gateway is given by
- [x] Gateway downlink rate
$    \tilde{R} = \tilde{B}\log_2\bigg(1 + \frac{\tilde{q} p^\text{sat}}{\tilde{B}\sigma^2}\bigg).$

Thus, the total latency and energy consumption for transmitting $M$ compressed images from the LEO satellite back to the gateway are $ \tilde{T} = \frac{\sum^M_{z = 1}l_z}{\tilde{R}}$ and $\tilde{E} = p^\text{sat}\frac{\sum^M_{z = 1}l_z}{\tilde{R}}$, respectively. 

- [x] Gateway transmission latency
- [x] Gateway transmission energy


### Metaverse Play-to-Earn model
Metaverse is essentially a digital environment in which MGUs generate content and interact with each other. With the idea of open economy and financial rewards, in the P2E model, MGUs add
value by playing and spending time in the P2E gaming system. 

We define a resolution-based earning function $\rho_k(r_k, R^\text{down}_k)$ where $R^\text{down}_k = R^\text{down\_sat}_{k, q}$ if $a_k = (1, q)$ and $R^\text{down}_k = R^\text{down\_bs}_{k, j}$ otherwise. $\rho_k(r_k, R^\text{down}_k)$ aims to approximate the MGUs' experience.

MGUs' Mean Opinion Scores (MOS) are a feasible way of addressing player engagement. Specifically, higher MOS indicates better gameplay, bringing higher profitability. Moreover, MOS can effectively reflect the willingness of MGUs to spend. Here, we consider three commonly used utility functions to estimate the MGUs' MOS when they are in different postures. 

Let a function $\rho^1_k(r_k)$ that aims to reflect the utility of the system in wireless networks as follows:
- [x] Earning function $\rho^1_k$
$\rho^1_k(r_k, R^\text{down}_k) = \alpha_1(r_k + R^\text{down}_k)^{\color{red}\beta_1},$
where $\alpha_1 \geq 0$ and $0 \leq \beta_1 \leq 1$. 

$\rho^2_k(r_k)$ captures the crowdsourcer's dwindling return in crowdsensing system, formulated as
- [x] Earning function $\rho^2_k$
$    \rho^2_k(r_k, R^\text{down}_k) = \alpha_2 \ln[1 + \beta_2(r_k + R^\text{down}_k)],
$
with $\alpha_2, \beta_2 \geq 0$. {We normalize the resolution $r_k$ and downlink data rate $R^\text{down}_k$ to the range of $0$ to $0.5$, respectively.}

$\rho^3_k(r_k)$ that models the relationship between the analytics accuracy and video frame resolution in MEC, which is given by
- [x] Earning function $\rho^3_k$
$\rho^3_k(r_k, R^\text{down}_k) = \alpha_3[1 - e^{-\beta_3(r_k + R^\text{down}_k)}],
$ 
where $\alpha_3, \beta_3 \geq 0$.

### H. Utility Function and Optimization Problem
Based on the communication model, computation offloading model and the Metaverse P2E model, the utility of MGU $u_k$ can be given as the weighted sum of energy consumption and earnings:
- [x] Utility function
$U_k = \eta_\text{earn} \cdot \rho_k(r_k) - \eta_\text{cons} \cdot \vartheta Q_k(a_k, a_{-k}),
$
with $\eta_\text{earn}$ and $\eta_\text{cons}$ being normalization factors, respectively. $\vartheta > 0$ serves as a weight parameter indicating the preference between delay and earnings. We can balance the trade-off between earnings and computation offloading cost by tuning $\vartheta$. In the Metaverse P2E game, we aim to minimize the offloading costs of all MGUs and maximize their earnings. \textcolor{red}{[Mention the total energy consumption for image compression and compressed image transmission]}
% Denote $w_{k} \in \{0, 1\} $ as an indicator variable that represents the association between $u_k$ and the LEO satellite or BS. In particular, $w_{k} = 1$ if $a_k = (1, q)$ and $w_{k} = 0$ if $a_k = (2, j)$.

Let $f^\text{tot}_1$ and $f^\text{tot}_2$ be the total computing capacity of the LEO satellite and the BS, respectively. Moreover, let $R_{\min}$ be the minimal transmission rates that guarantees proper communication. Thus, the scalarized optimization problem can be formulated as
<!-- \begin{subequations} -->
- [ ] Optimization objective and constraints
\[
\max_{r_k,\mathcal{A},\theta} \omega_U \eta_U \sum_{k=1}^{N} U_k - \omega_E \eta_E \left(E^{\mathrm{cpr}} + \tilde E\right) \\[0.6em]
\mathrm{s.t.}
R^{\mathrm{up,sat}}_{k,q} \ge R_{\min}, \;  R^{\mathrm{down,sat}}_{k,q} \ge R_{\min}, \ \forall u_k \in \mathcal{U}^{\mathrm{sat}}_q \\[0.4em]
R^{\mathrm{up,bs}}_{k,j} \ge R_{\min}, \; R^{\mathrm{down,bs}}_{k,j} \ge R_{\min}, \ \forall u_k \in \mathcal{U}^{\mathrm{bs}}_j \\[0.4em]
% \max\!\left( T^{\mathrm{sat}}_{k,q}, T^{\mathrm{cpr}} + \tilde T \right) \le T^{\max}_k, \ \forall u_k \in \mathcal{U}^{\mathrm{sat}}_q \\[0.4em]
r_{\min} \le r_k \le r_{\max} \\[0.4em]
\sum_{c_q \in \mathcal{C}} f^{\mathrm{sat}}_q + f^{\mathrm{cpr}} \le f^{\mathrm{tot}}_1 \\[0.4em]
\sum_{\hat c_j \in \hat{\mathcal{C}}} f^{\mathrm{bs}}_j \le f^{\mathrm{tot}}_2 \\[0.4em]
\theta \le \theta^{\max}
\]

where $\eta_U$ and $\eta_E$ are normalization factors, $\omega_{U}$ and $\omega_{E}$ are weight parameters and $\theta^{\max}$ is the maximum compression rate available. Constraint (\ref{opt}b) and (\ref{opt}c) ensure proper communication between MGUs and the LEO satellite or BS. Constraint (\ref{opt}d) guarantees that the LEO satellite must perform its tasks only when it can communicate with the MGUs. Constraint (\ref{opt}e) confines the resolution to a finite range. Constraint (\ref{opt}f) ensures that the total computing capacity distributed to execute the MGUs' tasks and the image compression task does not exceed what the LEO satellite provides. Similarly, constraint (\ref{opt}g) guarantees the total computing capacity the BS has to use to perform tasks is at most equal to its total capacity. Finally, constraint (\ref{opt}h) limits the compression capability of the LEO satellite edge server. 

### System parameters
We consider that there are $N=50$ MGUs randomly located in our system. We set the orbit height $H_e$ of the LEO satellite to $765$ km, and the elevation angle $\zeta_k$ between $u_k$ and the LEO satellite is $\sim[5.63^\circ, 85.94^\circ]$ (the angle between the LEO satellite and the gateway is drawn from the same uniform distribution). The LEO satellite and BS each possess $\nu^\text{sat} = 20$ and $\nu^\text{bs} = 15$ channels, respectively. The channel bandwidths $B^\text{sat}$ and $B^\text{bs}$ for each $c_q$ and $\hat{c}_j$ are set to $20$ MHz and $5$ MHz, respectively. The transmission power of each MGU $p_k$ is set to $23$ dBm. The total computational resources $f^\text{tot}_1$ of the LEO satellite is $20$ GHz (i.e. $20$ billion clock cycles per second) while that of the BS is $f^\text{tot}_2 = 50$ GHz. In addition, the number of computing resources consumed by the LEO satellite and the BS to complete one FLOP respectively are $C^\text{sat} = 1/32$ and $C^\text{bs} = 1/32$. During the allowed time, the LEO satellite captures $M = 100$ images. 

The earning function for each $u_k$ is chosen randomly between $\rho^1_k$, $\rho^2_k$ and $\rho^3_k$.  Other parameters can be found in Table~\ref{Table:sys_par}, and the parameters for earning functions can be found in Table~\ref{Table:earning}. 

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| $\chi_k$ | $\sim [300,600]$ | $R_{\min}$ | $1$ Mbps |
| $B_z$ | $\sim [1500,4500]$ Kb | $D^{\mathrm{up}}_k$ | $\sim [2000,6000]$ Kb |
| $\tau_{\mathrm{up}}$ | $\sim [1,10]$ | $\tau_{\mathrm{down}}$ | $\sim [1,100]$ |
| $\kappa^{\mathrm{sat}}$ | $1$ nF | $\sigma^2$ | $-134$ dBm |
| $p^{\mathrm{sat}}$ | $30$ dBm | $p^{\mathrm{bs}}$ | $46$ dBm |
| $G^{\mathrm{sat}}_q$, $\tilde G^{\mathrm{sat}}$ | $30$ dBi | $G^{\mathrm{user}}_{k,q}$ | $\sim [5,8]$ dBi |
| $\tilde G^{\mathrm{gate}}$ | $15$ dBi | $q^{\mathrm{bs}}_k$ | $\sim [-105,-80]$ dB |
| $d^{\mathrm{bs}}_k$ | $\sim [500,1000]$ m | $\Gamma$ | $4$ |
| $c$ | $3\times10^8$ m/s | $\mu$ | $3.986\times10^{14}$ m$^3$/s$^2$ |
| $\Psi_q$, $\tilde\Psi$ | $2$ GHz | $\theta^{\max}$ | $3$ |
| $r_{\min}$ | $1280\times720$ | $r_{\max}$ | $7680\times4320$ |
| $\varsigma$ | $4$ | $\omega_U,\omega_E$ | $0.5,\,0.5$ |

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| $\alpha_1$ | $4.268$ | $\beta_1$ | $0.2714$ |
| $\alpha_2$ | $1.159$ | $\beta_2$ | $91.92$ |
| $\alpha_3$ | $1.0$ | $\beta_3$ | $1.0$ |

