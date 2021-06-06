






## Factors
Let $X_{1}: \Omega_{1} \to E_{1}, X_{2}: \Omega_{2} \to E_{2}, \ldots, X_{k}: \Omega_{k} \to E_{k}$ be random
variables. A **_factor_** of **_scope_** $\{X_{1}, \ldots, X_{k}\}$ is defined simply as any function
$$\phi: X_{1}(\Omega_{1}) \times X_{2}(\Omega_{2}) \times X_{k}(\Omega_{k}) \to \mathbb{R}\,.$$

The following are a few examples of factors:


## Bayesian Networks
A **_bayesian network_** is

- a directed acyclic graph (DAG) whose nodes represent some random variables $X_{1}, \ldots, X_{n}$
- for each node $X_{i}$, we are given the conditional probability distribution $P(X_{i} | Parent_{G}(X_{i}))$
- with the joint distribution defined by 
