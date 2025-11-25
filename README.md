# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.

## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.

## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.


## POLICY IMPROVEMENT FUNCTION
### Name : DHARMARAJ S
### Register Number : 212222240025
```python
Include the policy improvement function
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s, a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: np.argmax(Q[s, :])
    return new_pi

```
## POLICY ITERATION FUNCTION
### Name : DHARMARAJ S
### Register Number : 212222240025
```python
Include the policy iteration function
def policy_iteration(P, gamma=1.0, theta=1e-10):
    num_states = len(P)
    num_actions = len(P[0])

    # Initialize an arbitrary policy (e.g., all actions are 0 - LEFT)
    pi = lambda s: 0

    while True:
        # Policy Evaluation
        V = policy_evaluation(pi, P, gamma, theta)

        # Policy Improvement
        new_pi_func = policy_improvement(V, P, gamma)

        # Check for policy convergence
        policy_stable = True
        for s in range(num_states):
            if new_pi_func(s) != pi(s):
                policy_stable = False
                break

        pi = new_pi_func

        if policy_stable:
            break

    return V, pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
## Policy
<img width="519" height="101" alt="image" src="https://github.com/user-attachments/assets/63a5d231-6063-4934-97ea-02983e2cef37" />

## success rate
<img width="752" height="80" alt="image" src="https://github.com/user-attachments/assets/bcc52b62-4632-4dc9-929c-d8ae2843a06d" />

## Value function
<img width="532" height="100" alt="image" src="https://github.com/user-attachments/assets/6622e97c-66b1-4bb6-8af6-b0dc38402060" />




### 2. Policy, Value function and success rate for the Improved Policy
## Policy
<img width="596" height="105" alt="image" src="https://github.com/user-attachments/assets/708f57e4-66f6-4671-9e26-137c586c4f19" />

## success rate
<img width="727" height="82" alt="image" src="https://github.com/user-attachments/assets/09a16ea1-3346-407a-ba92-667a5e45e81f" />

## Value function
<img width="469" height="105" alt="image" src="https://github.com/user-attachments/assets/5dacee2e-3b8e-429b-9528-1bedef66d813" />





### 3. Policy, Value function and success rate after policy iteration
## Policy
<img width="551" height="119" alt="image" src="https://github.com/user-attachments/assets/a7a02f30-e568-4500-ac44-616b4ba2860f" />


## success rate
<img width="701" height="82" alt="image" src="https://github.com/user-attachments/assets/446071d0-ec2e-49d4-b798-f6c2a22fe511" />


## Value function
<img width="796" height="80" alt="image" src="https://github.com/user-attachments/assets/551d9dbf-2952-4324-a87e-bab6e909dc05" />




## RESULT:

Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
