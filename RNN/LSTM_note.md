### RNN has two Major Problems:

1️⃣ Vanishing Gradient Problem
During training, gradients are propagated backward through time (BPTT).
In long sequences, gradients are multiplied many times by small numbers (derivatives < 1).As a result:

Gradient
→
0
Gradient→0
🔹 What is the effect?

Earlier time steps learn very slowly

Model cannot remember long-term dependencies

Old information fades away

🔹 Simple intuition

If each step multiplies gradient by 0.5:

After many steps:

0.5×0.5×0.5×...→0

So memory of early inputs disappears.RNN fails to capture long-range relationships.

Example:

In long sentences, the model forgets the beginning.

2️⃣ Exploding Gradient Problem

If gradients are multiplied by large values (>1) repeatedly:
Gradient→ ∞

🔹 What is the effect?

Weights update becomes extremely large

Training becomes unstable

Loss becomes NaN

Model may crash

🔹 Simple intuition

If each step multiplies by 2:

very large number
2×2×2×...→very large number

Problem	Cause	Effect:

Vanishing Gradient	Repeated multiplication by small numbers	Model forgets long-term info
Exploding Gradient	Repeated multiplication by large numbers	Training becomes unstable
✅ Solutions

LSTM / GRU (reduce vanishing gradient)

Gradient clipping (prevent exploding gradient)

Vanishing gradient makes learning very slow for early time steps, while exploding gradient makes training unstable due to very large updates.