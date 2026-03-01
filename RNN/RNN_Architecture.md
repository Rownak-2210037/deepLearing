# Architecture of RNN
**Goal**

1. RNN Architecture
2. RNN forward propagation
3. How to predict output from input

### why we can't handle sequential data with ANN?
ANN cannot properly handle sequential data because it processes inputs independently and has no memory of previous inputs. Sequential data requires context from earlier time steps, which ANN cannot capture. 
Problem:
1.text input--> varying size
2.Zero padding-->unnecessary computation
3.Prediction problem
4.sequence info are lost

1️⃣ No memory

A standard ANN (feedforward neural network) processes each input independently.

It does:
y=f(wx+b)

It only looks at the current input x.
It does NOT remember previous inputs.

2️⃣ Sequential data needs context

In sequential data (like text or time series),
the current element depends on previous ones.

Example:

“I love deep learning”

To understand “learning”,
the model needs to remember “I love deep”.

ANN cannot remember previous words.

2️⃣ Sequential data needs context

In sequential data (like text or time series),
the current element depends on previous ones.

Example:

“I love deep learning”

To understand “learning”,
the model needs to remember “I love deep”.

ANN cannot remember previous words.

**Data Input style**

[(timesteps,input feutures)] in this format we gave input.
for first word t=1
2nd word t=2.....
if vocab=5 then for one sentence "movie was good" the input will be (3,5) where 3=total timesteps
5=every vector size such as movie=[1,0,0,0,0], was=[0,1,0,0,0].........

suppose data exists:
          Review                      Sentiment
 x1    movie was good                    1          
 x2    movie was bad                     0
 x3    movie was not good                0

 structure is:  input-------->Recurrrent hidden---------->output
 difference is that we can't feed whole input at the same time.we send one by one(movie,was.....) according to timestep.

 ![alt text](image.png)


