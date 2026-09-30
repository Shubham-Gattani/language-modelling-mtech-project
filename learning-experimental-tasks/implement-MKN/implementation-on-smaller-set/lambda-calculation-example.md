
# Lambda Calculation Example: Goal First, Then Prerequisites

This is the same 3-sentence example, reorganized so that we first write the probability we want, identify everything it depends on, and only then calculate those pieces.

## 1. Corpus and fixed discounts

The corpus is:

```text
S1: <s> cat sat on mat </s>
S2: <s> cat sat on rug </s>
S3: <s> dog sat on mat </s>
```

We use these hardcoded Modified Kneser-Ney discounts:

```text
D1  = 0.50   when an observed n-gram count is 1
D2  = 0.70   when an observed n-gram count is 2
D3+ = 0.90   when an observed n-gram count is 3 or greater
```

## 2. Start with the final goal

We want probabilities after the trigram history:

```text
h3 = ("sat", "on")
```

We will calculate two candidate words:

```text
w = "mat"   -> observed trigram ("sat", "on", "mat")
w = "cat"   -> unseen trigram ("sat", "on", "cat")
```

The main formula is:

```text
P(w | "sat", "on")
= direct_trigram_contribution(w | "sat", "on")
  + lambda("sat", "on") * P(w | "on")
```

This formula tells us exactly what must be calculated:

```text
1. direct trigram contribution
2. lambda("sat", "on")
3. lower-order bigram probability P(w | "on")
```

The bigram probability itself is recursive:

```text
P(w | "on")
= direct_bigram_contribution(w | "on")
  + lambda("on") * P_cont(w)
```

Therefore, the dependency structure is:

```text
P(w | "sat", "on")
├── direct trigram contribution
├── lambda("sat", "on")
└── P(w | "on")
    ├── direct bigram contribution
    ├── lambda("on")
    └── P_cont(w)
```

We will calculate from the bottom of this dependency structure upward.

## 3. First calculate the base continuation probabilities

The base probability is:

```text
P_cont(w)
= number of distinct predecessors of w
  / total number of distinct bigram types
```

The distinct bigrams are:

```text
(<s>, cat)
(cat, sat)
(sat, on)
(on, mat)
(mat, </s>)
(on, rug)
(rug, </s>)
(<s>, dog)
(dog, sat)
```

There are 9 distinct bigram types.

For the words needed in our example:

```text
mat:
    predecessors = {on}
    P_cont(mat) = 1/9 = 0.1111...

cat:
    predecessors = {<s>}
    P_cont(cat) = 1/9 = 0.1111...

sat:
    predecessors = {cat, dog}
    P_cont(sat) = 2/9 = 0.2222...
```

We need P_cont(mat) and P_cont(cat) for the lower-order calculations.

## 4. Calculate the lower-order bigram model

The trigram formula needs:

```text
P(mat | on)
P(cat | on)
```

So first use the history:

```text
h2 = ("on",)
```

### 4.1 Count the history

The word on occurs three times:

```text
cat sat on mat
cat sat on rug
dog sat on mat
```

Therefore:

```text
c("on") = 3
```

The observed bigrams after on are:

```text
("on", "mat"): 2
("on", "rug"): 1
```

Their counts sum to the history count:

```text
2 + 1 = 3
```

### 4.2 Write the complete lambda formula before calculating it

For any history h:

```text
lambda(h)
= ([D1 * N1(h·) + D2 * N2(h·) + D3+ * N3+(h·)]) / c(h)
```

Here:

```text
N1(h·)  = number of continuations with count exactly 1
N2(h·)  = number of continuations with count exactly 2
N3+(h·) = number of continuations with count 3 or greater
```

For history on:

```text
("on", "rug") has count 1 -> N1(("on",)·) = 1
("on", "mat") has count 2 -> N2(("on",)·) = 1
no continuation has count 3 or greater -> N3+(("on",)·) = 0
```


### Intuition: why divide by c(h)?

Imagine that the three occurrences of "on" are three units of evidence:

```text
on -> mat
on -> mat
on -> rug
```

So the **total evidence** associated with the history "on" is:

```text
c("on") = 3 count units
```

Discounting removes:

```text
0.70 from the two-count continuation "on mat"
0.50 from the one-count continuation "on rug"
total removed = 1.20 count units
```

The number 1.20 is a count amount, not yet a probability. To ask **what fraction of the history's total evidence was removed**, compare it with all 3 units belonging to on:

```text
removed fraction
= removed count / total history count
= 1.20 / 3
= 0.40
```

Therefore, lambda is 0.40, meaning 40% of the probability mass after "on" is handed to the lower-order model. The remaining 60% stays with the direct bigram evidence.

If we divided by something other than c("on"), we would no longer be measuring the fraction of evidence removed from this particular history. This is why c(h) forms the denominator.

Now calculate:

```text
lambda("on")
= [0.50 * 1 + 0.70 * 1 + 0.90 * 0] / 3
= [0.50 + 0.70 + 0] / 3
= 1.20 / 3
= 0.4000
```

The division by c("on") = 3 converts the removed count units into a fraction of the total count associated with the history on.

### 4.3 Calculate P(mat | on)

The bigram ("on", "mat") has count 2, so it uses D2 = 0.70.

Direct bigram contribution:

```text
(2 - 0.70) / 3
= 1.30 / 3
= 0.4333...
```

Now add the lower-order contribution:

```text
P(mat | on)
= (2 - 0.70) / 3
  + lambda("on") * P_cont(mat)
= 0.4333... + 0.4000 * (1/9)
= 0.4333... + 0.0444...
= 0.4778...
```

### 4.4 Calculate P(cat | on)

The bigram ("on", "cat") was never observed, so its count is 0.

Its direct contribution is 0:

```text
max(0 - 0, 0) / 3 = 0
```

The probability comes entirely from the lower-order continuation probability:

```text
P(cat | on)
= 0
  + lambda("on") * P_cont(cat)
= 0.4000 * (1/9)
= 0.0444...
```

At this point, the lower-order values required by the trigram calculation are known:

```text
P(mat | on) = 0.4778...
P(cat | on) = 0.0444...
```

## 5. Calculate the trigram interpolation weight

Now return to the full history:

```text
h3 = ("sat", "on")
```

The history occurs three times:

```text
c("sat", "on") = 3
```

Its observed continuations are:

```text
("sat", "on", "mat"): 2
("sat", "on", "rug"): 1
```

Therefore:

```text
N1(("sat","on")·)  = 1
N2(("sat","on")·)  = 1
N3+(("sat","on")·) = 0
```

Use the complete lambda formula:

```text
lambda("sat", "on")
= [D1 * 1 + D2 * 1 + D3+ * 0] / 3
= [0.50 * 1 + 0.70 * 1 + 0.90 * 0] / 3
= 1.20 / 3
= 0.4000
```

This lambda is fixed for the history ("sat", "on"). It is used for both candidate words, mat and cat.

## 6. Calculate the direct trigram contributions

### 6.1 Target word mat

The trigram ("sat", "on", "mat") has count 2, so it uses D2 = 0.70.

```text
direct_trigram_contribution(mat | sat,on)
= (2 - 0.70) / 3
= 1.30 / 3
= 0.4333...
```

### 6.2 Target word cat

The trigram ("sat", "on", "cat") has count 0.

```text
direct_trigram_contribution(cat | sat,on)
= max(0 - 0, 0) / 3
= 0
```

The unseen trigram receives no direct trigram contribution.

## 7. Return to the original trigram formula

### 7.1 Final probability for mat

We now have every required component:

```text
direct trigram contribution = 0.4333...
lambda("sat", "on")         = 0.4000
P(mat | on)                 = 0.4778...
```

Substitute:

```text
P(mat | sat,on)
= 0.4333... + 0.4000 * 0.4778...
= 0.4333... + 0.1911...
= 0.6244...
```

Therefore:

```text
P(mat | sat,on) = 0.6244... = 62.44% approximately
```

### 7.2 Final probability for cat

We have:

```text
direct trigram contribution = 0
lambda("sat", "on")         = 0.4000
P(cat | on)                 = 0.0444...
```

Substitute:

```text
P(cat | sat,on)
= 0 + 0.4000 * 0.0444...
= 0.0178...
```

Therefore:

```text
P(cat | sat,on) = 0.0178... = 1.78% approximately
```

## 8. What this teaching order made visible

The final trigram probability was not calculated in one jump.

For mat, the dependency chain was:

```text
P(mat | sat,on)
= direct trigram contribution
  + lambda(sat,on) * P(mat | on)

P(mat | on)
= direct bigram contribution
  + lambda(on) * P_cont(mat)

P_cont(mat)
= distinct predecessors of mat
  / distinct bigram types
= 1/9
```

The same structure was used for cat.

The key mental model is:

```text
start with the target probability
-> write its formula
-> identify its prerequisites
-> calculate prerequisites from the bottom upward
-> substitute them into the original formula
```

Also remember:

```text
for one target word:
    use one discount based on c(h,w)

for lambda(h):
    sum discounts over all observed continuations after h
```
