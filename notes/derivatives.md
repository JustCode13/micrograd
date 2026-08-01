Perfect. I'll teach only what your `micrograd_clone` needs. Nothing extra.

---

# 3. Derivatives

## What is a derivative?

Imagine you're riding a bicycle.

At one moment you're riding slowly.

A few seconds later you're riding faster.

The derivative simply answers one question:

> **"How fast is something changing right now?"**

That's all.

Forget the scary math word.

Think:

> **Derivative = Change**

---

## Example 1

Suppose

```
Age : 5 years
Height : 100 cm

Age : 6 years
Height : 110 cm
```

Height changed.

Derivative asks

> "How quickly is height changing as age changes?"

---

## Example 2

Imagine a game score.

```
Second 1 → 5 points
Second 2 → 10 points
Second 3 → 20 points
```

The score is increasing.

Derivative measures

> **how quickly the score is increasing.**

---

# Local Rate of Change

Don't worry about the long name.

Break it.

```
Local
```

means

> **right here**

---

```
Rate
```

means

> **speed**

---

```
Change
```

means

> **something becoming different**

---

So

```
Local Rate of Change
```

means

> **How fast something is changing at this exact moment.**

---

Example

Imagine driving a car.

```
0 sec → 20 km/h

10 sec → 40 km/h

20 sec → 70 km/h
```

At exactly 10 seconds,

how fast are you going?

Not the whole trip.

Only

**RIGHT NOW.**

That's local rate of change.

---

# Slope Intuition

Imagine a hill.

```
       *
     *
   *
 *
```

Walking here is difficult.

Why?

Because the hill goes up quickly.

This is called a **steep slope.**

---

Now imagine

```
*
 *
  *
   *
```

Almost flat.

Easy to walk.

Small slope.

---

Now imagine

```
************
```

Perfectly flat.

Slope = 0

Nothing is changing.

---

Derivative is simply

> **the slope at one point.**

That's it.

---

# dy/dx

This scares everyone.

It is actually just a name.

```
dy
--
dx
```

Read it as

> **"change in y when x changes."**

Nothing more.

---

Suppose

```
x = Time

y = Money
```

Then

```
dy/dx
```

means

> **How fast money changes when time changes.**

---

Another example

```
x = Age

y = Height
```

Then

```
dy/dx
```

means

> **How fast height changes when age changes.**

---

In micrograd,

```
x
```

is usually

```
input
```

and

```
y
```

is

```
output
```

Derivative asks

> **If the input changes a tiny bit, how much does the output change?**

That's exactly what backpropagation needs.

---

# Derivative of a Constant

Constant means

> **A number that never changes.**

Examples

```
5

100

-8

1000
```

Suppose

```
y = 5
```

No matter what x is,

```
x = 1 → y = 5

x = 10 → y = 5

x = 100 → y = 5
```

Nothing changed.

Derivative measures change.

There is no change.

So

```
Derivative = 0
```

Always remember

```
d(5)/dx = 0
```

```
d(100)/dx = 0
```

```
d(-3)/dx = 0
```

Every constant has derivative **0**.

---

# Derivative of x

Now

```
y = x
```

Let's make a table.

| x | y |
| - | - |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |

Every time x increases by

```
1
```

y also increases by

```
1
```

The change is always

```
1
```

So

```
Derivative = 1
```

Remember

```
d(x)/dx = 1
```

---

# Derivative of x²

Now

```
y = x²
```

Table

| x | y  |
| - | -- |
| 1 | 1  |
| 2 | 4  |
| 3 | 9  |
| 4 | 16 |

Notice something.

The jumps become larger.

```
1 → 4
```

jump = 3

```
4 → 9
```

jump = 5

```
9 →16
```

jump = 7

The change is getting bigger.

The derivative tells us

```
Derivative = 2x
```

You don't need to prove this for micrograd.

Just remember it.

Examples

If

```
x=2
```

Derivative

```
2×2=4
```

---

If

```
x=5
```

Derivative

```
2×5=10
```

---

If

```
x=10
```

Derivative

```
2×10=20
```

---

# Derivative of xⁿ

You don't need all calculus.

Just one rule.

If

```
y = xⁿ
```

Then

```
Derivative

= n × xⁿ⁻¹
```

This is called the **power rule**.

Examples

---

```
x²
```

Derivative

```
2x
```

---

```
x³
```

Derivative

```
3x²
```

---

```
x⁴
```

Derivative

```
4x³
```

---

```
x⁵
```

Derivative

```
5x⁴
```

You will directly use this when implementing `__pow__` in your `Value` class.

---

# Derivative of ax + b

Looks scary.

It isn't.

Suppose

```
a = 3

b = 5
```

Then

```
y = 3x + 5
```

Table

| x | y  |
| - | -- |
| 1 | 8  |
| 2 | 11 |
| 3 | 14 |
| 4 | 17 |

Every time x increases by

```
1
```

y increases by

```
3
```

Always.

So the derivative is simply

```
3
```

The `+5` never changes, so it contributes **0**.

General rule:

```
y = ax + b

Derivative = a
```

Examples

```
2x + 10  → 2
```

```
7x + 100 → 7
```

```
-4x + 3 → -4
```

---

# The only formulas you need to remember for `micrograd`

```
Constant      → 0

x             → 1

x²            → 2x

xⁿ            → n × xⁿ⁻¹

ax + b        → a
```

That's all the derivative math you need before moving on to **partial derivatives** and then the **chain rule**, which are the core ideas behind backpropagation.
