# Partial Derivatives (Only what Micrograd needs)

Forget everything about college math. You only need enough to build the autograd engine.

---

# Step 1 — What is a derivative?

You already know a normal function.

```text
y = x²
```

If

```text
x = 3
```

then

```text
y = 9
```

A **derivative** simply answers one question:

> **If I change the input a tiny bit, how much will the output change?**

That's it.

Nothing more.

---

Example

```text
y = x²

x = 3
```

Output

```text
9
```

Now increase x just a little.

```text
x = 3.1
```

Now

```text
y = 9.61
```

The output changed because the input changed.

A derivative measures that change.

---

# Step 2 — But what if there are TWO inputs?

Suppose we have

```text
z = x + y
```

Now there isn't one input.

There are two.

```text
x
y
```

Both affect

```text
z
```

Example

```text
x = 5
y = 2

z = 7
```

Now ask

> What happens if x changes?

or

> What happens if y changes?

These are **different questions.**

So we need **two derivatives.**

---

# Step 3 — Why normal derivative is not enough

Normal derivative looks like

```text
dy/dx
```

This means

> How does y change when x changes?

But now we have

```text
z = x + y
```

There are two inputs.

So we need

```text
How does z change if x changes?

How does z change if y changes?
```

These answers are different.

---

# Step 4 — This is called a Partial Derivative

The fancy symbol

```text
∂
```

is just a signal.

It means

> "I'm only changing ONE variable."

Nothing else.

So

```text
∂z/∂x
```

means

> Change only x.
>
> Keep everything else fixed.

And

```text
∂z/∂y
```

means

> Change only y.
>
> Keep everything else fixed.

That's literally all the symbol means.

---

# Step 5 — "Holding other variables constant"

This sentence scares lots of people.

It is actually very simple.

Suppose

```text
z = x + y
```

Current values

```text
x = 5
y = 2

z = 7
```

Now we want

```text
∂z/∂x
```

We are only allowed to change x.

So

```text
y stays 2 forever.
```

Now change x.

```text
x = 6
y = 2

z = 8
```

Again

```text
x = 7
y = 2

z = 9
```

Notice

```text
Only x moved.
```

This is what

> Holding y constant

means.

Nothing mysterious.

---

Now do

```text
∂z/∂y
```

Now

```text
x never moves.
```

Keep

```text
x = 5
```

Change only

```text
y
```

```text
y = 3

z = 8
```

Then

```text
y = 4

z = 9
```

Again

Only y moved.

---

# Step 6 — Example 1

Function

```text
z = x + y
```

Question

```text
∂z/∂x
```

Increase x by 1.

Output also increases by 1.

So

```text
∂z/∂x = 1
```

Now

```text
∂z/∂y
```

Increase y by 1.

Output increases by 1.

So

```text
∂z/∂y = 1
```

---

# Step 7 — Another example

Suppose

```text
z = x × y
```

Current values

```text
x = 4
y = 3

z = 12
```

---

Find

```text
∂z/∂x
```

Keep

```text
y = 3
```

fixed.

Only move x.

```
x = 4

z = 12
```

```
x = 5

z = 15
```

Output increased

```text
12 → 15

+3
```

So every time x goes up by 1,

output goes up by 3.

Therefore

```text
∂z/∂x = 3
```

Notice

3 is exactly the value of y.

---

Now

```text
∂z/∂y
```

Keep

```text
x = 4
```

fixed.

Move y.

```
y = 3

z = 12
```

```
y = 4

z = 16
```

Output increased

```text
12 → 16

+4
```

So

```text
∂z/∂y = 4
```

Notice

4 is exactly the value of x.

---

# Step 8 — Why does Micrograd care?

Suppose your graph is

```text
a = 2
b = 3

c = a * b
```

Micrograd needs to know

```text
How much does c depend on a?

How much does c depend on b?
```

Those are exactly

```text
∂c/∂a

∂c/∂b
```

For multiplication,

Micrograd computes

```text
∂c/∂a = b

∂c/∂b = a
```

With the values

```text
a = 2
b = 3
```

it becomes

```text
∂c/∂a = 3

∂c/∂b = 2
```

These numbers are what the `.backward()` method uses to send gradients backward through the computation graph.

---

# Step 9 — Where you'll write this in code

When you implement multiplication in your `Value` class, you'll write logic equivalent to:

```python
self.grad += other.data * out.grad
other.grad += self.data * out.grad
```

Why?

Because mathematically,

```text
∂(a*b)/∂a = b
∂(a*b)/∂b = a
```

The code is just applying those partial derivatives during backpropagation.

---

# What you should remember

* A **partial derivative** is used when a function has **more than one input**.
* `∂z/∂x` means: **change only `x`; keep all other inputs fixed**.
* `∂z/∂y` means: **change only `y`; keep all other inputs fixed**.
* Micrograd computes these partial derivatives for every operation (`+`, `*`, `**`, `tanh`, etc.) so it knows how to propagate gradients backward through the computation graph.
