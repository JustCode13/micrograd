# Chain Rule (Only what Micrograd needs)

This is **the most important math in the whole project.**

If you understand this, you'll understand what `.backward()` is doing.

Forget big formulas.

We'll build it from zero.

---

# Step 1 — What problem does the chain rule solve?

Suppose you write

```python
a = 2
b = a * 3
c = b + 4
d = c * 5
```

Look carefully.

```text
a
│
▼
b
│
▼
c
│
▼
d
```

Notice something.

`d` does **not** depend directly on `a`.

Instead

```text
a
 ↓
b
 ↓
c
 ↓
d
```

There are **multiple steps**.

The chain rule answers one question:

> **If `a` changes a little, how much does the final answer `d` change?**

That's literally what the chain rule exists for.

---

# Step 2 — Nested functions

A **nested function** simply means

> One operation uses the result of another operation.

Example

```python
b = a * 3
c = b + 4
d = c * 5
```

Each line depends on the previous one.

Nothing more.

Your computation graph becomes

```text
a
│
▼
(*3)
│
▼
b
│
▼
(+4)
│
▼
c
│
▼
(*5)
│
▼
d
```

Micrograd builds exactly this graph.

---

# Step 3 — Forward pass

This is the easy part.

We simply calculate values.

Suppose

```python
a = 2
```

Compute forward.

```text
b = a * 3

b = 6
```

Next

```text
c = b + 4

c = 10
```

Next

```text
d = c * 5

d = 50
```

Forward pass means

```text
Start at inputs

↓

Keep calculating

↓

Reach final answer
```

That's all.

---

# Step 4 — Why forward is NOT enough

Imagine your neural network makes a wrong prediction.

You now want to improve it.

Question:

Which variable caused the mistake?

Maybe

```text
a
```

Maybe

```text
b
```

Maybe

```text
c
```

We need to know

> How much did each variable affect the final answer?

Forward pass cannot answer that.

We need to go backwards.

---

# Step 5 — Backward intuition

Now we start at the end.

```text
d
↑
c
↑
b
↑
a
```

Instead of computing values,

we compute

```text
gradients
```

Micrograd literally walks backwards through the graph.

---

# Step 6 — Local Gradient

This is an important word.

A **local gradient** means

> How much does the output of **this one operation** change if one of its inputs changes?

Only one operation.

Not the whole graph.

Example

```text
c = a + b
```

Question

```text
How much does c change if a changes?
```

Answer

```text
1
```

Question

```text
How much does c change if b changes?
```

Answer

```text
1
```

These are **local gradients**.

Only this operation.

---

Another example

```text
c = a * b
```

Question

```text
How much does c change if a changes?
```

Answer

```text
b
```

Question

```text
How much does c change if b changes?
```

Answer

```text
a
```

Again

Only this multiplication.

Nothing else.

---

# Step 7 — Why local gradients are useful

Look again.

```text
a
│
▼
(*3)
│
▼
b
│
▼
(+4)
│
▼
c
│
▼
(*5)
│
▼
d
```

Each arrow has its own local gradient.

Example

```text
a → b

gradient = 3
```

because

```text
b = a * 3
```

Next

```text
b → c

gradient = 1
```

because

```text
c = b + 4
```

Next

```text
c → d

gradient = 5
```

because

```text
d = c * 5
```

Each step only knows about itself.

---

# Step 8 — Multiplying gradients

Now comes the entire chain rule.

We want

```text
How much does d depend on a?
```

There isn't one step.

There are three.

```text
a

↓

b

↓

c

↓

d
```

Each step changes things.

Step 1

```text
×3
```

Step 2

```text
+4
```

Derivative

```text
1
```

Step 3

```text
×5
```

Now multiply.

```text
3 × 1 × 5 = 15
```

That is the chain rule.

Nothing more.

You multiply the local gradients.

---

# Step 9 — Another example

Suppose

```text
a = 4

b = a²

c = b * 10
```

Graph

```text
a

↓

b

↓

c
```

First local gradient

```text
b = a²
```

Derivative

```text
2a
```

Since

```text
a = 4
```

Gradient

```text
8
```

Second local gradient

```text
c = b * 10
```

Derivative

```text
10
```

Chain rule

```text
8 × 10 = 80
```

Final answer

```text
dc/da = 80
```

---

# Step 10 — How Micrograd uses this

Suppose you write

```python
a = Value(2)
b = a * 3
c = b + 4
d = c * 5

d.backward()
```

During

```python
d.backward()
```

Micrograd walks backward.

```text
d

↓

c

↓

b

↓

a
```

At every node it asks

> What is the local gradient here?

Then it multiplies it with the gradient coming from above.

Exactly like this.

```text
incoming gradient

×

local gradient

=

new gradient
```

This happens at every node.

---

# Step 11 — Why `out.grad` exists

Suppose we're at

```text
d = c * 5
```

When we're going backward,

`d` already has a gradient.

Maybe

```text
d.grad = 1
```

Now we compute

```text
c.grad
```

How?

Multiply

```text
incoming gradient

×

local gradient
```

Incoming gradient

```text
1
```

Local gradient

```text
5
```

Result

```text
5
```

So

```text
c.grad = 5
```

Then move to

```text
b
```

Incoming gradient is now

```text
5
```

Local gradient

```text
1
```

Result

```text
5
```

Then move to

```text
a
```

Incoming gradient

```text
5
```

Local gradient

```text
3
```

Result

```text
15
```

Notice what happened.

Every node received a gradient from above.

Then multiplied by its own local gradient.

That is exactly why Micrograd stores `out.grad`.

---

# Step 12 — This is the actual code you'll write

When implementing multiplication, you'll write something like:

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

Why?

For

```text
out = self * other
```

the local gradients are

```text
∂out/∂self = other

∂out/∂other = self
```

The chain rule says:

```text
my gradient

=

incoming gradient

×

local gradient
```

So the code becomes:

```text
self.grad += out.grad × other.data
other.grad += out.grad × self.data
```

That is the chain rule translated directly into Python.

---

# What you should remember

* **Nested functions**: One operation uses the result of another (`a → b → c → d`).
* **Forward pass**: Compute values from inputs to output.
* **Backward pass**: Start at the output and send gradients back to every input.
* **Local gradient**: How one operation's output changes when one of its inputs changes.
* **Chain rule**: To find how an early value affects the final output, **multiply the local gradients along the path**.
* **Micrograd's `.backward()`** is nothing more than repeatedly applying this multiplication while traversing the computation graph in reverse order.
