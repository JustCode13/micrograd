# 6. Basic Topological Order

First, one important thing:

**This is NOT actually a math topic.**

It is an **algorithm** (a way of deciding the order in which we do work).

Micrograd uses it because if we calculate gradients in the wrong order, we get the wrong answers.

---

# What problem are we trying to solve?

Suppose your code does this:

```python
a = Value(2)
b = Value(3)

c = a * b
d = c + 5
e = d * 2
```

The computer builds this graph.

```text
a      b
 \    /
  \  /
   c
   |
   d
   |
   e
```

Each variable depends on the one before it.

---

# What does "depends on" mean?


Look here:

```python
c = a * b
```

Can Python calculate `c` before it knows `a`?

No.

It must know

```
a
```

and

```
b
```

first.

So we say

> **c depends on a and b.**

---

Next

```python
d = c + 5
```

Can Python calculate `d` first?

No.

It first needs

```
c
```

So

> **d depends on c.**

---

Next

```python
e = d * 2
```

Can Python calculate `e` first?

No.

It first needs

```
d
```

So

> **e depends on d.**

---

The graph becomes

```text
a ---> c ---> d ---> e
b ---^
```

Everything is connected.

---

# Dependency Ordering

This is just a big name.

It simply means

> **Do things only after everything they need already exists.**

That's all.

Nothing more.

---

Example

This is impossible

```python
c = a * b

a = 2
b = 3
```

because

```
c
```

needs

```
a
```

and

```
b
```

first.

Correct order

```python
a = 2
b = 3

c = a * b
```

---

Another example

```python
x = 4

y = x + 1

z = y * 3
```

Correct order

```
x

↓

y

↓

z
```

Wrong order

```
z

↓

y

↓

x
```

because

```
z
```

needs

```
y
```

first.

---

# How is this used in Micrograd?

When you write

```python
c = a * b
```

Micrograd creates a node.

```
a

b

↓

c
```

Then

```python
d = c + 5
```

creates another node.

```
a

b

↓

c

↓

d
```

Every operation becomes another node.

Micrograd remembers

> "This node depends on these earlier nodes."

---

# Why backward starts at the output

Now suppose

```python
loss = ((a * b) + 5) * 2
```

Graph

```text
a      b
 \    /
  \  /
   *
   |
   +
   |
   *
   |
 loss
```

The final answer is

```
loss
```

---

When training a neural network, we only know one thing:

> "How bad was the final answer?"

That is the loss.

We do **not** know immediately how much

```
a
```

or

```
b
```

caused that loss.

So we must start from the end.

---

Suppose

```
loss = 20
```

We ask

> Where did 20 come from?

It came from

```
*
```

Then we ask

Where did that multiplication come from?

It came from

```
+
```

Then

Where did that plus come from?

It came from

```
a
```

and

```
b
```

So information moves

```text
loss

↑

*

↑

+

↑

a      b
```

This is called

**backward**.

---

# Why can't we start from a?

Suppose we try

```text
a

↓

*

↓

+

↓

loss
```

Can `a` know how much it should change?

No.

Because it doesn't yet know the final loss.

Only the output knows the loss.

So gradients must begin at

```
loss
```

and travel backwards.

---

# Reverse Traversal

Traversal means

> **Going through every node.**

Reverse means

> **Going from the last node to the first node.**

So reverse traversal simply means

```
last

↓

previous

↓

previous

↓

first
```

---

Example

Forward

```python
a = 2
b = 3

c = a * b
d = c + 5
e = d * 2
```

Order

```
a

b

↓

c

↓

d

↓

e
```

Backward

```
e

↓

d

↓

c

↓

a

b
```

Notice everything is reversed.

---

# Why do we reverse it?

Suppose we want the gradient of

```
c
```

Can we calculate it before we know the gradient of

```
d
```

No.

Because

```
d
```

uses

```
c
```

So first

```
e
```

must give information to

```
d
```

Then

```
d
```

gives information to

```
c
```

Then

```
c
```

gives information to

```
a
```

and

```
b
```

Everything flows backwards.

---

# Real project example

Imagine your `Value` objects are connected like this:

```python
a = Value(2)
b = Value(3)

c = a * b
d = c + 5
e = d * 2
```

When you later write

```python
e.backward()
```

Micrograd **cannot** immediately calculate gradients.

It first needs to know the correct order.

So it builds this list internally:

```text
a
b
c
d
e
```

This is the **topological order** (dependencies first).

Then it reverses that list:

```text
e
d
c
b
a
```

Now it performs the backward pass in exactly that order:

```python
e._backward()

d._backward()

c._backward()

b._backward()

a._backward()
```

This guarantees that when a node computes its gradient, all the nodes that depend on it have already sent their gradient information. If you skipped this ordering or visited nodes randomly, some gradients would still be missing, and the final answers would be incorrect.

---

# Everything in one picture

```text
FORWARD (building the graph)

a      b
 \    /
  \  /
   c
   |
   d
   |
   e


Topological order

a
b
c
d
e


Reverse it

e
d
c
b
a


Backward pass

e._backward()

↓

d._backward()

↓

c._backward()

↓

b._backward()

↓

a._backward()
```

That is the entire purpose of **basic topological order** in a micrograd clone: determine a dependency-respecting order, reverse it, and use that reversed order so gradients can correctly flow from the output back to every input.
