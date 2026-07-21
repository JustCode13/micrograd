Perfect. I'll teach only what you need for **micrograd**. Nothing extra.

---

# 1. Computational Graph

**This topic has almost no math.**

It is simply a way to **remember how a number was created.**

---

# Imagine LEGO Blocks

Suppose you have two LEGO blocks.

```
A = 2
B = 3
```

You join them together by adding them.

```
2 + 3 = 5
```

Normally Python only remembers

```
5
```

It forgets

* where 2 came from
* where 3 came from
* that they were added

A computational graph **doesn't forget**.

It remembers the whole story.

---

# What is a Node?

A **node** is simply **one thing** inside the graph.

That's all.

It can be

* a number
* an operation

Example

```
2
```

is a node.

```
3
```

is another node.

```
+
```

is also represented as a node (the result of the addition).

So if we do

```
2 + 3
```

our graph has

```
2
3
5
```

Three nodes.

---

Think of a node as

> "One box that stores one value."

Example

```
┌───┐
│ 2 │
└───┘
```

That box is a node.

---

# What is an Edge?

An **edge** is just a line.

The line says

> "This box helped create that box."

Example

```
2 ----\
        \
         5
        /
3 ----/
```

The lines are edges.

Nothing more.

---

Think of it like this.

Your parents helped create you.

```
Mom ----\
          \
           You
          /
Dad ----/
```

Those connecting lines are edges.

Exactly the same idea.

---

# Inputs → Operation → Output

Every calculation has

```
Inputs

↓

Operation

↓

Output
```

Example

```
2 + 3
```

Inputs

```
2

3
```

Operation

```
+
```

Output

```
5
```

Graph

```
    2
     \
      \
       +
      /
     /
    3
     \
      \
       5
```

The graph is simply showing

> "How did I get 5?"

Answer

```
2
+
3
```

---

Another example

```
4 × 5
```

Graph

```
4 ----\
        \
         *
        /
5 ----/

        ↓

       20
```

Again,

it remembers

```
4

5

multiply

20
```

---

Now a bigger one.

```
a = 2
b = 3

c = a * b

d = c + 4
```

Graph

```
2 ----\
        *
3 ----/ \
         \
          6
           \
            +
             \
              10
             /
            4
```

Read it slowly.

```
2

and

3

made

6

Then

6

and

4

made

10
```

That's the whole graph.

---

# Why do we even build this graph?

Imagine someone asks you

> "How did you get 10?"

Without the graph you only know

```
10
```

You forgot everything.

With the graph you can answer

```
10 came from

6 + 4

6 came from

2 × 3
```

The graph remembers the complete history.

---

# What is a DAG?

DAG sounds scary.

It isn't.

It means

**Directed Acyclic Graph**

Let's split the words.

---

## Directed

Directed means

The arrows have a direction.

```
2

↓

6

↓

10
```

Information only moves forward.

Not backward.

---

## Acyclic

Cycle means going in circles.

Example

```
A

↓

B

↓

C

↓

A
```

You're back where you started.

That's a cycle.

A computational graph **never** does this.

It always moves forward.

```
2

↓

6

↓

10
```

Finished.

Never comes back.

So it is

```
Acyclic
```

which simply means

> "No circles."

---

## Graph

Boxes connected with lines.

That's all.

---

So

**Directed Acyclic Graph**

means

> Boxes connected with arrows, where the arrows only move forward and never form a loop.

That's the only meaning you need.

---

# Why does the graph store dependencies?

Dependency means

> "I need this before I can make that."

Example

```
2 + 3
```

Can you make

```
5
```

before having

```
2

and

3
```

No.

So

```
5
```

depends on

```
2

3
```

Another example

```
a = 2
b = 3

c = a * b

d = c + 4
```

Can you make

```
d
```

first?

No.

Because you first need

```
c
```

Can you make

```
c
```

first?

No.

Because you first need

```
a

b
```

Everything depends on earlier things.

The graph stores those relationships.

---

# What micrograd actually stores

When you write

```python
c = a * b
```

micrograd remembers

```
a

b

*

↓

c
```

When you later write

```python
d = c + 4
```

it remembers

```
c

4

+

↓

d
```

Eventually, it has the whole story of how every value was created.

Later, during `.backward()`, it walks through this story **backwards** to compute gradients.

That's why building the graph is the very first step. In this lesson, you only need to understand that the graph is a record of **how each value was produced**. Backpropagation comes later.
