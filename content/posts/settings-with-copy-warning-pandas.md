---
title: "SettingsWithCopyWarning in pandas"
date: 2021-12-30T00:40:54+08:00
draft: false
summary: "What it is, why it crops up, and how to get rid of it"
tags: ["python", "pandas"]
---

(tl;dr: jump straight to [Getting rid of SettingWithCopyWarnings]({{< relref "#getting-rid-of-settingwithcopywarnings" >}} "Getting rid of SettingWithCopyWarnings") if you’re here for answers)

If you are a pandas user, chances are you’ve seen the SettingsWithCopyWarnings crop up when you’re assigning values to a `pd.DataFrame` or `pd.Series`.

```python
In [1]: import pandas as pd
   ...:
   ...: df = pd.DataFrame({
   ...:     "A": [1, 2, 3, 4, 5],
   ...:     "B": [6, 7, 8, 9, 10],
   ...:      }, index=range(5)
   ...: )
   ...: dfa = df.loc[3:5]
   ...: dfa["C"] = dfa["B"] * 50

<ipython-input-2-63497d1da3d9>:9: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfa["C"] = dfa["B"] * 50
```

pandas drops a warning here because the value assignment might or might not have worked as expected. 

To be clear, the value assignment did occur all right; the emphasis here is on “expected”. 

```python
In [2]: dfa
Out[2]:
   A   B    C
3  4   9  450
4  5  10  500

In [3]: df
Out[3]:
   A   B
0  1   6
1  2   7
2  3   8
3  4   9
4  5  10
```

Did you expect the contents of `df` to be affected by the value assignment in `dfa`?  pandas has internally consistent (albeit obtuse) rules on whether that happens or not. It’s just that the ambiguity in user expectations present in this situation warrants a warning, so that end users like you and me know where to look when our code misbehaves.

## Chained assignment with views and copies

The act of selecting rows or columns to access from a dataframe or series is called _indexing_. The flexibility of pandas allows for chained indexing, where you can repeatedly index the outcome of a previous indexing operation.

```python
# Select the 2nd to 4th row of data where col A > 3 and col B != 7
df[df["A"] > 3 & df["B"] != 7].iloc[3:5]
```

pandas will then return either a view or a copy of the dataframe. A view (shallow copy) references data from the original dataframe, while a copy (deep copy) is a separate instance of the same data. 

It is difficult to predict which will be returned by the indexing operation, as it depends on the memory layout of the underlying array. How exactly the indexing is chained can lead to different `__getitem__` and `__setitem__` calls being issued under the hood. Reproducing an example below:

```python
# Example borrowed from [1]

dfmi.loc[:, ('one', 'second')] = value
# becomes
dfmi.loc.__setitem__((slice(None), ('one', 'second')), value)

dfmi['one']['second'] = value
# becomes
dfmi.__getitem__('one').__setitem__('second', value)
```

Chain indexing inherently is not a problem, but assigning values using chained indexing, i.e. _chained assignment_, can be. Depending on the situation, chained assignment will either modify the original dataframe directly, or return a modified copy of the original dataframe. This can lead to insidious bugs when it is not obvious that chained indexing has occured. 

Chained indexing can take places across a few lines of code:

```python
# The following doesn't look like chain indexing, does it?
dfa = df.loc[row1:row2, col1:col2]
...
...
dfa[row2] = dfa[row1].apply(fn)
```

If pandas did not raise a warning in this scenario, it would not be obvious that `df` is not modified by the second value assignment. This is why the SettingWithCopyWarning exists.

pandas docs [^1] go into this with more detail. The warning message helpfully links to it, which is great because if you search `pandas settingwithcopywarning` on Google, the docs page is easy to miss! At time of writing, it is the 7th result on the first page of Google, and is crowded out by blogposts and StackOverflow questions.

## Peeking under the hood using the internal API

Chained indexing is a godsend of convenience for selecting the right data, but chained assignment is a minefield for assigning the correct values. The TowardsDataScience article in [^2] has a nice example where inverting the order of chained indexing alone is the difference between whether an assignment to the original dataframe occurs or not:

```python
# Example borrowed from [2]

# This updates `df`
df["product_group"][df["product_group"]=="PG4"] = "PG14"

# This doesn't!
df[df["product_group"]=="PG4"]["product_group"] = "PG14"

# pandas raises warnings for both
# the user needs to confirm the intended outcome
```

From [this StackOverflow post](https://stackoverflow.com/questions/26879073/checking-whether-data-frame-is-copy-or-view-in-pandas), `pd.DataFrame` and `pd.Series` objects have `_is_view` and `_is_copy` attributes as part of their internal API. `_is_view` returns True if the object is a view, and False if the object is not. `_is_copy` stores either a [weak reference](https://docs.python.org/3/library/weakref.html) to the dataframe it is copied from, or `None` if it is not associated to an existing dataframe.

Printing these internal attributes while poking around with chained assignment does reveal some interesting tidbits of info. On one hand, pandas uses `_is_copy` to decide if a SettingWithCopyWarning needs to be raised. On the other hand, modifying a dataframe with `_is_view` = True means that it will affect the original underlying dataframe. 

Before we begin, a disclaimer: internal APIs are not meant to be accessed by the end user and are subject to change, use them at your own risk.

```python
In [4]: pd.__version__
Out[4]: '1.3.3'
```

```python
# Setting up convenience functions
In [5]: def make_clean_df():
   ...:     df = pd.DataFrame({
   ...:         "A": [1, 2, 3, 4, 5],
   ...:         "B": [6, 7, 8, 9, 10],
   ...:         "C": [11, 12, 13, 14, 15],
   ...:          }, index=range(5)
   ...:     )
   ...:     return df

In [6]: def show_attrs(obj):
   ...:     print(f"view: {obj._is_view}, copy: {obj._is_copy}")
```

We’ll start by showing the `_is_view` and `_is_copy` attributes of a few common indexing methods.

```python
In [7]: df = make_clean_df()

In [8]: show_attrs(df.loc[3:5])
   ...: show_attrs(df.iloc[3:5])
   ...: show_attrs(df.loc[3:5, ["A", "B"]])
   ...: show_attrs(df.iloc[3:5, [0, 1]])
   ...: show_attrs(df["A"])
   ...: show_attrs(df.loc[:, "A"])

view: True, copy: <weakref at 0x7f4d648b2590; to 'DataFrame' at 0x7f4d648b54c0>
view: True, copy: <weakref at 0x7f4d648b2590; to 'DataFrame' at 0x7f4d648b54c0>
view: False, copy: None
view: False, copy: <weakref at 0x7f4d648be770; dead>
view: True, copy: None
view: True, copy: None
```

Let’s break this down:

- Both `df.loc[3:5]` and `df.iloc[3:5]` returned views and have references to the original dataframe.
- For `df.loc[3:5, ["A", "B"]]` and `df.iloc[3:5, [0, 1]]` , when the columns are additionally specified on top of the rows, copies of `df` are returned instead. Using `.loc` indexing has no references to the OG dataframe, while using `iloc` indexing results in a reference to a temporary dataframe that has been garbage collected, which is as good as `None` itself. We’ll see if this carries any significance.
- Referring to a column directly using either `df["A"]` or `df.loc[:, "A"]` returns a view, with no reference to the original dataframe. It might have to do with the fact that each dataframe column is actually stored as a pd Series.

What happens if we manually create copies of these indexed dataframes / series?

```python
In [9]: show_attrs(df.loc[3:5].copy())
   ...: show_attrs(df.iloc[3:5].copy())
   ...: show_attrs(df.loc[3:5, ["A", "B"]].copy())
   ...: show_attrs(df.iloc[3:5, [0, 1]].copy())
   ...: show_attrs(df["A"].copy())
   ...: show_attrs(df.loc[:, "A"].copy())

view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
view: False, copy: None
```

Explicitly calling `.copy` returns copies of data that have no reference to the original dataframe / series. Assigning data on these copies will not affect the original dataframe, and thus will not trigger SettingwithCopyWarnings. Given that `df.loc[3:5, ["A", "B"]]` and `df.iloc[3:5, [0, 1]]` above have similar attributes, we can expect that their behaviour under chained assignment should be similar to explicitly created copies.

Next, we’ll try a few chained assignment scenarios.

### Scenario 1: Specific rows indexed using loc

The following three chained assignments raise SettingWithCopyWarnings:

```python
In [10]: df = make_clean_df()
    ...: dfa = df.loc[3:5]
    ...: show_attrs(dfa)

view: True, copy: <weakref at 0x7fba308565e0; to 'DataFrame' at 0x7fba3084eac0>
```

```python
# (1a)
In [11]: dfa[dfa % 2 == 0] = 100

/tmp/ipykernel_34555/3321004726.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfa[dfa % 2 == 0] = 100
/home/tnwei/miniconda3/envs/ml/lib/python3.9/site-packages/pandas/core/
frame.py:3718: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  self._where(-key, value, inplace=True)
```

```python
# (1b)
In [12]: dfa["D"] = dfa["B"] * 10

/tmp/ipykernel_34555/447367411.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfa["D"] = dfa["B"] * 10 # 1b
```

```python
# (1c)
In [13]: dfa["A"][3] = 10

/tmp/ipykernel_34555/1338713145.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfa["A"][3] = 10
```

All of the value assignments took effect on `dfa` itself, but only (1a) and (1c) affected the original dataframe. (1b) did not. 

```python
In [14]: print(dfa)
 
A    B    C     D
3  10    9  100    90
4   5  100   15  1000

In [15]: print(df)

A    B    C
0   1    6   11
1   2    7   12
2   3    8   13
3  10    9  100
4   5  100   15
```

In addition, `dfa` is no longer a view, but a copy of the dataframe!

```python
In [16]: show_attrs(dfa) # view changed to False

view: False, copy: <weakref at 0x7fba308565e0; to 'DataFrame' at 0x7fba3084eac0>
```

What this tells us is that pandas will convert a view to a copy when necessary.  This further shows why figuring out chained assignment is inherently tricky, and is difficult to cater for automatically at the library level. 

### Scenario 2: Specific rows indexed using iloc

This is the same as scenario 1, but using `iloc` instead. 

```python
In [17]: df = make_clean_df()
    ...: dfb = df.iloc[3:5]
    ...: show_attrs(dfb)

view: True, copy: <weakref at 0x7fba30862040; to 'DataFrame' at 0x7fba30868c10>
```

```python
# (1a)
In [18]: dfb[dfb % 2 == 0] = 100

/tmp/ipykernel_34555/734837801.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfb[dfb % 2 == 0] = 100
/home/tnwei/miniconda3/envs/ml/lib/python3.9/site-packages/pandas/core/
frame.py:3718: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  self._where(-key, value, inplace=True)
```

```python
# (1b)
In [19]: dfb["D"] = dfb["B"] * 10

/tmp/ipykernel_34555/4288697762.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfb["D"] = dfb["B"] * 10
```

```python
# (1c)
In [20]: dfb["A"][3] = 10

/tmp/ipykernel_34555/2062795903.py:1: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/
stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  dfb["A"][3] = 10
```

The observed outcome is the same as Scenario 1.

```python
In [21]: print(dfb)

A    B    C     D
3  10    9  100    90
4   5  100   15  1000

In [22]: print(df)

A    B    C
0   1    6   11
1   2    7   12
2   3    8   13
3  10    9  100
4   5  100   15

In [23]: show_attrs(dfb)

view: False, copy: <weakref at 0x7fba30862040; to 'DataFrame' at 0x7fba30868c10>
```

### Scenario 3: Specific rows and columns indexed using loc

Same as Scenario 1, but the columns are specified as well.

```python
In [24]: df = make_clean_df()
    ...: dfc = df.loc[3:5, ["A", "B"]]
    ...: show_attrs(dfc)

view: False, copy: None

In [25]: dfc[dfc % 2 == 0] = 100 # No warnings raised
    ...: dfc["D"] = dfc["B"] * 10
    ...: dfc["A"][3] = 10
```

No warnings raised. All changes took effect on `dfc` without impacting `df`.

```python
In [26]: print(dfc)
 
A    B    D
3  10    9   90
4   5  100 1000

In [27]: print(df)

A    B    C
0   1    6   11
1   2    7   12
2   3    8   13
3  10    9  100
4   5  100   15
```

The chained assignment outcome is different, while the data indexed is the same as in Scenario 1. My guess is that a more complete description of the indexing operation prompted pandas to directly return a copy upfront, instead of a view that is linked to the original dataframe. 

### Scenario 4: Specific rows and columns indexed using iloc

This is similar to Scenario 3, but using iloc instead. Given the past few scenarios, it is no surprise that this scenario had the same outcome as Scenario 3.

```python
In [28]: df = make_clean_df()
    ...: dfd = df.iloc[3:5, [0, 1]]
    ...: show_attrs(dfd)

view: False, copy: <weakref at 0x7fba306f29f0; dead>

In [29]: dfd[dfd % 2 == 0] = 100  # No warnings raised
    ...: dfd["D"] = dfd["B"] * 10
    ...: dfd["A"][3] = 10

In [30]: print(dfd)

A    B     D
3  10    9    90
4   5  100  1000

In [31]: print(df)

A   B   C
0  1   6  11
1  2   7  12
2  3   8  13
3  4   9  14
4  5  10  15
```

In addition, `dfd` discarded the reference to the garbage-collected dataframe at the end of this code.

```python
In [32]: show_attrs(dfd)

view: False, copy: None
```

### Scenario 5: Directly referring to a column of a dataframe

This scenario tests chained assignment on series.

```python
In [33]: df = make_clean_df()
    ...: dfe = df["A"]
    ...: show_attrs(dfe)

view: True, copy: None

In [34]: dfe[1] = 99999 # No warnings raised
    ...: dfe.loc[2:4] = 88888
```

`dfe` remained a view of `df["A"]`all changes effected on `dfe` is reflected in `df["A"]` , which is still part of `df`. It appears that there’s not much to worry about for chained assignment on individual series. 

```python
In [35]: print(dfe)

0        1
1    99999
2    88888
3    88888
4    88888
Name: A, dtype: int64

In [36]: print(df)

A   B   C
0      1   6  11
1  99999   7  12
2  88888   8  13
3  88888   9  14
4  88888  10  15

In [37]: show_attrs(dfe)

view: True, copy: None
```

## Getting rid of SettingWithCopyWarnings

SettingWithCopyWarnings pop up when pandas isn’t sure if you want value assignment to affect the original dataframe or not. Therefore, getting rid of these warnings entails avoiding ambiguity in value assignment. As seen from the code samples above, getting pandas to return copies with no reference to the original dataframe is a clean way to ensure that values will not be written to the original dataframe unintended. 

I’ve found this to be a unifying thread across the solutions I came across when researching this topic. Summarizing them below:

### Disabling the warnings

If you know what you’re doing and your code is behaving as intended, you can choose to suppress the warnings by disabling them [^3]:

```python
# Example borrowed from [^3]

# Disables SettingWithCopyWarning globally
pd.set_option('mode.chained_assignment', None)

# Resets the warning option to default
pd.reset_option('mode.chained_assignment')

# Disables SettingWithCopyWarning locally within a context manager
with pd.option_context('mode.chained_assignment', None):
    # YOUR CODE HERE
```

Alternatively, you can suppress the warnings by setting the dataframe  `_is_copy` attribute to `None` [^3].

```python
# Example modified from [3]
In [38]: df = pd.DataFrame({
    ...:     "A": [1, 2, 3, 4, 5],
    ...:     "B": [6, 7, 8, 9, 10],
    ...:     "C": [11, 12, 13, 14, 15],
    ...: }, index=range(5))
    ...: 
    ...: dfa = df.loc[3:5]
    ...: print(dfa._is_copy)

<weakref at 0x7f4d64792810; to 'DataFrame' at 0x7f4d64784460>

In [39]: dfa._is_copy = None
    ...: dfa["D"] = dfa["B"] * 10  # No warning raised
```

Remember that making the warnings go away doesn’t resolve wonky chained assignment issues. Chained assignment is a minefield where you might or might not step on a landmine. Disabling the warnings is like removing the minefield warning signs. Food for thought.

### Making the warnings not show up in the first place

When you run into a SettingWithCopy warning, take a moment to trace the chained assignment and decide if you want to modify the original dataframe directly, or have values assigned to a copy of the dataframe. 

#### Working on the original dataframe

Use `.loc` indexing to directly assign values to the dataframe. 

```python
# Modified from examples in [2]
In [40]: df = pd.DataFrame({
    ...:     "A": [1, 2, 3, 4, 5],
    ...:     "B": [6, 7, 8, 9, 10],
    ...:     "C": [11, 12, 13, 14, 15],
    ...: }, index=range(5))
    ...:
    ...: df.loc[df["A"] % 2 != 0, "B"] = df.loc[df["A"] % 2 != 0, "B"] + 0.5
    ...: print(df)

   A     B   C
0  1   6.5  11
1  2   7.0  12
2  3   8.5  13
3  4   9.0  14
4  5  10.5  15
```

pandas docs recommend this method for two reasons: 

- using `.loc` is guaranteed to refer to the underlying dataframe it is called on. `.iloc` does not have this property.
- `.loc` indexing replaces what could be chain indexing into a single indexing step. If you refer to the TODO Example above, loc indexing results in a single setitem call

If you are selecting data using conditionals, you can consider returning a mask instead of a copy of the original dataframe. A mask is a boolean series or dataframe which can conveniently be used in `.loc` indexing, as the example below:

```python
# Modified from examples in [5]
In [41]: df = pd.DataFrame({
    ...:     "A": [1, 2, 3, 4, 5],
    ...:     "B": [6, 7, 8, 9, 10],
    ...:     "C": [11, 12, 13, 14, 15],
    ...: }, index=range(5))

In [42]: dfa = (df["A"] <= 3) & (df["C"] == 12)

In [43]: df.loc[dfa, "B"] = 99 # dfa can be fed into the loc index!

In [44]: print(df) # changes took place in the original dataframe
   A   B   C
0  1   6  11
1  2  99  12
2  3   8  13
3  4   9  14
4  5  10  15
```

Working on the original dataframe directly can be tricky if existing indexing logic is complex. In that case, you can always use one of the methods from the next section to return a copy, then assign it back to the original dataframe [^4].

#### Assigning values to an explicit copy of the dataframe

Use `assign`, `where` and `replace`:

```python
In [45]: df = pd.DataFrame({
    ...:     "A": [1, 2, 3, 4, 5],
    ...:     "B": [6, 7, 8, 9, 10],
    ...:     "C": [11, 12, 13, 14, 15],
    ...: }, index=range(5))

# 1. Use the `assign` method to add columns
In [46]: df = df.assign(D=df["C"] * 10)
    ...: df = df.assign(**{"D": df["C"] * 10}) # allows passing variables as names

# 2. Use the `where` method to select values using conditionals and replace them
# Modified from examples in [2]
In [47]: df["B"] = df["B"].where(
    ...:     df["A"] < 2, df["B"] * 10
    ...: )

# 3. Use the `replace` method to select and replace values in the dataframe
# Modified from examples in [2]
In [48]: df = df.replace({"A" : 1}, 100)

In [49]: print(df)
A    B   C    D
0  100    6  11  110
1    2   70  12  120
2    3   80  13  130
3    4   90  14  140
4    5  100  15  150
```

Break down chained assignment steps into single assignments [^5]. 

```python
# Examples borrowed from [4]
# Not these
df["z"][mask] = 0
df.loc[mask]["z"] = 0

# But this
df.loc[mask, "z"] = 0
```

A less elegant but foolproof method is to manually create a copy of the original dataframe and work on it instead [^2]. As long as you don’t introduce additional chained indexing, you will not see the SettingWithCopyWarning.

```python
In [50]: df = pd.DataFrame({
    ...:     "A": [1, 2, 3, 4, 5],
    ...:     "B": [6, 7, 8, 9, 10],
    ...:     "C": [11, 12, 13, 14, 15],
    ...: }, index=range(5))

In [51]: dfa = df.loc[3:5].copy() # Added .copy() here
    ...: dfa.loc[3, "A"] = 10     # causes this line to raise no warning
```

## Redoing some of the examples above without triggering SettingWithCopy warnings

Replace a chained assignment with `where`:

```python
# (i)
df = make_clean_df()
dfa = df.loc[3:5]

# Original that raises warning
# dfa[dfa % 2 == 0] = 100

dfa = dfa.where(dfa % 2 != 0, 100) # df is not affected
```

Replace creating a new column on the indexed dataframe with `assign`

```python
# (ii) 
df = make_clean_df()

# Original that raises warning
# dfa["D"] = dfa["B"] * 10

dfa = dfa.assign(D=dfa["B"]*10) # df is not affected
```

Creating a copy of the dataframe, before assigning values using `.loc` indexing.

```python
# (iii)
df = make_clean_df()

# Original that raises warnings
# dfa = df.loc[3:5]
# dfa["A"][3] = 10

# Create a copy then do loc indexing
dfa = df.loc[3:5].copy()
dfa.loc[3, "A"] = 10
```

Note that directly assigning values to `dfa` using `.loc` indexing will still raise a warning, as it is ambiguous if the assignment to `dfa` should also affect `df`. 

## Truly rooting out SettingWithCopyWarnings

Personally, I am a fan of promoting SettingWithCopyWarnings to SettingWithCopyExceptions for important scripts, using the following code:

```python
pd.set_option('mode.chained_assignment', "raise")
```

Doing this forces chained assignment to be dealt with, rather than allowing warnings to accumulate. 

In my experience, cleaning up notebooks with stderr clogged by SettingWithCopyWarnings is its special kind of zen. I wholeheartedly recommend it.


[^1]: Official `pandas` docs on chained assignment.
[https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy)

[^2]:  TowardsDataScience article that briefly touches on a few ways to deal with the SettingWithCopy warnings.
[https://scribe.rip/@towardsdatascience.com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe15d62de08](https://scribe.rip/@towardsdatascience.com/3-solutions-for-the-setting-with-copy-warning-of-python-pandas-dfe15d62de08)

[^3]:  In-depth article on this topic by DataQuest. Notably, there is a section dedicated to the history of dealing with chained assignment in `pandas`.
[https://www.dataquest.io/blog/settingwithcopywarning/](https://www.dataquest.io/blog/settingwithcopywarning/)

[^4]: StackOverflow post that contains more chained assignment examples. [https://stackoverflow.com/questions/48173980/pandas-knowing-when-an-operation-affects-the-original-dataframe](https://stackoverflow.com/questions/48173980/pandas-knowing-when-an-operation-affects-the-original-dataframe)

[^5]: RealPython article covering this topic. For me, RealPython is a trusted goto reference second to official library docs. This article further goes into depth on the underlying view vs copy mechanisms in pandas, and in numpy, which pandas depends on.
[https://realpython.com/pandas-settingwithcopywarning/](https://realpython.com/pandas-settingwithcopywarning/)
