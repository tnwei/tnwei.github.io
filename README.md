# My website

My personal site built using `hugo` and the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme.

## Commands cheat-sheet

+ `hugo server -D` from root repo to launch offline dev server at port 1313
+ Create new post with `hugo new -k post example-post.md`. This will create `example-post.md` in the `content/` dir, with a blank header block.

## On LaTeX

+ Latex requires using `math: true` in the front matter YAML block. 
+ Block equations: wrap in `\\[` and `\\]`, or between `$$`
+ Inline equations: wrap in `\\(` and `\\)`, or between `$`
+ Separators are defined in `layouts/partial/math.html`

References:

+ PaperMod docs itself
+ [ How to enable Math Typesetting in PaperMod? #236](https://github.com/adityatelange/hugo-PaperMod/issues/236)


