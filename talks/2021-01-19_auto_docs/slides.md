# Python Autodocs
2021-01-19

----
Outline:

* Autodocs
* Docstrings
* Docs extras

* Autoformatting

---
## Autodocs

----
### Motivation for docs

* Is the world's greatest library really the greatest if no-one knows how to use it?
* Docs help with getting up to speed
* Returning at a later date
* Write better structured code/ideas
----
### The problem?

* No one wants to spend time writing docs
* Let alone maintaining a website etc.

----
### Autodocs

* Little setup to get a website
* Sphinx/MKDocs
* ReST vs Markdown
* VScode support for markdown, extension for ReST
* Themes
----
### Examples

* [Numpy](https://numpy.org/doc/stable/)
* [Pandas](https://github.com/pandas-dev/pandas/tree/master/doc)
* [Requests](https://requests.readthedocs.io/en/master/)
  * [source](https://raw.githubusercontent.com/psf/requests/master/docs/index.rst)

----
### MKDocs

* Install
* Create project
* Build

```
pip install mkdocs
mkdocs new my-project
cd my-project
mkdocs serve
```
----

Quick example

----

### Markdown
[Cheat sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

----

More complete example

----
### Sphinx

* More complete
* Widely adopted in larger projects
* Slightly higher learning curve
* ReST

---
## Docstrings

Example:
```python
def process(x: pd.DataFrame) -> pd.DataFrame:
    """
    This function does something useful to `x`.

    Args:
        x (pd.DataFrame): input dataframe
    
    Returns:
        pd.DataFrame: output dataframe
    """
    y = pd.concat([x, x])
    return y
```

----
### Motivations
* Keeps docs next to the code
* Manged via git
* Helps with IDE autocomplete
* Ties in with `type hinting`
* Python `help` command

----

[Doc string formats](https://realpython.com/documenting-python-code/#google-docstrings-example)

----

VSCode example

* VSCode plugin
* autoDocstring
* Google style

---
### Autodocstring APIs

Built in reference manual with our `autodocs`!

----
### MKDocs plugin

Install:
```
pip install mkdocstrings
```
`mkdocs.yml`:
```
plugins:
  - mkdocstrings
```

----
Add a new page

`reference.md`:
```
# API Reference

::: path.to.a.package.module
```

----

Example...

---

## Extra stuff

Plugins to incorporate more features into docs.
* Mermaid flow diagrams
* Mathjax

----
### Mermaid
Makes nice flowcharts from markup

`mkdocs` extension: `mermaid2`

```
``` mermaid
graph TD
    A[(input_data)] --> B[function]
    B --> C[(output_data)]
```

Example...

----
### Mathjax
Latex rendering in docs

[link](https://squidfunk.github.io/mkdocs-material/reference/mathjax/)

---

## Autoformatting

Controversial topic...

----
Linters give style suggestions, autoformatters don't

Can be applied on save/on git push etc.

----

### [Black autoformatter](https://github.com/psf/black)

See `black_example.py`

---

# Questions & comments?

Slides - written in markdown :P
