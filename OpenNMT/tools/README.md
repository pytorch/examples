# Tools

This directory contains additional tools.

## Tokenization/Detokenization

### Tokenization
To tokenize a corpus:

```
th tools/tokenize.lua OPTIONS < file > file.tok
```

where the options are:

* `-mode`: can be `aggressive` or `conservative` (default). In conservative mode, letters, numbers and '_' are kept in sequence, hyphens are accepted as part of tokens. Finally inner characters `[.,]` are also accepted (url, numbers).
* `-sep_annotate`: indicate how to annotate non-separator tokenization - can be `marker` (default), `feature` or `none`:
  * `marker`: when a space is added for tokenization, add reversible separtor marks on one side (preference symbol, number, letter)
  * `feature`: generate separator feature `S` means that the token is preceded by a space, `N` means that there is not space in original corpus
  * `none`: don't annotate
* `-case_feature`: generate case feature - and convert all tokens to lowercase
  * `N`: not defined (for instance tokens without case)
  * `L`: token is lowercased (opennmt)
  * `U`: token is uppercased (OPENNMT)
  * `C`: token is capitalized (Opennmt)
  * `M`: token case is mixed (OpenNMT)

Note:

* `\|` is the feature separator symbol
* `\@` is the separator mark (generated in `-sep_annotate marker` mode)
* character `\` is also used to self-protect `\`: for instance the actual text sequence `\@` is represented by `\\@`

### Detokenization

If you activate `sep_annotate` marker, the tokenization is reversible - just use:

```
th tools/detokenize.lua [-case_feature] < file.tok > file.detok
```

## Release model

After training a model on the GPU, you may want to release it to run on the CPU with the `release_model.lua` script.

```
th tools/release_model.lua -model model.t7 -gpuid 1
```

By default, it will create a `model_release.t7` file. See `th tools/release_model.lua -h` for advanced options.
