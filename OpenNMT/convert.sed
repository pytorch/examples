#!/bin/sed
s/^\( \+\)/\1\1/
s/#\([^ $,)]\+\)/len(\1)/
s/cmd:option/parser.add_argument/
s/:/./
s/function \(.*\)$/def \1:/
s/local //
# s/table.insert(\([[:alpha:]]\+\), \([[:alpha:]]\+\))/\1 += [\2]/
s/ then$/:/
s/end//
s/require('\(.*\)')/import \1/
s/cmd = torch.CmdLine()/parser = argparse.ArgumentParser(description='')/
s/\[\[\(.*\)\]\]/"\1"/g
s/ do$/:/
s/--/#/
s/cmd.text(['"]\(.*\)['"])/## \1/
s/~=/!=/g
s/nil/None/g
s/ \.\. / + /g
s/true/True/g
s/false/False/g
s/else /else:/g
s/else$/else:/g
s/elseif/elif/g
