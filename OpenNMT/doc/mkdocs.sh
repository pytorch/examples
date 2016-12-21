rm -fr tree/
mkdir tree/
cp _markdown/onmt/README.md tree/index.md
cp _markdown/onmt/Quickstart.md tree/Quickstart.md


mkdir tree/code/
mkdir tree/code/modules
mkdir tree/code/train
mkdir tree/code/translate
mkdir tree/code/data

cp _markdown/onmt/onmt+modules+*.md tree/code/modules/
mv tree/code/modules/onmt+modules+init.md tree/code/modules/index.md

cp _markdown/onmt/onmt+train+*.md tree/code/train/
mv tree/code/train/onmt+train+init.md tree/code/train/index.md

cp _markdown/onmt/onmt+data+*.md tree/code/data/
mv tree/code/data/onmt+data+init.md tree/code/data/index.md

cp _markdown/onmt/onmt+translate+*.md tree/code/translate/
mv tree/code/translate/onmt+translate+init.md tree/code/translate/index.md

rm -fr tree/details
mkdir tree/details/
cd ../
th preprocess.lua --help | python doc/format.py >> doc/tree/details/preprocess.md
th train.lua --help | python doc/format.py >> doc/tree/details/train.md
th translate.lua --help | python doc/format.py >> doc/tree/details/translate.md
