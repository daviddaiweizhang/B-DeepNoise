tmp=uci_tmp
src="https://github.com/yaringal/DropoutUncertaintyExps"
git clone $src $tmp
mv $tmp/UCI_Datasets data/uci
rm -rf $tmp
