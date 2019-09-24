echo '.DS_Store
.ipynb_checkpoints/' > .gitignore

[ -d app ] || mkdir app/
[ -d app/utils ] || mkdir app/utils
[ -d app/utils/preprocessing ] || mkdir app/utils/preprocessing

[ -d config ] || mkdir config

[ -d data ] || mkdir data/
[ -d data/rawdata ] || mkdir data/rawdata
[ -d data/preprocessed_data ] || mkdir data/preprocessed_data
[ -d data/npy ] || mkdir data/npy
[ -d data/result ] || mkdir data/result

[ -d log ] || mkdir log && echo '*
!.gitignore' > log/.gitignore

[ -d model ] || mkdir model

[ -d tmp ] || mkdir tmp && echo '*
!.gitignore' > tmp/.gitignore