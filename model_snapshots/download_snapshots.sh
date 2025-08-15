model_folder=$1
mkdir -p "$model_folder"
scp -r lrz:~/mapra/project_folder/models/"$model_folder"/snapshots \
          "$model_folder"/

scp -r lrz:~/mapra/project_folder/models/"$model_folder"/references \
          "$model_folder"/
