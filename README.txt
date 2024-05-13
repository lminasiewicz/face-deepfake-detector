DATASET: https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces
DATASET 2: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
DATASET 3 (mixed with dataset2): https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

INSTRUKCJE:
Jeśli chcesz stworzyć nowy model za pomocą tego kodu, to ściągnij pierwszy powyższy dataset, i w folderze
backend stwórz folder dataset. Do folderu dataset wrzuć wszystkie dane z powyższego linku.
Potem w folderze backend wykonaj sort_dataset.py, oraz define_model.py w tej kolejności.

Dla (o wiele) większego modelu ściągnij dataset drugi, i w folderze backend stwórz folder dataset2. 
Do folderu dataset2 wrzuć wszystkie dane z powyższego linku.
W folderze backend w sort_dataset.py wykonaj funkcję relocate_and_split, z parametrami "dataset2", "dataset2/separated", train_test_split=True
Potem wykonaj define_model2.py, i poczekaj parę godzin.

Jeśli chcesz zmienić parametry swojego modelu, większość z nich jest dostępna w pliku .env.

Jeśli chcesz uruchomić aplikację, to na backendzie uruchom server.py, a na frontendzie
w folderze deepfake-detector-frontend odpal komendę "npm start".
Po chwili powinno odpalić się Web UI. Miłej zabawy!



ENG:
INSTRUCTIONS:
If you want to create a new AI model using thie code, then download the first dataset above, and in the folder labeled
"backend" create a folder named "dataset". Into the "dataset" folder, insert all the data downloaded from the link.
Then, from within the "backend" folder, execute sort_dataset.py and define_model.py, in that order.

For a much bigger model download the second dataset, and in the folder labeled "backend" create a folder labeled "dataset2". 
Into the "dataset2" folder, insert all the data downloaded from the link.
From within the "backend" folder, in sort_dataset.py execute the function relocate_and_split, with parameters "dataset2", "dataset2/separated", and train_test_split=True
Then execute define_model2.py, and wait several hours.

If you want to change the parameters of your model, most of them are available in the .env file.

If you want to launch the app, then on the backend execute server.py, and in a separate terminal, on the frontend
in the folder labeled "deepfake-detector-frontend" execute the command "npm start".
Afte a while, a Web UI should pop up. Have fun!
